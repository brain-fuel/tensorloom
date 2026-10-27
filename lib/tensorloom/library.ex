defmodule Tensorloom.Library do
  @moduledoc "Inspectable library recipes over arithmetic, indexing, windows and effects (ref:DEC-library)."
  alias Tensorloom, as: P

  def lower(op, types, prefix) do
    id = op["id"]
    name = op["op"]
    args = op["args"]
    a = op["attrs"] || %{}
    get_type = fn %{"ref" => r} -> Map.fetch!(types, r) end
    shape = fn x -> get_type.(x)["shape"] end
    node = fn key, n, xs, attrs -> P.op(prefix <> key, n, xs, attrs) end
    r = fn key -> P.ref(prefix <> key) end

    case name do
      "linear" ->
        [x, w | bias] = args

        [
          node.("wt", "transpose", [w], %{"dim0" => -2, "dim1" => -1}),
          node.("mm", "matmul", [x, r.("wt")], %{})
        ] ++
          [P.op(id, if(bias == [], do: "identity", else: "add"), [r.("mm") | bias])]

      "softmax" ->
        [x] = args
        d = Map.get(a, "dim", -1)

        [
          node.("max", "amax_or_zero", [x], %{"dim" => d, "keepdim" => true}),
          node.("shift", "sub", [x, r.("max")], %{}),
          node.("exp", "exp", [r.("shift")], %{}),
          node.("sum", "sum", [r.("exp")], %{"dim" => d, "keepdim" => true}),
          P.op(id, "div", [r.("exp"), r.("sum")])
        ]

      "embedding" ->
        [P.op(id, "index_rows", args, a)]

      "layer_norm" ->
        [x | rest] = args
        rank = length(shape.(x))
        nr = length(a["normalized_shape"])
        dims = Enum.to_list((rank - nr)..(rank - 1))
        stats = normalization(node, r, x, dims, Map.get(a, "eps", 1.0e-5))
        affine(stats, node, r, id, r.("norm"), rest)

      "attention" ->
        [q, k, v | mask] = args
        scale = Map.get(a, "scale") || :math.pow(List.last(shape.(q)), -0.5)

        if a["enable_gqa"],
          do: P.error(id, "expanded GQA is unsupported", "unsupported-operation")

        nodes = [
          node.("kt", "transpose", [k], %{"dim0" => -2, "dim1" => -1}),
          node.("scores", "matmul", [q, r.("kt")], %{}),
          node.("scaled", "mul", [r.("scores"), P.literal(scale)], %{})
        ]

        {nodes, scores} =
          if a["is_causal"] do
            {nodes ++
               [
                 node.("causal", "causal_mask", [r.("scaled")], %{}),
                 node.("causal_not", "logical_not", [r.("causal")], %{}),
                 node.("causal_scores", "masked_fill", [r.("scaled"), r.("causal_not")], %{
                   "value" => "-inf"
                 })
               ], r.("causal_scores")}
          else
            {nodes, r.("scaled")}
          end

        {nodes, scores} =
          if mask != [] do
            if get_type.(hd(mask))["dtype"] == "bool" do
              {nodes ++
                 [
                   node.("mask_not", "logical_not", mask, %{}),
                   node.("masked", "masked_fill", [scores, r.("mask_not")], %{"value" => "-inf"})
                 ], r.("masked")}
            else
              {nodes ++ [node.("masked", "add", [scores, hd(mask)], %{})], r.("masked")}
            end
          else
            {nodes, scores}
          end

        nodes ++
          [
            node.("infinite", "isneginf", [scores], %{}),
            node.("empty", "all", [r.("infinite")], %{"dim" => -1, "keepdim" => true}),
            node.("safe_scores", "masked_fill", [scores, r.("empty")], %{"value" => 0}),
            node.("prob", "softmax", [r.("safe_scores")], %{"dim" => -1}),
            node.("safe_prob", "masked_fill", [r.("prob"), r.("empty")], %{"value" => 0}),
            node.("drop", "dropout", [r.("safe_prob")], %{
              "p" => Map.get(a, "dropout_p", 0),
              "training" => true
            }),
            P.op(id, "matmul", [r.("drop"), v])
          ]

      "conv2d" ->
        [x, w | bias] = args
        [_o, _c, kh, kw] = shape.(w)

        window =
          Map.take(a, ["stride", "padding", "dilation"]) |> Map.put("kernel_size", [kh, kw])

        out = types[id]["shape"]

        nodes = [
          node.("windows", "unfold", [x], window),
          node.("contract", "grouped_matmul", [r.("windows"), w], %{
            "groups" => Map.get(a, "groups", 1)
          }),
          node.("spatial", "reshape", [r.("contract")], %{"shape" => out})
        ]

        if bias == [],
          do: nodes ++ [P.op(id, "identity", [r.("spatial")])],
          else:
            nodes ++
              [
                node.("bias", "reshape", bias, %{"shape" => [1, hd(shape.(w)), 1, 1]}),
                P.op(id, "add", [r.("spatial"), r.("bias")])
              ]

      n when n in ["max_pool2d", "avg_pool2d"] ->
        [x] = args
        attrs = Map.put(a, "kind", n)
        values = node.("windows", "pool_values", [x], attrs)

        if n == "avg_pool2d" do
          [
            values,
            node.("sum", "sum", [r.("windows")], %{"dim" => -1}),
            node.("divisor", "pool_divisor", [x], attrs),
            P.op(id, "div", [r.("sum"), r.("divisor")])
          ]
        else
          nodes = [
            values,
            node.("max", "reduce_max", [r.("windows")], %{"dim" => -1}),
            node.("values", "getitem", [r.("max")], %{"index" => 0})
          ]

          if a["return_indices"] do
            nodes ++
              [
                node.("choice", "getitem", [r.("max")], %{"index" => 1}),
                node.("indices", "pool_indices", [x], attrs),
                node.("selected", "gather", [r.("indices"), r.("choice")], %{"dim" => -1}),
                P.op(id, "tuple", [r.("values"), r.("selected")])
              ]
          else
            nodes ++ [P.op(id, "identity", [r.("values")])]
          end
        end

      n when n in ["rnn_cell", "gru_cell", "lstm_cell"] ->
        {x, h, c, wi, wh, bs} =
          if n == "lstm_cell" do
            [x, h, c, wi, wh | bs] = args
            {x, h, c, wi, wh, bs}
          else
            [x, h, wi, wh | bs] = args
            {x, h, nil, wi, wh, bs}
          end

        xi = [x, wi] ++ Enum.take(bs, 1)
        hh = [h, wh] ++ Enum.drop(bs, 1)
        nodes = [node.("xi", "linear", xi, %{}), node.("hh", "linear", hh, %{})]

        if n == "rnn_cell" do
          nodes ++
            [
              node.("sum", "add", [r.("xi"), r.("hh")], %{}),
              P.op(id, if(a["nonlinearity"] == "relu", do: "relu", else: "tanh"), [r.("sum")])
            ]
        else
          hidden = List.last(shape.(h))

          slices = fn input, labels ->
            Enum.with_index(labels)
            |> Enum.map(fn {key, i} ->
              node.(key, "slice", [input], %{
                "dim" => -1,
                "start" => i * hidden,
                "stop" => (i + 1) * hidden
              })
            end)
          end

          if n == "lstm_cell" do
            nodes ++
              [node.("gates", "add", [r.("xi"), r.("hh")], %{})] ++
              slices.(r.("gates"), ["i0", "f0", "g0", "o0"]) ++
              [
                node.("i", "sigmoid", [r.("i0")], %{}),
                node.("f", "sigmoid", [r.("f0")], %{}),
                node.("g", "tanh", [r.("g0")], %{}),
                node.("o", "sigmoid", [r.("o0")], %{}),
                node.("fc", "mul", [r.("f"), c], %{}),
                node.("ig", "mul", [r.("i"), r.("g")], %{}),
                node.("c", "add", [r.("fc"), r.("ig")], %{}),
                node.("tc", "tanh", [r.("c")], %{}),
                node.("h", "mul", [r.("o"), r.("tc")], %{}),
                P.op(id, "tuple", [r.("h"), r.("c")])
              ]
          else
            nodes ++
              slices.(r.("xi"), ["ir", "iz", "in"]) ++
              slices.(r.("hh"), ["hr", "hz", "hn"]) ++
              [
                node.("rs", "add", [r.("ir"), r.("hr")], %{}),
                node.("zs", "add", [r.("iz"), r.("hz")], %{}),
                node.("r", "sigmoid", [r.("rs")], %{}),
                node.("z", "sigmoid", [r.("zs")], %{}),
                node.("rh", "mul", [r.("r"), r.("hn")], %{}),
                node.("ns", "add", [r.("in"), r.("rh")], %{}),
                node.("n", "tanh", [r.("ns")], %{}),
                node.("one_z", "sub", [P.literal(1), r.("z")], %{}),
                node.("new", "mul", [r.("one_z"), r.("n")], %{}),
                node.("old", "mul", [r.("z"), h], %{}),
                P.op(id, "add", [r.("new"), r.("old")])
              ]
          end
        end

      "batch_norm" ->
        [x, _rm, _rv | tail] = args
        input_shape = shape.(x)
        rank = length(input_shape)
        channels = Enum.at(input_shape, 1)
        if rank < 2, do: P.error(id, "BatchNorm needs rank >= 2", "shape-error")
        dims = [0] ++ Enum.to_list(2..(rank - 1)//1)
        view = [1, channels] ++ List.duplicate(1, rank - 2)
        keys = ["x", "rm", "rv"] ++ Enum.take(["weight", "bias", "counter"], length(tail))

        parameters =
          Enum.zip(keys, args)
          |> Enum.map(fn {key, arg} ->
            P.input(
              key,
              if(key in ["rm", "rv", "counter"],
                do: Map.put(get_type.(arg), "storage", "buffer"),
                else: get_type.(arg)
              )
            )
          end)

        bn = fn key, n, xs, attrs -> P.op(key, n, xs, attrs) end
        br = fn key -> P.ref(key) end
        eps = Map.get(a, "eps", 1.0e-5)
        rate = Map.get(a, "momentum", 0.1)

        affine_nodes =
          Enum.zip(Enum.take(["weight", "bias"], min(length(tail), 2)), Enum.take(tail, 2))
          |> Enum.map(fn {key, _} ->
            bn.(key <> "_view", "reshape", [br.(key)], %{"shape" => view})
          end)

        wb = Enum.take([br.("weight_view"), br.("bias_view")], min(length(tail), 2))

        training_nodes =
          if(length(tail) == 3,
            do: [
              bn.("next_counter", "add", [br.("counter"), P.literal(1)], %{}),
              bn.("write_counter", "copy_buffer", [br.("counter"), br.("next_counter")], %{})
            ],
            else: []
          ) ++
            [bn.("valid", "require_batch", [br.("x")], %{})] ++
            normalization(bn, br, br.("valid"), dims, eps) ++ affine_nodes

        training_nodes = affine(training_nodes, bn, br, "output", br.("norm"), wb)

        {training_nodes, rate_ref} =
          if is_nil(rate) do
            if length(tail) != 3, do: P.error(id, "cumulative BatchNorm requires counter")

            {training_nodes ++ [bn.("rate", "reciprocal_counter", [br.("write_counter")], %{})],
             br.("rate")}
          else
            {training_nodes, P.literal(rate)}
          end

        training_nodes =
          training_nodes ++
            [
              bn.("mean_flat", "reshape", [br.("mean")], %{"shape" => [channels]}),
              bn.("var_flat", "reshape", [br.("variance")], %{"shape" => [channels]}),
              bn.("count", "batch_count", [br.("x")], %{}),
              bn.("count_minus", "sub", [br.("count"), P.literal(1)], %{}),
              bn.("correction", "div", [br.("count"), br.("count_minus")], %{}),
              bn.("unbiased", "mul", [br.("var_flat"), br.("correction")], %{})
            ]

        training_nodes =
          Enum.reduce([{"rm", "mean_flat"}, {"rv", "unbiased"}], training_nodes, fn {buffer, stat},
                                                                                    nodes ->
            nodes ++
              [
                bn.(buffer <> "_delta", "sub", [br.(stat), br.(buffer)], %{}),
                bn.(buffer <> "_update", "mul", [br.(buffer <> "_delta"), rate_ref], %{}),
                bn.(buffer <> "_next", "add", [br.(buffer), br.(buffer <> "_update")], %{}),
                bn.(buffer <> "_write", "copy_buffer", [br.(buffer), br.(buffer <> "_next")], %{})
              ]
          end)

        eval_nodes =
          [
            bn.("mean", "reshape", [br.("rm")], %{"shape" => view}),
            bn.("variance", "reshape", [br.("rv")], %{"shape" => view}),
            bn.("center", "sub", [br.("x"), br.("mean")], %{}),
            bn.("epsilon", "add", [br.("variance"), P.literal(eps)], %{}),
            bn.("denom", "sqrt", [br.("epsilon")], %{}),
            bn.("norm", "div", [br.("center"), br.("denom")], %{})
          ] ++ affine_nodes

        eval_nodes = affine(eval_nodes, bn, br, "output", br.("norm"), wb)
        true_region = P.graph(parameters, training_nodes, br.("output"))
        false_region = P.graph(parameters, eval_nodes, br.("output"))

        predicate =
          if Map.has_key?(a, "training"), do: P.literal(a["training"]), else: r.("training")

        if(Map.has_key?(a, "training"), do: [], else: [node.("training", "training", [], %{})]) ++
          [
            P.op(id, "branch", [predicate | args], %{}, %{
              "true" => true_region,
              "false" => false_region
            })
          ]

      "dropout" ->
        [x | mask] = args
        p = Map.get(a, "p", 0.5)

        if p == 0 or a["training"] == false do
          [P.op(id, "identity", [x])]
        else
          input_type = get_type.(x)

          body = fn nodes, result ->
            P.graph(
              [P.input("x", input_type)] ++
                if(mask == [], do: [], else: [P.input("mask", get_type.(hd(mask)))]),
              nodes,
              result
            )
          end

          active =
            cond do
              p == 1 ->
                body.([P.op("zero", "mul", [P.ref("x"), P.literal(0)])], P.ref("zero"))

              true ->
                mask_nodes =
                  if mask == [],
                    do: [
                      P.op("random", "rand_like", [P.ref("x")]),
                      P.op("mask", "ge", [P.ref("random"), P.literal(p)])
                    ],
                    else: []

                body.(
                  mask_nodes ++
                    [
                      P.op("cast", "cast", [P.ref("mask")], %{"dtype" => input_type["dtype"]}),
                      P.op("product", "mul", [P.ref("x"), P.ref("cast")]),
                      P.op("scale", "div", [P.ref("product"), P.literal(1 - p)])
                    ],
                  P.ref("scale")
                )
            end

          inactive = body.([], P.ref("x"))

          predicate =
            if Map.has_key?(a, "training"), do: P.literal(a["training"]), else: r.("training")

          if(Map.has_key?(a, "training"), do: [], else: [node.("training", "training", [], %{})]) ++
            [
              P.op(id, "branch", [predicate | args], %{}, %{"true" => active, "false" => inactive})
            ]
        end

      _ ->
        nil
    end
  end

  defp normalization(node, r, x, dims, eps) do
    attrs = %{"dim" => dims, "keepdim" => true}

    [
      node.("mean", "mean", [x], attrs),
      node.("center", "sub", [x, r.("mean")], %{}),
      node.("square", "mul", [r.("center"), r.("center")], %{}),
      node.("variance", "mean", [r.("square")], attrs),
      node.("epsilon", "add", [r.("variance"), P.literal(eps)], %{}),
      node.("denom", "sqrt", [r.("epsilon")], %{}),
      node.("norm", "div", [r.("center"), r.("denom")], %{})
    ]
  end

  defp affine(nodes, _node, _r, id, x, []), do: nodes ++ [P.op(id, "identity", [x])]
  defp affine(nodes, _node, _r, id, x, [w]), do: nodes ++ [P.op(id, "mul", [x, w])]

  defp affine(nodes, node, r, id, x, [w, b]),
    do: nodes ++ [node.("affine", "mul", [x, w], %{}), P.op(id, "add", [r.("affine"), b])]
end
