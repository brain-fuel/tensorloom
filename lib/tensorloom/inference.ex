defmodule Tensorloom.Inference do
  @moduledoc "Registry-directed type inference and explicit symbolic obligations."
  alias Tensorloom, as: P
  alias Tensorloom.Shape, as: S

  def run(g, bindings) do
    {result, types, guards} = region(g, %{}, bindings)
    {:ok, %{graph: g, result: result, types: types, constraints: Enum.uniq(guards)}}
  end

  defp bind(x, b) when is_binary(x), do: Map.get(b, x, x)
  defp bind(x, _), do: x

  defp region(g, outer, b) do
    env =
      Enum.reduce((g["inputs"] || []) ++ (g["state"] || []), outer, fn x, e ->
        t = x["type"]

        t =
          if t["kind"] == "tensor",
            do: Map.update!(t, "shape", &Enum.map(&1, fn d -> bind(d, b) end)),
            else: t

        Map.put(e, x["id"], Map.get(e, x["id"], t))
      end)

    {env, guards} =
      Enum.reduce(g["ops"], {env, []}, fn op, {env, guards} ->
        args = Enum.map(op["args"] || [], &value(&1, env))

        try do
          t = infer_op(op, args, b)

          relations =
            if Enum.any?(args, &symbolic?/1),
              do: [
                %{
                  "op" => op["id"],
                  "check" => "pytorch_shape_compatibility",
                  "args" => op["args"],
                  "attrs" => op["attrs"] || %{}
                }
              ],
              else: []

          {Map.put(env, op["id"], t), guards ++ relations}
        rescue
          e -> P.error(op["id"], Exception.message(e), "invalid-operation")
        catch
          {:shape, msg} -> P.error(op["id"], msg, "shape-error")
        end
      end)

    {value(g["result"], env), env, guards}
  end

  defp symbolic?(%{"kind" => "tensor", "shape" => s}), do: Enum.any?(s, &(!is_integer(&1)))
  defp symbolic?(_), do: false
  defp value(%{"ref" => r}, e), do: Map.fetch!(e, r)

  defp value(%{"lit" => x}, _),
    do: %{
      "kind" => "scalar",
      "dtype" =>
        cond do
          is_boolean(x) -> "bool"
          is_integer(x) -> "int64"
          true -> "float32"
        end,
      "value" => x
    }

  defp value(%{"tuple" => xs}, e),
    do: %{"kind" => "tuple", "items" => Enum.map(xs, &value(&1, e))}

  defp value(%{"dict" => xs}, e),
    do: %{"kind" => "dict", "items" => Map.new(xs, fn {k, v} -> {k, value(v, e)} end)}

  defp shape(%{"kind" => "tensor", "shape" => s}), do: s
  defp shape(%{"kind" => "scalar"}), do: []
  defp shape(_), do: S.fail("expected tensor or scalar")

  defp tensor(t, s),
    do:
      t
      |> Map.put("kind", "tensor")
      |> Map.put("shape", s)
      |> Map.put(
        "axes",
        if(length(t["axes"] || []) == length(s),
          do: t["axes"] || [],
          else: List.duplicate(nil, length(s))
        )
      )
      |> Map.delete("value")

  defp dim(a, t, key, default) do
    d = Map.get(a, key, default)

    if is_binary(d) do
      Enum.find_index(t["axes"] || [], &(&1 == d)) || S.fail("unknown named axis #{d}")
    else
      d
    end
  end

  defp pair(n) when is_integer(n), do: [n, n]
  defp pair([_, _] = n), do: n

  defp infer_op(op, args, b) do
    a = op["attrs"] || %{}
    name = op["op"]
    t = List.first(args)
    s = if t && t["kind"] in ["tensor", "scalar"], do: shape(t), else: []
    policy = P.registry()[name]["shape"]

    if not is_nil(t) and
         name in [
           "softmax",
           "layer_norm",
           "batch_norm",
           "dropout",
           "attention",
           "conv2d",
           "max_pool2d",
           "avg_pool2d",
           "rnn_cell",
           "gru_cell",
           "lstm_cell",
           "mean",
           "var"
         ] and t["dtype"] not in ["float32", "float64"],
       do: S.fail("#{name} needs a floating tensor")

    if name == "rand_like" and t["dtype"] not in ["float32", "float64"],
      do: S.fail("random generation requires floating dtype")

    if name == "masked_fill" do
      mask = Enum.at(args, 1)
      unless mask["dtype"] == "bool", do: S.fail("masked_fill needs a boolean mask")
      S.equal!(S.broadcast(s, shape(mask)), s)
    end

    if name == "dropout" and length(args) == 2 do
      mask = Enum.at(args, 1)
      unless mask["dtype"] == "bool", do: S.fail("dropout mask must be boolean")
      S.equal!(S.broadcast(s, shape(mask)), s)
    end

    if name == "layer_norm" do
      normalized = a["normalized_shape"] || S.fail("normalized_shape is required")
      if normalized == [], do: S.fail("normalized_shape must not be empty")
      S.equal!(Enum.take(s, -length(normalized)), normalized)
      Enum.each(tl(args), &S.equal!(shape(&1), normalized))
    end

    if name == "batch_norm" do
      if length(s) < 2, do: S.fail("BatchNorm rank must be at least two")
      args |> Enum.drop(1) |> Enum.take(4) |> Enum.each(&S.equal!(shape(&1), [Enum.at(s, 1)]))
      if length(args) == 6, do: S.equal!(shape(Enum.at(args, 5)), [])
    end

    if name == "cast" and a["dtype"] not in ["float32", "float64", "int64", "bool"],
      do: S.fail("unsupported cast dtype")

    case policy do
      "empty_sequence" ->
        tensor(t, List.insert_at(s, Map.get(a, "dim", 0), 0))

      "training" ->
        P.scalar()

      "bool_unary" ->
        Map.put(t, "dtype", "bool")

      "count" ->
        P.scalar(t["dtype"])

      "rate" ->
        P.scalar("float64")

      "grouped_matmul" ->
        [n, ck, l] = s
        [oc, ci, kh, kw] = shape(Enum.at(args, 1))
        groups = Map.get(a, "groups", 1)
        S.equal!(ck, ci * kh * kw * groups)
        tensor(t, [n, oc, l])

      "pool_values" ->
        out = pool_shape(s, a)
        [kh, kw] = pair(a["kernel_size"])
        result = tensor(t, out ++ [kh * kw])
        if name == "pool_indices", do: Map.put(result, "dtype", "int64"), else: result

      "pool_divisor" ->
        out = pool_shape(s, a)
        tensor(t, Enum.take(out, -2))

      "reduce_max" ->
        out = tensor(t, S.reduce(s, Map.get(a, "dim", -1)))
        %{"kind" => "tuple", "items" => [out, Map.put(out, "dtype", "int64")]}

      "gather" ->
        tensor(t, shape(Enum.at(args, 1)))

      "unary" ->
        if name != "identity", do: shape(t)

        if name == "dropout" and (Map.get(a, "p", 0.5) < 0 or Map.get(a, "p", 0.5) > 1),
          do: S.fail("dropout probability")

        cond do
          name in ["exp", "log", "sqrt", "sigmoid", "tanh"] and t["dtype"] in ["int64", "bool"] ->
            Map.put(t, "dtype", "float32")

          name in ["neg", "abs", "relu"] and t["dtype"] == "bool" ->
            S.fail("#{name} does not support Bool")

          true ->
            t
        end

      "broadcast" ->
        out = S.broadcast(s, shape(Enum.at(args, 1)))
        base = Enum.find(args, &(&1["kind"] == "tensor")) || t

        if Enum.all?(args, &(&1["kind"] == "tensor")) and
             Enum.any?(args, &(&1["dtype"] != t["dtype"])),
           do: S.fail("mixed tensor dtypes need explicit cast")

        result =
          if Enum.all?(args, &(&1["kind"] == "scalar")),
            do: Map.delete(base, "value"),
            else: tensor(base, out)

        cond do
          name in ["eq", "lt", "le", "gt", "ge", "logical_and"] ->
            Map.put(result, "dtype", "bool")

          name == "div" and result["dtype"] in ["int64", "bool"] ->
            Map.put(result, "dtype", "float32")

          result["dtype"] in ["int64", "bool"] and
              Enum.any?(args, &(&1["kind"] == "scalar" and &1["dtype"] in ["float32", "float64"])) ->
            Map.put(result, "dtype", "float32")

          result["dtype"] == "bool" and
              Enum.any?(args, &(&1["kind"] == "scalar" and &1["dtype"] == "int64")) ->
            Map.put(result, "dtype", "int64")

          true ->
            result
        end

      "matmul" ->
        if t["dtype"] != Enum.at(args, 1)["dtype"], do: S.fail("matmul dtype mismatch")
        right = shape(Enum.at(args, 1))
        out = S.matmul(s, right)

        if t["dtype"] == "bool" and
             ((length(s) == 1 and length(right) == 1) or
                (Enum.all?(s ++ right, &is_integer/1) and !(0 in out or List.last(s) == 0))),
           do: S.fail("nonempty Bool matmul is unsupported on CPU")

        tensor(t, out)

      "reshape" ->
        tensor(t, S.reshape(s, Enum.map(a["shape"], &bind(&1, b))))

      "transpose" ->
        d0 = dim(a, t, "dim0", 0)
        d1 = dim(a, t, "dim1", 1)

        tensor(t, S.transpose(s, d0, d1))
        |> Map.put("axes", S.transpose(t["axes"] || List.duplicate(nil, length(s)), d0, d1))

      "permute" ->
        ds = Enum.map(a["dims"], &S.axis(&1, s))

        unless Enum.sort(ds) == Enum.to_list(0..(length(s) - 1)//1),
          do: S.fail("invalid permutation")

        tensor(t, Enum.map(ds, &Enum.at(s, &1)))
        |> Map.put(
          "axes",
          Enum.map(ds, &Enum.at(t["axes"] || List.duplicate(nil, length(s)), &1))
        )

      "index" ->
        case name do
          "getitem" ->
            items = t["items"] || S.fail("getitem expects structured input")

            if(is_list(items), do: Enum.at(items, a["index"]), else: items[a["index"]]) ||
              S.fail("invalid structured index")

          "slice" ->
            tensor(
              t,
              S.slice(s, dim(a, t, "dim", 0), a["start"], a["stop"], Map.get(a, "step", 1))
            )

          "select" ->
            d = S.axis(dim(a, t, "dim", 0), s)
            n = Enum.at(s, d)
            i = a["index"]
            if is_integer(n) and (i < -n or i >= n), do: S.fail("index out of bounds")
            tensor(t, List.delete_at(s, d))
        end

      "cat" ->
        if name == "stack" do
          Enum.each(args, &if(shape(&1) != s, do: S.fail("stack shape mismatch")))
          d = S.axis(dim(a, t, "dim", 0), s ++ [1])
          tensor(t, List.insert_at(s, d, length(args)))
        else
          tensor(t, S.cat(Enum.map(args, &shape/1), dim(a, t, "dim", 0)))
        end

      "reduce" ->
        dims = Map.get(a, "dim")
        dims = if is_binary(dims), do: dim(a, t, "dim", 0), else: dims

        if name in ["amax", "amin"] and s != [] do
          ds =
            if is_nil(dims) or dims == [],
              do: s,
              else: Enum.map(List.wrap(dims), &Enum.at(s, S.axis(&1, s)))

          if 0 in ds, do: S.fail("empty reduction has no identity")
        end

        out = tensor(t, S.reduce(s, dims, Map.get(a, "keepdim", false)))

        cond do
          name in ["sum", "prod"] and t["dtype"] == "bool" -> Map.put(out, "dtype", "int64")
          name == "all" -> Map.put(out, "dtype", "bool")
          true -> out
        end

      "linear" ->
        w = shape(Enum.at(args, 1))
        if length(w) != 2 or s == [], do: S.fail("linear rank")
        S.equal!(List.last(s), List.last(w))
        if length(args) == 3, do: S.equal!(shape(Enum.at(args, 2)), [hd(w)])
        tensor(t, List.replace_at(s, -1, hd(w)))

      "attention" ->
        [q, k, v | _] = Enum.map(args, &shape/1)
        scores = S.matmul(q, S.transpose(k, -2, -1))
        if length(args) == 4, do: S.broadcast(scores, shape(Enum.at(args, 3)))
        tensor(t, S.matmul(scores, v))

      "embedding" ->
        w = shape(Enum.at(args, 1))

        if length(w) != 2 or t["dtype"] != "int64",
          do: S.fail("embedding expects integer indices and rank-two weights")

        tensor(Enum.at(args, 1), s ++ [List.last(w)])

      "conv2d" ->
        w = shape(Enum.at(args, 1))
        if length(s) != 4 or length(w) != 4, do: S.fail("conv2d expects NCHW/OIHW")
        [n, c, h, width] = s
        [o, ci, kh, kw] = w
        groups = Map.get(a, "groups", 1)
        unless is_integer(groups) and groups > 0, do: S.fail("invalid convolution groups")
        S.equal!(c, ci * groups)
        if rem(o, groups) != 0, do: S.fail("output channels/groups")
        [sh, sw] = pair(Map.get(a, "stride", 1))
        [ph, pw] = pair(Map.get(a, "padding", 0))
        [dh, dw] = pair(Map.get(a, "dilation", 1))
        tensor(t, [n, o, S.window(h, kh, sh, ph, dh), S.window(width, kw, sw, pw, dw)])

      "pool" ->
        out = pool_shape(s, a)
        result = tensor(t, out)

        if a["return_indices"],
          do: %{"kind" => "tuple", "items" => [result, Map.put(result, "dtype", "int64")]},
          else: result

      "recurrent" ->
        h = Enum.at(args, 1)
        hs = shape(h)
        hdim = List.last(hs)

        if length(s) not in [1, 2] or length(hs) != length(s),
          do: S.fail("recurrent input/state rank mismatch")

        S.equal!(Enum.drop(s, -1), Enum.drop(hs, -1))
        offset = if name == "lstm_cell", do: 3, else: 2
        wi = shape(Enum.at(args, offset))
        wh = shape(Enum.at(args, offset + 1))

        gates =
          case name do
            "rnn_cell" -> 1
            "gru_cell" -> 3
            _ -> 4
          end

        S.equal!(wi, [gates * hdim, List.last(s)])
        S.equal!(wh, [gates * hdim, hdim])
        if name == "lstm_cell", do: S.equal!(shape(Enum.at(args, 2)), hs)
        Enum.drop(args, offset + 2) |> Enum.each(&S.equal!(shape(&1), [gates * hdim]))

        if name == "lstm_cell",
          do: %{"kind" => "tuple", "items" => [h, Enum.at(args, 2)]},
          else: h

      "where" ->
        [condition, x, y] = args
        unless condition["dtype"] == "bool", do: S.fail("where needs a boolean condition")
        tensor(x, S.broadcast(s, S.broadcast(shape(x), shape(y))))

      "pad" ->
        pads = a["pad"]
        if rem(length(pads), 2) != 0 or length(pads) > length(s) * 2, do: S.fail("padding rank")

        out =
          Enum.chunk_every(pads, 2)
          |> Enum.with_index()
          |> Enum.reduce(s, fn {[l, r], i}, out ->
            List.update_at(out, length(s) - 1 - i, &S.arith("add", &1, l + r))
          end)

        if Enum.any?(out, &(is_integer(&1) and &1 < 0)), do: S.fail("negative padded dimension")
        tensor(t, out)

      "unfold" ->
        [n, c, h, w] = s
        [kh, kw] = pair(a["kernel_size"])
        [sh, sw] = pair(Map.get(a, "stride", 1))
        [ph, pw] = pair(Map.get(a, "padding", 0))
        [dh, dw] = pair(Map.get(a, "dilation", 1))
        tensor(t, [n, c * kh * kw, S.window(h, kh, sh, ph, dh) * S.window(w, kw, sw, pw, dw)])

      "cast" ->
        Map.put(t, "dtype", a["dtype"])

      "copy" ->
        S.equal!(s, shape(Enum.at(args, 1)))
        t

      "structure" ->
        if name == "tuple",
          do: %{"kind" => "tuple", "items" => args},
          else: %{"kind" => "dict", "items" => Map.new(Enum.zip(a["keys"], args))}

      "region" ->
        rs = op["regions"]

        run = fn r, values ->
          if length(r["inputs"]) != length(values), do: S.fail("region argument count")

          captures =
            Map.new(Enum.zip(r["inputs"], values), fn {x, v} ->
              if x["type"] != v and !compatible?(x["type"], v),
                do: S.fail("region input type mismatch")

              {x["id"], v}
            end)

          {result, _, _} = region(r, captures, b)
          result
        end

        case name do
          "call" ->
            run.(rs["body"], args)

          "branch" ->
            unless t["dtype"] == "bool" and s == [],
              do: S.fail("branch predicate must be boolean scalar")

            left = run.(rs["true"], tl(args))
            right = run.(rs["false"], tl(args))
            unless compatible?(left, right), do: S.fail("incompatible branch results")
            left

          "loop" ->
            if length(args) != 1, do: S.fail("loop carries one structured value")
            out = run.(rs["body"], args)
            unless compatible?(out, t), do: S.fail("loop changes carried type")

            if rs["condition"] do
              cond_type = run.(rs["condition"], args)

              unless cond_type["dtype"] == "bool" and shape(cond_type) == [],
                do: S.fail("loop condition must be boolean scalar")
            end

            out
        end
    end
  end

  defp pool_shape(s, a) do
    if length(s) not in [3, 4], do: S.fail("pool rank")
    [kh, kw] = pair(a["kernel_size"])
    [sh, sw] = pair(Map.get(a, "stride", a["kernel_size"]))
    [ph, pw] = pair(Map.get(a, "padding", 0))
    [dh, dw] = pair(Map.get(a, "dilation", 1))
    if ph > div(kh, 2) or pw > div(kw, 2), do: S.fail("pool padding exceeds half kernel")

    Enum.drop(s, -2) ++
      [
        S.window(Enum.at(s, -2), kh, sh, ph, dh, Map.get(a, "ceil_mode", false)),
        S.window(List.last(s), kw, sw, pw, dw, Map.get(a, "ceil_mode", false))
      ]
  end

  defp compatible?(a, b),
    do: Map.drop(a, ["axes", "value", "storage"]) == Map.drop(b, ["axes", "value", "storage"])
end
