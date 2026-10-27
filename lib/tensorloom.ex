defmodule Tensorloom do
  @moduledoc """
  Ordered typed graph compiler (ref:DEC-graph, ref:DEC-shapes).

  Values use JSON maps. A reference is `%{"ref" => id}` and a literal is
  `%{"lit" => value}`. Regions receive all dependencies as explicit arguments.
  Every public transformation validates its input and output.
  """
  alias Tensorloom.{Inference, Rewrite}

  @external_resource Path.expand("../priv/operations.json", __DIR__)
  @operations Tensorloom.JSON.decode(File.read!(@external_resource))
  def registry, do: @operations

  def tensor(shape, opts \\ []),
    do:
      Map.merge(
        %{
          "kind" => "tensor",
          "shape" => shape,
          "dtype" => "float32",
          "device" => "cpu",
          "layout" => "strided",
          "axes" => List.duplicate(nil, length(shape))
        },
        Map.new(opts, fn {k, v} -> {to_string(k), v} end)
      )

  def scalar(dtype \\ "bool"), do: %{"kind" => "scalar", "dtype" => dtype}
  def ref(id), do: %{"ref" => id}
  def literal(v), do: %{"lit" => v}

  def graph(inputs, ops, result, state \\ []),
    do: %{"version" => 1, "inputs" => inputs, "ops" => ops, "result" => result, "state" => state}

  def input(id, type), do: %{"id" => id, "type" => type}

  def op(id, name, args, attrs \\ %{}, regions \\ %{}),
    do: %{"id" => id, "op" => name, "args" => args, "attrs" => attrs, "regions" => regions}

  def validate(g) do
    protect(fn ->
      unless is_map(g) and g["version"] == 1, do: error("graph", "unsupported graph version")
      validate_region(g, %{}, true)
      routers = router_ids(g)

      unless Enum.uniq(routers) == routers,
        do: error("graph", "router IDs must be unambiguous across regions")

      Tensorloom.Inference.run(g, %{})
      {:ok, g}
    end)
  end

  defp router_ids(g),
    do:
      Enum.flat_map(g["ops"], fn op ->
        if(op["op"] == "branch", do: [op["id"]], else: []) ++
          Enum.flat_map(op["regions"] || %{}, fn {_, r} -> router_ids(r) end)
      end)

  defp validate_region(g, inherited, root) do
    unless is_list(g["inputs"]) and is_list(g["ops"]),
      do: error("graph", "inputs and ops must be lists")

    env =
      Enum.reduce(g["inputs"], %{}, fn x, env ->
        type!(x["type"], x["id"])
        put!(env, x["id"], Map.get(inherited, x["id"], %{storage: nil}))
      end)

    env =
      if root do
        Enum.reduce(g["state"] || [], env, fn x, env ->
          type!(x["type"], x["id"])

          if x["kind"] == "parameter" and x["type"]["dtype"] not in ["float32", "float64"],
            do: error(x["id"], "parameters must be floating point")

          unless x["kind"] in ["parameter", "constant", "buffer"],
            do: error(x["id"], "invalid state kind")

          if x["alias"] do
            target = Enum.find(g["state"], &(&1["id"] == x["alias"]))

            unless target && !target["alias"] && target["kind"] == x["kind"] &&
                     target["type"] == x["type"],
                   do: error(x["id"], "invalid state alias")
          else
            unless Map.has_key?(x, "data"), do: error(x["id"], "state needs data or alias")

            unless x["type"]["kind"] == "tensor" and Enum.all?(x["type"]["shape"], &is_integer/1),
              do: error(x["id"], "state needs a concrete tensor type")

            count = Enum.reduce(x["type"]["shape"], 1, &*/2)
            data = List.flatten(List.wrap(x["data"]))

            unless length(data) == count and Enum.all?(data, &(is_number(&1) or is_boolean(&1))),
              do: error(x["id"], "state data does not match shape")
          end

          put!(env, x["id"], %{storage: x["kind"]})
        end)
      else
        if (g["state"] || []) != [], do: error("region", "state must be captured explicitly")
        env
      end

    env =
      Enum.reduce(g["ops"], env, fn op, env ->
        id = op["id"]

        rule =
          registry()[op["op"]] ||
            error(id, "unsupported operation #{op["op"]}", "unsupported-operation")

        args = op["args"] || []
        [lo, hi] = rule["arity"]

        unless is_list(args) and length(args) >= lo and length(args) <= hi,
          do: error(id, "invalid arity for #{op["op"]}")

        Enum.each(args, &value!(&1, env, id))
        regions = op["regions"] || %{}

        required =
          case op["op"] do
            "call" -> ["body"]
            "branch" -> ["true", "false"]
            "loop" -> ["body"]
            _ -> []
          end

        unless Enum.all?(required, &Map.has_key?(regions, &1)), do: error(id, "missing regions")

        if op["op"] == "loop" do
          attrs = op["attrs"] || %{}

          if Map.has_key?(attrs, "count") do
            unless is_integer(attrs["count"]) and attrs["count"] >= 0,
              do: error(id, "invalid loop count")
          else
            unless is_integer(attrs["budget"]) and attrs["budget"] >= 0 and regions["condition"],
              do: error(id, "convergence loop requires condition and execution budget")
          end
        end

        storage = fn value -> Map.get(value_cap(value, env), :storage) end

        if op["op"] == "copy_buffer" and storage.(hd(args)) != "buffer",
          do: error(id, "mutation target must be a mutable buffer")

        if op["op"] == "batch_norm" do
          indexes = [1, 2] ++ if(length(args) == 6, do: [5], else: [])

          unless Enum.all?(indexes, &(storage.(Enum.at(args, &1)) == "buffer")),
            do: error(id, "BatchNorm state must reference mutable buffers")
        end

        region_caps =
          Map.new(regions, fn {key, r} ->
            values = if op["op"] == "branch", do: tl(args), else: args

            captures =
              Map.new(Enum.zip(r["inputs"], values), fn {param, arg} ->
                {param["id"], value_cap(arg, env)}
              end)

            if op["op"] not in ["call", "branch", "loop"],
              do: error(id, "unexpected region #{key}")

            {key, validate_region(r, captures, false)}
          end)

        cap =
          case op["op"] do
            n when n in ["identity", "copy_buffer"] ->
              value_cap(hd(args), env)

            "tuple" ->
              %{items: Enum.map(args, &value_cap(&1, env))}

            "dict" ->
              %{
                items: Map.new(Enum.zip(op["attrs"]["keys"], Enum.map(args, &value_cap(&1, env))))
              }

            "getitem" ->
              items = Map.get(value_cap(hd(args), env), :items, [])
              index = op["attrs"]["index"]

              if(is_list(items) and is_integer(index),
                do: Enum.at(items, index),
                else: if(is_map(items), do: Map.get(items, index), else: nil)
              ) || %{storage: nil}

            "call" ->
              region_caps["body"]

            "branch" ->
              if region_caps["true"] == region_caps["false"],
                do: region_caps["true"],
                else: %{storage: nil}

            "loop" ->
              initial = value_cap(hd(args), env)

              if op["attrs"]["count"] == 0,
                do: initial,
                else: if(initial == region_caps["body"], do: initial, else: %{storage: nil})

            _ ->
              %{storage: nil}
          end

        put!(env, id, cap)
      end)

    value!(g["result"], env, "result")
    value_cap(g["result"], env)
  end

  defp value_cap(%{"ref" => id}, env), do: Map.fetch!(env, id)
  defp value_cap(%{"tuple" => xs}, env), do: %{items: Enum.map(xs, &value_cap(&1, env))}

  defp value_cap(%{"dict" => xs}, env),
    do: %{items: Map.new(xs, fn {k, v} -> {k, value_cap(v, env)} end)}

  defp value_cap(_, _), do: %{storage: nil}

  defp put!(env, id, t) do
    unless is_binary(id) and byte_size(id) > 0 and !Map.has_key?(env, id),
      do: error(id, "duplicate or invalid value id")

    Map.put(env, id, t)
  end

  defp type!(%{"kind" => "tensor", "shape" => s} = t, id) when is_list(s) do
    unless t["dtype"] in ["float32", "float64", "int64", "bool"] and t["device"] == "cpu" and
             t["layout"] == "strided",
           do: error(id, "unsupported tensor profile", "unsupported-type")

    unless Enum.all?(s, &(is_binary(&1) or (is_integer(&1) and &1 >= 0))),
      do: error(id, "invalid extent")

    axes = t["axes"] || List.duplicate(nil, length(s))
    named = Enum.reject(axes, &is_nil/1)

    unless length(axes) == length(s) and Enum.uniq(named) == named,
      do: error(id, "invalid axis names")
  end

  defp type!(%{"kind" => "scalar", "dtype" => d}, _)
       when d in ["bool", "int64", "float32", "float64"], do: :ok

  defp type!(%{"kind" => "tuple", "items" => ts}, id), do: Enum.each(ts, &type!(&1, id))

  defp type!(%{"kind" => "dict", "items" => ts}, id),
    do: Enum.each(ts, fn {_, t} -> type!(t, id) end)

  defp type!(_, id), do: error(id, "invalid value type")

  defp value!(%{"ref" => r}, env, id),
    do: unless(Map.has_key?(env, r), do: error(id, "dangling or forward reference #{r}"))

  defp value!(%{"lit" => v}, _, id),
    do:
      unless(is_number(v) or is_boolean(v),
        do: error(id, "invalid literal")
      )

  defp value!(%{"tuple" => xs}, env, id), do: Enum.each(xs, &value!(&1, env, id))
  defp value!(%{"dict" => xs}, env, id), do: Enum.each(xs, fn {_, v} -> value!(v, env, id) end)
  defp value!(_, _, id), do: error(id, "invalid value expression")

  def infer(g, bindings \\ %{}) do
    with {:ok, g} <- validate(g), do: protect(fn -> Inference.run(g, bindings) end)
  end

  def expand(g), do: with_valid(g, &Rewrite.expand/1)
  def collapse(g, selection), do: with_valid(g, &Rewrite.collapse(&1, selection))
  def resolve(g, decisions), do: with_valid(g, &Rewrite.resolve(&1, decisions))

  defp with_valid(g, f) do
    with {:ok, _} <- infer(g),
         {:ok, out} <- protect(fn -> {:ok, f.(g)} end),
         {:ok, _} <- infer(out),
         do: {:ok, out}
  end

  def encode(g) do
    with {:ok, _} <- infer(g), do: {:ok, IO.iodata_to_binary(Tensorloom.JSON.encode(g))}
  end

  def decode(s),
    do:
      protect(fn ->
        g = Tensorloom.JSON.decode(s)
        with {:ok, _} <- infer(g), do: {:ok, g}
      end)

  def compile(g, options \\ []) do
    with {:ok, _} <- infer(g), do: Tensorloom.Compiler.compile(g, options)
  end

  def protect(f) do
    try do
      f.()
    rescue
      e -> {:error, [%{op: "graph", code: "invalid-graph", message: Exception.message(e)}]}
    catch
      {:diagnostic, id, code, msg} -> {:error, [%{op: id, code: code, message: msg}]}
      {:shape, msg} -> {:error, [%{op: "shape", code: "shape-error", message: msg}]}
    end
  end

  def error(id, msg, code \\ "invalid-graph"), do: throw({:diagnostic, id, code, msg})
end
