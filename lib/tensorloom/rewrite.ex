defmodule Tensorloom.Rewrite do
  @moduledoc "Dependency-preserving region rewrites (ref:DEC-rewrites)."
  alias Tensorloom, as: P
  def refs(%{"ref" => r}), do: [r]
  def refs(x) when is_map(x), do: Enum.flat_map(x, fn {_, v} -> refs(v) end)
  def refs(x) when is_list(x), do: Enum.flat_map(x, &refs/1)
  def refs(_), do: []
  def subst(%{"ref" => r}, m), do: Map.get(m, r, P.ref(r))
  def subst(x, m) when is_map(x), do: Map.new(x, fn {k, v} -> {k, subst(v, m)} end)
  def subst(x, m) when is_list(x), do: Enum.map(x, &subst(&1, m))
  def subst(x, _), do: x

  def expand(g) do
    {:ok, info} = Tensorloom.Inference.run(g, %{})
    ids = Enum.map(g["ops"] ++ g["inputs"] ++ (g["state"] || []), & &1["id"])

    ops =
      Enum.flat_map(g["ops"], fn op ->
        regions = Map.new(op["regions"] || %{}, fn {k, r} -> {k, expand(r)} end)
        op = Map.put(op, "regions", regions)
        prefix = prefix(op["id"] <> "/", ids)

        case op["op"] do
          "call" ->
            inline(op, regions["body"], op["args"], prefix)

          "loop" ->
            if Map.has_key?(op["attrs"], "count") do
              {nodes, last} =
                Enum.reduce(
                  Enum.to_list(0..(op["attrs"]["count"] - 1)//1),
                  {[], hd(op["args"])},
                  fn i, {nodes, last} ->
                    id = prefix <> "iteration_#{i}"

                    {nodes ++ [P.op(id, "call", [last], %{}, %{"body" => regions["body"]})],
                     P.ref(id)}
                  end
                )

              nodes ++ [P.op(op["id"], "identity", [last])]
            else
              [op]
            end

          _ ->
            Tensorloom.Library.lower(op, info.types, prefix) || [op]
        end
      end)

    out = Map.put(g, "ops", ops)

    if out == g do
      if Map.has_key?(out, "router_domain"),
        do:
          Map.put(
            out,
            "router_domain",
            Enum.uniq(router_ids(out) ++ Map.keys(out["decisions"] || %{}))
          ),
        else: out
    else
      expand(out)
    end
  end

  defp prefix(candidate, ids),
    do:
      if(Enum.any?(ids, &String.starts_with?(&1, candidate)),
        do: prefix(candidate <> "_", ids),
        else: candidate
      )

  defp inline(op, body, args, prefix) do
    params = Map.new(Enum.zip(body["inputs"], args), fn {x, v} -> {x["id"], v} end)
    renames = Map.new(body["ops"], fn x -> {x["id"], P.ref(prefix <> x["id"])} end)
    m = Map.merge(params, renames)

    nodes =
      Enum.map(body["ops"], fn x ->
        x
        |> Map.put("id", prefix <> x["id"])
        |> Map.update!("args", &subst(&1, m))
        |> Map.put(
          "regions",
          Map.new(x["regions"] || %{}, fn {key, r} ->
            {key, rename_region(r, prefix <> x["id"] <> "/" <> key <> "/")}
          end)
        )
      end)

    nodes ++ [P.op(op["id"], "identity", [subst(body["result"], m)])]
  end

  defp rename_region(g, prefix) do
    renames = Map.new(g["ops"], fn op -> {op["id"], P.ref(prefix <> op["id"])} end)

    ops =
      Enum.map(g["ops"], fn op ->
        op
        |> Map.put("id", prefix <> op["id"])
        |> Map.update!("args", &subst(&1, renames))
        |> Map.put(
          "regions",
          Map.new(op["regions"] || %{}, fn {key, r} ->
            {key, rename_region(r, prefix <> op["id"] <> "/" <> key <> "/")}
          end)
        )
      end)

    g |> Map.put("ops", ops) |> Map.update!("result", &subst(&1, renames))
  end

  def collapse(g, selection) do
    chosen = MapSet.new(selection)
    nodes = Enum.filter(g["ops"], &MapSet.member?(chosen, &1["id"]))

    if nodes == [] or length(nodes) != MapSet.size(chosen),
      do: P.error("collapse", "unknown or empty selection")

    positions = Map.new(Enum.with_index(g["ops"]), fn {n, i} -> {n["id"], i} end)
    last = nodes |> List.last() |> Map.fetch!("id") |> then(&positions[&1])
    first = nodes |> hd() |> Map.fetch!("id") |> then(&positions[&1])
    between = Enum.slice(g["ops"], first..last) |> Enum.reject(&MapSet.member?(chosen, &1["id"]))

    if Enum.any?(between, fn n -> Enum.any?(refs(n["args"]), &MapSet.member?(chosen, &1)) end),
      do: P.error("collapse", "selection would create a dependency cycle")

    effect? = fn n -> effects(n) != [] end

    if Enum.any?(between, effect?) or (between != [] and Enum.any?(nodes, effect?)),
      do: P.error("collapse", "selection crosses an effect")

    incoming =
      Enum.flat_map(nodes, &refs(&1["args"]))
      |> Enum.reject(&MapSet.member?(chosen, &1))
      |> Enum.uniq()

    outside = Enum.reject(g["ops"], &MapSet.member?(chosen, &1["id"]))
    uses = Enum.flat_map(outside, &refs(&1["args"])) ++ refs(g["result"])
    outgoing = Enum.filter(Enum.map(nodes, & &1["id"]), &(&1 in uses))
    outgoing = if outgoing == [], do: [List.last(nodes)["id"]], else: outgoing
    {:ok, info} = P.infer(g)
    states = Map.new(g["state"] || [], &{&1["id"], &1["kind"]})

    params =
      Enum.map(incoming, fn id ->
        type = Map.fetch!(info.types, id)
        type = if states[id], do: Map.put(type, "storage", states[id]), else: type
        P.input(id, type)
      end)

    id =
      fresh(
        "collapsed",
        Map.keys(positions) ++
          Enum.map(g["inputs"], & &1["id"]) ++ Enum.map(g["state"] || [], & &1["id"])
      )

    body = P.graph(params, nodes, %{"tuple" => Enum.map(outgoing, &P.ref/1)})
    call = P.op(id, "call", Enum.map(incoming, &P.ref/1), %{}, %{"body" => body})

    extracts =
      Enum.with_index(outgoing)
      |> Enum.map(fn {r, i} -> P.op(r, "getitem", [P.ref(id)], %{"index" => i}) end)

    ops =
      Enum.with_index(g["ops"])
      |> Enum.flat_map(fn {n, i} ->
        cond do
          i == last -> [call | extracts]
          MapSet.member?(chosen, n["id"]) -> []
          true -> [n]
        end
      end)

    Map.put(g, "ops", ops)
  end

  def effects(n) do
    own = (P.registry()[n["op"]] || %{})["effects"] || []

    own ++
      Enum.flat_map(n["regions"] || %{}, fn {_, r} -> Enum.flat_map(r["ops"], &effects/1) end)
  end

  defp fresh(id, ids), do: if(id in ids, do: fresh(id <> "_", ids), else: id)

  def resolve(g, decisions) do
    known = g["router_domain"] || router_ids(g)
    previous = g["decisions"] || %{}

    unless Enum.all?(decisions, fn {k, v} ->
             k in known and is_boolean(v) and (!Map.has_key?(previous, k) or previous[k] == v)
           end),
           do: P.error("resolve", "unknown router or incompatible decision")

    g
    |> do_resolve(decisions)
    |> Map.put("decisions", Map.merge(previous, decisions))
    |> Map.put("router_domain", known)
  end

  defp router_ids(g),
    do:
      Enum.flat_map(g["ops"], fn n ->
        if(n["op"] == "branch", do: [n["id"]], else: []) ++
          Enum.flat_map(n["regions"] || %{}, fn {_, r} -> router_ids(r) end)
      end)

  defp do_resolve(g, decisions) do
    ops =
      Enum.flat_map(g["ops"], fn n ->
        rs = Map.new(n["regions"] || %{}, fn {k, r} -> {k, do_resolve(r, decisions)} end)
        n = Map.put(n, "regions", rs)

        if n["op"] == "branch" and Map.has_key?(decisions, n["id"]) do
          [
            P.op(n["id"], "call", tl(n["args"]), %{}, %{
              "body" => rs[if(decisions[n["id"]], do: "true", else: "false")]
            })
          ]
        else
          [n]
        end
      end)

    Map.put(g, "ops", ops)
  end
end
