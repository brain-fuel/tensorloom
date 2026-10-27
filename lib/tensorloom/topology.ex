defmodule Tensorloom.Topology do
  @moduledoc "Ordered wiring without physical tensor copies (ref:DEC-graph)."
  alias Tensorloom, as: P

  def pipe(value, node, position) when is_integer(position) and position >= 0 do
    args = node["args"]
    if position >= length(args), do: raise(ArgumentError, "argument position out of bounds")
    Map.put(node, "args", List.replace_at(args, position, value))
  end

  def fork(value, count) when is_integer(count) and count >= 0, do: List.duplicate(value, count)
  def join(id, operation, values, attrs \\ %{}), do: P.op(id, operation, values, attrs)

  def swap(values, permutation) do
    n = length(values)

    unless Enum.sort(permutation) == Enum.to_list(0..(n - 1)//1),
      do: raise(ArgumentError, "invalid permutation")

    Enum.map(permutation, &Enum.at(values, &1))
  end

  def inverse(permutation) do
    _ = swap(permutation, permutation)
    permutation |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
  end

  def parallel(inputs, tracks, state \\ []) do
    P.protect(fn ->
      available = Enum.map(inputs ++ state, & &1["id"])

      ops =
        Enum.with_index(tracks)
        |> Enum.map(fn {%{region: region, args: args}, i} ->
          if Enum.any?(region["ops"], &(Tensorloom.Rewrite.effects(&1) != [])),
            do: P.error("parallel", "regions must be pure")

          unless Enum.all?(Tensorloom.Rewrite.refs(args), &(&1 in available)),
            do: P.error("parallel", "tracks cannot depend on another track")

          P.op("parallel_#{i}", "call", args, %{}, %{"body" => region})
        end)

      g = P.graph(inputs, ops, %{"tuple" => Enum.map(ops, &P.ref(&1["id"]))}, state)
      P.validate(g)
    end)
  end
end
