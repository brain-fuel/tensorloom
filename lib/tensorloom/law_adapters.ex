defmodule Tensorloom.MatrixShape do
  @moduledoc false
  defstruct [:rows, :cols]
end

defmodule Tensorloom.LawAdapters do
  @moduledoc "Native bindings exercise public graph APIs; exact reference functions live in LawSpec (ref:DEC-contract-names)."
  alias Tensorloom, as: P
  alias Tensorloom.Shape, as: S

  def broadcast_extent(a, b) do
    try do
      hd(S.broadcast([a], [b]))
    catch
      {:shape, _} -> -1
    end
  end

  def transpose_shape(%{rows: r, cols: c}), do: S.transpose([r, c], 0, 1)
  def matmul_shape(%{rows: r, cols: c}, n), do: S.matmul([r, c], [c, n])
  def reshape_shape(%{rows: r, cols: c}), do: S.reshape([r, c], [r * c])
  def concatenate_shape(%{rows: r, cols: c}, n), do: S.cat([[r, c], [r, n]], 1)
  def reduce_shape(%{rows: r, cols: c}), do: S.reduce([r, c], 1)
  def slice_shape(%{rows: r, cols: c}, n), do: S.slice([r, c], 1, 0, n)

  defp g(n) do
    P.graph(
      [P.input("x", P.tensor([n]))],
      [P.op("a", "relu", [P.ref("x")]), P.op("b", "add", [P.ref("a"), P.ref("x")])],
      %{"tuple" => [P.ref("a"), P.ref("b")]}
    )
  end

  def codec(n) do
    graph = g(n)
    {:ok, json} = P.encode(graph)
    {:ok, roundtrip} = P.decode(json)
    graph == roundtrip
  end

  def rewrites(n) do
    graph = g(n)
    {:ok, c} = P.collapse(graph, ["a", "b"])
    {:ok, e} = P.expand(c)
    {:ok, again} = P.expand(e)
    {:ok, left} = P.infer(graph)
    {:ok, right} = P.infer(e)
    left.result == right.result and e == again
  end

  def permutation(xs) do
    perm = Enum.to_list(0..(length(xs) - 1)//1) |> Enum.reverse()

    xs
    |> Tensorloom.Topology.swap(perm)
    |> Tensorloom.Topology.swap(Tensorloom.Topology.inverse(perm))
  end

  def join(a, b) do
    Tensorloom.Topology.join("joined", "sub", [P.literal(a), P.literal(b)])["args"]
    |> Enum.map(& &1["lit"])
  end

  def resolve_independent(a, b) do
    t = P.tensor([2])
    body = P.graph([P.input("v", t)], [], P.ref("v"))
    inputs = [P.input("x", t), P.input("p", P.scalar())]
    rs = %{"true" => body, "false" => body}

    graph =
      P.graph(
        inputs,
        [
          P.op("r1", "branch", [P.ref("p"), P.ref("x")], %{}, rs),
          P.op("r2", "branch", [P.ref("p"), P.ref("r1")], %{}, rs)
        ],
        P.ref("r2")
      )

    {:ok, left} = P.resolve(graph, %{"r1" => a})
    {:ok, left} = P.resolve(left, %{"r2" => b})
    {:ok, right} = P.resolve(graph, %{"r2" => b})
    {:ok, right} = P.resolve(right, %{"r1" => a})
    left == right
  end
end
