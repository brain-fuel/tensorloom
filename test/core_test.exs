defmodule Tensorloom.CoreTest do
  use ExUnit.Case, async: true
  use ExUnitProperties
  alias Tensorloom, as: P
  alias Tensorloom.Shape, as: S

  defp one(shape) do
    P.graph([P.input("x", P.tensor(shape))], [P.op("a", "relu", [P.ref("x")])], P.ref("a"))
  end

  test "zero extents broadcast with one but not arbitrary axes" do
    assert S.broadcast([0, 3], [1, 3]) == [0, 3]
    assert catch_throw(S.broadcast([0], [2])) == {:shape, "incompatible broadcast {0, 2}"}

    for a <- 0..4, b <- 0..4 do
      expected =
        cond do
          a == b -> a
          a == 1 -> b
          b == 1 -> a
          true -> :invalid
        end

      if expected == :invalid do
        assert catch_throw(S.broadcast([a], [b]))
      else
        assert S.broadcast([a], [b]) == [expected]
      end
    end
  end

  test "vector and batched matmul including zero inner dimension" do
    assert S.matmul([3], [3]) == []
    assert S.matmul([2, 3], [3]) == [2]
    assert S.matmul([3], [2, 3, 5]) == [2, 5]
    assert S.matmul([7, 1, 2, 0], [4, 0, 5]) == [7, 4, 2, 5]
    assert catch_throw(S.matmul([], [1]))
    assert catch_throw(S.matmul([2], [3]))
  end

  test "reshape disallows ambiguous zeros and inexact products" do
    assert S.reshape([0, 3], [3, 0]) == [3, 0]
    assert S.reshape([2, 3], [-1, 2]) == [3, 2]
    assert S.reshape([], []) == []
    assert catch_throw(S.reshape([0, 3], [0, -1]))
    assert catch_throw(S.reshape([2, 3], [4, -1]))
    assert catch_throw(S.reshape([2, 3], [-1, -1]))
  end

  property "constructed compatible batches preserve outer matrix dimensions" do
    check all(
            batch <- list_of(integer(0..5), max_length: 3),
            m <- integer(0..5),
            k <- integer(0..5),
            n <- integer(0..5)
          ) do
      assert S.matmul(batch ++ [m, k], [k, n]) == batch ++ [m, n]
    end
  end

  property "transpose is involutive on constructed valid axes" do
    check all(s <- list_of(integer(0..9), min_length: 1, max_length: 5), seed <- integer(0..100)) do
      a = rem(seed, length(s))
      b = rem(div(seed, 2), length(s))
      assert S.transpose(S.transpose(s, a, b), a, b) == s
    end
  end

  test "slice clips Python bounds and positive strides" do
    assert S.slice([5], 0, -10, 10) == [5]
    assert S.slice([5], 0, 4, 1) == [0]
    assert S.slice([5], 0, 0, 5, 2) == [3]
    assert catch_throw(S.slice([5], 0, 0, 5, -1))
  end

  test "empty scalar reductions retain rank zero" do
    assert S.reduce([], nil) == []
    assert S.reduce([2, 0, 3], 1) == [2, 3]
    assert S.reduce([2, 0, 3], 1, true) == [2, 1, 3]
  end

  test "ordered graphs reject cycles dangling captures and unknown operations" do
    g = one([2])
    assert {:ok, ^g} = P.validate(g)

    assert {:error, [%{op: "a"}]} =
             P.validate(put_in(g, ["ops", Access.at(0), "args"], [P.ref("missing")]))

    assert {:error, [%{code: "unsupported-operation"}]} =
             P.validate(put_in(g, ["ops", Access.at(0), "op"], "surprise"))

    assert {:error, _} = P.validate(Map.put(g, "ops", g["ops"] ++ g["ops"]))
    r = P.graph([], [], P.ref("x"))
    call = P.graph(g["inputs"], [P.op("a", "call", [], %{}, %{"body" => r})], P.ref("a"))
    assert {:error, _} = P.validate(call)
  end

  test "validation includes shape incompatibilities and operation identity" do
    g =
      P.graph(
        [P.input("a", P.tensor([2, 3])), P.input("b", P.tensor([4, 5]))],
        [P.op("bad", "matmul", [P.ref("a"), P.ref("b")])],
        P.ref("bad")
      )

    assert {:error, [%{op: "bad", code: "shape-error"}]} = P.validate(g)
  end

  test "JSON round trips actual nulls and preserves arbitrary metadata" do
    g = one([0, 3]) |> Map.put("source", %{"line" => 4, "path" => "example"})
    assert {:ok, json} = P.encode(g)
    assert json =~ "null"
    refute json =~ "\"nil\""
    assert {:ok, ^g} = P.decode(json)
    assert {:error, _} = P.decode("{")
    assert {:error, _} = P.decode("{\"version\":2}")
  end

  test "symbolic dimensions emit constraints and concrete bindings discharge them" do
    g =
      P.graph(
        [P.input("x", P.tensor(["batch", 3])), P.input("y", P.tensor(["other", 3]))],
        [P.op("z", "add", [P.ref("x"), P.ref("y")])],
        P.ref("z")
      )

    assert {:ok, %{constraints: [_]}} = P.infer(g)

    assert {:ok, %{constraints: [], result: %{"shape" => [4, 3]}}} =
             P.infer(g, %{"batch" => 4, "other" => 1})

    assert {:error, [%{op: "z"}]} = P.infer(g, %{"batch" => 4, "other" => 2})
  end

  test "incompatible branch results and unbudgeted loops are rejected" do
    t = P.tensor([2])
    a = P.graph([P.input("x", t)], [], P.ref("x"))
    b = P.graph([P.input("x", t)], [P.op("r", "sum", [P.ref("x")])], P.ref("r"))
    inputs = [P.input("x", t), P.input("p", P.scalar())]

    g =
      P.graph(
        inputs,
        [P.op("r", "branch", [P.ref("p"), P.ref("x")], %{}, %{"true" => a, "false" => b})],
        P.ref("r")
      )

    assert {:error, [%{op: "r"}]} = P.validate(g)

    loop =
      P.graph([hd(inputs)], [P.op("r", "loop", [P.ref("x")], %{}, %{"body" => a})], P.ref("r"))

    assert {:error, _} = P.validate(loop)
  end

  test "collapse rejects dependence cycles and crossed effects" do
    t = P.tensor([2])
    inputs = [P.input("x", t)]

    g =
      P.graph(
        inputs,
        [
          P.op("a", "relu", [P.ref("x")]),
          P.op("b", "relu", [P.ref("a")]),
          P.op("c", "add", [P.ref("a"), P.ref("b")])
        ],
        P.ref("c")
      )

    assert {:error, _} = P.collapse(g, ["a", "c"])

    g =
      P.graph(
        inputs,
        [
          P.op("a", "rand_like", [P.ref("x")]),
          P.op("b", "rand_like", [P.ref("x")]),
          P.op("c", "rand_like", [P.ref("x")])
        ],
        %{"tuple" => [P.ref("a"), P.ref("b"), P.ref("c")]}
      )

    assert {:error, _} = P.collapse(g, ["a", "c"])
  end

  property "collapse preserves multiple outgoing interfaces and state identity" do
    check all(n <- integer(0..8)) do
      g = one([n])
      g = Map.put(g, "result", %{"tuple" => [P.ref("a"), P.ref("x")]})
      assert {:ok, c} = P.collapse(g, ["a"])
      assert {:ok, e} = P.expand(c)
      assert {:ok, left} = P.infer(g)
      assert {:ok, right} = P.infer(e)
      assert left.result == right.result
      assert {:ok, ^e} = P.expand(e)
    end
  end
end

defmodule Tensorloom.NegativeControlsTest do
  use ExUnit.Case, async: true

  test "broadcast conformance rejects the max extent mutant" do
    cases = [{0, 1, 0}, {1, 0, 0}, {0, 0, 0}, {2, 1, 2}, {1, 3, 3}]
    conforms = fn f -> Enum.all?(cases, fn {a, b, c} -> f.([a], [b]) == [c] end) end
    assert conforms.(&Tensorloom.Shape.broadcast/2)
    refute conforms.(fn [a], [b] -> [max(a, b)] end)
  end
end

defmodule Tensorloom.AdditionalCoreTest do
  use ExUnit.Case, async: true
  alias Tensorloom, as: P

  test "collapse cannot move a pure buffer read across mutation" do
    t = P.tensor([2])
    state = [%{"id" => "b", "kind" => "buffer", "type" => t, "data" => [1, 2]}]

    g =
      P.graph(
        [P.input("x", t)],
        [
          P.op("a", "add", [P.ref("b"), P.ref("x")]),
          P.op("write", "copy_buffer", [P.ref("b"), P.ref("x")]),
          P.op("c", "relu", [P.ref("a")])
        ],
        P.ref("c"),
        state
      )

    assert {:ok, _} = P.validate(g)
    assert {:error, _} = P.collapse(g, ["a", "c"])
    constant = put_in(g, ["state", Access.at(0), "kind"], "constant")
    assert {:error, [%{op: "write"}]} = P.validate(constant)
  end

  test "resolved decisions are compatible idempotent and reject conflicts" do
    t = P.tensor([2])
    body = P.graph([P.input("x", t)], [], P.ref("x"))

    g =
      P.graph(
        [P.input("x", t), P.input("p", P.scalar())],
        [P.op("r", "branch", [P.ref("p"), P.ref("x")], %{}, %{"true" => body, "false" => body})],
        P.ref("r")
      )

    assert {:ok, a} = P.resolve(g, %{"r" => true})
    assert {:ok, ^a} = P.resolve(a, %{"r" => true})
    assert {:error, _} = P.resolve(a, %{"r" => false})
  end

  test "fixed count loops expand to stable acyclic wiring including zero" do
    t = P.tensor([2])
    body = P.graph([P.input("x", t)], [P.op("a", "relu", [P.ref("x")])], P.ref("a"))

    for n <- [0, 1, 4] do
      g =
        P.graph(
          [P.input("x", t)],
          [P.op("loop", "loop", [P.ref("x")], %{"count" => n}, %{"body" => body})],
          P.ref("loop")
        )

      assert {:ok, e} = P.expand(g)
      refute Enum.any?(e["ops"], &(&1["op"] in ["call", "loop"]))
      assert {:ok, ^e} = P.expand(e)
      assert {:ok, %{result: ^t}} = P.infer(e)
    end
  end
end

defmodule Tensorloom.TopologyTest do
  use ExUnit.Case, async: true
  alias Tensorloom, as: P
  alias Tensorloom.Topology, as: T

  test "wiring preserves fanout values and join argument order" do
    v = P.ref("x")
    assert T.fork(v, 3) == [v, v, v]
    assert T.fork(v, 0) == []
    node = T.join("subtract", "sub", [P.literal(2), P.literal(7)])
    assert node["args"] == [P.literal(2), P.literal(7)]
    assert T.pipe(v, node, 1)["args"] == [P.literal(2), v]
    assert T.swap([:a, :b, :c], [2, 0, 1]) == [:c, :a, :b]
    assert T.swap(T.swap([:a, :b, :c], [2, 0, 1]), T.inverse([2, 0, 1])) == [:a, :b, :c]
    assert_raise ArgumentError, fn -> T.swap([:a, :b], [0, 0]) end
  end

  test "pure parallel regions preserve order and reject effects or cross dependencies" do
    t = P.tensor([2])
    inputs = [P.input("x", t)]
    body = P.graph([P.input("arg", t)], [P.op("a", "neg", [P.ref("arg")])], P.ref("a"))
    tracks = [%{region: body, args: [P.ref("x")]}, %{region: body, args: [P.ref("x")]}]
    assert {:ok, g} = T.parallel(inputs, tracks)
    assert {:ok, %{result: %{"kind" => "tuple", "items" => [^t, ^t]}}} = P.infer(g)
    effect = put_in(body, ["ops", Access.at(0), "op"], "rand_like")
    assert {:error, _} = T.parallel(inputs, [%{region: effect, args: [P.ref("x")]}])
    assert {:error, _} = T.parallel(inputs, [%{region: body, args: [P.ref("parallel_0")]}])
  end
end

defmodule Tensorloom.BufferRewriteTest do
  use ExUnit.Case, async: true
  alias Tensorloom, as: P

  test "collapse and expansion preserve mutable buffer capabilities through structured results" do
    t = P.tensor([2])
    state = [%{"id" => "buffer", "kind" => "buffer", "type" => t, "data" => [1, 2]}]

    g =
      P.graph(
        [P.input("x", t)],
        [
          P.op("alias", "identity", [P.ref("buffer")]),
          P.op("write", "copy_buffer", [P.ref("alias"), P.ref("x")])
        ],
        P.ref("write"),
        state
      )

    assert {:ok, ^g} = P.validate(g)
    assert {:ok, c} = P.collapse(g, ["alias"])
    assert {:ok, e} = P.expand(c)
    assert {:ok, _} = P.validate(e)

    for variant <- [g, c, e] do
      constant = put_in(variant, ["state", Access.at(0), "kind"], "constant")
      assert {:error, _} = P.validate(constant)
    end
  end
end
