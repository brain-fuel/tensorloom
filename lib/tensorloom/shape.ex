defmodule Tensorloom.Shape do
  @moduledoc "Exact fixed-rank shape algebra. Unresolved relations become explicit guards (ref:DEC-shapes)."
  def broadcast(a, b) do
    n = max(length(a), length(b))

    Enum.zip(List.duplicate(1, n - length(a)) ++ a, List.duplicate(1, n - length(b)) ++ b)
    |> Enum.map(fn
      {x, x} ->
        x

      {1, y} ->
        y

      {x, 1} ->
        x

      {x, y} when is_integer(x) and is_integer(y) ->
        fail("incompatible broadcast #{inspect({x, y})}")

      {x, y} ->
        %{"expr" => "broadcast", "args" => [x, y]}
    end)
  end

  def matmul([], _), do: fail("matmul needs rank >= 1")
  def matmul(_, []), do: fail("matmul needs rank >= 1")

  def matmul(a, b) do
    av = length(a) == 1
    bv = length(b) == 1
    aa = if av, do: [1 | a], else: a
    bb = if bv, do: b ++ [1], else: b
    equal!(List.last(aa), Enum.at(bb, -2))
    batch = broadcast(Enum.drop(aa, -2), Enum.drop(bb, -2))
    batch ++ if(av, do: [], else: [Enum.at(aa, -2)]) ++ if bv, do: [], else: [List.last(bb)]
  end

  def reshape(shape, target) do
    unless Enum.all?(target, &(is_binary(&1) or (is_integer(&1) and &1 >= -1))),
      do: fail("invalid reshape extent")

    if Enum.count(target, &(&1 == -1)) > 1, do: fail("multiple inferred dimensions")
    n = product(shape)
    known = product(Enum.reject(target, &(&1 == -1)))

    cond do
      -1 in target and known == 0 ->
        fail("ambiguous zero reshape")

      -1 in target and is_integer(n) and is_integer(known) ->
        if rem(n, known) != 0, do: fail("reshape element count")
        Enum.map(target, &if(&1 == -1, do: div(n, known), else: &1))

      -1 in target ->
        Enum.map(target, &if(&1 == -1, do: %{"expr" => "div", "args" => [n, known]}, else: &1))

      true ->
        equal!(n, known)
        target
    end
  end

  def product(xs), do: Enum.reduce(xs, 1, &arith("mul", &2, &1))
  def arith("mul", a, b) when is_integer(a) and is_integer(b), do: a * b
  def arith("add", a, b) when is_integer(a) and is_integer(b), do: a + b
  def arith("mul", 1, b), do: b
  def arith("mul", a, 1), do: a
  def arith(op, a, b), do: %{"expr" => op, "args" => [a, b]}
  def equal!(x, x), do: :ok

  def equal!(x, y) when is_list(x) and is_list(y) do
    if length(x) != length(y), do: fail("shape ranks differ")
    Enum.zip(x, y) |> Enum.each(fn {a, b} -> equal!(a, b) end)
  end

  def equal!(x, y) when is_integer(x) and is_integer(y),
    do: fail("dimensions differ: #{x} and #{y}")

  def equal!(_, _), do: :guard

  def axis(d, shape) when is_integer(d) do
    n = length(shape)
    if d < -n or d >= n, do: fail("axis #{d} outside rank #{n}")
    Integer.mod(d, n)
  end

  def transpose([], a, b) when a in [0, -1] and b in [0, -1], do: []

  def transpose(s, a, b) do
    a = axis(a, s)
    b = axis(b, s)
    s |> List.replace_at(a, Enum.at(s, b)) |> List.replace_at(b, Enum.at(s, a))
  end

  def reduce(s, dims, keep \\ false) do
    if s == [] and (is_nil(dims) or dims in [0, -1, [], [0], [-1]]),
      do: [],
      else: reduce_axes(s, dims, keep)
  end

  defp reduce_axes(s, dims, keep) do
    dims =
      if is_nil(dims) or dims == [],
        do: Enum.to_list(0..(length(s) - 1)//1),
        else: List.wrap(dims)

    ds = Enum.map(dims, &axis(&1, s))
    if Enum.uniq(ds) != ds, do: fail("duplicate reduction axis")

    s
    |> Enum.with_index()
    |> Enum.flat_map(fn {v, i} -> if i in ds, do: if(keep, do: [1], else: []), else: [v] end)
  end

  def cat([s | rest], dim) do
    d = axis(dim, s)

    Enum.reduce(rest, s, fn t, acc ->
      if length(t) != length(s), do: fail("concatenation rank mismatch")
      Enum.with_index(t) |> Enum.each(fn {v, i} -> if i != d, do: equal!(v, Enum.at(s, i)) end)
      List.update_at(acc, d, &arith("add", &1, Enum.at(t, d)))
    end)
  end

  def cat([], _), do: fail("empty concatenation")

  def slice(s, dim, start, stop, step \\ 1) do
    if step <= 0, do: fail("slice step must be positive")
    d = axis(dim, s)
    n = Enum.at(s, d)

    value =
      if is_integer(n) do
        clamp = fn v -> min(n, max(0, if(v < 0, do: n + v, else: v))) end
        a = clamp.(start || 0)
        b = clamp.(stop || n)
        div(max(b - a, 0) + step - 1, step)
      else
        %{"expr" => "slice", "args" => [n, start, stop, step]}
      end

    List.replace_at(s, d, value)
  end

  def window(n, k, s, p, d, ceil_mode \\ false) do
    if Enum.any?([n, k, s, p, d], &(not is_integer(&1))),
      do: fail("symbolic spatial dimensions need bindings")

    if k <= 0 or s <= 0 or p < 0 or d <= 0, do: fail("invalid window geometry")
    numerator = n + 2 * p - d * (k - 1) - 1
    out = Integer.floor_div(numerator + if(ceil_mode, do: s - 1, else: 0), s) + 1
    out = if ceil_mode and (out - 1) * s >= n + p, do: out - 1, else: out
    if out <= 0, do: fail("window has no output")
    out
  end

  def fail(message), do: throw({:shape, message})
end
