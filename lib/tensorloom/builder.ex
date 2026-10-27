defmodule Tensorloom.Builder do
  @moduledoc "Elixir graph builders for fixed-length recurrent composition (ref:DEC-library)."
  alias Tensorloom, as: P

  def sequence(kind, x_type, h_type, state, opts \\ [])
      when kind in ["rnn_cell", "gru_cell", "lstm_cell"] do
    steps = Keyword.fetch!(opts, :steps)
    if not is_integer(steps) or steps < 0, do: raise(ArgumentError, "steps must be nonnegative")
    time = Keyword.get(opts, :time_dim, 0)

    inputs =
      [P.input("x", x_type), P.input("h", h_type)] ++
        if(kind == "lstm_cell", do: [P.input("c", h_type)], else: [])

    weights = Enum.map(state, &P.ref(&1["id"]))
    initial = if kind == "lstm_cell", do: [P.ref("h"), P.ref("c")], else: [P.ref("h")]

    {ops, last, ys} =
      Enum.reduce(Enum.to_list(0..(steps - 1)//1), {[], initial, []}, fn i, {ops, carry, ys} ->
        id = "step_#{i}"

        nodes = [
          P.op(id <> "_x", "select", [P.ref("x")], %{"dim" => time, "index" => i}),
          P.op(id, kind, [P.ref(id <> "_x")] ++ carry ++ weights, Keyword.get(opts, :attrs, %{}))
        ]

        if kind == "lstm_cell" do
          nodes =
            nodes ++
              [
                P.op(id <> "_h", "getitem", [P.ref(id)], %{"index" => 0}),
                P.op(id <> "_c", "getitem", [P.ref(id)], %{"index" => 1})
              ]

          {ops ++ nodes, [P.ref(id <> "_h"), P.ref(id <> "_c")], ys ++ [P.ref(id <> "_h")]}
        else
          {ops ++ nodes, [P.ref(id)], ys ++ [P.ref(id)]}
        end
      end)

    output =
      if steps == 0,
        do: P.op("sequence", "empty_sequence", [P.ref("h")], %{"dim" => time}),
        else: P.op("sequence", "stack", ys, %{"dim" => time})

    final = if kind == "lstm_cell", do: %{"tuple" => last}, else: hd(last)
    P.graph(inputs, ops ++ [output], %{"tuple" => [P.ref("sequence"), final]}, state)
  end
end
