[action, path | rest] = System.argv()
g = Tensorloom.JSON.decode(File.read!(path))

result =
  case action do
    "validate" ->
      Tensorloom.validate(g)

    "infer_region" ->
      Tensorloom.Inference.run(g, %{})

    "infer_many" ->
      {:ok,
       Enum.map(g, fn graph ->
         case Tensorloom.infer(graph) do
           {:ok, info} -> %{"result" => info.result}
           {:error, ds} -> %{"diagnostics" => ds}
         end
       end)}

    "infer" ->
      Tensorloom.infer(g)

    "expand" ->
      Tensorloom.expand(g)

    "collapse" ->
      Tensorloom.collapse(g, rest)

    "resolve" ->
      Tensorloom.resolve(g, Tensorloom.JSON.decode(hd(rest)))

    "compile" ->
      Tensorloom.compile(g, lowering: String.to_existing_atom(List.first(rest) || "expanded"))
  end

case result do
  {:ok, %{types: types, result: value, constraints: cs}} ->
    IO.puts(Tensorloom.JSON.encode(%{types: types, result: value, constraints: cs}))

  {:ok, v} when is_binary(v) ->
    IO.write(v)

  {:ok, v} ->
    IO.puts(Tensorloom.JSON.encode(v))

  {:error, diagnostics} ->
    IO.puts(Tensorloom.JSON.encode(%{diagnostics: diagnostics}))
    System.halt(2)
end
