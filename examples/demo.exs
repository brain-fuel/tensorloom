alias Tensorloom, as: T

graph =
  T.graph(
    [T.input("x", T.tensor([2, 2]))],
    [
      T.op("linear", "linear", [T.ref("x"), T.ref("weight"), T.ref("bias")]),
      T.op("output", "relu", [T.ref("linear")])
    ],
    T.ref("output"),
    [
      %{
        "id" => "weight",
        "kind" => "parameter",
        "type" => T.tensor([1, 2]),
        "data" => [[2.0, -1.0]]
      },
      %{"id" => "bias", "kind" => "parameter", "type" => T.tensor([1]), "data" => [0.5]}
    ]
  )

{:ok, inferred} = T.infer(graph)
{:ok, json} = T.encode(graph)
{:ok, source} = T.compile(graph, lowering: :expanded)
directory = System.get_env("TENSORLOOM_DEMO_DIR", Path.expand("artifacts/demo"))
File.mkdir_p!(directory)
File.write!(Path.join(directory, "graph.json"), json)
File.write!(Path.join(directory, "model.py"), source)
IO.puts("Elixir graph output shape: #{inspect(inferred.result["shape"])}")
IO.puts("Generated graph.json and model.py in #{directory}")
