---
id: tensorloom.readme
kind: tutorial
title: Executable Elixir–PyTorch graph compiler
---
# Tensorloom

An Elixir graph compiler with a portable JSON format, explicit FX/export capture,
executable LawSpec contracts, and generated PyTorch modules. It implements the
dense CPU profile described in [the notation](graph_notation.txt), including
structured values, shared parameters, mutable buffers, control flow, shapes,
library expansions and native lowerings. See ref:DEC-compiler.

Use Python 3.13 / PyTorch 2.14.1, Elixir 1.20.4 / OTP 29 and Node 22+.
The sibling LawSpec checkout at `../lawspec` is used without modification.

```sh
uv sync --locked
mix deps.get
MIX_ENV=test mix deps.compile
mix compile
node scripts/setup_canon.mjs
node scripts/verify.mjs
```

The checked-in LawSpec runtime/tests are generated from `laws/` and
`python/laws/conformance.lawspec`. The Elixir frontend and Python backend have
separate component manifests. `scripts/verify.mjs` checks definitions, generates
and executes both through the LawSpec CLI, then validates the root project.
It returns exit code 2 while Canon review remains pending.
Mandatory definition validation is independent from selected examples,
properties and finite exhaustive checks. [Validation evidence](docs/reference/validation.md) and
[conformance ownership](docs/reference/law-owned-conformance.md)
separates automation from pending human review. Canon's grammar installation can
be recreated with the pinned Rice's Tax setup; its local cache is ignored.

Build a graph in Elixir:

```elixir
alias Tensorloom, as: P
x = P.input("x", P.tensor([2, 3]))
g = P.graph([x], [P.op("y", "relu", [P.ref("x")])], P.ref("y"))
{:ok, _validated} = P.validate(g)
{:ok, %{result: type, constraints: guards}} = P.infer(g)
{:ok, json} = P.encode(g)
{:ok, ^g} = P.decode(json)
{:ok, source} = P.compile(g, lowering: :expanded)
File.write!("generated_model.py", source)
```

Generated source imports `tensorloom`; run it with `PYTHONPATH=python`.
`expand/1`, `collapse/2` and `resolve/2` return validated graphs. Every diagnostic
contains the failing operation where available. `Tensorloom.Builder.sequence/5`
composes fixed-length recurrent cells and handles zero passes.

Capture and compile an ordinary PyTorch module:

```python
import json
import torch
from tensorloom import capture, compile_graph, export_graph

model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU()).double()
x = torch.randn(2, 3, dtype=torch.float64)
graph = capture(model, (x,), method="fx")  # explicit alternative: method="export"
graph = json.loads(json.dumps(graph))
generated = compile_graph(graph, lowering="expanded")  # or "native"
torch.testing.assert_close(generated(x), model(x))
export_graph(graph, (x,), "model.pt2", dynamic_shapes=(
    {0: torch.export.Dim("batch", min=1, max=8)},
))
loaded = torch.export.load("model.pt2").module()
```

Use `PYTHONPATH=python uv run python ...` for Python examples. Python APIs invoke
the compiled Elixir validator; run `mix compile` first. There is no automatic
capture-method fallback. [The source-linked PyTorch comparison](docs/reference/pytorch-comparison.md)
lists implemented families and explicit restrictions, including fixed sequence
length, one-layer sequence capture and unsupported device/layout profiles.

Elixir can save an export directly using JSON example values:

```elixir
{:ok, artifact} = P.compile(g, profile: :export, lowering: :native,
  example_inputs: [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], path: "model.pt2",
  dynamic_dims: %{"0" => %{"0" => %{"name" => "batch", "min" => 1, "max" => 8}}})
```

[examples/mlp.json](examples/mlp.json) is a captured MLP. Inspect a graph from the
command line with `mix run scripts/graph.exs infer examples/mlp.json`, or generate
source with `mix run scripts/graph.exs compile examples/mlp.json native`.
