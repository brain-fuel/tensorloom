---
id: tensorloom.readme
kind: tutorial
title: Executable Elixir–PyTorch graph compiler
---
# Tensorloom

Tensorloom builds and checks computation graphs in Elixir, exchanges them as
JSON, and compiles them into ordinary PyTorch modules. It can also capture
supported PyTorch models through FX or export. Executable LawSpec contracts
check shapes, graph transformations, numerical results, gradients and state.
See ref:DEC-compiler and ref:DEC-build-artifacts.

## Install and run the demo

Install these tools first:

- [Node.js](https://nodejs.org/en/download) 22 or later, with npm.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the
  locked Python environment. Setup uses Python 3.13.15 and PyTorch 2.14.1.
- [Elixir and Erlang/OTP](https://elixir-lang.org/install.html). The tested
  versions are Elixir 1.20.4 and OTP 29.1, recorded in `.tool-versions`.
- Git.

Then run:

```sh
git clone https://github.com/brain-fuel/tensorloom.git
cd tensorloom
npm ci
mix local.hex --force
npm run setup
npm run demo
```

`mix local.hex` is a one-time installation of Elixir's package manager. Setup
installs the locked Python dependencies into `.venv`, fetches Elixir dependencies,
generates the LawSpec runtime and tests, compiles Elixir, and restores the local
Canon grammar cache. The first run needs network access and disk space for
PyTorch. No sibling repositories or Haskell installation are needed.

The demo builds a small neural network in Elixir, generates Python, runs it,
checks its gradient, captures an equivalent PyTorch model, compiles both lowering
modes, and saves and reloads a PyTorch export. Its main results are:

```text
Elixir graph output shape: [2, 1]
Elixir -> generated PyTorch: [[2.5], [0.0]]
Weight gradient: [[3.0, 4.0]]
Captured PyTorch -> native: [[2.5], [0.0]]
Captured PyTorch -> expanded: [[2.5], [0.0]]
Saved export -> reloaded: [[2.5], [0.0]]
```

The network computes `max(0, 2*x1 - x2 + 0.5)` for each row. The first input row,
`[3, 4]`, gives `2.5`; the second, `[-1, 2]`, gives `0` after ReLU clips the
negative value. The gradient says how a small change to each weight would change
the sum of the outputs. Only the first row contributes here, giving `[3, 4]`.

Read or edit [examples/demo.exs](examples/demo.exs) and
[examples/demo.py](examples/demo.py), then rerun `npm run demo`. Its generated
`graph.json`, `model.py` and `model.pt2` are in ignored `artifacts/demo/`.

## Reproduce generation and validation

The repository keeps authored contracts, implementation, fixtures and lockfiles.
LawSpec **0.31.0**, published from tag
[v0.31.0](https://github.com/brain-fuel/lawspec/tree/v0.31.0), is pinned in
`package.json`; `package-lock.json` fixes the installed dependency versions and
integrity hashes. Canon setup pins Rice's Tax 0.3.2. Use `npm ci` to reproduce
these tools. There is no dependency on a developer's LawSpec checkout.
The released LawSpec CLI has no `project status` or `project validate` verbs;
our verification script composes its definition/native checks with the released
Canon commands instead.

Generated files stay local. Run `npm run generate` after changing a contract or
`npm run setup` to recreate the environment and generated code. Generation
writes Elixir sources under `generated/elixir/`, test support under
`test/support/`, and `test/*_lawspec_test.exs`. Python outputs go under
`python/lawspec_generated/` and `python/lawspec_tests/`. Both components keep
local ownership metadata in ignored `.lawspec/` directories. Edit the `.lawspec`
source contracts and authored adapters, then regenerate.

```sh
npm run verify
mix test
```

Verification checks definitions, regenerates and executes both LawSpec components,
then lists Canon review status and runs Canon validation directly. The supported
profile currently has 190 Elixir LawSpec cases and 314 Python cases; `mix test` also runs the authored
ExUnit tests. Reports are written to ignored `artifacts/` and `.lawspec/` folders.
Verification runs fresh tests by default. `npm run verify -- --cached` lets
LawSpec reuse current evidence; changed inputs still require execution.

**Canon human review is pending.** `npm run verify` returns **2** when executable
checks pass but Canon validation remains nonpassing; inspect
`artifacts/verification-summary.json`, `artifacts/canon-review.log` and
`artifacts/canon-validation.log` for review status and any findings. Test or
definition failures return **1**. Setup and the working demo do not require a
human sign-off. See [validation evidence](docs/reference/validation.md) and
[conformance ownership](docs/reference/law-owned-conformance.md).

## Use the APIs

Build a graph in Elixir:

```elixir
alias Tensorloom, as: T
x = T.input("x", T.tensor([2, 3]))
graph = T.graph([x], [T.op("y", "relu", [T.ref("x")])], T.ref("y"))
{:ok, _validated} = T.validate(graph)
{:ok, %{result: type, constraints: guards}} = T.infer(graph)
{:ok, json} = T.encode(graph)
{:ok, ^graph} = T.decode(json)
{:ok, source} = T.compile(graph, lowering: :expanded)
File.write!("artifacts/relu.py", source)
```

Generated modules import `tensorloom`. For Python scripts, use
`PYTHONPATH=python uv run --locked python your_script.py` from the repository root.
Python APIs invoke the compiled Elixir validator, so run setup first.

```python
import torch
from tensorloom import capture, compile_graph, export_graph

model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU()).double()
x = torch.randn(2, 3, dtype=torch.float64)
graph = capture(model, (x,), method="fx")  # explicit alternative: "export"
generated = compile_graph(graph, lowering="expanded")  # or "native"
torch.testing.assert_close(generated(x), model(x))
export_graph(graph, (x,), "artifacts/model.pt2", dynamic_shapes=(
    {0: torch.export.Dim("batch", min=1, max=8)},
))
loaded = torch.export.load("artifacts/model.pt2").module()
```

`expand/1`, `collapse/2` and `resolve/2` transform validated graphs.
`Tensorloom.Builder.sequence/5` composes fixed-length recurrent cells, including
zero passes. Capture methods never silently fall back to another method.

The first numerical profile is dense CPU tensors with Float32, Float64, Int64
and Bool, with fixed rank. See [the notation](graph_notation.txt) and
[the PyTorch comparison](docs/reference/pytorch-comparison.md) for supported
families, export restrictions and unsupported capabilities.
[examples/mlp.json](examples/mlp.json) is a captured MLP fixture. Inspect
it with `mix run scripts/graph.exs infer examples/mlp.json`, or emit Python with
`mix run scripts/graph.exs compile examples/mlp.json native`.
