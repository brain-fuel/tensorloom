"""Run the Elixir-built graph, capture PyTorch, and reload a saved export."""
import importlib.util
import json
import os
from pathlib import Path

import torch
from tensorloom import capture, compile_graph, export_graph


def main():
    directory = Path(os.environ.get("TENSORLOOM_DEMO_DIR", "artifacts/demo"))
    spec = importlib.util.spec_from_file_location("tensorloom_demo", directory / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    emitted = module.GeneratedGraph()
    x = torch.tensor([[3.0, 4.0], [-1.0, 2.0]])
    expected = torch.tensor([[2.5], [0.0]])
    output = emitted(x)
    torch.testing.assert_close(output, expected)
    print("Elixir -> generated PyTorch:", output.detach().tolist())

    output.sum().backward()
    gradient = next(emitted.parameters()).grad
    torch.testing.assert_close(gradient, torch.tensor([[3.0, 4.0]]))
    print("Weight gradient:", gradient.tolist())

    reference = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.ReLU())
    with torch.no_grad():
        reference[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
        reference[0].bias.fill_(0.5)
    captured = capture(reference, (x,), method="fx")
    for lowering in ("native", "expanded"):
        compiled = compile_graph(captured, lowering=lowering)
        actual = compiled(x)
        torch.testing.assert_close(actual, reference(x))
        print(f"Captured PyTorch -> {lowering}:", actual.detach().tolist())

    graph = json.loads((directory / "graph.json").read_text())
    path = directory / "model.pt2"
    export_graph(graph, (x,), path, dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},))
    loaded = torch.export.load(path).module()
    torch.testing.assert_close(loaded(x), expected)
    print("Saved export -> reloaded:", loaded(x).detach().tolist())
    print("Demo passed. Generated files are in", directory)


if __name__ == "__main__":
    main()
