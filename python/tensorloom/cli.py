import argparse
import json
from pathlib import Path
import torch
from .compiler import compile_source, export_graph, Diagnostic
from .kernels import DTYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["compile", "export"])
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        if args.action == "compile":
            print(
                compile_source(
                    request["graph"],
                    request.get("lowering", "expanded"),
                    request.get("profile", "eager"),
                ),
                end="",
            )
        else:

            def value(t, v):
                if t["kind"] == "tensor":
                    return torch.tensor(v, dtype=DTYPES[t["dtype"]])
                if t["kind"] == "tuple":
                    return tuple(value(c, x) for c, x in zip(t["items"], v))
                if t["kind"] == "dict":
                    return {k: value(c, v[k]) for k, c in t["items"].items()}
                return v

            examples = [
                value(s["type"], v)
                for s, v in zip(request["graph"]["inputs"], request["examples"])
            ]
            dynamic = None
            if request.get("dynamic_dims"):
                symbols = {}

                def dimension(spec):
                    name = spec["name"]
                    if name not in symbols:
                        symbols[name] = torch.export.Dim(
                            name, min=spec.get("min", 1), max=spec.get("max")
                        )
                    return symbols[name]

                dynamic = tuple(
                    {
                        int(k): dimension(v)
                        for k, v in request["dynamic_dims"].get(str(i), {}).items()
                    }
                    for i in range(len(examples))
                )
            export_graph(
                request["graph"],
                examples,
                request["path"],
                request.get("lowering", "native"),
                dynamic,
            )
            print(
                json.dumps(
                    {
                        "path": request["path"],
                        "profile": "export",
                        "lowering": request.get("lowering", "native"),
                    }
                )
            )
    except (Diagnostic, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "code": getattr(exc, "code", "backend-error"),
                    "op": getattr(exc, "op", None),
                    "message": str(exc),
                }
            )
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
