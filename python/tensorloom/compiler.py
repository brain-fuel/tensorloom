"""Source generation, input guards and a separately restricted export profile."""

from pathlib import Path
import copy
import json
import os
import subprocess
import tempfile
import torch
from .kernels import DTYPES

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = json.loads((ROOT / "priv/operations.json").read_text())


class Diagnostic(ValueError):
    def __init__(self, code, message, op=None):
        super().__init__(f"{code}{' at ' + op if op else ''}: {message}")
        self.code, self.op, self.message = code, op, message


def elixir(action, graph, *arguments):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(graph, f, allow_nan=False)
        f.flush()
        env = dict(os.environ, MIX_ENV="dev")
        proc = subprocess.run(
            [
                "mix",
                "run",
                "--no-compile",
                "scripts/graph.exs",
                action,
                f.name,
                *arguments,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    if proc.returncode:
        try:
            failure = json.loads(proc.stdout)["diagnostics"][0]
        except (ValueError, KeyError, IndexError):
            raise Diagnostic("validator-failed", proc.stdout + proc.stderr) from None
        raise Diagnostic(failure["code"], failure["message"], failure.get("op"))
    return json.loads(proc.stdout)


def guard_inputs(specs, values, constraints):
    symbols = {}

    def check(t, v):
        if t["kind"] == "tensor":
            if not isinstance(v, torch.Tensor):
                raise Diagnostic("input-type", "expected tensor")
            if (
                v.dtype != DTYPES[t["dtype"]]
                or v.device.type != t["device"]
                or v.layout != torch.strided
            ):
                raise Diagnostic("input-type", "dtype, device or layout mismatch")
            if v.ndim != len(t["shape"]):
                raise Diagnostic("input-shape", "rank mismatch")
            for expected, actual in zip(t["shape"], v.shape):
                if isinstance(expected, int):
                    if actual != expected:
                        raise Diagnostic(
                            "input-shape", f"expected {expected}, got {actual}"
                        )
                elif expected in symbols:
                    if actual != symbols[expected]:
                        raise Diagnostic("symbolic-guard", f"inconsistent {expected}")
                else:
                    symbols[expected] = actual
        elif t["kind"] == "tuple":
            if not isinstance(v, tuple) or len(v) != len(t["items"]):
                raise Diagnostic("input-type", "tuple structure")
            for child, item in zip(t["items"], v):
                check(child, item)
        elif t["kind"] == "dict":
            if not isinstance(v, dict) or v.keys() != t["items"].keys():
                raise Diagnostic("input-type", "dict structure")
            for k, child in t["items"].items():
                check(child, v[k])
        elif t["kind"] == "scalar":
            expected = (
                bool
                if t["dtype"] == "bool"
                else int
                if t["dtype"] == "int64"
                else float
            )
            if type(v) is not expected:
                raise Diagnostic("input-type", f"expected {expected.__name__}")

    for t, v in zip(specs, values):
        check(t, v)
    for symbol, bounds in constraints.items():
        if symbol in symbols and not bounds.get("min", 0) <= symbols[
            symbol
        ] <= bounds.get("max", 2**63 - 1):
            raise Diagnostic("symbolic-guard", f"{symbol} outside declared bounds")


def _walk(graph):
    for op in graph["ops"]:
        yield op
        for region in op.get("regions", {}).values():
            yield from _walk(region)


def _profile(graph, profile):
    if profile not in ("eager", "export"):
        raise Diagnostic("unsupported-profile", profile)
    if profile == "export":
        for op in _walk(graph):
            if op["op"] == "loop" and "count" not in op.get("attrs", {}):
                raise Diagnostic(
                    "export-restriction", "convergence loops are eager only", op["id"]
                )
            if op["op"] == "copy_buffer":
                raise Diagnostic(
                    "export-restriction", "explicit mutation is eager only", op["id"]
                )
            if (
                op["op"] == "batch_norm"
                and op.get("attrs", {}).get("momentum", 0.1) is None
            ):
                raise Diagnostic(
                    "export-restriction",
                    "cumulative BatchNorm needs scalar extraction",
                    op["id"],
                )


class Emitter:
    def __init__(self, lowering, profile, types, symbols):
        self.lowering, self.profile, self.types = lowering, profile, types
        self.lines = []
        self.counter = 0
        self.symbols = symbols

    def fresh(self, prefix="v"):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def line(self, indent, s):
        self.lines.append("    " * indent + s)

    def expr(self, value, env):
        if "ref" in value:
            return env[value["ref"]]
        if "lit" in value:
            return repr(value["lit"])
        if "tuple" in value:
            return "(" + "".join(self.expr(x, env) + "," for x in value["tuple"]) + ")"
        if "dict" in value:
            return (
                "{"
                + ",".join(
                    repr(k) + ":" + self.expr(v, env) for k, v in value["dict"].items()
                )
                + "}"
            )
        raise Diagnostic("invalid-value", str(value))

    def region(self, g, env, indent):
        env = dict(env)
        local_types = dict(self.types)
        local_types.update({x["id"]: x["type"] for x in g["inputs"]})
        if any(
            isinstance(node.get("attrs", {}).get(key), str)
            for node in g["ops"]
            for key in ("dim", "dim0", "dim1")
        ):
            local_types = elixir("infer_region", g)["types"]
        for op in g["ops"]:
            name = op["op"]
            v = self.fresh()
            attrs = copy.deepcopy(op.get("attrs", {}))
            argv = [self.expr(x, env) for x in op.get("args", [])]
            rs = op.get("regions", {})
            if name == "call":
                body = rs["body"]
                captures = dict(zip((x["id"] for x in body["inputs"]), argv))
                result = self.region(body, captures, indent)
                self.line(indent, f"{v} = {result}")
            elif name == "branch":

                def arm(key, ind):
                    body = rs[key]
                    captures = dict(zip((x["id"] for x in body["inputs"]), argv[1:]))
                    return self.region(body, captures, ind)

                if self.profile == "eager":
                    self.line(indent, f"if bool({argv[0]}):")
                    result = arm("true", indent + 1)
                    self.line(indent + 1, f"{v} = {result}")
                    self.line(indent, "else:")
                    result = arm("false", indent + 1)
                    self.line(indent + 1, f"{v} = {result}")
                else:
                    funcs = []
                    for key in ("true", "false"):
                        fn = self.fresh("branch")
                        funcs.append(fn)
                        formal = [self.fresh("arg") for _ in argv[1:]]
                        self.line(indent, f"def {fn}({', '.join(formal)}):")
                        captures = dict(
                            zip((x["id"] for x in rs[key]["inputs"]), formal)
                        )
                        result = self.region(rs[key], captures, indent + 1)
                        self.line(indent + 1, f"return {result}")
                    self.line(
                        indent,
                        f"{v} = torch.cond({argv[0]}, {funcs[0]}, {funcs[1]}, ({','.join(argv[1:])},))",
                    )
            elif name == "loop":
                self.line(indent, f"{v} = {argv[0]}")
                body = rs["body"]
                var = body["inputs"][0]["id"]
                if "count" in attrs:
                    self.line(indent, f"for _ in range({attrs['count']}):")
                    result = self.region(body, {var: v}, indent + 1)
                    self.line(indent + 1, f"{v} = {result}")
                else:
                    c = rs["condition"]
                    cv = c["inputs"][0]["id"]
                    idx = self.fresh("iteration")
                    self.line(indent, f"for {idx} in range({attrs['budget'] + 1}):")
                    pred = self.region(c, {cv: v}, indent + 1)
                    self.line(indent + 1, f"if not bool({pred}): break")
                    self.line(
                        indent + 1,
                        f'if {idx} == {attrs["budget"]}: raise Diagnostic("loop-budget", "execution budget exhausted", {op["id"]!r})',
                    )
                    result = self.region(body, {var: v}, indent + 1)
                    self.line(indent + 1, f"{v} = {result}")
            else:
                if name not in REGISTRY:
                    raise Diagnostic("unsupported-operation", name, op["id"])
                # Axis names select axes; PyTorch broadcasting stays positional.
                if op.get("args") and "ref" in op["args"][0]:
                    t = local_types.get(op["args"][0]["ref"], {})
                    for key in ("dim", "dim0", "dim1"):
                        if isinstance(attrs.get(key), str):
                            attrs[key] = t.get("axes", []).index(attrs[key])
                attrs_source = repr(attrs)
                if name == "reshape":

                    def extent(d):
                        if isinstance(d, int):
                            return repr(d)
                        if isinstance(d, str):
                            if d not in self.symbols:
                                raise Diagnostic(
                                    "symbolic-guard",
                                    f"unbound shape symbol {d}",
                                    op["id"],
                                )
                            return self.symbols[d]
                        if isinstance(d, dict) and d.get("expr") in (
                            "mul",
                            "add",
                            "div",
                        ):
                            left, right = map(extent, d["args"])
                            sign = {"mul": "*", "add": "+", "div": "//"}[d["expr"]]
                            return f"({left}{sign}{right})"
                        raise Diagnostic(
                            "symbolic-guard",
                            f"unsupported extent expression {d}",
                            op["id"],
                        )

                    attrs_source = (
                        "{'shape': ["
                        + ", ".join(extent(d) for d in attrs["shape"])
                        + "]}"
                    )
                self.line(
                    indent,
                    f"{v} = execute({name!r}, [{', '.join(argv)}], {attrs_source}, self.training, {self.lowering!r})",
                )
            env[op["id"]] = v
        return self.expr(g["result"], env)


def compile_source(graph, lowering="expanded", profile="eager"):
    if lowering not in ("expanded", "native"):
        raise Diagnostic("unsupported-lowering", lowering)
    info = elixir("infer", graph)
    _profile(graph, profile)
    for op in _walk(graph):
        if (
            lowering == "expanded"
            and op["op"] == "attention"
            and op.get("attrs", {}).get("enable_gqa")
        ):
            raise Diagnostic(
                "unsupported-lowering", "expanded GQA is unsupported", op["id"]
            )
    specs = [x["type"] for x in graph["inputs"]]
    symbols = {}

    def collect_symbols(t, expression):
        if t["kind"] == "tensor":
            for i, d in enumerate(t["shape"]):
                if isinstance(d, str):
                    symbols.setdefault(d, f"{expression}.shape[{i}]")
        elif t["kind"] in ("tuple", "dict"):
            items = (
                enumerate(t["items"]) if t["kind"] == "tuple" else t["items"].items()
            )
            for k, child in items:
                collect_symbols(child, f"{expression}[{k!r}]")

    for i, t in enumerate(specs):
        collect_symbols(t, f"x{i}")
    emitter = Emitter(lowering, profile, info["types"], symbols)
    header = [
        "import torch",
        "from tensorloom.kernels import execute, DTYPES",
        "from tensorloom.compiler import guard_inputs, Diagnostic",
        "",
        "class GeneratedGraph(torch.nn.Module):",
        f"    graph = {graph!r}",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    env = {}
    state = graph.get("state", [])
    names = {x["id"]: f"s{i}" for i, x in enumerate(state)}
    # Canonical state is initialized once, and aliases bind to that exact object.
    for x in sorted(state, key=lambda x: bool(x.get("alias"))):
        n = names[x["id"]]
        env[x["id"]] = "self." + n
        if x.get("alias"):
            header.append(f"        self.{n} = self.{names[x['alias']]}")
        else:
            t = x["type"]
            expression = f"torch.tensor({x['data']!r}, dtype=DTYPES[{t['dtype']!r}]).reshape({t['shape']!r})"
            if x["kind"] == "parameter":
                header.append(
                    f"        self.register_parameter({n!r}, torch.nn.Parameter({expression}, requires_grad={x.get('requires_grad', True)!r}))"
                )
            else:
                header.append(
                    f"        self.register_buffer({n!r}, {expression}, persistent={x.get('persistent', True)!r})"
                )
    header.append(
        f"        self.train({graph.get('capture', {}).get('training', True)!r})"
    )
    args = [f"x{i}" for i in range(len(specs))]
    header += [f"    def forward(self{', ' if args else ''}{', '.join(args)}):"]
    if profile == "eager":
        header += [
            f"        guard_inputs({specs!r}, [{', '.join(args)}], {graph.get('constraints', {})!r})"
        ]
    env.update(zip((x["id"] for x in graph["inputs"]), args))
    result = emitter.region(graph, env, 2)
    return "\n".join(header + emitter.lines + [f"        return {result}", ""])


def compile_graph(graph, lowering="expanded", profile="eager"):
    source = compile_source(graph, lowering, profile)
    namespace = {}
    exec(compile(source, "<tensorloom-generated>", "exec"), namespace)
    module = namespace["GeneratedGraph"]()
    module.generated_source = source
    return module


def export_graph(graph, example_inputs, path, lowering="native", dynamic_shapes=None):
    module = compile_graph(graph, lowering, profile="export").eval()
    try:
        program = torch.export.export(
            module, tuple(example_inputs), dynamic_shapes=dynamic_shapes, strict=True
        )
        torch.export.save(program, path)
    except Exception as exc:
        raise Diagnostic("export-failed", str(exc)) from exc
    return program
