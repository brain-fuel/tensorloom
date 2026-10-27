"""LawSpec native adapters: construct fixtures and return typed observations.

There are no conformance assertions here. The checked LawSpec definitions own
comparison, tolerances, nonfinite semantics, case enumeration, and mutation laws.
"""

from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy
import json
import math
import torch
from torch import nn
from . import capture, compile_graph, export_graph, Diagnostic
from .capture import tensor_type, value_type
from .compiler import elixir, REGISTRY
from . import kernels
from . import law_domain as D


def name(value):
    return type(value).__name__


def r(key):
    return {"ref": key}


def lit(value):
    return {"lit": value}


def op(key, operation, args, attrs=None, regions=None):
    return dict(
        id=key, op=operation, args=args, attrs=attrs or {}, regions=regions or {}
    )


def graph(inputs, ops, result, state=None):
    return dict(
        version=1,
        inputs=[dict(id=k, type=value_type(v)) for k, v in inputs.items()],
        ops=ops,
        result=result,
        state=state or [],
    )


def state(key, value):
    return dict(id=key, kind="parameter", type=tensor_type(value), data=value.tolist())


def sample(x):
    x = float(x)
    if math.isnan(x):
        return D.NotANumber()
    if math.isinf(x):
        return D.PositiveInfinity() if x > 0 else D.NegativeInfinity()
    return D.Finite(Fraction(x))


class Recorder:
    def __init__(self):
        self.tensors = []
        self.facts = []

    def add(self, path, value):
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1).tolist()
            samples = [
                D.Finite(Fraction(int(v)))
                if not value.is_floating_point()
                else sample(v)
                for v in flat
            ]
            self.tensors.append(
                D.TensorObservation(
                    path=path,
                    shape=list(value.shape),
                    dtype=str(value.dtype).split(".")[-1],
                    samples=samples,
                )
            )
        elif isinstance(value, (tuple, list)):
            self.facts.append(f"{path}:{type(value).__name__}:{len(value)}")
            for i, v in enumerate(value):
                self.add(f"{path}/{i}", v)
        elif isinstance(value, dict):
            self.facts.append(f"{path}:dict:{json.dumps(sorted(value))}")
            for k, v in sorted(value.items()):
                self.add(f"{path}/{k}", v)
        else:
            self.facts.append(f"{path}:{value!r}")

    def result(self):
        return D.Observation(tensors=self.tensors, facts=self.facts)


def energy(value, mode="square"):
    if isinstance(value, dict):
        return sum(energy(x, mode) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(energy(x, mode) for x in value)
    return (
        (value.sum() if mode == "sum" else value.square().sum())
        if value.is_floating_point()
        else 0
    )


def leaves(value):
    if isinstance(value, (tuple, list)):
        return [x for v in value for x in leaves(v)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in leaves(v)]
    return [value] if isinstance(value, torch.Tensor) else []


def snapshot(rec, module, xs, label="output", gradient=True, loss_mode="square"):
    out = module(*xs)
    rec.add(label, out)
    if gradient:
        loss = energy(out, loss_mode)
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            rec.add(
                label + "/input_gradients",
                [
                    x.grad
                    for x in leaves(xs)
                    if x.is_floating_point() and x.requires_grad
                ],
            )
            rec.add(
                label + "/parameter_gradients",
                [p.grad for p in module.parameters() if p.requires_grad],
            )
    return out


def compiled(g, mode, defect=None):
    model = compile_graph(json.loads(json.dumps(g)), mode)
    if defect == "DuplicateParameter":
        source = model.generated_source.replace(
            "self.s1 = self.s0",
            "self.s1 = torch.nn.Parameter(self.s0.detach().clone())",
        )
        namespace = {}
        exec(compile(source, "<lawspec-parameter-mutant>", "exec"), namespace)
        model = namespace["GeneratedGraph"]()
    return model


class Unary(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(3, 3, bias=False)
        self.b = nn.Linear(3, 3, bias=False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return {"result": self.a(x) + self.b(x) + x, "pair": (x, x)}


def layer_fixture(kind, dtype, expansion=False):
    if kind in ("Mlp", "LinearExpansion"):
        return (
            (nn.Linear(3, 4), (2, 3))
            if expansion
            else (nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 3)), (2, 5))
        )
    if kind in ("Convolution", "ConvExpansion"):
        return (
            (nn.Conv2d(4, 6, 3, padding=2, dilation=2, groups=2), (2, 4, 6, 7))
            if expansion
            else (
                nn.Conv2d(
                    4,
                    6,
                    (3, 2),
                    stride=(2, 1),
                    padding=(2, 1),
                    dilation=(2, 1),
                    groups=2,
                ),
                (2, 4, 8, 7),
            )
        )
    if kind in ("Normalization", "LayerNormExpansion"):
        return nn.LayerNorm((3, 4)), (2, 3, 4)
    if kind in ("Softmax", "SoftmaxExpansion"):
        return nn.Softmax(-1), (2, 3, 4)
    if kind == "EmptySoftmaxExpansion":
        return nn.Softmax(-1), (2, 0)
    if kind in ("MaxPool", "MaxPoolExpansion"):
        return nn.MaxPool2d(3, 2, 1, ceil_mode=True, return_indices=True), (2, 3, 6, 7)
    if kind in ("AverageExcludePad", "AvgPoolExpansion"):
        return nn.AvgPool2d(3, 2, 1, ceil_mode=True, count_include_pad=False), (
            2,
            3,
            6,
            7,
        )
    if kind == "AverageIncludePad":
        return nn.AvgPool2d(3, 2, 1, ceil_mode=True, count_include_pad=True), (
            1,
            2,
            5,
            6,
        )
    if kind == "BatchNormExpansion":
        return nn.BatchNorm2d(3), (2, 3, 4, 5)
    raise ValueError(kind)


def module_case(case, actual, rec, defect=None):
    family = name(case)
    value = case.value
    dtype = (
        torch.float32
        if name(getattr(value, "precision", None)) == "Float32"
        else torch.float64
    )
    mode = (
        "native" if name(getattr(value, "lowering", value)) == "Native" else "expanded"
    )
    expansion = family == "ExpandedExecution"
    kind = name(value.expansion if expansion else value.kernel)
    if kind == "AttentionExpansion":
        return attention_case(
            D.AttentionExecution(
                D.AttentionCase(
                    mask=D.FullyMasked(), precision=D.Float64(), lowering=value.lowering
                )
            ),
            actual,
            rec,
            True,
        )
    if kind == "DropoutExpansion":
        return dropout_case(value.lowering, actual, rec, True)
    layer, shape = layer_fixture(kind, dtype, expansion)
    original = Unary(layer).to(dtype)
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    model = original
    if actual:
        g = capture(original, (x,))
        if expansion:
            e = elixir("expand", g)
            rec.add("expansion_stable", elixir("expand", e) == e)

            def all_expanded(region):
                return all(
                    not REGISTRY[n["op"]]["expansion"]
                    and all(all_expanded(r) for r in n.get("regions", {}).values())
                    for n in region["ops"]
                )

            rec.add("fully_expanded", all_expanded(e))
            g = e
        model = compiled(g, mode, defect)
    elif expansion:
        rec.add("expansion_stable", True)
        rec.add("fully_expanded", True)
    snapshot(rec, model, (x,))
    if expansion:
        model.eval()
        snapshot(rec, model, (x,), label="eval", gradient=False)
        rec.add("parameter_shapes", [list(p.shape) for p in model.parameters()])
        rec.add("buffers", list(model.buffers()))


def attention_case(case, actual, rec, expansion=False):
    c = case.value
    dtype = torch.float32 if name(c.precision) == "Float32" else torch.float64
    mode = "native" if name(c.lowering) == "Native" else "expanded"
    q = torch.randn(2, 2, 3, 4, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 5, 4, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 2, 5, 6, dtype=dtype, requires_grad=True)
    mask = torch.rand(3, 5) > 0.3
    kind = name(c.mask)
    if kind == "FullyMasked":
        mask[:] = False
    if kind == "AdditiveMask":
        mask = torch.zeros(3, 5, dtype=dtype).masked_fill(~mask, float("-inf"))
    xs = [q, k, v] + ([] if kind == "CausalMask" else [mask])
    attrs = {"is_causal": True} if kind == "CausalMask" else {}
    if actual:
        g = graph(
            {str(i): x for i, x in enumerate(xs)},
            [op("y", "attention", [r(str(i)) for i in range(len(xs))], attrs)],
            r("y"),
        )
        if expansion:
            e = elixir("expand", g)
            rec.add("expansion_stable", elixir("expand", e) == e)
            g = e
        out = compiled(g, mode)(*xs)
    else:
        if expansion:
            rec.add("expansion_stable", True)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, None if kind == "CausalMask" else mask, **attrs
        )
    rec.add("output", out)
    out.sum().backward()
    rec.add("gradients", [q.grad, k.grad, v.grad])


def dropout_case(lowering, actual, rec, expansion=False):
    mode = "native" if name(lowering) == "Native" else "expanded"
    x = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    g = graph(
        dict(x=x, mask=mask),
        [op("y", "dropout", [r("x"), r("mask")], {"p": 0.5})],
        r("y"),
    )
    if actual:
        if expansion:
            e = elixir("expand", g)
            rec.add("expansion_stable", elixir("expand", e) == e)
            g = e
        model = compiled(g, mode)
        out = model(x, mask)
    else:
        if expansion:
            rec.add("expansion_stable", True)
        out = x * mask / 0.5
    rec.add("output", out)
    out.sum().backward()
    rec.add("gradient", x.grad)
    if actual:
        model.eval()
        rec.add("eval_alias", model(x, mask) is x)
    else:
        rec.add("eval_alias", True)
    g["ops"] = [op("y", "dropout", [r("x"), r("mask")], {"p": 1.0})]
    rec.add("p_one", compiled(g, mode)(x, mask) if actual else x * 0)


def error_code(fn):
    try:
        fn()
    except Diagnostic as e:
        return e.code
    except IndexError:
        return "index-error"
    except (RuntimeError, AssertionError):
        return "runtime-error"
    return "no-error"


def run_case(case, actual, defect=None):
    torch.set_num_threads(1)
    rec = Recorder()
    family = name(case)
    value = getattr(case, "value", None)
    mode = (
        "native" if name(getattr(value, "lowering", value)) == "Native" else "expanded"
    )
    if family in ("ModuleExecution", "ExpandedExecution"):
        module_case(case, actual, rec, defect)
    elif family == "AttentionExecution":
        attention_case(case, actual, rec)
    elif family == "DropoutExecution":
        dropout_case(value, actual, rec)
    elif family == "SharedExecution":
        dtype = torch.float32 if name(value.precision) == "Float32" else torch.float64
        original = Shared().to(dtype)
        x = torch.randn(4, 3, dtype=dtype, requires_grad=True)
        model = compiled(capture(original, (x,)), mode, defect) if actual else original
        rec.add("parameter_count", len(list(model.parameters())))
        rec.add(
            "shared_identity",
            model.s0 is model.s1 if actual else model.a.weight is model.b.weight,
        )
        out = snapshot(rec, model, (x,))
        rec.add("fanout_aliases", [out["pair"][0] is x, out["pair"][1] is x])
    elif family == "BatchState":
        original = Unary(nn.BatchNorm2d(3)).double()
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        g = capture(original, (x,)) if actual else None
        model = compiled(g, mode) if actual else original
        for i in range(3):
            snapshot(rec, model, (x,), label=f"train{i}", gradient=False)
        rec.add("buffers", list(model.buffers()))
        model.eval()
        snapshot(rec, model, (x,), label="eval", gradient=False)
        restored = compiled(g, mode) if actual else copy.deepcopy(original)
        restored.load_state_dict(model.state_dict())
        restored.eval()
        snapshot(rec, restored, (x,), label="restored", gradient=False)
    elif family in ("CellExecution", "SequenceExecution"):
        cls = {"Rnn": nn.RNN, "Gru": nn.GRU, "Lstm": nn.LSTM}[name(value.cell)]
        if family == "CellExecution":
            cls = {"Rnn": nn.RNNCell, "Gru": nn.GRUCell, "Lstm": nn.LSTMCell}[
                name(value.cell)
            ]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = cls(
                    3,
                    4,
                    **({"batch_first": True} if family == "SequenceExecution" else {}),
                ).double()

            def forward(self, x, h):
                return self.layer(x, h)

        original = Model()
        x = torch.randn(
            (2, 3, 3) if family == "SequenceExecution" else (2, 3),
            dtype=torch.float64,
            requires_grad=True,
        )
        h = torch.randn(
            (1, 2, 4) if family == "SequenceExecution" else (2, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
        h = (
            (h, h.detach().clone().requires_grad_())
            if name(value.cell) == "Lstm"
            else h
        )
        g = capture(original, (x, h)) if actual else None
        model = compiled(g, mode) if actual else original
        snapshot(rec, model, (x, h))
        if family == "SequenceExecution":
            e = elixir("expand", g) if actual else None
            model = compiled(e, mode) if actual else original
            snapshot(rec, model, (x, h), label="expanded_sequence", gradient=False)
    elif family == "EmbeddingExecution":
        original = Unary(nn.Embedding(7, 3, padding_idx=0)).double()
        x = torch.tensor([[0, 2, 2], [1, 3, 6]])
        model = compiled(capture(original, (x,)), mode) if actual else original
        snapshot(rec, model, (x,), loss_mode="sum")
        rec.add(
            "invalid_index",
            error_code(lambda: model(torch.tensor([[-1, 2, 2], [1, 3, 6]]))),
        )
    elif family == "SoftmaxBoundaries":
        for i, x in enumerate(
            (
                torch.tensor([[10000.0, 10001.0], [-float("inf"), -float("inf")]]),
                torch.empty(2, 0),
                torch.tensor([[float("inf"), 1.0]]),
            )
        ):
            g = graph(dict(x=x), [op("y", "softmax", [r("x")], {"dim": -1})], r("y"))
            rec.add(
                f"boundary{i}", compiled(g, mode)(x) if actual else torch.softmax(x, -1)
            )
    elif family == "ControlExecution":
        control_case(actual, mode, rec, defect)
    elif family == "CollapseExecution":
        x = torch.randn(2, 3)
        w = torch.randn(3, 3)
        g = graph(
            dict(x=x),
            [
                op("a", "linear", [r("x"), r("w")]),
                op("b", "relu", [r("a")]),
                op("c", "add", [r("a"), r("b")]),
            ],
            {"tuple": [r("a"), r("c")]},
            [state("w", w)],
        )
        expected = (
            nn.functional.linear(x, w),
            nn.functional.linear(x, w) + nn.functional.linear(x, w).relu(),
        )
        if actual:
            collapsed = elixir("collapse", g, "a", "b")
            expanded = elixir("expand", collapsed)
            rec.add("collapsed", compiled(collapsed, mode)(x))
            rec.add("expanded", compiled(expanded, mode)(x))
            rec.add("stable", elixir("expand", expanded) == expanded)
            rec.add(
                "invalid_collapse", error_code(lambda: elixir("collapse", g, "a", "c"))
            )
        else:
            rec.add("collapsed", expected)
            rec.add("expanded", expected)
            rec.add("stable", True)
            rec.add("invalid_collapse", "invalid-graph")
    elif family in (
        "SavedExport",
        "CapturedExport",
        "SavedConditional",
        "SavedRecurrence",
        "CapturedConditional",
        "ImportedExecution",
    ):
        export_case(case, actual, mode, rec)
    elif family == "RejectedExecution":
        rejection_case(value, actual, rec)
    elif family == "NamedAxes":
        x = torch.randn(2, 3)
        t = tensor_type(x)
        t["axes"] = ["batch", "feature"]
        body = graph(
            dict(x=x),
            [
                op("a", "transpose", [r("x")], {"dim0": "batch", "dim1": "feature"}),
                op("b", "sum", [r("a")], {"dim": "feature"}),
            ],
            r("b"),
        )
        body["inputs"][0]["type"] = t
        g = graph(
            dict(x=x), [op("call", "call", [r("x")], regions={"body": body})], r("call")
        )
        g["inputs"][0]["type"] = t
        rec.add("output", compiled(g, mode)(x) if actual else x.sum(1))
    elif family == "FrozenState":

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2), requires_grad=False)
                self.register_buffer("offset", torch.ones(2), persistent=False)

            def forward(self, x):
                return x * self.weight + self.offset

        original = Model().eval()
        x = torch.randn(2)
        model = compiled(capture(original, (x,)), mode) if actual else original
        rec.add("training", model.training)
        rec.add("requires_grad", next(model.parameters()).requires_grad)
        rec.add("persistent_count", len(model.state_dict()))
        rec.add("output", model(x))
    else:
        raise ValueError(f"unknown conformance case {family}")
    return rec.result()


def control_case(actual, mode, rec, defect=None):
    x = torch.tensor([2.0, 3.0])
    yes = graph(dict(a=x), [op("yes", "mul", [r("a"), lit(2)])], r("yes"))
    no = graph(dict(a=x), [op("no", "log", [r("a")])], r("no"))
    g = graph(
        dict(x=x, p=True),
        [op("r", "branch", [r("p"), r("x")], regions={"true": yes, "false": no})],
        r("r"),
    )
    for flag in (False, True):
        rec.add(
            f"branch{flag}",
            compiled(g, mode)(x, flag) if actual else (x * 2 if flag else x.log()),
        )
        resolved = elixir("resolve", g, json.dumps({"r": flag})) if actual else None
        rec.add(
            f"resolved{flag}",
            compiled(resolved, mode)(x, flag)
            if actual
            else (x * 2 if flag else x.log()),
        )
    body = graph(dict(a=x), [op("v", "add", [r("a"), lit(1)])], r("v"))
    for count in (0, 1, 4):
        loop = graph(
            dict(x=x),
            [op("r", "loop", [r("x")], {"count": count}, {"body": body})],
            r("r"),
        )
        rec.add(f"loop{count}", compiled(loop, mode)(x) if actual else x + count)
    scalar = torch.tensor(0.0)
    body = graph(dict(a=scalar), [op("v", "add", [r("a"), lit(1)])], r("v"))
    condition = graph(dict(a=scalar), [op("v", "lt", [r("a"), lit(3)])], r("v"))
    loop = graph(
        dict(x=scalar),
        [
            op(
                "r",
                "loop",
                [r("x")],
                {"budget": 2},
                {"body": body, "condition": condition},
            )
        ],
        r("r"),
    )
    rec.add(
        "budget_failure",
        error_code(lambda: compiled(loop, mode)(scalar)) if actual else "loop-budget",
    )
    loop["ops"][0]["attrs"]["budget"] = 3
    rec.add("budget_success", compiled(loop, mode)(scalar) if actual else scalar + 3)
    rec.add(
        "export_restriction",
        error_code(lambda: compile_graph(loop, mode, profile="export"))
        if actual
        else "export-restriction",
    )
    trace = []
    if actual:
        execute = kernels.execute

        def traced(operation, *args):
            trace.append(operation)
            return execute(operation, *args)

        with patch.object(kernels, "execute", traced):
            source = compiled(g, mode).generated_source
            if defect == "ExecuteBothBranches":
                source = source.replace(
                    "        if bool(x1):",
                    f"        execute('log', [x0], {{}}, self.training, {mode!r})\n        if bool(x1):",
                )
            namespace = {}
            exec(source, namespace)
            namespace["GeneratedGraph"]()(x, True)
    else:
        trace = ["mul"]
    rec.add("exclusive_trace", trace)


def rejection_case(value, actual, rec):
    kind = name(value)
    if not actual:
        rec.add(
            "diagnostic",
            {
                "DynamicCapture": "capture-failed",
                "UnknownCapture": "unsupported-operation",
                "WrongRank": "input-shape",
                "WrongDtype": "input-type",
            }[kind],
        )
        return
    if kind == "DynamicCapture":

        class Model(nn.Module):
            def forward(self, x):
                return x + 1 if x.sum() > 0 else x - 1

        rec.add(
            "diagnostic",
            error_code(lambda: capture(Model(), (torch.ones(2),), method="fx")),
        )
    elif kind == "UnknownCapture":

        class Model(nn.Module):
            def forward(self, x):
                return torch.fft.fft(x)

        rec.add("diagnostic", error_code(lambda: capture(Model(), (torch.ones(2),))))
    else:
        x = torch.zeros(2, 3)
        g = graph(dict(x=x), [op("y", "relu", [r("x")])], r("y"))
        model = compiled(g, "expanded")
        rec.add(
            "diagnostic",
            error_code(
                lambda: model(torch.zeros(3, 3) if kind == "WrongRank" else x.double())
            ),
        )


def export_case(case, actual, mode, rec):
    family = name(case)
    value = case.value
    if family in ("SavedExport", "CapturedExport"):
        original = (
            (
                Unary(nn.Linear(3, 2))
                if family == "SavedExport"
                else nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
            )
            .double()
            .eval()
        )
        x = torch.randn(2, 3, dtype=torch.float64)
        dynamic = ({0: torch.export.Dim("batch", min=1, max=8)},)
        batches = [(n, torch.randn(n, 3, dtype=torch.float64)) for n in (1, 4, 8)]
        invalid = torch.randn(9, 3, dtype=torch.float64)
        if actual:
            g = capture(
                original,
                (x,),
                method="fx" if family == "SavedExport" else "export",
                dynamic_shapes=None if family == "SavedExport" else dynamic,
            )
            if family == "CapturedExport":
                rec.add("constraints_present", bool(g["constraints"]))
                model = compiled(g, mode)
            else:
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt2"
                    export_graph(g, (x,), path, mode, dynamic_shapes=dynamic)
                    model = torch.export.load(path).module()
        else:
            model = original
            if family == "CapturedExport":
                rec.add("constraints_present", True)
        for batch, xx in batches:
            rec.add(f"batch{batch}", model(xx))
        rec.add(
            "bound_violation",
            error_code(lambda: model(invalid))
            if actual
            else ("runtime-error" if family == "SavedExport" else "symbolic-guard"),
        )
    elif family in ("SavedConditional", "CapturedConditional"):

        class Model(nn.Module):
            def forward(self, p, x):
                return {"value": torch.cond(p, lambda a: a * 2, lambda a: a - 1, (x,))}

        x = torch.randn(2, 3)
        p = torch.tensor(True)
        if actual:
            if family == "CapturedConditional":
                model = compiled(capture(Model(), (p, x), method="export"), mode)
            else:
                yes = graph(
                    dict(x=x),
                    [op("v", "mul", [r("x"), lit(2)])],
                    {"dict": {"value": r("v")}},
                )
                no = graph(
                    dict(x=x),
                    [op("v", "sub", [r("x"), lit(1)])],
                    {"dict": {"value": r("v")}},
                )
                g = graph(
                    dict(p=p, x=x),
                    [
                        op(
                            "r",
                            "branch",
                            [r("p"), r("x")],
                            regions={"true": yes, "false": no},
                        )
                    ],
                    r("r"),
                )
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "conditional.pt2"
                    export_graph(g, (p, x), path, mode)
                    model = torch.export.load(path).module()
        for flag in (False, True):
            rec.add(
                f"branch{flag}",
                model(torch.tensor(flag), x)
                if actual
                else {"value": x * 2 if flag else x - 1},
            )
    elif family == "SavedRecurrence":
        x = torch.randn(2, 3)
        body = graph(dict(x=x), [op("v", "tanh", [r("x")])], r("v"))
        g = graph(
            dict(x=x), [op("r", "loop", [r("x")], {"count": 3}, {"body": body})], r("r")
        )
        if actual:
            with TemporaryDirectory() as directory:
                path = Path(directory) / "loop.pt2"
                export_graph(g, (x,), path, mode)
                rec.add("output", torch.export.load(path).module()(x))
        else:
            rec.add("output", x.tanh().tanh().tanh())
    else:
        kind = name(value.library)
        if kind == "ImportedBatchNorm":
            original = Unary(nn.BatchNorm2d(3)).double().eval()
            x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        elif kind == "ImportedEmbedding":
            original = Unary(nn.Embedding(5, 3, padding_idx=0)).double().eval()
            x = torch.tensor([[0, 2, 4]])
        else:
            original = Unary(nn.MaxPool2d(3, 2, 1, return_indices=True)).eval()
            x = torch.randn(2, 3, 6, 7)
        model = (
            compiled(capture(original, (x,), method="export"), mode)
            if actual
            else original
        )
        rec.add("output", model(x))


def observe(case):
    with torch.random.fork_rng():
        torch.manual_seed(91)
        return run_case(case, True)


def reference(case):
    with torch.random.fork_rng():
        torch.manual_seed(91)
        return run_case(case, False)


def mutation_case(defect):
    kind = name(defect)
    if kind == "ReverseMask":
        return D.AttentionExecution(
            D.AttentionCase(
                mask=D.BooleanMask(), precision=D.Float64(), lowering=D.Expanded()
            )
        )
    if kind == "DuplicateParameter":
        return D.SharedExecution(
            D.SharedCase(precision=D.Float64(), lowering=D.Expanded())
        )
    if kind == "BiasedVariance":
        return D.BatchState(D.Expanded())
    return D.ControlExecution(D.Expanded())


def mutated(defect):
    kind = name(defect)
    execute = kernels.execute
    batch = kernels.batch_expanded

    def reversed_mask(operation, args, *rest):
        if operation == "attention":
            args = args[:3] + [~args[3]]
        return execute(operation, args, *rest)

    def biased(*args, **kwargs):
        before = args[2].clone()
        result = batch(*args, **kwargs)
        rate = kwargs.get("momentum", 0.1)
        count = args[0].numel() // args[0].shape[1]
        with torch.no_grad():
            args[2].copy_(
                before * (1 - rate)
                + (args[2] - before * (1 - rate)) * (count - 1) / count
            )
        return result

    context = (
        patch.object(kernels, "execute", reversed_mask)
        if kind == "ReverseMask"
        else patch.object(kernels, "batch_expanded", biased)
        if kind == "BiasedVariance"
        else nullcontext()
    )
    with torch.random.fork_rng(), context:
        torch.manual_seed(91)
        return run_case(mutation_case(defect), True, kind)


def mutation_reference(defect):
    return reference(mutation_case(defect))


def shape_request(case):
    operation = {
        "BroadcastShape": "add",
        "MatmulShape": "matmul",
        "ReshapeShape": "reshape",
        "SumShape": "sum",
        "MeanShape": "mean",
        "MaximumShape": "amax",
        "MinimumShape": "amin",
        "ExpShape": "exp",
        "SqrtShape": "sqrt",
        "DivideShape": "div",
    }[name(case.operation)]
    dtype = {
        "ShapeFloat32": torch.float32,
        "ShapeFloat64": torch.float64,
        "ShapeInt64": torch.int64,
        "ShapeBool": torch.bool,
    }[name(case.dtype)]
    x = torch.ones(tuple(case.left), dtype=dtype)
    xs = [x]
    attrs = {}
    if operation in ("add", "matmul", "div"):
        xs.append(torch.ones(tuple(case.right), dtype=dtype))
    if operation == "reshape":
        attrs["shape"] = list(case.target)
    if operation in ("sum", "mean", "amax", "amin"):
        axis = name(case.axis)
        if axis != "AllAxes":
            attrs["dim"] = {"FirstAxis": 0, "LastAxis": -1, "EmptyAxes": []}[axis]
    return operation, xs, attrs


def infer_shape(case):
    operation, xs, attrs = shape_request(case)
    g = graph(
        {str(i): x for i, x in enumerate(xs)},
        [op("out", operation, [r(str(i)) for i in range(len(xs))], attrs)],
        r("out"),
    )
    result = elixir("infer_many", [g])[0]
    if "diagnostics" in result:
        return D.Observation(tensors=[], facts=["rejected"])
    t = result["result"]
    return D.Observation(
        tensors=[], facts=["accepted", json.dumps(t["shape"]), t["dtype"]]
    )


def reference_shape(case):
    operation, xs, attrs = shape_request(case)
    try:
        out = (
            xs[0].reshape(attrs["shape"])
            if operation == "reshape"
            else getattr(torch, operation)(*xs, **attrs)
        )
    except (RuntimeError, IndexError):
        return D.Observation(tensors=[], facts=["rejected"])
    return D.Observation(
        tensors=[],
        facts=["accepted", json.dumps(list(out.shape)), str(out.dtype).split(".")[-1]],
    )
