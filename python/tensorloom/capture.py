"""Explicit FX/export capture with state identity and source metadata (ref:DEC-import)."""

import copy
import torch
from torch import fx
from torch.utils import _pytree as pytree
from .compiler import Diagnostic, REGISTRY, elixir

DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.bool: "bool",
}


def tensor_type(t):
    if (
        t.dtype not in DTYPE_NAMES
        or t.device.type != "cpu"
        or t.layout != torch.strided
    ):
        raise Diagnostic("unsupported-type", f"{t.dtype}/{t.device}/{t.layout}")
    if any(not isinstance(d, int) and not str(d).isidentifier() for d in t.shape):
        raise Diagnostic("unsupported-symbolic-expression", str(t.shape))
    return dict(
        kind="tensor",
        shape=[int(d) if isinstance(d, int) else str(d) for d in t.shape],
        dtype=DTYPE_NAMES[t.dtype],
        device="cpu",
        layout="strided",
        axes=[None] * t.ndim,
    )


def value_type(v):
    if isinstance(v, torch.Tensor):
        return tensor_type(v)
    if isinstance(v, tuple):
        return dict(kind="tuple", items=[value_type(x) for x in v])
    if isinstance(v, dict):
        return dict(kind="dict", items={k: value_type(x) for k, x in v.items()})
    if type(v) in (bool, int, float):
        return dict(
            kind="scalar", dtype={bool: "bool", int: "int64", float: "float64"}[type(v)]
        )
    raise Diagnostic("unsupported-type", str(type(v)))


def capture(module, examples, method="fx", dynamic_shapes=None):
    if method not in ("fx", "export"):
        raise Diagnostic("unsupported-capture", method)
    try:
        # Shape propagation and export may run the model. Neither may mutate it.
        if any(
            m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
            for m in module.modules()
        ):
            raise Diagnostic("unsupported-feature", "module hooks are not portable")
        model = copy.deepcopy(module)
        if method == "fx":
            tracer = fx.Tracer()
            tracer.record_stack_traces = True
            gm = fx.GraphModule(model, tracer.trace(model))
            from torch.fx.passes.shape_prop import ShapeProp

            snapshot = copy.deepcopy(gm.state_dict())
            with torch.random.fork_rng():
                ShapeProp(gm).propagate(*examples)
            gm.load_state_dict(snapshot)
            program = None
        else:
            with torch.random.fork_rng():
                program = torch.export.export(
                    model, tuple(examples), dynamic_shapes=dynamic_shapes, strict=True
                )
            gm = program.graph_module
        result = _convert(gm, model, examples, program)
        elixir("infer", result)
        return result
    except Diagnostic:
        raise
    except Exception as exc:
        raise Diagnostic(
            "capture-failed", f"{method}: {type(exc).__name__}: {exc}"
        ) from exc


def _convert(gm, model, examples, program):
    graph = dict(
        version=1,
        inputs=[],
        state=[],
        ops=[],
        result=None,
        capture={"method": "export" if program else "fx", "training": model.training},
    )
    state_by_name = {}
    identities = {}
    refs = {}
    input_index = 0

    def state(name, tensor, kind):
        if name in state_by_name:
            return {"ref": state_by_name[name]}
        ident = "state:" + name
        entry = dict(id=ident, kind=kind, type=tensor_type(tensor))
        if kind == "parameter":
            entry["requires_grad"] = tensor.requires_grad
        if kind == "buffer":
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            entry["persistent"] = parts[-1] not in parent._non_persistent_buffers_set
        if id(tensor) in identities:
            entry["alias"] = identities[id(tensor)]
        else:
            entry["data"] = tensor.detach().tolist()
            identities[id(tensor)] = ident
        graph["state"].append(entry)
        state_by_name[name] = ident
        return {"ref": ident}

    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    for name, t in params.items():
        state(name, t, "parameter")
    for name, t in buffers.items():
        state(name, t, "buffer")

    def attr(path):
        value = model
        for part in path.split("."):
            value = getattr(value, part)
        return value

    def encode(v):
        if isinstance(v, fx.Node):
            return refs[v]
        if isinstance(v, tuple):
            return {"tuple": [encode(x) for x in v]}
        if isinstance(v, list):
            return {"tuple": [encode(x) for x in v]}
        if isinstance(v, dict):
            return {"dict": {k: encode(x) for k, x in v.items()}}
        if v is None or isinstance(v, (bool, int, float, str)):
            return {"lit": v}
        raise Diagnostic("unsupported-value", repr(v))

    def add(ident, name, args, attrs=None, regions=None, source=None):
        op = dict(
            id=ident, op=name, args=args, attrs=attrs or {}, regions=regions or {}
        )
        if source:
            op["source"] = source
        graph["ops"].append(op)
        return {"ref": ident}

    sig = (
        {s.arg.name: s for s in program.graph_signature.input_specs} if program else {}
    )
    for node in gm.graph.nodes:
        ident = node.name
        source = node.meta.get("stack_trace")
        if node.op == "placeholder":
            spec = sig.get(node.name)
            if spec and spec.kind.name in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                name = spec.target
                if name in params:
                    refs[node] = state(name, params[name], "parameter")
                elif name in buffers:
                    refs[node] = state(name, buffers[name], "buffer")
                else:
                    refs[node] = state(name, program.constants[name], "constant")
            else:
                example = examples[input_index]
                input_index += 1
                typ = (
                    value_type(node.meta.get("val", example))
                    if program
                    else value_type(example)
                )
                graph["inputs"].append(dict(id=ident, type=typ))
                refs[node] = {"ref": ident}
        elif node.op == "get_attr":
            value = gm
            for part in node.target.split("."):
                value = getattr(value, part)
            if isinstance(value, fx.GraphModule):
                refs[node] = value
            elif isinstance(value, torch.Tensor):
                kind = (
                    "parameter"
                    if node.target in params
                    else "buffer"
                    if node.target in buffers
                    else "constant"
                )
                refs[node] = state(
                    node.target,
                    attr(node.target)
                    if node.target in params or node.target in buffers
                    else value,
                    kind,
                )
            else:
                raise Diagnostic("unsupported-operation", "non-tensor get_attr", ident)
        elif node.op == "output":
            output = node.args[0]
            if program:
                # Export outputs are flat. Restore the public call's structure.
                user_specs = [
                    s
                    for s in program.graph_signature.output_specs
                    if s.kind.name == "USER_OUTPUT"
                ]
                mutations = [
                    s
                    for s in program.graph_signature.output_specs
                    if s.kind.name != "USER_OUTPUT"
                ]
                if mutations:
                    raise Diagnostic(
                        "unsupported-mutation",
                        "export buffer/input mutation capture is not supported",
                        ident,
                    )
                output = pytree.tree_unflatten(list(output), program.call_spec.out_spec)
            graph["result"] = encode(output)
        elif node.op == "call_module":
            m = gm.get_submodule(node.target)
            name = type(m).__name__
            args = list(node.args)
            kw = dict(node.kwargs)

            def st(key):
                path = node.target + "." + key
                t = attr(path)
                return state(path, t, "parameter" if path in params else "buffer")

            def basic(op, a=None, extra=()):
                return add(
                    ident, op, [encode(x) for x in args] + list(extra), a, source=source
                )

            if isinstance(m, torch.nn.Linear):
                refs[node] = basic(
                    "linear",
                    extra=[st("weight")] + ([st("bias")] if m.bias is not None else []),
                )
            elif isinstance(
                m, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Identity)
            ):
                if getattr(m, "inplace", False):
                    raise Diagnostic(
                        "unsupported-mutation", "inplace activation", ident
                    )
                refs[node] = basic(
                    {
                        "ReLU": "relu",
                        "Sigmoid": "sigmoid",
                        "Tanh": "tanh",
                        "Identity": "identity",
                    }[name]
                )
            elif isinstance(m, torch.nn.Embedding):
                if m.max_norm is not None or m.sparse:
                    raise Diagnostic(
                        "unsupported-operation",
                        "embedding renorm or sparse gradients",
                        ident,
                    )
                refs[node] = basic(
                    "embedding",
                    dict(
                        padding_idx=m.padding_idx,
                        scale_grad_by_freq=m.scale_grad_by_freq,
                    ),
                    [st("weight")],
                )
            elif isinstance(m, torch.nn.Softmax):
                refs[node] = basic("softmax", dict(dim=m.dim))
            elif isinstance(m, torch.nn.Conv2d):
                if m.padding_mode != "zeros":
                    raise Diagnostic(
                        "unsupported-operation", "conv padding_mode", ident
                    )
                refs[node] = basic(
                    "conv2d",
                    dict(
                        stride=list(m.stride),
                        padding=list(m.padding),
                        dilation=list(m.dilation),
                        groups=m.groups,
                    ),
                    [st("weight")] + ([st("bias")] if m.bias is not None else []),
                )
            elif isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                keys = ["kernel_size", "stride", "padding", "ceil_mode"] + (
                    ["dilation", "return_indices"]
                    if isinstance(m, torch.nn.MaxPool2d)
                    else ["count_include_pad", "divisor_override"]
                )
                a = {k: getattr(m, k) for k in keys if getattr(m, k) is not None}
                refs[node] = basic(
                    "max_pool2d" if isinstance(m, torch.nn.MaxPool2d) else "avg_pool2d",
                    a,
                )
            elif isinstance(m, torch.nn.LayerNorm):
                extras = [st("weight")] if m.weight is not None else []
                if m.bias is not None:
                    extras.append(st("bias"))
                refs[node] = basic(
                    "layer_norm",
                    dict(normalized_shape=list(m.normalized_shape), eps=m.eps),
                    extras,
                )
            elif isinstance(
                m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
            ):
                if not m.track_running_stats or not m.affine:
                    raise Diagnostic(
                        "unsupported-operation",
                        "BatchNorm requires running state and affine bindings",
                        ident,
                    )
                refs[node] = basic(
                    "batch_norm",
                    dict(momentum=m.momentum, eps=m.eps),
                    [
                        st(k)
                        for k in (
                            "running_mean",
                            "running_var",
                            "weight",
                            "bias",
                            "num_batches_tracked",
                        )
                    ],
                )
            elif isinstance(m, torch.nn.Dropout):
                if m.inplace:
                    raise Diagnostic("unsupported-mutation", "inplace dropout", ident)
                refs[node] = basic("dropout", dict(p=m.p))
            elif isinstance(m, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                if m.num_layers != 1 or m.bidirectional or getattr(m, "proj_size", 0):
                    raise Diagnostic(
                        "unsupported-operation",
                        "sequence capture supports one layer, one direction, no projection",
                        ident,
                    )
                x = args[0]
                meta = x.meta.get("val", x.meta.get("tensor_meta"))
                if meta is None or len(meta.shape) != 3:
                    raise Diagnostic(
                        "unsupported-operation",
                        "sequence input must have fixed rank three",
                        ident,
                    )
                time = 1 if m.batch_first else 0
                steps = meta.shape[time]
                batch = meta.shape[1 - time]
                if not isinstance(steps, int):
                    raise Diagnostic(
                        "unsupported-operation", "sequence length must be fixed", ident
                    )
                carried = []
                if len(args) > 1 and args[1] is not None:
                    initial = encode(args[1])
                    for j in range(2 if isinstance(m, torch.nn.LSTM) else 1):
                        item = (
                            add(
                                ident + f"/initial{j}",
                                "getitem",
                                [initial],
                                {"index": j},
                            )
                            if isinstance(m, torch.nn.LSTM)
                            else initial
                        )
                        carried.append(
                            add(
                                ident + f"/initial_select{j}",
                                "select",
                                [item],
                                {"dim": 0, "index": 0},
                            )
                        )
                else:
                    for j in range(2 if isinstance(m, torch.nn.LSTM) else 1):
                        zero = torch.zeros(
                            batch, m.hidden_size, dtype=m.weight_ih_l0.dtype
                        )
                        carried.append(state(ident + f"/zero{j}", zero, "constant"))
                weights = [st("weight_ih_l0"), st("weight_hh_l0")] + (
                    [st("bias_ih_l0"), st("bias_hh_l0")] if m.bias else []
                )
                outputs = []
                cell_name = {"RNN": "rnn_cell", "GRU": "gru_cell", "LSTM": "lstm_cell"}[
                    name
                ]
                for i in range(steps):
                    xx = add(
                        ident + f"/x{i}",
                        "select",
                        [encode(x)],
                        {"dim": time, "index": i},
                    )
                    result = add(
                        ident + f"/step{i}",
                        cell_name,
                        [xx] + carried + weights,
                        {"nonlinearity": m.nonlinearity} if name == "RNN" else {},
                    )
                    if isinstance(m, torch.nn.LSTM):
                        carried = [
                            add(
                                ident + f"/carry{i}_{j}",
                                "getitem",
                                [result],
                                {"index": j},
                            )
                            for j in range(2)
                        ]
                    else:
                        carried = [result]
                    outputs.append(carried[0])
                sequence = (
                    add(ident + "/sequence", "stack", outputs, {"dim": time})
                    if outputs
                    else add(
                        ident + "/sequence",
                        "empty_sequence",
                        [carried[0]],
                        {"dim": time},
                    )
                )
                final = [
                    add(ident + f"/final{j}", "stack", [v], {"dim": 0})
                    for j, v in enumerate(carried)
                ]
                refs[node] = add(
                    ident,
                    "tuple",
                    [
                        sequence,
                        {"tuple": final} if isinstance(m, torch.nn.LSTM) else final[0],
                    ],
                    source=source,
                )
            elif isinstance(m, (torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell)):
                if len(args) != 2:
                    raise Diagnostic(
                        "unsupported-operation",
                        "recurrent capture needs explicit state",
                        ident,
                    )
                encoded = [encode(args[0])]
                if isinstance(m, torch.nn.LSTMCell):
                    for i in range(2):
                        encoded.append(
                            add(
                                ident + f"/state{i}",
                                "getitem",
                                [encode(args[1])],
                                {"index": i},
                            )
                        )
                else:
                    encoded.append(encode(args[1]))
                encoded += [st("weight_ih"), st("weight_hh")] + (
                    [st("bias_ih"), st("bias_hh")] if m.bias else []
                )
                refs[node] = add(
                    ident,
                    {
                        "RNNCell": "rnn_cell",
                        "GRUCell": "gru_cell",
                        "LSTMCell": "lstm_cell",
                    }[name],
                    encoded,
                    {"nonlinearity": m.nonlinearity} if name == "RNNCell" else {},
                    source=source,
                )
            else:
                raise Diagnostic("unsupported-operation", f"module {name}", ident)
        elif node.op in ("call_function", "call_method"):
            target = node.target
            raw = str(target)
            target_name = getattr(target, "__name__", raw)
            full = getattr(target, "__module__", "") + "." + target_name
            if raw == "cond":
                predicate, left, right, operands = node.args
                regions = {}
                for key, body in [("true", refs[left]), ("false", refs[right])]:
                    values = [v.meta["val"] for v in operands]
                    region = _convert(body, torch.nn.Module(), values, None)
                    if region["state"]:
                        raise Diagnostic(
                            "unsupported-capture",
                            "conditional state must be lifted as operands",
                            ident,
                        )
                    region.pop("capture", None)
                    regions[key] = region
                refs[node] = add(
                    ident,
                    "branch",
                    [encode(predicate)] + [encode(v) for v in operands],
                    regions=regions,
                    source=source,
                )
                continue
            lookup = {
                mapping: name
                for name, rule in REGISTRY.items()
                for mapping in rule["imports"]
            }
            name = lookup.get(raw) or lookup.get(full) or lookup.get(target_name)
            if not name:
                raise Diagnostic("unsupported-operation", raw, ident)
            args = list(node.args)
            a = dict(node.kwargs)
            if name == "getitem":
                if isinstance(args[1], int):
                    source_val = args[0].meta.get(
                        "val", args[0].meta.get("tensor_meta")
                    )
                    if isinstance(source_val, torch.Tensor) or hasattr(
                        source_val, "shape"
                    ):
                        name = "select"
                        a = {"dim": 0, "index": args[1]}
                    else:
                        a = {"index": args[1]}
                    args = args[:1]
                elif isinstance(args[1], str):
                    a = {"index": args[1]}
                    args = args[:1]
                else:
                    raise Diagnostic(
                        "unsupported-operation",
                        "advanced Python indexing; use select/slice",
                        ident,
                    )
            elif name == "reshape":
                a = {
                    "shape": list(args[1])
                    if len(args) == 2 and isinstance(args[1], (list, tuple))
                    else args[1:]
                }
                args = args[:1]
                if any(isinstance(x, fx.Node) for x in a["shape"]):
                    raise Diagnostic(
                        "unsupported-operation",
                        "runtime reshape size expression",
                        ident,
                    )
            elif name == "transpose":
                a = {
                    "dim0": args[1] if len(args) > 1 else -2,
                    "dim1": args[2] if len(args) > 2 else -1,
                }
                args = args[:1]
            elif name == "permute":
                a = {
                    "dims": list(args[1])
                    if isinstance(args[1], (list, tuple))
                    else args[1:]
                }
                args = args[:1]
            elif name in ("cat", "stack"):
                a = {"dim": args[1] if len(args) > 1 else a.get("dim", 0)}
                args = list(args[0])
            elif name in ("sum", "mean", "amax", "amin", "prod", "softmax", "var"):
                if len(args) > 1:
                    a["dim"] = args[1]
                if len(args) > 2:
                    a["keepdim"] = args[2]
                args = args[:1]
                if a.pop("dtype", None) is not None:
                    raise Diagnostic(
                        "unsupported-operation",
                        "reduction dtype override; use explicit cast",
                        ident,
                    )
            elif name == "select":
                a = {"dim": args[1], "index": args[2]}
                args = args[:1]
            elif name == "slice":
                a = {
                    "dim": args[1],
                    "start": args[2],
                    "stop": args[3],
                    "step": args[4] if len(args) > 4 else 1,
                }
                args = args[:1]
            elif name == "linear":
                if len(args) < 3 and "bias" in a:
                    args.append(a.pop("bias"))
                args = [x for x in args if x is not None]
            elif name == "embedding":
                if raw.startswith("aten."):
                    padding = args[2] if len(args) > 2 else -1
                    a = {
                        "padding_idx": None if padding == -1 else padding,
                        "scale_grad_by_freq": args[3] if len(args) > 3 else False,
                    }
                    if len(args) > 4 and args[4]:
                        raise Diagnostic(
                            "unsupported-operation", "sparse embedding gradients", ident
                        )
                    args = [args[1], args[0]]
                else:
                    for key, v in zip(
                        (
                            "padding_idx",
                            "max_norm",
                            "norm_type",
                            "scale_grad_by_freq",
                            "sparse",
                        ),
                        args[2:],
                    ):
                        a[key] = v
                    if a.pop("max_norm", None) is not None or a.pop("sparse", False):
                        raise Diagnostic(
                            "unsupported-operation",
                            "embedding renorm or sparse gradients",
                            ident,
                        )
                    a.pop("norm_type", None)
                    args = args[:2]
            elif name == "batch_norm":
                if raw.startswith("aten."):
                    x, w, b, rm, rv, *settings = args
                    a = dict(
                        zip(("training", "momentum", "eps", "cudnn_enabled"), settings)
                    )
                    a.pop("cudnn_enabled", None)
                    args = [x, rm, rv] + (
                        [w, b] if w is not None and b is not None else []
                    )
                    if (w is None) != (b is None):
                        raise Diagnostic(
                            "unsupported-operation", "partial affine BatchNorm", ident
                        )
                else:
                    for key, v in zip(("training", "momentum", "eps"), args[5:]):
                        a[key] = v
                    args = args[:5]
                    if any(v is None for v in args[1:3]):
                        raise Diagnostic(
                            "unsupported-operation",
                            "BatchNorm needs running buffers",
                            ident,
                        )
                    args = [v for v in args if v is not None]
            elif name == "dropout":
                if len(args) > 1:
                    a["p"] = args[1]
                if len(args) > 2:
                    a["training"] = args[2]
                if len(args) > 3 and args[3]:
                    raise Diagnostic("unsupported-mutation", "inplace dropout", ident)
                a.pop("inplace", None)
                args = args[:1]
            elif name == "conv2d":
                for key, v in zip(
                    ("stride", "padding", "dilation", "groups"), args[3:]
                ):
                    a[key] = v
                args = [x for x in args[:3] if x is not None]
            elif name in ("max_pool2d", "avg_pool2d"):
                keys = (
                    (
                        "kernel_size",
                        "stride",
                        "padding",
                        "dilation",
                        "ceil_mode",
                        "return_indices",
                    )
                    if name == "max_pool2d"
                    else (
                        "kernel_size",
                        "stride",
                        "padding",
                        "ceil_mode",
                        "count_include_pad",
                        "divisor_override",
                    )
                )
                for key, v in zip(keys, args[1:]):
                    if v is not None and v != []:
                        a[key] = v
                if raw == "aten.max_pool2d_with_indices.default":
                    a["return_indices"] = True
                args = args[:1]
            elif name == "layer_norm":
                a["normalized_shape"] = list(args[1])
                a["eps"] = args[4] if len(args) > 4 else a.get("eps", 1e-5)
                args = [args[0]] + [x for x in args[2:4] if x is not None]
            elif name == "attention":
                mask = a.pop("attn_mask", None)
                if len(args) < 4 and mask is not None:
                    args.append(mask)
                for key, v in zip(
                    ("dropout_p", "is_causal", "scale", "enable_gqa"), args[4:]
                ):
                    a[key] = v
                args = [x for x in args[:4] if x is not None]
            elif len(args) > REGISTRY[name]["arity"][1]:
                raise Diagnostic(
                    "unsupported-operation", f"unsupported signature {raw}", ident
                )
            if any(
                isinstance(v, (torch.dtype, torch.device, fx.Node)) for v in a.values()
            ):
                raise Diagnostic(
                    "unsupported-operation", "nonportable keyword argument", ident
                )
            refs[node] = add(ident, name, [encode(x) for x in args], a, source=source)
        else:
            raise Diagnostic("unsupported-operation", node.op, ident)
    if program:
        graph["constraints"] = {}
        for symbol, bounds in program.range_constraints.items():
            entry = {}
            for key, bound in [("min", bounds.lower), ("max", bounds.upper)]:
                if str(bound) not in ("int_oo", "-int_oo", "oo", "-oo"):
                    entry[key] = int(bound)
            graph["constraints"][str(symbol)] = entry
        graph["capture"]["input_spec"] = pytree.treespec_dumps(
            program.call_spec.in_spec
        )
        graph["capture"]["output_spec"] = pytree.treespec_dumps(
            program.call_spec.out_spec
        )
    return graph
