"""Independent expanded algorithms and native lowerings (ref:DEC-library).

All mutation is restricted to explicit buffer arguments. Autograd remains owned
by PyTorch. Floating reductions retain their documented numerical tolerance.
"""

import operator
import torch
from torch.nn import functional as F

DTYPES = {k: getattr(torch, k) for k in ("float32", "float64", "int64", "bool")}


def pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def stable_softmax(x, dim=-1):
    # Empty softmax is well-defined; max has no identity on an empty axis.
    if x.shape[dim] == 0:
        return x.clone()
    e = torch.exp(x - torch.amax(x, dim=dim, keepdim=True))
    return e / torch.sum(e, dim=dim, keepdim=True)


def conv_expanded(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    n, _, h, width = x.shape
    oc, ci, kh, kw = w.shape
    cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    cols = cols.reshape(n, groups, ci * kh * kw, -1)
    weights = w.reshape(groups, oc // groups, ci * kh * kw)
    out = torch.matmul(weights.unsqueeze(0), cols)
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    out = out.reshape(n, oc, oh, ow)
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


def pool_expanded(
    x,
    kind,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    return_indices=False,
):
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride or kernel_size)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    if kind == "avg_pool2d" and (dh, dw) != (1, 1):
        raise ValueError("avg_pool2d does not support dilation")
    if ph > kh // 2 or pw > kw // 2:
        raise ValueError("pool padding exceeds half kernel")
    unbatched = x.ndim == 3
    if unbatched:
        x = x.unsqueeze(0)
    n, c, h, w = x.shape

    def output(n, k, s, p, d):
        out = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        if ceil_mode and (out - 1) * s >= n + p:
            out -= 1
        return out

    oh = output(h, kh, sh, ph, dh)
    ow = output(w, kw, sw, pw, dw)
    if oh <= 0 or ow <= 0:
        raise ValueError("pool has no output")
    # Index grids also preserve native flattened indices and first-maximum ties.
    yy = (
        torch.arange(oh, device=x.device)[:, None] * sh
        - ph
        + torch.arange(kh, device=x.device)[None, :] * dh
    )
    xx = (
        torch.arange(ow, device=x.device)[:, None] * sw
        - pw
        + torch.arange(kw, device=x.device)[None, :] * dw
    )
    y = yy[:, None, :, None].expand(oh, ow, kh, kw)
    z = xx[None, :, None, :].expand(oh, ow, kh, kw)
    valid = (y >= 0) & (y < h) & (z >= 0) & (z < w)
    samples = x[:, :, y.clamp(0, h - 1), z.clamp(0, w - 1)]
    if kind == "max_pool2d":
        samples = samples.masked_fill(~valid, float("-inf")).flatten(-2)
        out, choice = torch.max(samples, -1)
        indices = (y * w + z).flatten(-2).expand(n, c, -1, -1, -1)
        idx = torch.gather(indices, -1, choice.unsqueeze(-1)).squeeze(-1)
        if unbatched:
            out, idx = out.squeeze(0), idx.squeeze(0)
        return (out, idx) if return_indices else out
    samples = samples.masked_fill(~valid, 0)
    total = samples.sum((-1, -2))
    inside_padding = (y >= -ph) & (y < h + ph) & (z >= -pw) & (z < w + pw)
    divisor = (
        divisor_override
        if divisor_override is not None
        else (inside_padding if count_include_pad else valid).sum((-1, -2))
    )
    out = total / divisor
    return out.squeeze(0) if unbatched else out


def moments(x, dims, eps):
    mean = x.mean(dims, keepdim=True)
    variance = ((x - mean) ** 2).mean(dims, keepdim=True)
    return (x - mean) * torch.rsqrt(variance + eps), mean, variance


def layer_expanded(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    shape = tuple(normalized_shape)
    if tuple(x.shape[-len(shape) :]) != shape:
        raise ValueError("LayerNorm normalized_shape")
    dims = tuple(range(x.ndim - len(shape), x.ndim))
    out, _, _ = moments(x, dims, eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def batch_expanded(
    x,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    counter=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    dims = (0,) + tuple(range(2, x.ndim))
    view = (1, -1) + (1,) * (x.ndim - 2)
    count = x.numel() // x.shape[1]
    if training:
        if count <= 1:
            raise ValueError("BatchNorm needs more than one value per channel")
        out, mean, variance = moments(x, dims, eps)
        with torch.no_grad():
            if counter is not None:
                counter.add_(1)
            rate = momentum if momentum is not None else 1.0 / counter.item()
            running_mean.lerp_(mean.reshape(-1), rate)
            # Native normalization uses biased variance; persistent state is unbiased.
            running_var.lerp_(variance.reshape(-1) * count / (count - 1), rate)
    else:
        out = (x - running_mean.reshape(view)) * torch.rsqrt(
            running_var.reshape(view) + eps
        )
    if weight is not None:
        out = out * weight.reshape(view)
    if bias is not None:
        out = out + bias.reshape(view)
    return out


def recurrent(name, args, attrs):
    if name == "lstm_cell":
        x, h, c, wi, wh, *bs = args
    else:
        x, h, wi, wh, *bs = args
    bi = bs[0] if bs else None
    bh = bs[1] if len(bs) > 1 else None
    xi = F.linear(x, wi, bi)
    hh = F.linear(h, wh, bh)
    if name == "rnn_cell":
        return (
            torch.relu(xi + hh)
            if attrs.get("nonlinearity") == "relu"
            else torch.tanh(xi + hh)
        )
    if name == "lstm_cell":
        i, f, g, o = (xi + hh).chunk(4, -1)
        cn = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(cn), cn
    ir, iz, inn = xi.chunk(3, -1)
    hr, hz, hn = hh.chunk(3, -1)
    r = torch.sigmoid(ir + hr)
    z = torch.sigmoid(iz + hz)
    n = torch.tanh(inn + r * hn)
    return (1 - z) * n + z * h


def execute(name, args, attrs, training, lowering):
    """Registry operations; dispatch is constant in each generated statement."""
    a = dict(attrs)
    expanded = lowering == "expanded"
    if name == "empty_sequence":
        return args[0].unsqueeze(a.get("dim", 0)).narrow(a.get("dim", 0), 0, 0)
    if name == "training":
        return training
    if name == "logical_not":
        return torch.logical_not(args[0])
    if name == "isneginf":
        return torch.isneginf(args[0])
    if name == "causal_mask":
        return torch.ones_like(args[0], dtype=torch.bool).tril()
    if name == "all":
        return torch.all(args[0], **a)
    if name == "amax_or_zero":
        return (
            args[0].sum(**a)
            if args[0].shape[a["dim"]] == 0
            else torch.amax(args[0], **a)
        )
    if name == "grouped_matmul":
        cols, w = args
        n, _, l = cols.shape
        oc, ci, kh, kw = w.shape
        groups = a.get("groups", 1)
        return (
            w.reshape(groups, oc // groups, -1).unsqueeze(0)
            @ cols.reshape(n, groups, ci * kh * kw, l)
        ).reshape(n, oc, l)
    if name in ("pool_values", "pool_indices", "pool_divisor"):
        parts = pool_components(args[0], **a)
        return parts[{"pool_values": 0, "pool_indices": 1, "pool_divisor": 2}[name]]
    if name == "reduce_max":
        return tuple(torch.max(args[0], **a))
    if name == "gather":
        return torch.gather(args[0], a["dim"], args[1].unsqueeze(a["dim"])).squeeze(
            a["dim"]
        )
    if name == "batch_count":
        return args[0].numel() // args[0].shape[1]
    if name == "require_batch":
        if args[0].numel() // args[0].shape[1] <= 1:
            raise ValueError("BatchNorm needs more than one value per channel")
        return args[0]
    if name == "reciprocal_counter":
        return 1.0 / args[0].item()
    if name in ("identity", "clone", "contiguous"):
        return args[0] if name == "identity" else getattr(args[0], name)()
    if name in (
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "eq",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
    ) and all(isinstance(x, (int, float, bool)) for x in args):
        functions = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
            "pow": operator.pow,
            "maximum": max,
            "minimum": min,
            "eq": operator.eq,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
            "logical_and": operator.and_,
        }
        return functions[name](*args)
    if name in (
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "maximum",
        "minimum",
        "matmul",
        "eq",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
        "relu",
        "sigmoid",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "neg",
    ):
        return getattr(torch, name)(*args)
    if name == "reshape":
        return args[0].reshape(a["shape"])
    if name == "transpose":
        return args[0].transpose(a.get("dim0", 0), a.get("dim1", 1))
    if name == "permute":
        return args[0].permute(a["dims"])
    if name == "slice":
        slices = [slice(None)] * args[0].ndim
        slices[a.get("dim", 0)] = slice(a.get("start"), a.get("stop"), a.get("step", 1))
        return args[0][tuple(slices)]
    if name == "select":
        return args[0].select(a.get("dim", 0), a["index"])
    if name == "getitem":
        return args[0][a["index"]]
    if name in ("cat", "stack"):
        return getattr(torch, name)(args, dim=a.get("dim", 0))
    if name in ("sum", "mean", "amax", "amin", "prod", "var"):
        if "dim" in a and isinstance(a["dim"], list):
            a["dim"] = tuple(a["dim"])
        return getattr(torch, name)(args[0], **a)
    if name == "softmax":
        return stable_softmax(*args, **a) if expanded else F.softmax(*args, **a)
    if name == "linear":
        x, w, *bias = args
        return (
            (x @ w.transpose(-2, -1) + (bias[0] if bias else 0))
            if expanded
            else F.linear(*args)
        )
    if name in ("embedding", "index_rows"):
        return F.embedding(args[0], args[1], **a)
    if name == "conv2d":
        return (conv_expanded if expanded else F.conv2d)(*args, **a)
    if name in ("max_pool2d", "avg_pool2d"):
        return (
            pool_expanded(*args, kind=name, **a)
            if expanded
            else getattr(F, name)(*args, **a)
        )
    if name in ("rnn_cell", "gru_cell", "lstm_cell"):
        if expanded:
            return recurrent(name, args, a)
        if name == "lstm_cell":
            x, h, c, wi, wh, *bs = args
            return torch._VF.lstm_cell(x, (h, c), wi, wh, *(bs or [None, None]))
        x, h, wi, wh, *bs = args
        fn = (
            torch._VF.gru_cell
            if name == "gru_cell"
            else (
                torch._VF.rnn_relu_cell
                if a.get("nonlinearity") == "relu"
                else torch._VF.rnn_tanh_cell
            )
        )
        return fn(x, h, wi, wh, *(bs or [None, None]))
    if name == "layer_norm":
        x, *wb = args
        return (layer_expanded if expanded else F.layer_norm)(
            x, a.pop("normalized_shape"), *wb, **a
        )
    if name == "batch_norm":
        a["training"] = a.get("training", training)
        if expanded:
            return batch_expanded(*args, **a)
        x, rm, rv, *tail = args
        weight = tail[0] if tail else None
        bias = tail[1] if len(tail) > 1 else None
        counter = tail[2] if len(tail) > 2 else None
        momentum = a.get("momentum", 0.1)
        if a["training"] and counter is not None:
            counter.add_(1)
            if momentum is None:
                momentum = 1.0 / counter.item()
        return F.batch_norm(
            x,
            rm,
            rv,
            weight,
            bias,
            training=a["training"],
            momentum=momentum or 0.0,
            eps=a.get("eps", 1e-5),
        )
    if name == "dropout":
        x = args[0]
        p = a.get("p", 0.5)
        active = a.get("training", training)
        if not active or p == 0:
            return x
        if p == 1:
            return x * 0
        if len(args) > 1:
            return x * args[1].to(x.dtype) / (1 - p)
        return (
            x * (torch.rand_like(x) >= p) / (1 - p)
            if expanded
            else F.dropout(x, p, True)
        )
    if name == "attention":
        q, k, v, *mask = args
        if not expanded:
            return F.scaled_dot_product_attention(
                q, k, v, mask[0] if mask else None, **a
            )
        if a.get("enable_gqa"):
            raise ValueError("GQA requires native lowering")
        scale = a.get("scale") or q.shape[-1] ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        if a.get("is_causal"):
            causal = torch.ones(
                scores.shape[-2:], dtype=torch.bool, device=q.device
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask:
            scores = (
                scores.masked_fill(~mask[0], float("-inf"))
                if mask[0].dtype == torch.bool
                else scores + mask[0]
            )
        empty_row = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty_row, 0.0)
        probs = stable_softmax(scores, -1).masked_fill(empty_row, 0.0)
        probs = F.dropout(probs, a.get("dropout_p", 0.0), True)
        return probs @ v
    if name == "where":
        return torch.where(*args)
    if name == "masked_fill":
        return args[0].masked_fill(
            args[1], float(a["value"]) if isinstance(a["value"], str) else a["value"]
        )
    if name == "pad":
        return F.pad(
            args[0], a["pad"], mode=a.get("mode", "constant"), value=a.get("value", 0)
        )
    if name == "unfold":
        return F.unfold(*args, **a)
    if name == "cast":
        return args[0].to(DTYPES[a["dtype"]])
    if name == "rand_like":
        return torch.rand_like(args[0])
    if name == "copy_buffer":
        with torch.no_grad():
            args[0].copy_(args[1])
        return args[0]
    if name == "tuple":
        return tuple(args)
    if name == "dict":
        return dict(zip(a["keys"], args))
    raise ValueError(f"unsupported-operation: {name}")


def pool_components(
    x,
    kind,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    **ignored,
):
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride or kernel_size)
    ph, pw = pair(padding)
    dh, dw = pair(dilation)
    h, w = x.shape[-2:]

    def out(n, k, s, p, d):
        o = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
        return o - 1 if ceil_mode and (o - 1) * s >= n + p else o

    oh = out(h, kh, sh, ph, dh)
    ow = out(w, kw, sw, pw, dw)
    y = (
        torch.arange(oh, device=x.device)[:, None] * sh
        - ph
        + torch.arange(kh, device=x.device)[None, :] * dh
    )[:, None, :, None].expand(oh, ow, kh, kw)
    z = (
        torch.arange(ow, device=x.device)[:, None] * sw
        - pw
        + torch.arange(kw, device=x.device)[None, :] * dw
    )[None, :, None, :].expand(oh, ow, kh, kw)
    valid = (y >= 0) & (y < h) & (z >= 0) & (z < w)
    values = (
        x[..., y.clamp(0, h - 1), z.clamp(0, w - 1)]
        .masked_fill(~valid, float("-inf") if kind == "max_pool2d" else 0)
        .flatten(-2)
    )
    indices = (y * w + z).flatten(-2).expand_as(values)
    padding_valid = (y >= -ph) & (y < h + ph) & (z >= -pw) & (z < w + pw)
    divisor = (
        divisor_override
        if divisor_override is not None
        else (padding_valid if count_include_pad else valid).sum((-1, -2))
    )
    return values, indices, divisor
