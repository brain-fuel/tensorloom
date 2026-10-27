---
id: tensorloom.pytorch-comparison
kind: reference
title: PyTorch subsystem comparison and capability boundaries
---
# PyTorch subsystem comparison and capability boundaries

Reviewed against the [PyTorch 2.14 documentation index](https://docs.pytorch.org/docs/2.14/index.html)
and verified locally with PyTorch **2.14.1** on Python **3.13.15**. The index
includes a much larger runtime ecosystem than an architecture notation. The
table distinguishes the chosen compiler profile from potential notation mappings.
See ref:DEC-compiler and ref:DEC-library.

| Subsystem and source | Notation mapping and current capability |
| --- | --- |
| [Tensor arithmetic and manipulation](https://docs.pytorch.org/docs/2.14/torch.html) | Registry operations cover elementwise arithmetic, comparisons, matmul, reductions, reshaping, transposition, slicing, selection, concatenation, stacking, padding and windows. Arbitrary Python scalar functions and unregistered operators fail explicitly. |
| [Tensor dtype/device/layout](https://docs.pytorch.org/docs/2.14/tensor_attributes.html) | Typed metadata includes all three. Dense CPU Float32/Float64/Int64/Bool are admitted. Mixed tensor arithmetic requires explicit casts; this intentionally supports less than PyTorch's promotion system. Axis names select positions, not named-tensor alignment. |
| [Broadcasting](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html) | Right alignment and singleton expansion, including zero extents. Fixed rank with symbolic sizes is allowed; unresolved compatibility is guarded at execution. |
| [Neural network layers](https://docs.pytorch.org/docs/2.14/nn.html) | Linear/MLP/residual, embedding, softmax, attention, Conv2d, pooling, cells and fixed recurrence, LayerNorm, BatchNorm and dropout. Containers disappear into wiring. Other dimensions, losses and full Transformer modules require additional registry/capture mappings. |
| [Autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html) | Generated operations use ordinary autograd. Aliased parameter uses accumulate into one leaf. Custom autograd Functions, hooks, higher-order functional transforms and checkpoint policies are outside capture support. |
| [Module state](https://docs.pytorch.org/docs/2.14/notes/modules.html) | Parameters, frozen flags, immutable constants, mutable buffers, state_dict persistence and train/eval are distinct. Supported mutations are explicit buffer writes. Optimizer state and module hooks are not architecture nodes. |
| [Randomness/reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | Random and state effects are tracked independently from routers. Seeded capture preserves global RNG state. Deterministic mask-driven dropout is compared exactly; fused and expanded random kernels need not consume identical streams. |
| [FX](https://docs.pytorch.org/docs/2.14/fx.html) | Default capture for static Python control flow, preserving structured outputs and registered state. Data-dependent Python branches fail with an FX diagnostic; capture does not switch methods. |
| [Export IR](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/export/ir_spec.html) | The graph signature distinguishes lifted parameters, buffers and mutations. Capture restores user output structure, aliases and finite dynamic bounds; unsupported mutations fail. Shape source and call-spec metadata are retained where supplied. |
| [Export API](https://docs.pytorch.org/docs/2.14/torch.export.html) | Separate inference profile uses strict export, save and load. Declared dynamic dimensions, fixed recurrence and compatible tensor conditionals are tested. Convergence loops and explicit writes are restricted. An export failure is a diagnostic, not eager fallback. |
| [Attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Boolean True means allowed, additive masks add to scores, and causal masks are lower triangular. Fully masked rows produce zero attention. Expanded grouped-query attention is rejected; dropout probability is an explicit attention effect. |
| [BatchNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html) | Training normalizes with biased variance but stores the unbiased estimator. Running mean, variance and batch counter persist. Affine running-stat modules are captured; direct graphs may omit affine terms. |
| [MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html), [AvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html) | Stride, padding, dilation for max, ceil boundaries, first-maximum indices, average padding inclusion and divisor overrides. Average pooling with dilation is unsupported. |
| [CUDA/devices](https://docs.pytorch.org/docs/2.14/cuda.html) | Device fields make transfers and placement representable in a future profile. CPU is implemented; CUDA, MPS, XPU, MTIA, accelerator streams/events, AMP and backend controls have no admitted execution profile. |
| [Sparse tensors](https://docs.pytorch.org/docs/2.14/sparse.html) | Layout is explicit, but sparse construction, sparse kernels and sparse gradients are rejected. COO/CSR/CSC/BSR/BSC are not silently densified. Nested/ragged, masked tensor subclasses, complex and quantized tensors likewise need another profile. |
| [Distributed execution](https://docs.pytorch.org/docs/2.14/distributed.html) | Collectives would be ordered communication effects, not pure fan-out. DDP, DTensor, FSDP, RPC, elastic execution, pipeline/tensor parallelism and distributed checkpointing are outside this CPU compiler. |
| [Profiler](https://docs.pytorch.org/docs/2.14/profiler.html) | Generated modules remain ordinary PyTorch objects usable with profiling/testing utilities. This repository does not implement a profiler, benchmark runner, data loader, dashboard or deployment runtime. |
| [Other public families in the index](https://docs.pytorch.org/docs/2.14/index.html) | FFT, linalg beyond matmul, special/signal/distribution functions, optimizers, initialization policies, torch.func/library extensions, ONNX/JIT/compile/package/hub, storage interop and monitoring are not registry coverage claims. Unknown graph operators fail explicitly. |

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
explains why floating computations cannot be compared by mathematical
associativity. Structural shape models use integers. Numerical tests compare
Float32 at rtol 2e-5 / atol 2e-6 and Float64 at rtol 1e-9 / atol 1e-10,
with equal NaNs and matching infinity sign. Checked LawSpec definitions own
these comparisons; integer and boolean entries compare exactly. These are conformance tolerances,
not proofs of universal error bounds.

Inspect `priv/operations.json` for exact supported operator names and
`python/tensorloom/capture.py` for accepted signatures. Shape inference
rejects symbolic spatial windows; supply concrete spatial extents first. Export capture of
nonlinear symbolic expressions, arbitrary Python input containers, multi-layer
or bidirectional sequence modules, custom module hooks, storage views sharing
only part of a parameter, and unregistered mutation are explicit limitations.
