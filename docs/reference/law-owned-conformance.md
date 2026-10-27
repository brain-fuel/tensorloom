---
id: tensorloom.law.owned.conformance
kind: reference
title: LawSpec ownership of compiler conformance
---
# LawSpec ownership of compiler conformance

All former handwritten Python conformance cases are represented by
`python/laws/conformance.lawspec`. LawSpec owns the finite case domain, examples,
assertions, numeric tolerance, nonfinite behavior, generated shape requests and
mutation rejection. Generated pytest/Hypothesis files are execution artifacts;
there is no separate handwritten pytest suite. See ref:DEC-law-owned-conformance.

The Elixir frontend uses `lawspec.edn`; the independently built Python backend
uses `python/lawspec.edn`. Both run through `node scripts/verify.mjs`. Native
bindings return typed `Observation` values containing every output/gradient/state
sample and exact structure, identity or diagnostic facts. Python adapters do not
assert conformance, invoke pytest, or return a cached success flag. The native
PyTorch oracle is assumed; its fidelity is tested by independently known shape
examples and deliberately defective implementations, not claimed as a proof.

| Former pytest family | LawSpec execution-case constructors |
| --- | --- |
| Seven layer families × two dtypes × two lowerings | `ModuleExecution (KernelCase ...)` |
| Residual/shared weights and fan-out aliases | `SharedExecution` |
| BatchNorm updates, eval and state_dict reload | `BatchState` |
| RNN/GRU/LSTM transitions and gradients | `CellExecution` |
| Embedding gradients and invalid indices | `EmbeddingExecution` |
| Boolean, additive, causal and fully masked attention | `AttentionExecution` |
| Deterministic dropout, gradients and eval identity | `DropoutExecution` |
| Large logits, empty axes, NaN and infinities | `SoftmaxBoundaries` |
| Branches, resolutions, fixed loops, budgets and exclusivity | `ControlExecution` |
| Collapse/expand, shared state and invalid selections | `CollapseExecution` |
| Dynamic saved exports and export capture/JSON/compile | `SavedExport`, `CapturedExport` |
| Capture failures and input guards | Four `RejectedExecution` alternatives |
| Recurrent sequence composition | `SequenceExecution` |
| Ten inspected expansions, stability, gradients and eval | `ExpandedExecution` |
| Saved and captured structured conditionals, saved recurrence | `SavedConditional`, `CapturedConditional`, `SavedRecurrence` |
| Export capture of BatchNorm, embedding and pooling | `ImportedExecution` |
| Named axes and frozen/nonpersistent state | `NamedAxes`, `FrozenState` |
| Four deliberately faulty implementations | The finite `Mutation` domain and its rejection law |
| Previous 156-request shape/dtype batch | 156 explicit LawSpec example rows, six regression rows and generated bounded `ShapeCase` inputs |

The finite `ConformanceCase` domain contains **114 cases**. It separates the four
capture/input rejections that were formerly combined into two pytest cases.
Together with the four negative controls and the previous aggregate shape test,
it preserves the behavior of the former 117-case suite. Cases with several
observations, such as the three softmax boundaries or successive BatchNorm
updates, retain all of those observations. Expanded sequence execution and
inspected expansion gradients are included.

`closeSample`, `closeSamples`, `closeTensor`, `closeTensors` and `equivalent` are
checked LawSpec definitions. Finite values cross as exact rationals; NaN and
signed infinities are separate constructors. Floating values use dtype-specific
relative/absolute tolerances; integer and boolean values compare exactly. Length,
shape, dtype and result structure are checked so empty or truncated observations
cannot satisfy conformance. Sequence iteration checks every element, rather than
sampling tensor entries.

Shape ranks are bounded by four, source extents by eight, and target extents by
minus one through eight. These restrictions are part of the LawSpec type, so
mandatory boundaries and shrinking cannot introduce unbounded allocations.
Shape requests include intentional incompatibilities whose expected outcome is
rejection. The native Hypothesis generator constructs valid requests directly,
independently of the compiler; generator exhaustion would be incomplete evidence.

No LawSpec compiler delta is required for this suite. The initial recursive
comparator hit Python's host recursion limit on convolution-sized lists. The
existing `prelude.any` lowering provides finite element iteration; comparing
lengths and rejecting any differing element expresses full comparison without
that recursion. A general improvement to stack-safe structural recursion in the
Python dynamic-definition fallback would make the original recursive spelling
usable at larger sizes. A `prelude.all` convenience alias would be ergonomic;
it is not needed for the contract. The sibling checkout was not changed.

The inspected sources establish the tool boundary: ref:lawspec-native-bindings
and ref:lawspec-harness cover in-process native adapters and executable checks;
ref:lawspec-iteration implements finite iteration. Ironframe owns resolved
service topology, environment bindings, deployment and integration attestations
(ref:ironframe-planes). LawTwin owns independently checked stand-ins for external
protocol dependencies (ref:lawtwin-planes). This compiler needs neither an
external protocol stand-in nor environment/deployment orchestration. They would
be relevant if a future compiler depended on a remote model service or required
real/twin deployment conformance; they do not solve a local comparison-loop
recursion limit.
