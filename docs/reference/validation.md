---
id: tensorloom.validation
kind: reference
title: Automated validation and review status
---
# Automated validation and review status

Validation date: 2026-10-26. Toolchain: Python 3.13.15, PyTorch 2.14.1,
pytest 8.4.2, Hypothesis 6.170.12, Elixir 1.20.4, Erlang/OTP 29 and StreamData
1.4.0. Generation uses the published LawSpec 0.31.0 npm release;
`package-lock.json` fixes its dependencies and integrity hashes. No sibling
checkout is required. See ref:DEC-compiler, ref:DEC-law-owned-conformance and
ref:DEC-build-artifacts.

Run `npm run verify` after setup in README.md. This checks definitions,
generates and runs both independently built LawSpec components, then runs
Canon review status (`ricestax canon vet`) and validation (`ricestax canon check`).
The released LawSpec 0.31.0 CLI does not include the development checkout's
`project status`/`project validate` wrappers. `scripts/verify.mjs` composes the
released tools directly and writes `artifacts/verification-summary.json`.
`lawspec.edn` owns frontend contracts;
`python/lawspec.edn` owns backend conformance. Examples, properties and finite
exhaustive checking are selected explicitly; mandatory definition validation
remains independent. No solver or proof checker is selected.

Fresh-checkout reproduction was checked from a source-only copy outside the
project directory, with no sibling repositories, generated outputs, local
ownership manifests, virtual environment or installed dependencies. The README
commands `npm ci`, `mix local.hex --force`, `npm run setup` and `npm run demo`
passed. The demo checked Elixir-to-PyTorch output and gradients, both FX
lowerings, and saved export reload; `mix test` passed all 211 cases. All 57
compiler-owned files are excluded from the Git index. Local ownership metadata
is recreated alongside ignored outputs rather than tracked separately.

The Python suite is generated entirely from
`python/laws/conformance.lawspec`. The fresh LawSpec CLI run passed all 314 native cases:

- 114 finite execution cases covering the former numerical, gradient, state,
  identity, train/eval, capture, rewrite, control-flow and saved-export matrix.
- Four finite mutation cases. Checked comparison must reject reversed masks,
  duplicated shared parameters, biased running variance and both-branch execution.
- 162 independently known shape/dtype examples: the former 156 inputs plus six
  empty/nonempty Bool-matmul regressions exposed during migration.
- 31 shape boundary cases, one bounded Hypothesis property and one replay slot.
- One checked comparison law covering NaN, signed infinities, unequal lengths
  and exact equal values.

Replay slots are reported separately from ordinary examples; they do not imply
new input coverage. Numerical observations contain every output, gradient and
state entry. Checked LawSpec definitions compare shape, dtype, structure and
identity facts, Float32/Float64 tolerances and exact integer/boolean values.
Native adapters construct fixtures and return observations; they contain no
conformance assertions. The old handwritten pytest files and coarse cached
Boolean probes are removed. [Conformance ownership](law-owned-conformance.md)
maps every former test family and explains the tool-boundary assessment.

The Elixir LawSpec runner has 190 native cases for exact shape models, JSON,
ordered wiring, permutations, interfaces, expansion stability and compatible
resolution. Supplementary ExUnit tests cover diagnostics, constructive shape
properties, pure parallel regions, mutation permissions and buffer-preserving
collapse. These supplement rather than replace executable contracts.

The bounded shape generator and its shrinking remain inside declared LawSpec
types: rank at most four, source extents at most eight, target extents minus one
through eight. Inputs come from the reference domain rather than the compiler.
Intentional incompatibilities have rejection outcomes. Generator exhaustion
would remain incomplete evidence. Finite execution covers the declared case
matrix and seeded fixtures; it is not a proof over all tensors or models.
The native PyTorch reference and native codecs remain explicit trust boundaries.

The migration exposed two useful boundaries: ordinary recursive Python fallback
comparison exceeded the host stack, so the contract uses existing finite
sequence iteration; unbounded shape types admitted impossible host extents, so
the shape domain is now declared explicitly. Generated boundaries then exposed
PyTorch's empty Bool-matmul behavior, which inference now preserves. No LawSpec
compiler change or Ironframe/LawTwin integration is required for this suite.

**Human review is pending.** Canon material is ingested as pending; no verdict is
forged. Strict Canon checking and the aggregate verification exit status
remain nonpassing while sign-off is pending, independently of executable-check
outcomes. The verification script returns 2 for that review condition and keeps
reports in ignored `artifacts/` and `.lawspec/` directories. Regenerate reports
for the current checkout rather than treating an old result as current proof.
