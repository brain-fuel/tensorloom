---
id: tensorloom.discovery
kind: explanation
title: Intent and starting evidence
---
# Intent and starting evidence

The supplied implementation plan is the source of settled intent. It requests
an Elixir compiler with builder, JSON and PyTorch capture inputs, executable
contracts, shared state, structured control flow, inspectable expansions, and
separate eager and export evidence. Canon records those decisions without
claiming human vetting. See ref:DEC-graph and ref:DEC-compiler.

Shape contracts follow ref:DEC-shapes; dependency-preserving transformations
follow ref:DEC-rewrites; branch and loop semantics follow ref:DEC-control;
state identity follows ref:DEC-state; explicit capture follows ref:DEC-import.

At discovery, `lib/tensorloom.ex` contained only a module description,
`laws/atoi_codec.lawspec` was the generated integer codec starter, and
`laws/PROPOSED.md` held nonexecutable proposals. `graph_notation.txt` version
0.4 asserted tensor-only results, physical copying at forks, and a general
hypercube interpretation. These are observations of the initial files, not
requirements to preserve their limitations.

Implementation assumptions: JSON maps are the public Elixir value model;
region arguments are explicit; CPU layouts and dtypes follow the selected
profile; symbolic batch dimensions may remain dynamic while spatial window
geometry needs concrete dimensions. Collapse may conservatively reject some
acyclic selections. Sequence capture initially supports one layer and direction;
cell builders support arbitrary fixed sequence composition.

Human review remains pending. Automated tests check implementation conformance;
they do not establish that the plan captures every future use case. No solver or
proof checker is selected. Generator exhaustion is incomplete evidence, never
success or a proof of an empty domain.
