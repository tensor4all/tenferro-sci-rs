# AGENTS.md

Before acting, read the latest shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`,
and load only the rule files relevant to the task. If the remote cannot be
reached, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.

Then read, in this order:

1. [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md): the rules specific to this
   repository. Re-read it before implementation and before opening a PR.
2. The design: [tensor4all-meta · docs/design/tenferro-sci.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro-sci.md).
3. [`CONTRIBUTING.md`](CONTRIBUTING.md), which points to the rules shared by
   every tensor4all Rust repository (issues, PR evidence, gates, review).

Short version of the repository rules:

- Two kinds of crates. Independent cores are pure Rust and never depend on a
  tenferro crate. `tenferro-<module>` crates depend on released tenferro crates
  and add tensors, autodiff rules, batching and GPU.
- Depend on tenferro only through the versions pinned once in the root
  `Cargo.toml`. No git or path dependency on tenferro-rs; the
  `upstream-main.yml` workflow covers its `main`.
- Every numerical claim is checked against SciPy or mpmath reference values
  with a stated tolerance. A test that only checks finiteness is not
  validation.
- Ported code records its source project, file and commit, and keeps the
  upstream license.
- Contractions, GEMM, decompositions and solves go through tenferro, tcontract,
  tlinalg or colmat; no handwritten inner loops.
- When work here reveals a shortcoming in tenferro's design or API (a
  missing primitive, an awkward contract, a performance gap), do not work
  around it ad hoc in this repository. Report it to the user with the
  evidence, and after the user confirms, improve tenferro-rs itself (issue,
  then PR there). Until that lands, the change here waits or stays on its
  branch.
- The PR body carries `## Verification` with a command and its output, and a
  line-initial `Model: <provider>/<model>`.
