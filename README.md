# tenferro-sci-rs

General numerical algorithms on top of [tenferro-rs](https://github.com/tensor4all/tenferro-rs),
in the role SciPy plays for NumPy: Krylov solvers, optimizers, special
functions, integrators and interpolation. CPU first, autodiff wherever
possible, GPU where it pays.

**Status:** just started. No crate is published yet. The design is in
[tensor4all-meta · docs/design/tenferro-sci.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro-sci.md).

![tenferro-sci-rs crates sit on tenferro and on pure Rust algorithm cores that any Rust code can use](https://raw.githubusercontent.com/tensor4all/tensor4all-meta/main/docs/design/figures/sci_layers.svg)

The figure is maintained with the design document in tensor4all-meta
(`docs/design/figures/make_sci_figures.py`).

## Planned crates

| Crate | SciPy counterpart | Contents | Core |
| --- | --- | --- | --- |
| `tenferro-krylov` | `sparse.linalg` (iterative part) | eigsolve, linsolve, exponentiate, svdsolve, Davidson | [`krylov`](https://github.com/Ryo-wtnb11/krylov-rs) |
| `tenferro-optimize` | `optimize` | L-BFGS, Levenberg–Marquardt, trust region, root finding | own |
| `tenferro-special` | `special` | gamma, Bessel, error functions, orthogonal polynomials | [`specials`](crates/specials) (in this workspace) |
| `tenferro-integrate` | `integrate` | adaptive quadrature, ODE solvers with adjoints | own |
| `tenferro-interpolate` | `interpolate` | splines, grid interpolation, Chebyshev | own |

All crates live in this one workspace and are released together at one
version. Algorithms that need only vector operations sit in independent,
pure Rust cores that do not depend on tenferro, so any Rust code can use them;
the `tenferro-*` crates add tenferro tensors (CPU and GPU), autodiff rules,
batching and traced use on top.

## Independent cores in this workspace

| Crate | Contents | Status |
| --- | --- | --- |
| [`specials`](crates/specials) | Special functions for `f64` (a port of SciPy's [xsf](https://github.com/scipy/xsf)) and double-double `xprec::Df64` (precision-adaptive algorithms). Gamma family so far. | not published |

## Validation

Results are compared numerically against SciPy (for parity) and mpmath (for
ground truth), with derivatives checked against reference data or finite
differences. See [REPOSITORY_RULES.md](REPOSITORY_RULES.md).

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Coding agents start from
[AGENTS.md](AGENTS.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Code ported from other projects
keeps its upstream license as well; each such crate declares it.
