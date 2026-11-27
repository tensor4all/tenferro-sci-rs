//! Optimization in pure Rust: unconstrained L-BFGS and the Moré–Thuente line search it is built on.
//!
//! [`lbfgs`] is the unconstrained path (`bounds=None`) of SciPy's `L-BFGS-B`, as an ask/tell state machine
//! ([`lbfgs::LbfgsState`]) and a closure driver ([`minimize`]), on any [`Vector`] type (`Vec<f64>` is implemented;
//! the scalar is `f64`, provisionally). [`linesearch`] is a port of SciPy's Moré–Thuente strong-Wolfe line search
//! (MINPACK-2 `dcsrch`/`dcstep`, as translated in `scipy.optimize._dcsrch`) and of the thin driver
//! `scipy.optimize.scalar_search_wolfe1` around it; it works on plain `f64` scalar functions φ(α) and φ'(α) and is the
//! search L-BFGS uses. The crate depends on no other crate beyond `thiserror`.
//!
//! # SciPy mapping
//!
//! The goal is mathematical parity with SciPy 1.16.3, not bit-for-bit agreement: the tests replay SciPy's recorded
//! runs, compare requested points within a stated tolerance and compare evaluation counts exactly.
//!
//! | `scipy.optimize` | `minimize` | Agreement |
//! |---|---|---|
//! | `_dcsrch.DCSRCH` (`_iterate`) | [`linesearch::MoreThuente`] | Steps within 1e-12 relative, evaluation counts equal (dcstep Case 1 tie: see below) |
//! | `_linesearch.scalar_search_wolfe1` | [`linesearch::scalar_search_wolfe1`] | The same, plus `None` ↔ `alpha: None` |
//! | `minimize(method="L-BFGS-B", bounds=None)` (`_lbfgsb_py._minimize_lbfgsb`, `__lbfgsb.c`) | [`minimize`], [`lbfgs::LbfgsState`], [`lbfgs::LbfgsConfig::scipy_default`] | On every recorded run: counts of evaluations, iterations, stored pairs and restarts and the termination equal; every requested point within `32 ε` times the absolute-value scale of its computation plus the measured deviation of SciPy's own direction from the exact one (`C_X · ε · x_scale + x_slack`; `tests/data/lbfgs/README.md` derives both; five long small-memory runs are cut at a recorded `maxiter`, because last-digit differences are amplified past that point). Live runs on closed-form problems: the same counts as SciPy on the committed problems |
//!
//! `maxcor`, `gtol`, `ftol`, `maxiter`, `maxfun` and `maxls` are the fields of [`lbfgs::LbfgsConfig`] of the same
//! meaning; `LbfgsConfig::scipy_default()` has SciPy's defaults. SciPy's `ftol` is the relative tolerance of the
//! function test (`factr = ftol / eps`), here too.
//!
//! Deliberate differences, each documented where it is implemented:
//!
//! - dcstep Case 1 compares with a strict `<` as the paper, the MINPACK Fortran and SciPy's C
//!   copy do; `_dcsrch.py` has `<=` (`docs/upstream-issues.md`). The two differ only on an exact
//!   tie, which no recorded run hits.
//! - NaN and infinite inputs are rejected up front ([`linesearch::ConfigError::NonFinite`],
//!   [`linesearch::StartError::NonFiniteStart`]) instead of being compared as "not negative".
//! - A non-finite φ or φ' at a trial ends the search with [`linesearch::Reason::NonFiniteValue`]
//!   (SciPy keeps iterating on NaN until `maxiter`), which changes the counts on those paths only.
//! - A non-finite next step ends the search with [`linesearch::Reason::NonFiniteStep`], and every
//!   termination reports a `best` point that satisfies the sufficient decrease condition.
//!
//! For L-BFGS:
//!
//! - `maxiter = 0` and `maxfun = 0` are refused ([`lbfgs::ConfigError::ZeroMaxiter`], [`lbfgs::ConfigError::ZeroMaxfun`]);
//!   SciPy accepts both and runs one iteration. A NaN or infinite `gtol` or `ftol`
//!   ([`lbfgs::ConfigError::NonFinite`]) and a negative `gtol` ([`lbfgs::ConfigError::NegativeGtol`]) are refused too;
//!   SciPy accepts them silently (a NaN tolerance never triggers its test). `maxcor = 0`, `maxls = 0` and a negative
//!   `ftol` are errors in SciPy as well.
//! - A NaN or infinite `f` or gradient entry ends the run with [`lbfgs::Termination::NonFiniteValue`] at the last
//!   finite iterate (the decision taken for the line search on #38). SciPy resets its memory and keeps trying (NaN at
//!   `x0` costs it 21 evaluations before `ABNORMAL`), so the counts differ on those paths only; they are tested with
//!   hand-fed values, not replayed.
//! - The trial point is always `x + α d`. SciPy copies a precomputed point when `α = 1`; the two differ in rounding only.
//! - The direction is the two-loop recursion, not SciPy's compact representation, so there is no restart when SciPy's
//!   Cholesky factorization of its middle matrix fails (`formt`). The two agree in exact arithmetic; SciPy's direction
//!   is up to 44 ε times the scale of the computation away from the exact one on the recorded runs (the two-loop
//!   recursion: 1.4 ε), which is why the replay tolerance is measured against SciPy's own error.
//! - After a failed line search `LbfgsState::f` is `f(x)` at the restored `x`; SciPy returns the `f` of the last
//!   rejected trial (`docs/upstream-issues.md`).
//! - Not provided: bounds, `callback`, `hess_inv` and the SciPy options
//!   `eps`, `iprint`, `disp`. The scalar is `f64` for now: `LbfgsConfig` is not generic, and becomes so when the scalar
//!   trait of #38 is public and a second precision has parity data.
//!
//! Not in this crate yet: bounds (the Cauchy point and subspace minimization of L-BFGS-B), BFGS, `line_search_wolfe1`
//! on vectors, the zoom search (`scalar_search_wolfe2`), Armijo and Hager–Zhang searches.
//!
//! # Example
//!
//! ```
//! use minimize::lbfgs::{LbfgsConfig, Termination};
//!
//! // Rosenbrock's function from the classical start, with SciPy's defaults.
//! let rosen = |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
//!     let (a, b) = (x[0], x[1]);
//!     g[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
//!     g[1] = 200.0 * (b - a * a);
//!     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
//! };
//! let rep = minimize::minimize(rosen, vec![-1.2, 1.0], &LbfgsConfig::scipy_default(), &mut ());
//! assert!(matches!(rep.termination, Termination::GradTol | Termination::FTol));
//! assert!((rep.x[0] - 1.0).abs() < 1e-3 && (rep.x[1] - 1.0).abs() < 1e-3);
//! ```
//!
//! and the scalar line search:
//!
//! ```
//! use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Options};
//!
//! let phi = |a: f64| (a - 5.0) * (a - 5.0);
//! let dphi = |a: f64| 2.0 * (a - 5.0);
//! let rep = scalar_search_wolfe1(phi, dphi, None, None, None, &Wolfe1Options::default())?;
//! assert!(rep.alpha.is_some());
//! # Ok::<(), minimize::linesearch::Wolfe1Error>(())
//! ```

pub mod bfgs;
pub mod lbfgs;
pub mod linesearch;
mod real;
pub mod vector;

pub use lbfgs::minimize;
pub use vector::Vector;
