//! Scalar line searches in pure Rust.
//!
//! This crate holds [`linesearch`], a port of SciPy's Moré–Thuente strong-Wolfe line search
//! (MINPACK-2 `dcsrch`/`dcstep`, as translated in `scipy.optimize._dcsrch`) and of the thin
//! driver `scipy.optimize.scalar_search_wolfe1` around it. It depends on no other crate beyond
//! `thiserror`, and works on plain `f64` scalar functions φ(α) and φ'(α).
//!
//! # SciPy mapping
//!
//! The goal is mathematical parity with SciPy 1.16.3, not bit-for-bit agreement: the tests
//! replay SciPy's recorded runs, compare steps within a stated relative tolerance and compare
//! evaluation counts exactly.
//!
//! | `scipy.optimize` | `minimize` | Agreement |
//! |---|---|---|
//! | `_dcsrch.DCSRCH` (`_iterate`) | [`linesearch::MoreThuente`] | Steps within 1e-12 relative, evaluation counts equal (dcstep Case 1 tie: see below) |
//! | `_linesearch.scalar_search_wolfe1` | [`linesearch::scalar_search_wolfe1`] | The same, plus `None` ↔ `alpha: None` |
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
//! Not in this crate yet: `line_search_wolfe1` on vectors, the zoom search
//! (`scalar_search_wolfe2`), Armijo and Hager–Zhang searches.
//!
//! # Example
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

pub mod linesearch;
mod real;
