//! Unconstrained BFGS in ask/tell form: the port of SciPy's `minimize(method="BFGS")`.
//!
//! Ported from SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34),
//! `scipy/optimize/_optimize.py` (`_minimize_bfgs`, `_line_search_wolfe12`), with the line search
//! of [`crate::linesearch`]. The update is written independently from the textbook formula, see
//! [`inverse_update`]. The dense inverse Hessian approximation is a `colmat` matrix.
//!
//! # SciPy mapping
//!
//! | `scipy.optimize` | `minimize::bfgs` | Agreement |
//! |---|---|---|
//! | `_minimize_bfgs` (options `gtol`, `xrtol`, `c1`, `c2`, `maxiter`; `norm = inf`, `hess_inv0 = None`) | [`BfgsState`], [`minimize`], [`BfgsConfig`] | Evaluation points within the stated tolerance, evaluation and iteration counts equal, termination equal, **on runs where SciPy's `wolfe2` fallback never fires** |
//! | `_line_search_wolfe12` → `line_search_wolfe1` | [`crate::linesearch::MoreThuente`] | as the line search |
//! | `_line_search_wolfe12` → `line_search_wolfe2` (fallback) | not ported | [`Termination::LineSearchFailed`], no step |
//!
//! **Parity is claimed only where the fallback never fires.** SciPy calls `wolfe2` (the zoom search
//! of Nocedal & Wright, `scalar_search_wolfe2`) when `wolfe1` returns `None`; this crate has no
//! zoom search yet, so there the run stops with [`Termination::LineSearchFailed`] and takes no step,
//! where SciPy would try `wolfe2`. In the recorded cases `wolfe2` fails as well on noisy, nonsmooth
//! and inconsistent objectives (SciPy ends with status 2, after more evaluations), but it **returns
//! a step** on an objective that is unbounded below (its last trial after `maxiter`, with a
//! `LineSearchWarning` that SciPy silences, so not a Wolfe step; SciPy takes it and goes on): there the
//! runs diverge for good, and the recorded case `w2_unbounded` pins where. The data in
//! `tests/data/bfgs/README.md` records, per run, whether the fallback fired.
//!
//! **Red stage:** the types and the Vec-based plumbing exist; the algorithm is not written, so
//! [`BfgsState::ask`] ends every run at once.

mod config;
mod driver;
mod state;
mod update;

pub use config::{BfgsConfig, ConfigError};
pub use driver::{minimize, Report};
pub use state::{BfgsState, Step, Termination};
pub use update::{inverse_update, UpdateError};
