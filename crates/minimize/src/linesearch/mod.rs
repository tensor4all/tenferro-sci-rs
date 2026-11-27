//! Moré–Thuente strong-Wolfe line search.
//!
//! Given a descent direction, i.e. a function φ(α) of the step α ≥ 0 with φ'(0) < 0, the search
//! looks for a step satisfying the strong Wolfe conditions (Nocedal & Wright, *Numerical
//! Optimization*, 2nd ed., §3.1; Moré & Thuente (1.1)–(1.2)):
//!
//! ```text
//! (1.1)  φ(α) ≤ φ(0) + μ α φ'(0)          sufficient decrease        (μ = ftol)
//! (1.2)  |φ'(α)| ≤ η |φ'(0)|              curvature                  (η = gtol)
//! ```
//!
//! It works with the modified function
//!
//! ```text
//! ψ(α) = φ(α) − φ(0) − μ α φ'(0),
//! ```
//!
//! for which (1.1) reads ψ(α) ≤ 0. Stage 1 ([`Stage::Auxiliary`]) keeps a safeguarded interval
//! that contains a minimizer of ψ; after the first trial with ψ ≤ 0 and φ' ≥ 0 it works on φ
//! itself ([`Stage::Function`]). The interval endpoint with the least value always satisfies
//! (1.1) (invariant (2.1), Theorem 2.1), which is why every termination reports a usable
//! `best` point. If μ < η and φ is bounded below, a step satisfying both conditions exists.
//!
//! # Use
//!
//! * [`MoreThuente`] is the search as an ask/tell state machine: it asks for φ and φ' at a trial
//!   step, you evaluate them wherever and however you like (a vector problem, a GPU, a batch)
//!   and pass them to `tell`, which returns the next trial or the [`Termination`].
//!   [`MoreThuenteConfig`] holds the validated parameters and is passed to every call.
//! * [`scalar_search_wolfe1`] is the loop around it for closures, with the semantics of SciPy's
//!   function of the same name.
//!
//! ```
//! use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Step};
//!
//! let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
//! let phi = |a: f64| (a - 5.0) * (a - 5.0);
//! let dphi = |a: f64| 2.0 * (a - 5.0);
//! let (mut ls, mut step) = MoreThuente::start(&cfg, phi(0.0), dphi(0.0), 1.0)?;
//! while let Step::Evaluate { alpha } = step {
//!     step = ls.tell(&cfg, phi(alpha), dphi(alpha));
//! }
//! let Step::Done(t) = step else { unreachable!() };
//! assert!(t.reason.is_converged());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # References
//!
//! - J. J. Moré and D. J. Thuente, *Line search algorithms with guaranteed sufficient decrease*,
//!   ACM Trans. Math. Software 20(3), 286–307, 1994,
//!   [doi:10.1145/192115.192132](https://doi.org/10.1145/192115.192132). The equation, case and
//!   page numbers in the comments of this module are those of the paper.
//! - MINPACK-2, the Fortran routines `dcsrch` and `dcstep` (Argonne National Laboratory and
//!   University of Minnesota), through SciPy's translation `scipy/optimize/_dcsrch.py`.
//! - J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, §3.1
//!   (the strong Wolfe conditions).

mod config;
mod dcstep;
mod driver;
mod more_thuente;

pub use config::{ConfigError, MoreThuenteConfig};
pub use driver::{scalar_search_wolfe1, Wolfe1Error, Wolfe1Options, Wolfe1Outcome, Wolfe1Report};
pub use more_thuente::{MoreThuente, Point, Reason, Stage, StartError, Step, Termination};
