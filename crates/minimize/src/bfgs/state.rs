//! The BFGS state machine in ask/tell form.
//!
//! Semantics of SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! `scipy/optimize/_optimize.py::_minimize_bfgs` and `_line_search_wolfe12` (the `wolfe1` branch),
//! with the line search of [`crate::linesearch`]; the update is written independently, see
//! [`inverse_update`](crate::bfgs::inverse_update). Optim.jl and LineSearches.jl are references
//! only; nothing is ported from them.
//!
//! **Red stage:** the state allocates its workspace and reports `x0`; [`BfgsState::ask`] says
//! [`Termination::MaxIter`] at once and [`BfgsState::tell`] does nothing, so every behavioural
//! test fails.

use super::config::BfgsConfig;
use crate::Vector;
use colmat::{Mat, MatRef};

/// Why the iteration ended.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::Termination;
///
/// assert_ne!(Termination::GradTol, Termination::StepTol);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// `‖g‖∞ <= gtol`.
    GradTol,
    /// The last step satisfied `α‖p‖₂ <= xrtol (xrtol + ‖x‖₂)`.
    StepTol,
    /// `maxiter` iterations were done. As in SciPy this wins over [`Termination::GradTol`] when the
    /// last allowed iteration is the converging one.
    MaxIter,
    /// The line search found no step. SciPy would try `wolfe2` here (decision 3 on #40); no step
    /// is taken and `x` is the last accepted iterate.
    LineSearchFailed,
    /// `f` or `g` was NaN or infinite at a trial point (#38, option (b)): the iteration stops at
    /// the last finite iterate, which is what `x`, `f` and `g` hold.
    NonFiniteValue,
}

/// What the caller does next.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::{BfgsConfig, BfgsState, Step};
///
/// let cfg = BfgsConfig::scipy_default(2);
/// let state = BfgsState::new(&cfg, &mut (), vec![1.0, -1.0]);
/// match state.ask() {
///     Step::Evaluate { x } => assert_eq!(x, &vec![1.0, -1.0]),
///     Step::Converged(t) => panic!("nothing is evaluated yet, got {t:?}"),
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step<'a, V> {
    /// Compute `f(x)` and its gradient, then call [`BfgsState::tell`].
    Evaluate {
        /// The point to evaluate, borrowed from the state.
        x: &'a V,
    },
    /// The iteration is over.
    Converged(Termination),
}

/// The state of a BFGS run: iterate, gradient, direction and the dense inverse Hessian
/// approximation (a `colmat` matrix, host memory).
///
/// `V` must expose its entries (`AsRef<[f64]>`, `AsMut<[f64]>`) because the dense matrix
/// product `H g` needs them; this is a deviation from the design sketch's `V: Vector` (the
/// trait has no dimension or coordinate access), asked about in the PR.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::{BfgsConfig, BfgsState};
///
/// let cfg = BfgsConfig::scipy_default(2);
/// let state = BfgsState::new(&cfg, &mut (), vec![1.0, -1.0]);
/// assert_eq!(state.x(), &vec![1.0, -1.0]);
/// assert_eq!((state.iterations(), state.evaluations()), (0, 0));
/// assert_eq!(state.inverse_hessian().nrows(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct BfgsState<V> {
    x: V,
    f: f64,
    g: V,
    p: V,
    h: Mat<f64>,
    iterations: usize,
    nfev: usize,
}

impl<V: Vector + AsRef<[f64]> + AsMut<[f64]>> BfgsState<V> {
    /// A state at `x0` with `H = I`; nothing is evaluated yet. The workspace is allocated here
    /// by cloning `x0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::bfgs::{BfgsConfig, BfgsState};
    ///
    /// let cfg = BfgsConfig::scipy_default(3);
    /// let state = BfgsState::new(&cfg, &mut (), vec![0.0; 3]);
    /// assert_eq!(state.inverse_hessian().ncols(), 3);
    /// ```
    pub fn new(_cfg: &BfgsConfig, _ctx: &mut V::Ctx, x0: V) -> Self {
        let n = x0.as_ref().len();
        Self {
            g: x0.clone(),
            p: x0.clone(),
            x: x0,
            f: f64::NAN,
            h: Mat::identity(n, n),
            iterations: 0,
            nfev: 0,
        }
    }

    /// What to do next: evaluate a point, or the run is over.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::bfgs::{BfgsConfig, BfgsState, Step};
    ///
    /// let cfg = BfgsConfig::scipy_default(1);
    /// let state = BfgsState::new(&cfg, &mut (), vec![2.0]);
    /// assert!(matches!(state.ask(), Step::Evaluate { .. }));
    /// ```
    pub fn ask(&self) -> Step<'_, V> {
        Step::Converged(Termination::MaxIter)
    }

    /// Passes `f` and the gradient at the point last asked.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::bfgs::{BfgsConfig, BfgsState, Step};
    ///
    /// let cfg = BfgsConfig::scipy_default(1);
    /// let mut state = BfgsState::new(&cfg, &mut (), vec![2.0]);
    /// // f(x) = x^2 / 2 at x = 2: f = 2, g = 2
    /// state.tell(&cfg, &mut (), 2.0, &vec![2.0]);
    /// assert!(matches!(state.ask(), Step::Evaluate { x } if x[0] != 2.0));
    /// ```
    pub fn tell(&mut self, _cfg: &BfgsConfig, _ctx: &mut V::Ctx, _f: f64, _grad: &V) {}

    /// The last accepted iterate (`x0` before the first iteration).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(1), &mut (), vec![3.0]);
    /// assert_eq!(state.x(), &vec![3.0]);
    /// ```
    pub fn x(&self) -> &V {
        &self.x
    }

    /// `f` at [`x`](Self::x); NaN before the first `tell`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(1), &mut (), vec![3.0]);
    /// assert!(state.f().is_nan());
    /// ```
    pub fn f(&self) -> f64 {
        self.f
    }

    /// The gradient at [`x`](Self::x).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(2), &mut (), vec![3.0, 4.0]);
    /// assert_eq!(state.grad().len(), 2);
    /// ```
    pub fn grad(&self) -> &V {
        &self.g
    }

    /// The search direction `p = −H g` of the current iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(2), &mut (), vec![3.0, 4.0]);
    /// assert_eq!(state.direction().len(), 2);
    /// ```
    pub fn direction(&self) -> &V {
        &self.p
    }

    /// Accepted iterations (SciPy's `nit`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(1), &mut (), vec![3.0]);
    /// assert_eq!(state.iterations(), 0);
    /// ```
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Evaluations passed to [`tell`](Self::tell) so far (SciPy's `nfev`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(1), &mut (), vec![3.0]);
    /// assert_eq!(state.evaluations(), 0);
    /// ```
    pub fn evaluations(&self) -> usize {
        self.nfev
    }

    /// Why the run ended, `None` while it is running.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(1), &mut (), vec![3.0]);
    /// assert_eq!(state.termination(), None);
    /// ```
    pub fn termination(&self) -> Option<super::Termination> {
        None
    }

    /// The current inverse Hessian approximation `H` (`H = I` before the first update).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::{BfgsConfig, BfgsState};
    /// let state = BfgsState::new(&BfgsConfig::scipy_default(2), &mut (), vec![3.0, 4.0]);
    /// let h = state.inverse_hessian();
    /// assert_eq!((h[(0, 0)], h[(0, 1)], h[(1, 1)]), (1.0, 0.0, 1.0));
    /// ```
    pub fn inverse_hessian(&self) -> MatRef<'_, f64> {
        self.h.as_ref()
    }
}
