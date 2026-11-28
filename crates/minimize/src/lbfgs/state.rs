//! The L-BFGS iteration as an ask/tell state machine.
//!
//! The driver semantics are derived from reading SciPy v1.16.3 (commit `b9105ccc2237f57acb1060202cd77f6dd264fb34`)
//! `scipy/optimize/__lbfgsb.c` (`mainlb`, `lnsrlb`) and `scipy/optimize/_lbfgsb_py.py` (`_minimize_lbfgsb`); see the
//! module docs of [`super`] and `tests/data/lbfgs/README.md`. The shape (`new`/`ask`/`tell`, `Step`) is the design
//! sketch of tensor4all-meta `docs/design/tenferro-sci.md` § Optimizers.
//!
//! **Red stage:** [`LbfgsState::ask`] returns [`Step::Converged`] immediately and [`LbfgsState::tell`] ignores its
//! arguments, so every behavioural test fails.

use super::config::LbfgsConfig;
use super::memory::Memory;
use crate::vector::Vector;

/// Why the iteration stopped.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::Termination;
///
/// assert_ne!(Termination::GradTol, Termination::FTol);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// `‖g‖∞ ≤ gtol`, tested at the start and after every accepted iteration.
    GradTol,
    /// `f_old − f ≤ ftol · max(|f_old|, |f|, 1)` after an accepted iteration (one-sided: an increase also stops).
    FTol,
    /// `maxiter` accepted iterations. Tested before [`Termination::GradTol`]: a run that satisfies the gradient test
    /// at iteration `maxiter` reports `MaxIter` (SciPy's driver does the same).
    MaxIter,
    /// More than `maxfun` evaluations after an accepted iteration (strict `>`; a search in progress is not interrupted).
    MaxEval,
    /// The line search failed (not a descent direction, `maxls` trials without success, or a non-finite step) and
    /// the steepest-descent retry after a memory reset failed too.
    LineSearchFailed,
    /// `f` or a gradient entry was NaN or infinite. The state holds the last finite iterate
    /// (the departure from SciPy decided on #38, option (b)).
    NonFiniteValue,
}

/// What the iteration wants next.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step};
///
/// let cfg = LbfgsConfig::scipy_default();
/// let state = LbfgsState::new(&cfg, &mut (), vec![1.0, 2.0]);
/// // the first request is the starting point itself
/// assert!(matches!(state.ask(), Step::Evaluate { x } if *x == vec![1.0, 2.0]));
/// # Ok::<(), minimize::lbfgs::ConfigError>(())
/// ```
#[derive(Debug)]
pub enum Step<'a, V> {
    /// Evaluate `f` and `∇f` at `x` and pass them to [`LbfgsState::tell`]. `x` borrows the state.
    Evaluate {
        /// The point to evaluate.
        x: &'a V,
    },
    /// The iteration is over.
    Converged(Termination),
}

/// The state of an L-BFGS run: plain data, no closures, workspace allocated once in [`LbfgsState::new`].
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step};
///
/// // minimize f(x) = |x|² / 2: ask for a point, evaluate, tell.
/// let cfg = LbfgsConfig::scipy_default();
/// let mut state = LbfgsState::new(&cfg, &mut (), vec![3.0, -4.0]);
/// let mut evals = 0;
/// loop {
///     let (f, g) = match state.ask() {
///         Step::Evaluate { x } => (0.5 * x.iter().map(|t| t * t).sum::<f64>(), x.clone()),
///         Step::Converged(_) => break,
///     };
///     state.tell(&cfg, &mut (), f, &g);
///     evals += 1;
/// }
/// assert!(state.x().iter().all(|t| t.abs() < 1e-5));
/// assert_eq!(evals, state.evaluations());
/// # Ok::<(), minimize::lbfgs::ConfigError>(())
/// ```
#[derive(Clone, Debug)]
pub struct LbfgsState<V: Vector> {
    x: V,
    f: f64,
    g: V,
    // Workspace of the iteration, allocated in `new` (cloned from `x0`).
    #[allow(dead_code)]
    x_trial: V,
    #[allow(dead_code)]
    d: V,
    #[allow(dead_code)]
    mem: Memory<V>,
    #[allow(dead_code)]
    alpha: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    skipped: usize,
    restarts: usize,
    terminated: Option<Termination>,
}

impl<V: Vector> LbfgsState<V> {
    /// Takes the starting point `x0` (it becomes the state's iterate), allocates the rest of the workspace by cloning
    /// it, and returns a state whose first request is `x0` itself. Nothing is evaluated.
    ///
    /// `ctx` is threaded through for uniformity with the other calls; the host path (`Ctx = ()`) ignores it.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let state = LbfgsState::new(&cfg, &mut (), vec![1.0]);
    /// assert_eq!((state.iterations(), state.evaluations()), (0, 0));
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn new(cfg: &LbfgsConfig, _ctx: &mut V::Ctx, x0: V) -> Self {
        Self {
            g: x0.clone(),
            x_trial: x0.clone(),
            d: x0.clone(),
            mem: Memory::new(cfg.m(), &x0),
            alpha: vec![0.0; cfg.m()],
            x: x0,
            f: f64::NAN,
            iterations: 0,
            evaluations: 0,
            skipped: 0,
            restarts: 0,
            // Stub: ends at once.
            terminated: Some(Termination::LineSearchFailed),
        }
    }

    /// The next request: a point to evaluate, or the reason the iteration is over.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let state = LbfgsState::new(&cfg, &mut (), vec![0.5]);
    /// assert!(matches!(state.ask(), Step::Evaluate { .. }));
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn ask(&self) -> Step<'_, V> {
        match self.terminated {
            Some(t) => Step::Converged(t),
            None => Step::Evaluate { x: &self.x_trial },
        }
    }

    /// Passes `f` and `∇f` at the point of the last [`Step::Evaluate`]; the next [`ask`](Self::ask) says what is
    /// next. Calling it after [`Step::Converged`] does nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let mut state = LbfgsState::new(&cfg, &mut (), vec![0.0]);
    /// state.tell(&cfg, &mut (), 0.0, &vec![0.0]); // stationary at the start
    /// assert_eq!((state.evaluations(), state.iterations()), (1, 0));
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn tell(&mut self, _cfg: &LbfgsConfig, _ctx: &mut V::Ctx, _f: f64, _grad: &V) {
        // Stub: ignores the evaluation.
    }

    /// The current accepted iterate (the starting point before the first iteration; after
    /// [`Termination::NonFiniteValue`] or [`Termination::LineSearchFailed`] the last finite accepted iterate).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let state = LbfgsState::new(&cfg, &mut (), vec![1.5]);
    /// assert_eq!(state.x(), &vec![1.5]);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn x(&self) -> &V {
        &self.x
    }

    /// `f` at [`x`](Self::x); NaN before the first evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let state = LbfgsState::new(&cfg, &mut (), vec![1.5]);
    /// assert!(state.f().is_nan());
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn f(&self) -> f64 {
        self.f
    }

    /// `∇f` at [`x`](Self::x) (the starting point's slot, unspecified, before the first evaluation).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// let state = LbfgsState::new(&cfg, &mut (), vec![1.5, 2.5]);
    /// assert_eq!(state.grad().len(), 2);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn grad(&self) -> &V {
        &self.g
    }

    /// The number of accepted iterations (SciPy's `nit`).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!(LbfgsState::new(&cfg, &mut (), vec![1.0]).iterations(), 0);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The number of evaluations passed to [`tell`](Self::tell) (SciPy's `nfev`). A line-search trial that was
    /// aborted by `maxls` was never requested, so it is not counted.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!(LbfgsState::new(&cfg, &mut (), vec![1.0]).evaluations(), 0);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn evaluations(&self) -> usize {
        self.evaluations
    }

    /// The number of stored `(s, y)` pairs (SciPy's `col`). Diagnostic but stable: part of the public API.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!(LbfgsState::new(&cfg, &mut (), vec![1.0]).stored_pairs(), 0);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn stored_pairs(&self) -> usize {
        self.mem.len()
    }

    /// The number of updates skipped by the `yᵀs ≤ eps · (−gᵀs)` rule (SciPy's `nskip`). Diagnostic but stable: part
    /// of the public API.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!(LbfgsState::new(&cfg, &mut (), vec![1.0]).skipped_updates(), 0);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn skipped_updates(&self) -> usize {
        self.skipped
    }

    /// The number of memory resets after a failed line search (the retry is steepest descent with step 1). Diagnostic
    /// but stable: part of the public API.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{LbfgsConfig, LbfgsState};
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!(LbfgsState::new(&cfg, &mut (), vec![1.0]).restarts(), 0);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn restarts(&self) -> usize {
        self.restarts
    }
}
