//! The L-BFGS iteration as an ask/tell state machine on [`MoreThuente`].
//!
//! Driver semantics derived from reading SciPy `__lbfgsb.c` `mainlb`/`lnsrlb` (v1.16.3, b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! and `_lbfgsb_py.py` (`_minimize_lbfgsb`): the order of the tests after an accepted step (maxiter, maxfun, gradient,
//! function), the counting of trials against `maxls`, the reset to steepest descent with a unit step after a failed
//! search, the first trial step `min(1/‖d‖, 1e10)`, and the acceptance of a Moré–Thuente warning as the new iterate.
//! The Rust control flow follows SciPy's but is not a line-by-line translation (SciPy works in the compact
//! representation with bounds; this is the two-loop recursion on the unconstrained path). SciPy is BSD-3-Clause, and
//! the L-BFGS-B code it translates is by Zhu, Byrd, Lu, Nocedal and Morales: see `LICENSE-NOTICE-lbfgsb` and
//! `LICENSE-BSD-3-Clause-scipy` in the crate. The two-loop recursion and the ring buffer are written independently
//! (`twoloop.rs`, `memory.rs`). The shape (`new`/`ask`/`tell`, `Step`) is the design sketch of tensor4all-meta
//! `docs/design/tenferro-sci.md` § Optimizers.
//!
//! Departures from SciPy, each listed in the crate docs: non-finite values stop the run at the last finite iterate
//! (decision (b) of #38); the trial point is always `x + α d` (SciPy copies a precomputed `z` when `α = 1`); no
//! restart when SciPy's Cholesky factorization of the middle matrix fails (the two-loop form has none).
//!
//! # Order of work in `tell`
//!
//! 1. the first evaluation (at `x0`) is checked for finiteness and the gradient test, then the first search starts
//!    with `d = −g` and the step `min(1/‖d‖, 1e10)`;
//! 2. later evaluations go to the line search (φ = f, φ' = gᵀd). If it asks for another trial and `maxls` trials have
//!    been evaluated, the search fails (the next trial is never requested); a non-finite φ or φ' ends the run;
//! 3. if it ends in a convergence or a warning, the last trial is the new iterate and the tests run in SciPy's order:
//!    `maxiter`, `maxfun` (strict), `‖g‖∞ ≤ gtol`, one-sided `ftol`; then the pair `(s, y)` is offered to the memory
//!    and the next direction and search start;
//! 4. a failed search (or a direction with `gᵀd ≥ 0`) restores nothing (the accepted iterate was never overwritten)
//!    and either ends the run (empty memory) or empties the memory and retries with `d = −g` and the unit step.

use super::config::LbfgsConfig;
use super::config::H0;
use super::memory::{Memory, Push};
use super::twoloop::twoloop;
use crate::linesearch::{MoreThuente, Reason, Step as LsStep};
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
    x_trial: V,
    d: V,
    s: V,
    y: V,
    mem: Memory<V>,
    alpha: Vec<f64>,
    // The search in progress, `φ'(0) = gᵀd` of its start and the trials evaluated in it.
    ls: Option<MoreThuente>,
    gd0: f64,
    trials: usize,
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
            s: x0.clone(),
            y: x0.clone(),
            mem: Memory::new(cfg.m(), &x0),
            alpha: vec![0.0; cfg.m()],
            x: x0,
            f: f64::NAN,
            ls: None,
            gd0: 0.0,
            trials: 0,
            iterations: 0,
            evaluations: 0,
            skipped: 0,
            restarts: 0,
            terminated: None,
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
    pub fn tell(&mut self, cfg: &LbfgsConfig, ctx: &mut V::Ctx, f: f64, grad: &V) {
        if self.terminated.is_some() {
            return;
        }
        self.evaluations += 1;
        if self.evaluations == 1 {
            self.tell_start(cfg, ctx, f, grad);
        } else {
            self.tell_trial(cfg, ctx, f, grad);
        }
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

impl<V: Vector> LbfgsState<V> {
    /// The evaluation at `x0`: finiteness, gradient test, first search.
    fn tell_start(&mut self, cfg: &LbfgsConfig, ctx: &mut V::Ctx, f: f64, grad: &V) {
        // `norm_inf` is NaN for a NaN entry and infinite for an infinite one.
        let gnorm = grad.norm_inf(ctx);
        if !(f.is_finite() && gnorm.is_finite()) {
            self.terminated = Some(Termination::NonFiniteValue);
            return;
        }
        self.f = f;
        self.g.clone_from(grad);
        if gnorm <= cfg.gtol() {
            self.terminated = Some(Termination::GradTol);
            return;
        }
        self.start_search(cfg, ctx);
    }

    /// An evaluation at a trial point of the search in progress.
    fn tell_trial(&mut self, cfg: &LbfgsConfig, ctx: &mut V::Ctx, f: f64, grad: &V) {
        let dphi = grad.dot(ctx, &self.d);
        self.trials += 1;
        let step = match self.ls.as_mut() {
            Some(ls) => ls.tell(cfg.line_search(), f, dphi),
            None => return,
        };
        match step {
            LsStep::Evaluate { alpha } => {
                if self.trials >= cfg.maxls() {
                    // SciPy's `iback >= maxls`: the next trial is not requested.
                    self.search_failed(cfg, ctx);
                } else {
                    self.set_trial(ctx, alpha);
                }
            }
            LsStep::Done(t) => match t.reason {
                Reason::NonFiniteValue => self.terminated = Some(Termination::NonFiniteValue),
                Reason::NonFiniteStep => self.search_failed(cfg, ctx),
                // convergence and every warning: the last trial is the new iterate
                _ => self.accept(cfg, ctx, f, grad, t.alpha, t.dphi),
            },
        }
    }

    /// `x_trial = x + α d`.
    fn set_trial(&mut self, ctx: &mut V::Ctx, alpha: f64) {
        self.x_trial.clone_from(&self.x);
        self.x_trial.axpy(ctx, alpha, &self.d);
    }

    /// Takes the last trial `(x_trial, f, grad)` as the iterate, runs the termination tests in SciPy's order, offers the
    /// pair to the memory and starts the next search. `alpha` and `dphi` are the step and `gᵀd` of that trial.
    fn accept(
        &mut self,
        cfg: &LbfgsConfig,
        ctx: &mut V::Ctx,
        f: f64,
        grad: &V,
        alpha: f64,
        dphi: f64,
    ) {
        // y = g_new − g_old, s = α d (before g and x are overwritten)
        self.y.clone_from(grad);
        self.y.axpy(ctx, -1.0, &self.g);
        self.s.clone_from(&self.d);
        self.s.scale(ctx, alpha);
        let f_old = self.f;
        self.x.clone_from(&self.x_trial);
        self.g.clone_from(grad);
        self.f = f;
        self.ls = None;
        self.iterations += 1;

        if self.iterations >= cfg.maxiter() {
            self.terminated = Some(Termination::MaxIter);
            return;
        }
        if self.evaluations > cfg.maxfun() {
            self.terminated = Some(Termination::MaxEval);
            return;
        }
        if self.g.norm_inf(ctx) <= cfg.gtol() {
            self.terminated = Some(Termination::GradTol);
            return;
        }
        if f_old - f <= cfg.ftol() * f_old.abs().max(f.abs()).max(1.0) {
            self.terminated = Some(Termination::FTol);
            return;
        }

        // SciPy: dr = (gd − gdold) stp, ddum = −gdold stp, rr = yᵀy.
        let dr = (dphi - self.gd0) * alpha;
        let ddum = -self.gd0 * alpha;
        let rr = self.y.dot(ctx, &self.y);
        if self.mem.push(&self.s, &self.y, dr, ddum, rr) == Push::Skipped {
            self.skipped += 1;
        }
        self.start_search(cfg, ctx);
    }

    /// Computes `d = −H g` from the memory and starts a line search from the current iterate; if that is impossible
    /// (`gᵀd ≥ 0`, or a non-finite start), it is a failed search.
    fn start_search(&mut self, cfg: &LbfgsConfig, ctx: &mut V::Ctx) {
        let gamma = match cfg.h0() {
            H0::ScaledGamma => self.mem.gamma(),
            H0::FixedIdentity => 1.0,
        };
        twoloop(&self.mem, ctx, gamma, &self.g, &mut self.d, &mut self.alpha);
        self.d.scale(ctx, -1.0);
        self.gd0 = self.g.dot(ctx, &self.d);
        self.trials = 0;
        // SciPy: the first search of a run starts at min(1/‖d‖, stpmax), every other one (a retry included) at 1.
        let alpha0 = if self.iterations == 0 {
            (1.0 / self.d.norm(ctx)).min(cfg.line_search().stpmax())
        } else {
            1.0
        };
        match MoreThuente::start(cfg.line_search(), self.f, self.gd0, alpha0) {
            Ok((ls, LsStep::Evaluate { alpha })) => {
                self.ls = Some(ls);
                self.set_trial(ctx, alpha);
            }
            Ok((_, LsStep::Done(_))) | Err(_) => self.search_failed(cfg, ctx),
        }
    }

    /// A failed search: the iterate was never overwritten, so nothing is restored. With an empty memory the run ends;
    /// otherwise the memory is emptied and the search restarts from steepest descent without counting an iteration.
    fn search_failed(&mut self, cfg: &LbfgsConfig, ctx: &mut V::Ctx) {
        self.ls = None;
        if self.mem.len() == 0 {
            self.terminated = Some(Termination::LineSearchFailed);
        } else {
            self.mem.clear();
            self.restarts += 1;
            self.start_search(cfg, ctx);
        }
    }
}
