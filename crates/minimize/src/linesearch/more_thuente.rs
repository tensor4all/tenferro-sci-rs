//! The Moré–Thuente search as an ask/tell state machine (MINPACK-2 `dcsrch`).
//!
//! Ported from SciPy v1.16.3 (b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! scipy/optimize/_dcsrch.py (`DCSRCH._iterate`), a 2023 translation of MINPACK-2 dcsrch
//! (Argonne National Laboratory / University of Minnesota; see LICENSE-NOTICE-minpack2).
//! Deviations:
//!
//! - Inversion of control: SciPy's `task` string and its `START` / `FG` re-entry become
//!   [`MoreThuente::start`] and [`MoreThuente::tell`], returning a [`Step`]. The configuration is
//!   not stored; `gtest = ftol * ginit` is recomputed from the `&cfg` passed to every `tell`.
//! - The warning tests are evaluated in the reverse of `_iterate`'s order (a later test
//!   overwrites an earlier one there) and the first match is returned: convergence, stpmin,
//!   stpmax, xtol, rounding errors. The result is the same without mutable overwrites.
//! - `start` additionally rejects a non-finite `alpha0`, φ(0) or φ'(0) ([`StartError::NonFiniteStart`]).
//!   SciPy has no such check.
//! - A non-finite φ or φ' at a trial ends the search with [`Reason::NonFiniteValue`].
//! - A non-finite next step ends the search with [`Reason::NonFiniteStep`] inside `tell`
//!   (SciPy's driver tests `isfinite(stp)` after `_iterate`).
//! - NaN propagates ([`crate::real`]); the final clip into `[stpmin, stpmax]` lets NaN through so
//!   that it reaches the non-finite check. Python's `min`/`max` and `np.clip` are not emulated.
//! - The state is generic over a private scalar trait and instantiated for `f64` only.

use super::config::MoreThuenteConfig;
use super::dcstep::{dcstep, Pt};
use crate::real::{clip, max, min, Real};

/// δ of the bisection rule (p. 293): if the bracket has not shrunk to this fraction of its width
/// two steps ago, the next trial is its midpoint.
const BISECT_FRACTION: f64 = 0.66;
/// Lower factor of the extrapolation interval (2.2), `stmin = stp + 1.1 (stp - stx)`.
const XTRAP_LOWER: f64 = 1.1;
/// Upper factor of the extrapolation interval (2.2), `stmax = stp + 4 (stp - stx)` (δ = 4, p. 304).
const XTRAP_UPPER: f64 = 4.0;

/// A point `(α, φ(α), φ'(α))`.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::Point;
///
/// let p = Point { alpha: 1.0, phi: -0.5, dphi: 0.0 };
/// assert_eq!(p, Point { alpha: 1.0, phi: -0.5, dphi: 0.0 });
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// The step α.
    pub alpha: f64,
    /// φ(α).
    pub phi: f64,
    /// φ'(α).
    pub dphi: f64,
}

/// Which function the search currently works on (Moré–Thuente §3–§4, p. 297–298).
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Stage};
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// let (ls, _) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0)?;
/// assert_eq!(ls.stage(), Stage::Auxiliary);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Stage 1: the interval contains a minimizer of the modified function
    /// ψ(α) = φ(α) − φ(0) − μ α φ'(0) (§3). The search starts here.
    Auxiliary,
    /// Stage 2: after the first trial with ψ ≤ 0 and φ' ≥ 0, the interval contains a minimizer
    /// of φ itself (Theorem 3.2, p. 297).
    Function,
}

/// Why a search ended.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::Reason;
///
/// assert!(Reason::Converged.is_converged());
/// assert!(!Reason::StpMax.is_converged());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Both strong-Wolfe conditions (1.1) and (1.2) hold at the last trial.
    Converged,
    /// Rounding errors prevent progress: the trial step reached an end of the interval.
    RoundingErrors,
    /// The bracket is narrower than `xtol` relative to its upper end.
    XtolSatisfied,
    /// The step reached `stpmax` while (1.1) holds and φ' stays below `ftol φ'(0)`.
    StpMax,
    /// The step reached `stpmin` without sufficient decrease.
    StpMin,
    /// The trial-step formula produced a non-finite step. This does not only happen with
    /// non-finite φ: on Moré–Thuente's example (3.1) a cubic through two collinear points has
    /// q = 0. The `alpha`/`phi`/`dphi` of the [`Termination`] are the last evaluated (finite)
    /// trial; the non-finite next step is not reported.
    NonFiniteStep,
    /// φ or φ' at the last trial is NaN or infinite. Detected in `tell` before the trial is used,
    /// so `best` is untouched and still satisfies (1.1). `alpha`/`phi`/`dphi` are that trial.
    /// SciPy does not check this (a NaN φ burns the remaining iterations).
    NonFiniteValue,
}

impl Reason {
    /// Whether this is [`Reason::Converged`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::Reason;
    /// assert!(Reason::Converged.is_converged());
    /// assert!(!Reason::RoundingErrors.is_converged());
    /// ```
    pub fn is_converged(&self) -> bool {
        matches!(self, Reason::Converged)
    }
}

/// Why [`MoreThuente::start`] refused its input.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, StartError};
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// let e = MoreThuente::start(&cfg, 1.0, 1.0, 1.0).unwrap_err();
/// assert_eq!(e, StartError::NotDescent);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StartError {
    /// `alpha0 < stpmin`.
    #[error("the initial step is below stpmin")]
    StepBelowStpmin,
    /// `alpha0 > stpmax`.
    #[error("the initial step is above stpmax")]
    StepAboveStpmax,
    /// `phi'(0) >= 0`: not a descent direction.
    #[error("phi'(0) is not negative: not a descent direction")]
    NotDescent,
    /// `alpha0`, `phi(0)` or `phi'(0)` is NaN or infinite. A documented addition to SciPy.
    #[error("alpha0, phi(0) and phi'(0) must be finite")]
    NonFiniteStart,
}

/// The end of a search.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Reason, Step};
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// // phi(a) = (a - 1)^2, phi(0) = 1, phi'(0) = -2: the first trial a = 1 converges.
/// let (mut ls, _) = MoreThuente::start(&cfg, 1.0, -2.0, 1.0)?;
/// let Step::Done(t) = ls.tell(&cfg, 0.0, 0.0) else { unreachable!() };
/// assert_eq!(t.reason, Reason::Converged);
/// assert_eq!((t.alpha, t.evaluations), (1.0, 1));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Termination {
    /// Why the search ended.
    pub reason: Reason,
    /// The last trial step, as MINPACK returns it.
    pub alpha: f64,
    /// φ at the last trial.
    pub phi: f64,
    /// φ' at the last trial.
    pub dphi: f64,
    /// Number of `tell` calls, i.e. evaluated and tested trials.
    pub evaluations: usize,
    /// The best point (stx, fx, gx). By invariant (2.1) it always satisfies (1.1), so on a
    /// warning the caller still has a usable point.
    pub best: Point,
}

/// What the search wants next.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Step};
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// let (_, step) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0)?;
/// assert_eq!(step, Step::Evaluate { alpha: 1.0 });
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Evaluate φ and φ' at `alpha` and pass them to [`MoreThuente::tell`].
    Evaluate {
        /// The trial step.
        alpha: f64,
    },
    /// The search is over.
    Done(Termination),
}

/// What MINPACK's `isave`/`dsave` keep between calls, apart from `gtest` (recomputed from the
/// configuration). Fixed size, nothing is allocated.
#[derive(Clone, Debug)]
struct State<T: Real> {
    brackt: bool,
    stage: Stage,
    /// φ(0) and φ'(0).
    finit: T,
    ginit: T,
    /// stx, fx, gx: the best point.
    x: Pt<T>,
    /// sty, fy, gy: the other end of the interval.
    y: Pt<T>,
    /// The interval the next step must stay in (the bracket, or the extrapolation interval).
    stmin: T,
    stmax: T,
    /// Bracket width now and two steps ago (the 0.66 rule).
    width: T,
    width1: T,
    /// The last asked step.
    stp: T,
    evaluations: usize,
    /// Set once the search is over; `tell` returns it again.
    done: Option<Termination>,
}

/// (α, f − α·gtest, g − gtest): the modified function ψ of §3 (the paper's ψ has gtest = μ φ'(0),
/// and f − φ(0), which is a constant shift that leaves every comparison unchanged).
fn to_psi<T: Real>(p: Pt<T>, gtest: T) -> Pt<T> {
    Pt {
        a: p.a,
        f: p.f - p.a * gtest,
        g: p.g - gtest,
    }
}

/// The inverse of [`to_psi`].
fn from_psi<T: Real>(p: Pt<T>, gtest: T) -> Pt<T> {
    Pt {
        a: p.a,
        f: p.f + p.a * gtest,
        g: p.g + gtest,
    }
}

fn point<T: Real>(p: Pt<T>) -> Point {
    Point {
        alpha: p.a.to_f64(),
        phi: p.f.to_f64(),
        dphi: p.g.to_f64(),
    }
}

impl<T: Real> State<T> {
    /// The START block of `_iterate`: argument checks, then the initial interval (0, φ(0), φ'(0))
    /// with extrapolation bound 5 α0 and `width1 = 2 width`. Returns the first trial step.
    fn start(
        cfg: &MoreThuenteConfig,
        phi0: T,
        dphi0: T,
        alpha0: T,
    ) -> Result<(Self, T), StartError> {
        let zero = T::from_f64(0.0);
        if !(alpha0.is_finite() && phi0.is_finite() && dphi0.is_finite()) {
            return Err(StartError::NonFiniteStart);
        }
        // `_iterate` reports the last failing check: not descent over above over below.
        if dphi0 >= zero {
            return Err(StartError::NotDescent);
        }
        if alpha0 > T::from_f64(cfg.stpmax()) {
            return Err(StartError::StepAboveStpmax);
        }
        if alpha0 < T::from_f64(cfg.stpmin()) {
            return Err(StartError::StepBelowStpmin);
        }
        let p = Pt {
            a: zero,
            f: phi0,
            g: dphi0,
        };
        let width = T::from_f64(cfg.stpmax()) - T::from_f64(cfg.stpmin());
        let state = State {
            brackt: false,
            stage: Stage::Auxiliary,
            finit: phi0,
            ginit: dphi0,
            x: p,
            y: p,
            stmin: zero,
            stmax: alpha0 + T::from_f64(XTRAP_UPPER) * alpha0,
            width,
            width1: width / T::from_f64(0.5),
            stp: alpha0,
            evaluations: 0,
            done: None,
        };
        Ok((state, alpha0))
    }

    /// `gtest = ftol φ'(0)`, recomputed from the configuration (the state holds none).
    fn gtest(&self, cfg: &MoreThuenteConfig) -> T {
        T::from_f64(cfg.ftol()) * self.ginit
    }

    /// The termination tests of `_iterate` for the trial `t`; pure. `_iterate` runs them in the
    /// order rounding, xtol, stpmax, stpmin, convergence and lets each overwrite the last; the
    /// first match of the reverse order is the same answer.
    fn check(&self, cfg: &MoreThuenteConfig, t: Pt<T>) -> Option<Reason> {
        let gtest = self.gtest(cfg);
        let ftest = self.finit + t.a * gtest;
        let (stpmin, stpmax) = (T::from_f64(cfg.stpmin()), T::from_f64(cfg.stpmax()));
        // (1.1) and (1.2): f <= ftest and |g| <= gtol |φ'(0)|  (ginit < 0)
        if t.f <= ftest && t.g.abs() <= T::from_f64(cfg.gtol()) * -self.ginit {
            return Some(Reason::Converged);
        }
        if t.a == stpmin && (t.f > ftest || t.g >= gtest) {
            return Some(Reason::StpMin);
        }
        if t.a == stpmax && t.f <= ftest && t.g <= gtest {
            return Some(Reason::StpMax);
        }
        if self.brackt && self.stmax - self.stmin <= T::from_f64(cfg.xtol()) * self.stmax {
            return Some(Reason::XtolSatisfied);
        }
        if self.brackt && (t.a <= self.stmin || t.a >= self.stmax) {
            return Some(Reason::RoundingErrors);
        }
        None
    }

    /// Everything after the termination tests: the modified-function branch or plain `dcstep`,
    /// the bisection rule, the new interval and the bounds. Returns the next step.
    fn advance(&mut self, cfg: &MoreThuenteConfig, t: Pt<T>) -> T {
        let half = T::from_f64(0.5);
        let gtest = self.gtest(cfg);
        let ftest = self.finit + t.a * gtest;
        let (stpmin, stpmax) = (T::from_f64(cfg.stpmin()), T::from_f64(cfg.stpmax()));

        // A modified function is used to predict the step during the first stage if a lower
        // function value has been obtained but the decrease is not sufficient (§3).
        let step = if self.stage == Stage::Auxiliary && t.f <= self.x.f && t.f > ftest {
            let d = dcstep(
                to_psi(self.x, gtest),
                to_psi(self.y, gtest),
                to_psi(t, gtest),
                self.brackt,
                self.stmin,
                self.stmax,
            );
            let mut d = d;
            d.x = from_psi(d.x, gtest);
            d.y = from_psi(d.y, gtest);
            d
        } else {
            dcstep(self.x, self.y, t, self.brackt, self.stmin, self.stmax)
        };
        self.x = step.x;
        self.y = step.y;
        self.brackt = step.brackt;
        let mut stp = step.stp;

        // Decide if a bisection step is needed: the bracket must shrink by BISECT_FRACTION every
        // two steps (p. 292–293).
        if self.brackt {
            let w = (self.y.a - self.x.a).abs();
            if w >= T::from_f64(BISECT_FRACTION) * self.width1 {
                stp = self.x.a + half * (self.y.a - self.x.a);
            }
            self.width1 = self.width;
            self.width = w;
        }

        // The interval the next step must stay in: the bracket, or the extrapolation interval
        // [stp + 1.1 (stp - stx), stp + 4 (stp - stx)] of (2.2).
        if self.brackt {
            self.stmin = min(self.x.a, self.y.a);
            self.stmax = max(self.x.a, self.y.a);
        } else {
            self.stmin = stp + T::from_f64(XTRAP_LOWER) * (stp - self.x.a);
            self.stmax = stp + T::from_f64(XTRAP_UPPER) * (stp - self.x.a);
        }

        // Force the step to be within stpmin and stpmax (NaN passes through).
        stp = clip(stp, stpmin, stpmax);

        // If further progress is not possible, let stp be the best point obtained so far.
        if self.brackt
            && (stp <= self.stmin
                || stp >= self.stmax
                || self.stmax - self.stmin <= T::from_f64(cfg.xtol()) * self.stmax)
        {
            stp = self.x.a;
        }
        stp
    }

    fn finish(&mut self, reason: Reason, t: Pt<T>) -> Step {
        let done = Termination {
            reason,
            alpha: t.a.to_f64(),
            phi: t.f.to_f64(),
            dphi: t.g.to_f64(),
            evaluations: self.evaluations,
            best: point(self.x),
        };
        self.done = Some(done);
        Step::Done(done)
    }

    fn tell(&mut self, cfg: &MoreThuenteConfig, f: T, g: T) -> Step {
        if let Some(done) = self.done {
            return Step::Done(done);
        }
        self.evaluations += 1;
        let t = Pt { a: self.stp, f, g };
        // Deviation from SciPy: a non-finite φ or φ' ends the search before it can enter the
        // interval; `x` stays the last point satisfying (1.1).
        if !(f.is_finite() && g.is_finite()) {
            return self.finish(Reason::NonFiniteValue, t);
        }
        let ftest = self.finit + t.a * self.gtest(cfg);
        // If psi(stp) <= 0 and f'(stp) >= 0 for some step, the algorithm enters the second stage
        // (§4, p. 298: f <= ftest and g >= 0).
        if self.stage == Stage::Auxiliary && f <= ftest && g >= T::from_f64(0.0) {
            self.stage = Stage::Function;
        }
        if let Some(reason) = self.check(cfg, t) {
            return self.finish(reason, t);
        }
        let stp = self.advance(cfg, t);
        if !stp.is_finite() {
            return self.finish(Reason::NonFiniteStep, t);
        }
        self.stp = stp;
        Step::Evaluate {
            alpha: stp.to_f64(),
        }
    }
}

/// The Moré–Thuente line search as an ask/tell state machine.
///
/// [`start`](MoreThuente::start) returns the state and the first trial step; evaluate φ and φ'
/// there and pass them to [`tell`](MoreThuente::tell), which returns the next trial or the
/// [`Termination`]. The state holds no configuration: pass the same
/// [`MoreThuenteConfig`] to every call.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Step};
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// let phi = |a: f64| (a - 3.0) * (a - 3.0);
/// let dphi = |a: f64| 2.0 * (a - 3.0);
/// let (mut ls, mut step) = MoreThuente::start(&cfg, phi(0.0), dphi(0.0), 1.0)?;
/// while let Step::Evaluate { alpha } = step {
///     step = ls.tell(&cfg, phi(alpha), dphi(alpha));
/// }
/// let Step::Done(t) = step else { unreachable!() };
/// assert!(t.reason.is_converged());
/// assert!(dphi(t.alpha).abs() <= 0.9 * dphi(0.0).abs());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct MoreThuente {
    s: State<f64>,
}

impl MoreThuente {
    /// Starts a search from `phi0 = φ(0)`, `dphi0 = φ'(0)` and the first trial step `alpha0`.
    ///
    /// # Errors
    ///
    /// - [`StartError::NonFiniteStart`] if `alpha0`, `phi0` or `dphi0` is NaN or infinite
    ///   (checked first),
    /// - [`StartError::NotDescent`] if `dphi0 >= 0`,
    /// - [`StartError::StepAboveStpmax`] if `alpha0 > cfg.stpmax()`,
    /// - [`StartError::StepBelowStpmin`] if `alpha0 < cfg.stpmin()`.
    ///
    /// When several of the last three hold, the first of this list is reported (MINPACK reports
    /// the last failing check, which is the same).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Step};
    ///
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// let (ls, step) = MoreThuente::start(&cfg, 1.0, -1.0, 2.0)?;
    /// assert_eq!(step, Step::Evaluate { alpha: 2.0 });
    /// assert_eq!(ls.best().alpha, 0.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(
        cfg: &MoreThuenteConfig,
        phi0: f64,
        dphi0: f64,
        alpha0: f64,
    ) -> Result<(Self, Step), StartError> {
        let (s, stp) = State::<f64>::start(cfg, phi0, dphi0, alpha0)?;
        Ok((Self { s }, Step::Evaluate { alpha: stp }))
    }

    /// Passes φ and φ' at the step last asked (the `alpha` of the last [`Step::Evaluate`]) and
    /// returns the next step. After a [`Step::Done`] it returns the same `Done` again; it never
    /// panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Step};
    ///
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// let (mut ls, _) = MoreThuente::start(&cfg, 1.0, -2.0, 1.0)?;
    /// let done = ls.tell(&cfg, 0.0, 0.0);
    /// assert!(matches!(done, Step::Done(_)));
    /// assert_eq!(ls.tell(&cfg, 7.0, 7.0), done); // idempotent
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn tell(&mut self, cfg: &MoreThuenteConfig, phi: f64, dphi: f64) -> Step {
        self.s.tell(cfg, phi, dphi)
    }

    /// The best point so far, (stx, fx, gx): `(0, φ(0), φ'(0))` right after `start`. It always
    /// satisfies the sufficient decrease condition (1.1).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Point};
    ///
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// let (ls, _) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0)?;
    /// assert_eq!(ls.best(), Point { alpha: 0.0, phi: 1.0, dphi: -1.0 });
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn best(&self) -> Point {
        point(self.s.x)
    }

    /// The bracket `(lo, hi)` (sorted) once a minimizer is bracketed, `None` before.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{MoreThuente, MoreThuenteConfig};
    ///
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// let (ls, _) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0)?;
    /// assert_eq!(ls.bracket(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn bracket(&self) -> Option<(f64, f64)> {
        self.s
            .brackt
            .then(|| (min(self.s.x.a, self.s.y.a), max(self.s.x.a, self.s.y.a)))
    }

    /// The current [`Stage`].
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Stage};
    ///
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// let (ls, _) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0)?;
    /// assert_eq!(ls.stage(), Stage::Auxiliary);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stage(&self) -> Stage {
        self.s.stage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_values_stop_the_search_with_a_valid_best() {
        let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0).unwrap();
        for (f, g) in [
            (f64::NAN, -1.0),
            (f64::INFINITY, 1.0),
            (0.5, f64::NAN),
            (0.5, f64::NEG_INFINITY),
        ] {
            let (mut ls, _) = MoreThuente::start(&cfg, 1.0, -1.0, 1.0).unwrap();
            let Step::Done(t) = ls.tell(&cfg, f, g) else {
                panic!("expected Done for ({f}, {g})")
            };
            assert_eq!(t.reason, Reason::NonFiniteValue);
            assert_eq!(t.evaluations, 1);
            assert_eq!(t.alpha, 1.0);
            assert_eq!(
                t.best,
                Point {
                    alpha: 0.0,
                    phi: 1.0,
                    dphi: -1.0
                }
            );
            assert_eq!(ls.stage(), Stage::Auxiliary);
            // sticky (compared field-wise: the NaN trial value is not equal to itself)
            let Step::Done(u) = ls.tell(&cfg, 0.0, 0.0) else {
                panic!("not sticky")
            };
            assert_eq!((u.reason, u.evaluations, u.best), (t.reason, 1, t.best));
        }
    }
}
