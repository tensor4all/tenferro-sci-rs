//! The Moré–Thuente search as an ask/tell state machine.
//!
//! Stub: `start` and `tell` return a placeholder `Done` until the algorithm lands.
#![allow(dead_code)]

use super::config::MoreThuenteConfig;
use super::dcstep::Pt;
use crate::real::Real;

/// A point `(α, φ(α), φ'(α))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// The step α.
    pub alpha: f64,
    /// φ(α).
    pub phi: f64,
    /// φ'(α).
    pub dphi: f64,
}

/// Which function the search currently works on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Stage 1: the modified function ψ.
    Auxiliary,
    /// Stage 2: φ itself.
    Function,
}

/// Why a search ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Both strong-Wolfe conditions hold at the last trial.
    Converged,
    /// Rounding errors prevent progress.
    RoundingErrors,
    /// The bracket is narrower than `xtol`.
    XtolSatisfied,
    /// The step reached `stpmax`.
    StpMax,
    /// The step reached `stpmin`.
    StpMin,
    /// The trial-step formula produced a non-finite step.
    NonFiniteStep,
}

impl Reason {
    /// Whether this is [`Reason::Converged`].
    pub fn is_converged(&self) -> bool {
        matches!(self, Reason::Converged)
    }
}

/// Why [`MoreThuente::start`] refused its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StartError {
    /// `alpha0 < stpmin`.
    #[error("the initial step is below stpmin")]
    StepBelowStpmin,
    /// `alpha0 > stpmax`.
    #[error("the initial step is above stpmax")]
    StepAboveStpmax,
    /// `phi'(0) >= 0`.
    #[error("phi'(0) is not negative")]
    NotDescent,
    /// `alpha0`, `phi(0)` or `phi'(0)` is NaN or infinite.
    #[error("alpha0, phi(0) and phi'(0) must be finite")]
    NonFiniteStart,
}

/// The end of a search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Termination {
    /// Why the search ended.
    pub reason: Reason,
    /// The last trial step.
    pub alpha: f64,
    /// φ at the last trial.
    pub phi: f64,
    /// φ' at the last trial.
    pub dphi: f64,
    /// Number of `tell` calls so far.
    pub evaluations: usize,
    /// The best point (stx, fx, gx); it satisfies (1.1).
    pub best: Point,
}

/// What the search wants next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    /// Evaluate φ and φ' at `alpha` and call [`MoreThuente::tell`].
    Evaluate {
        /// The trial step.
        alpha: f64,
    },
    /// The search is over.
    Done(Termination),
}

#[derive(Clone, Debug)]
struct State<T: Real> {
    brackt: bool,
    stage: Stage,
    finit: T,
    ginit: T,
    x: Pt<T>,
    y: Pt<T>,
    stmin: T,
    stmax: T,
    width: T,
    width1: T,
    stp: T,
    evaluations: usize,
    done: Option<Termination>,
}

/// The search state.
#[derive(Clone, Debug)]
pub struct MoreThuente {
    s: State<f64>,
}

impl MoreThuente {
    /// Starts a search (stub).
    ///
    /// # Errors
    ///
    /// See [`StartError`].
    pub fn start(
        _cfg: &MoreThuenteConfig,
        phi0: f64,
        dphi0: f64,
        alpha0: f64,
    ) -> Result<(Self, Step), StartError> {
        let p = Pt {
            a: 0.0,
            f: phi0,
            g: dphi0,
        };
        let s = State {
            brackt: false,
            stage: Stage::Auxiliary,
            finit: phi0,
            ginit: dphi0,
            x: p,
            y: p,
            stmin: 0.0,
            stmax: alpha0,
            width: 0.0,
            width1: 0.0,
            stp: alpha0,
            evaluations: 0,
            done: None,
        };
        let step = Step::Done(Termination {
            reason: Reason::NonFiniteStep,
            alpha: alpha0,
            phi: phi0,
            dphi: dphi0,
            evaluations: 0,
            best: Point {
                alpha: 0.0,
                phi: phi0,
                dphi: dphi0,
            },
        });
        Ok((Self { s }, step))
    }

    /// Feeds φ and φ' at the last asked step (stub).
    pub fn tell(&mut self, _cfg: &MoreThuenteConfig, phi: f64, dphi: f64) -> Step {
        Step::Done(Termination {
            reason: Reason::NonFiniteStep,
            alpha: self.s.stp,
            phi,
            dphi,
            evaluations: 0,
            best: self.best(),
        })
    }

    /// The best point so far.
    pub fn best(&self) -> Point {
        Point {
            alpha: self.s.x.a,
            phi: self.s.x.f,
            dphi: self.s.x.g,
        }
    }

    /// The bracket `(lo, hi)` once a minimizer is bracketed.
    pub fn bracket(&self) -> Option<(f64, f64)> {
        None
    }

    /// The current stage.
    pub fn stage(&self) -> Stage {
        self.s.stage
    }
}
