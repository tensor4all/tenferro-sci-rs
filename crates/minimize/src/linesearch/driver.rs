//! `scalar_search_wolfe1`: the SciPy-style driver.
//!
//! Stub: the driver lands after the state machine.

use super::config::ConfigError;
use super::more_thuente::{Point, Reason, StartError};

/// Options of [`scalar_search_wolfe1`] (SciPy's defaults).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wolfe1Options {
    /// Sufficient-decrease parameter.
    pub c1: f64,
    /// Curvature parameter.
    pub c2: f64,
    /// Largest step.
    pub amax: f64,
    /// Smallest step.
    pub amin: f64,
    /// Relative bracket-width tolerance.
    pub xtol: f64,
    /// Iteration limit, counting the start call.
    pub maxiter: usize,
}

impl Default for Wolfe1Options {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            amax: 50.0,
            amin: 1e-8,
            xtol: 1e-14,
            maxiter: 100,
        }
    }
}

/// Why the options were refused.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum Wolfe1Error {
    /// `0 < c1 < c2 < 1` does not hold.
    #[error("c1 and c2 must satisfy 0 < c1 < c2 < 1")]
    InvalidWolfeParameters,
    /// `amin`, `amax` or `xtol` is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// How the driver ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wolfe1Outcome {
    /// Strong Wolfe holds at `alpha`.
    Converged,
    /// The search stopped with a warning.
    Warning(Reason),
    /// `maxiter` ran out.
    MaxIter,
    /// The start was refused.
    Error(StartError),
}

/// Result of [`scalar_search_wolfe1`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wolfe1Report {
    /// The step; `None` exactly where SciPy returns `None`.
    pub alpha: Option<f64>,
    /// φ at the last evaluated point.
    pub phi: f64,
    /// φ(0).
    pub phi0: f64,
    /// The best point found.
    pub best: Point,
    /// How the search ended.
    pub outcome: Wolfe1Outcome,
    /// Calls of φ, including φ(0) if the driver evaluated it.
    pub phi_evals: usize,
    /// Calls of φ', including φ'(0) if the driver evaluated it.
    pub dphi_evals: usize,
}

/// Strong-Wolfe search on a scalar function (stub).
///
/// # Errors
///
/// See [`Wolfe1Error`].
pub fn scalar_search_wolfe1(
    mut phi: impl FnMut(f64) -> f64,
    mut dphi: impl FnMut(f64) -> f64,
    phi0: Option<f64>,
    _old_phi0: Option<f64>,
    dphi0: Option<f64>,
    _opts: &Wolfe1Options,
) -> Result<Wolfe1Report, Wolfe1Error> {
    let phi0 = phi0.unwrap_or_else(|| phi(0.0));
    let dphi0 = dphi0.unwrap_or_else(|| dphi(0.0));
    Ok(Wolfe1Report {
        alpha: None,
        phi: phi0,
        phi0,
        best: Point {
            alpha: 0.0,
            phi: phi0,
            dphi: dphi0,
        },
        outcome: Wolfe1Outcome::Warning(Reason::NonFiniteStep),
        phi_evals: 0,
        dphi_evals: 0,
    })
}
