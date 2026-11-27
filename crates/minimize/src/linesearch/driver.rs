//! `scalar_search_wolfe1`: the SciPy-style driver around [`MoreThuente`].
//!
//! Ported from SciPy v1.16.3 (b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! scipy/optimize/_linesearch.py (`scalar_search_wolfe1`) and the `DCSRCH.__call__` loop of
//! scipy/optimize/_dcsrch.py, a 2023 translation of MINPACK-2 dcsrch (Argonne National
//! Laboratory / University of Minnesota; see LICENSE-NOTICE-minpack2). Deviations:
//!
//! - Invalid options are an error of the driver ([`Wolfe1Error`]), raised before anything is
//!   evaluated; SciPy raises `ValueError` for `c1`/`c2` only and reports the other violations
//!   as a DCSRCH task after evaluating φ(0). Start errors stay outcomes of the report, as in
//!   SciPy, which returns `None` with an `ERROR` task for them.
//! - `maxiter` is an option here (SciPy fixes it at 100) and keeps SciPy's counting: START counts
//!   as an iteration, so at most `maxiter` trials are evaluated and the last one is never tested.
//! - The report always carries `best` and the evaluation counts; `alpha` is `None` exactly where
//!   SciPy returns `None`.
//! - A NaN `old_phi0` gives the initial step 1, as Python's `min(1.0, nan)` does.

use super::config::{ConfigError, MoreThuenteConfig};
use super::more_thuente::{MoreThuente, Point, Reason, StartError, Step};

/// Options of [`scalar_search_wolfe1`]; the [`Default`] is SciPy's.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::Wolfe1Options;
///
/// let o = Wolfe1Options::default();
/// assert_eq!((o.c1, o.c2, o.amin, o.amax, o.xtol, o.maxiter), (1e-4, 0.9, 1e-8, 50.0, 1e-14, 100));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wolfe1Options {
    /// Parameter μ of the sufficient decrease condition (1.1); `0 < c1 < c2 < 1`.
    pub c1: f64,
    /// Parameter η of the curvature condition (1.2); `0 < c1 < c2 < 1`.
    pub c2: f64,
    /// Largest step (`stpmax`).
    pub amax: f64,
    /// Smallest step (`stpmin`).
    pub amin: f64,
    /// Relative bracket-width tolerance (`xtol`).
    pub xtol: f64,
    /// Iteration limit, counting the start call as SciPy does.
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

/// Why [`scalar_search_wolfe1`] refused its options.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Error, Wolfe1Options};
///
/// let o = Wolfe1Options { c1: 0.9, c2: 0.9, ..Wolfe1Options::default() };
/// let e = scalar_search_wolfe1(|a| a, |_| 1.0, Some(0.0), None, Some(-1.0), &o).unwrap_err();
/// assert_eq!(e, Wolfe1Error::InvalidWolfeParameters);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum Wolfe1Error {
    /// `0 < c1 < c2 < 1` does not hold (SciPy's `_check_c1_c2`; this includes NaN).
    #[error("c1 and c2 must satisfy 0 < c1 < c2 < 1")]
    InvalidWolfeParameters,
    /// `amin`, `amax` or `xtol` (with `c1` as `ftol` and `c2` as `gtol`) is not a valid
    /// [`MoreThuenteConfig`].
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// How [`scalar_search_wolfe1`] ended.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Options, Wolfe1Outcome};
///
/// let rep = scalar_search_wolfe1(|a| (a - 2.0) * (a - 2.0), |a| 2.0 * (a - 2.0),
///     None, None, None, &Wolfe1Options::default())?;
/// assert_eq!(rep.outcome, Wolfe1Outcome::Converged);
/// # Ok::<(), minimize::linesearch::Wolfe1Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Wolfe1Outcome {
    /// Strong Wolfe holds at `alpha`.
    Converged,
    /// The search stopped with a warning; `alpha` is `None`, `best` still satisfies (1.1).
    Warning(Reason),
    /// `maxiter` ran out before a step was accepted.
    MaxIter,
    /// The start was refused; no trial was evaluated.
    Error(StartError),
}

/// Result of [`scalar_search_wolfe1`].
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Options};
///
/// let rep = scalar_search_wolfe1(|a| (a - 2.0) * (a - 2.0), |a| 2.0 * (a - 2.0),
///     None, None, None, &Wolfe1Options::default())?;
/// assert_eq!(rep.phi0, 4.0);
/// // phi(0) and phi'(0) are evaluated by the driver and counted, one call each per trial
/// assert_eq!((rep.phi_evals, rep.dphi_evals), (2, 2));
/// # Ok::<(), minimize::linesearch::Wolfe1Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wolfe1Report {
    /// The accepted step; `None` exactly where SciPy returns `None` (warnings, `maxiter`,
    /// start errors).
    pub alpha: Option<f64>,
    /// φ at the last evaluated point (φ(0) if none was evaluated), SciPy's second return value.
    pub phi: f64,
    /// φ(0), SciPy's third return value.
    pub phi0: f64,
    /// The best point found: it satisfies (1.1) on every outcome, also when `alpha` is `None`.
    pub best: Point,
    /// How the search ended.
    pub outcome: Wolfe1Outcome,
    /// Calls of φ, including φ(0) if the driver had to evaluate it (SciPy's counting).
    pub phi_evals: usize,
    /// Calls of φ', including φ'(0) if the driver had to evaluate it.
    pub dphi_evals: usize,
}

/// SciPy's initial step: `min(1, 1.01 * 2 (φ0 - φ0_old) / φ'(0))`, or 1 if that is negative, if
/// `old_phi0` is unknown or `φ'(0) = 0`.
fn initial_step(phi0: f64, old_phi0: Option<f64>, dphi0: f64) -> f64 {
    match old_phi0 {
        Some(old) if dphi0 != 0.0 => {
            let a = 1.01 * 2.0 * (phi0 - old) / dphi0;
            let a = if a < 1.0 { a } else { 1.0 }; // NaN gives 1, as Python's min(1.0, nan)
            if a < 0.0 {
                1.0
            } else {
                a
            }
        }
        _ => 1.0,
    }
}

/// Strong-Wolfe search for a step on the scalar functions `phi` and `dphi` (= φ'), with the
/// semantics of SciPy's `scipy.optimize.scalar_search_wolfe1`.
///
/// φ(0) and φ'(0) are evaluated if not given. If `old_phi0` is given, the first trial step is
/// `min(1, 1.01 * 2 (φ(0) - old_phi0) / φ'(0))` (1 if negative), else 1. The step is a descent
/// step: φ'(0) < 0 is required, a violation is [`Wolfe1Outcome::Error`]. The search stops at
/// the first trial satisfying (1.1) and (1.2) with `c1` and `c2`, see [`MoreThuente`].
///
/// SciPy's counting of `maxiter` is kept: the start call is one of the iterations, so the last
/// evaluated point is never tested (a run with `maxiter = 2` evaluates two trials and tests the
/// first).
///
/// # Errors
///
/// - [`Wolfe1Error::InvalidWolfeParameters`] if `0 < c1 < c2 < 1` does not hold,
/// - [`Wolfe1Error::Config`] if `amin`, `amax` or `xtol` is negative, non-finite (`amax = +inf`
///   excepted) or `amax < amin`.
///
/// Problems of the start values and of the search are not errors but outcomes of the report.
///
/// # Examples
///
/// ```
/// use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Options, Wolfe1Outcome};
///
/// let phi = |a: f64| (a - 5.0) * (a - 5.0);
/// let dphi = |a: f64| 2.0 * (a - 5.0);
/// let rep = scalar_search_wolfe1(phi, dphi, Some(25.0), None, Some(-10.0), &Wolfe1Options::default())?;
/// assert_eq!(rep.outcome, Wolfe1Outcome::Converged);
/// let a = rep.alpha.expect("converged");
/// assert!(phi(a) <= 25.0 + 1e-4 * a * -10.0 && dphi(a).abs() <= 0.9 * 10.0);
/// # Ok::<(), minimize::linesearch::Wolfe1Error>(())
/// ```
pub fn scalar_search_wolfe1(
    mut phi: impl FnMut(f64) -> f64,
    mut dphi: impl FnMut(f64) -> f64,
    phi0: Option<f64>,
    old_phi0: Option<f64>,
    dphi0: Option<f64>,
    opts: &Wolfe1Options,
) -> Result<Wolfe1Report, Wolfe1Error> {
    // _check_c1_c2; written so that NaN fails
    if !(0.0 < opts.c1 && opts.c1 < opts.c2 && opts.c2 < 1.0) {
        return Err(Wolfe1Error::InvalidWolfeParameters);
    }
    let cfg = MoreThuenteConfig::new(opts.c1, opts.c2, opts.xtol, opts.amin, opts.amax)?;

    let (mut phi_evals, mut dphi_evals) = (0, 0);
    let phi0 = phi0.unwrap_or_else(|| {
        phi_evals += 1;
        phi(0.0)
    });
    let dphi0 = dphi0.unwrap_or_else(|| {
        dphi_evals += 1;
        dphi(0.0)
    });
    let start_point = Point {
        alpha: 0.0,
        phi: phi0,
        dphi: dphi0,
    };
    let report = |alpha, phi, best, outcome, phi_evals, dphi_evals| Wolfe1Report {
        alpha,
        phi,
        phi0,
        best,
        outcome,
        phi_evals,
        dphi_evals,
    };
    // SciPy's `for i in range(maxiter)` never enters with maxiter = 0: nothing is started.
    if opts.maxiter == 0 {
        return Ok(report(
            None,
            phi0,
            start_point,
            Wolfe1Outcome::MaxIter,
            phi_evals,
            dphi_evals,
        ));
    }

    let alpha1 = initial_step(phi0, old_phi0, dphi0);
    let (mut ls, mut step) = match MoreThuente::start(&cfg, phi0, dphi0, alpha1) {
        Ok(started) => started,
        Err(e) => {
            return Ok(report(
                None,
                phi0,
                start_point,
                Wolfe1Outcome::Error(e),
                phi_evals,
                dphi_evals,
            ));
        }
    };

    let mut trials = 0;
    let mut last_phi = phi0;
    loop {
        match step {
            Step::Evaluate { alpha } => {
                let (f, g) = (phi(alpha), dphi(alpha));
                trials += 1;
                last_phi = f;
                // Iteration 0 was START: the `maxiter`-th trial is evaluated but never told.
                if trials >= opts.maxiter {
                    return Ok(report(
                        None,
                        last_phi,
                        ls.best(),
                        Wolfe1Outcome::MaxIter,
                        phi_evals + trials,
                        dphi_evals + trials,
                    ));
                }
                step = ls.tell(&cfg, f, g);
            }
            Step::Done(t) => {
                let (alpha, outcome) = if t.reason.is_converged() {
                    (Some(t.alpha), Wolfe1Outcome::Converged)
                } else {
                    (None, Wolfe1Outcome::Warning(t.reason))
                };
                return Ok(report(
                    alpha,
                    last_phi,
                    t.best,
                    outcome,
                    phi_evals + trials,
                    dphi_evals + trials,
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_step_follows_scipy() {
        // no old value, or dphi0 = 0: 1
        assert_eq!(initial_step(1.0, None, -1.0), 1.0);
        assert_eq!(initial_step(1.0, Some(2.0), 0.0), 1.0);
        // min(1, 1.01 * 2 * (1 - 1.1) / -1) = 0.202
        let a = initial_step(1.0, Some(1.1), -1.0);
        assert!((a - 1.01 * 2.0 * 0.1).abs() < 1e-15, "{a}");
        // capped at 1
        assert_eq!(initial_step(10.0, Some(1.0), -1.0), 1.0);
        // negative: 1; NaN: 1
        assert_eq!(initial_step(1.0, Some(0.0), -1.0), 1.0);
        assert_eq!(initial_step(1.0, Some(f64::NAN), -1.0), 1.0);
    }
}
