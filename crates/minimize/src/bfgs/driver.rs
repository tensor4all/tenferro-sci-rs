//! The convenience driver: a thin loop over ask and tell.
//!
//! **Red stage:** it returns the report of an immediately ended run.

use super::config::BfgsConfig;
use super::state::{BfgsState, Step, Termination};
use crate::Vector;

/// The result of [`minimize`].
///
/// # Examples
///
/// ```
/// use minimize::bfgs::{minimize, BfgsConfig};
///
/// let cfg = BfgsConfig::scipy_default(1);
/// let rep = minimize(|_, x: &Vec<f64>, g: &mut Vec<f64>| { g[0] = x[0]; 0.5 * x[0] * x[0] }, vec![0.0], &cfg, &mut ());
/// assert_eq!((rep.iterations, rep.nfev), (0, 1));
/// ```
#[derive(Clone, Debug)]
pub struct Report<V> {
    /// The last accepted iterate.
    pub x: V,
    /// `f` at `x`.
    pub f: f64,
    /// The gradient at `x`.
    pub g: V,
    /// Why the run ended.
    pub termination: Termination,
    /// Accepted iterations (SciPy's `nit`).
    pub iterations: usize,
    /// Objective evaluations (SciPy's `nfev`).
    pub nfev: usize,
}

/// Minimizes with BFGS: evaluates `f(ctx, x, &mut grad) -> value` wherever the state asks.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::{minimize, BfgsConfig, Termination};
///
/// // f(x) = |x - c|^2 / 2
/// let c = [1.0, -2.0];
/// let f = |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
///     let mut v = 0.0;
///     for i in 0..2 {
///         g[i] = x[i] - c[i];
///         v += 0.5 * g[i] * g[i];
///     }
///     v
/// };
/// let rep = minimize(f, vec![0.0, 0.0], &BfgsConfig::scipy_default(2), &mut ());
/// assert_eq!(rep.termination, Termination::GradTol);
/// assert!((rep.x[0] - 1.0).abs() < 1e-5 && (rep.x[1] + 2.0).abs() < 1e-5);
/// ```
pub fn minimize<V, F>(mut f: F, x0: V, cfg: &BfgsConfig, ctx: &mut V::Ctx) -> Report<V>
where
    V: Vector + AsRef<[f64]> + AsMut<[f64]>,
    F: FnMut(&mut V::Ctx, &V, &mut V) -> f64,
{
    let mut state = BfgsState::new(cfg, ctx, x0.clone());
    let mut grad = x0;
    let termination = loop {
        match state.ask() {
            Step::Evaluate { x } => {
                let value = f(ctx, x, &mut grad);
                state.tell(cfg, ctx, value, &grad);
            }
            Step::Converged(t) => break t,
        }
    };
    Report {
        x: state.x().clone(),
        f: state.f(),
        g: state.g().clone(),
        termination,
        iterations: state.iterations(),
        nfev: state.nfev(),
    }
}
