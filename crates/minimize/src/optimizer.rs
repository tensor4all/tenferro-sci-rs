//! Static dispatch over the optimizers of this crate: [`Optimizer`] and the top-level [`minimize`].
//!
//! Decision 5 on tensor4all/tenferro-sci-rs#40: no runtime `Method` enum; the configuration type selects the
//! algorithm at compile time, through an associated report type. `minimize::minimize(f, x0, &cfg, ctx)` therefore
//! returns an [`lbfgs::Report`](crate::lbfgs::Report) for an [`LbfgsConfig`] and a
//! [`bfgs::Report`](crate::bfgs::Report) for a [`BfgsConfig`]; both are also callable as
//! [`lbfgs::minimize`](crate::lbfgs::minimize) and [`bfgs::minimize`](crate::bfgs::minimize).

use crate::bfgs::{self, BfgsConfig};
use crate::lbfgs::{self, LbfgsConfig};
use crate::Vector;

/// A configuration that knows how to run its algorithm: the closure driver of the algorithm it configures.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, Termination};
/// use minimize::Optimizer;
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
/// let rep = LbfgsConfig::scipy_default().minimize(f, vec![0.0, 0.0], &mut ());
/// assert!(matches!(rep.termination, Termination::GradTol | Termination::FTol));
/// ```
pub trait Optimizer<V: Vector> {
    /// What the driver returns.
    type Report;

    /// Runs the algorithm this configuration belongs to, evaluating `f(ctx, x, &mut grad) -> value`.
    fn minimize<F>(&self, f: F, x0: V, ctx: &mut V::Ctx) -> Self::Report
    where
        F: FnMut(&mut V::Ctx, &V, &mut V) -> f64;
}

impl<V: Vector> Optimizer<V> for LbfgsConfig {
    type Report = lbfgs::Report<V>;

    fn minimize<F>(&self, f: F, x0: V, ctx: &mut V::Ctx) -> Self::Report
    where
        F: FnMut(&mut V::Ctx, &V, &mut V) -> f64,
    {
        lbfgs::minimize(f, x0, self, ctx)
    }
}

impl<V: Vector + AsRef<[f64]> + AsMut<[f64]>> Optimizer<V> for BfgsConfig {
    type Report = bfgs::Report<V>;

    fn minimize<F>(&self, f: F, x0: V, ctx: &mut V::Ctx) -> Self::Report
    where
        F: FnMut(&mut V::Ctx, &V, &mut V) -> f64,
    {
        bfgs::minimize(f, x0, self, ctx)
    }
}

/// Minimizes with the algorithm `cfg` belongs to ([`LbfgsConfig`] or [`BfgsConfig`]); static dispatch, no enum.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, Termination};
///
/// let f = |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
///     g[0] = x[0] - 3.0;
///     0.5 * (x[0] - 3.0) * (x[0] - 3.0)
/// };
/// let rep = minimize::minimize(f, vec![0.0], &LbfgsConfig::scipy_default(), &mut ());
/// assert!(matches!(rep.termination, Termination::GradTol | Termination::FTol));
/// assert!((rep.x[0] - 3.0).abs() < 1e-4);
/// ```
pub fn minimize<V, F, C>(f: F, x0: V, cfg: &C, ctx: &mut V::Ctx) -> C::Report
where
    V: Vector,
    C: Optimizer<V>,
    F: FnMut(&mut V::Ctx, &V, &mut V) -> f64,
{
    cfg.minimize(f, x0, ctx)
}
