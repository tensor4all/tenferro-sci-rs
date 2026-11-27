//! The closure driver around [`LbfgsState`]: `minimize`.
//!
//! Written for tensor4all/tenferro-sci-rs#40 after the design sketch of tensor4all-meta `docs/design/tenferro-sci.md`
//! (§ Optimizers, "Interface: ask and tell").

use super::config::LbfgsConfig;
use super::state::{LbfgsState, Step, Termination};
use crate::vector::Vector;

/// The result of [`minimize`].
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, Termination};
///
/// let cfg = LbfgsConfig::scipy_default()?;
/// // f(x) = (x - 1)², started at the minimizer: zero iterations, one evaluation.
/// let rep = minimize::minimize(
///     |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
///         g[0] = 2.0 * (x[0] - 1.0);
///         (x[0] - 1.0) * (x[0] - 1.0)
///     },
///     vec![1.0],
///     &cfg,
///     &mut (),
/// );
/// assert_eq!(rep.termination, Termination::GradTol);
/// assert_eq!((rep.iterations, rep.evaluations), (0, 1));
/// # Ok::<(), minimize::lbfgs::ConfigError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Report<V> {
    /// The final iterate (the last finite one after [`Termination::NonFiniteValue`]).
    pub x: V,
    /// `f(x)`.
    pub f: f64,
    /// `∇f(x)`.
    pub grad: V,
    /// Why the iteration stopped.
    pub termination: Termination,
    /// Accepted iterations (SciPy's `nit`).
    pub iterations: usize,
    /// Evaluations of `f` and `∇f` (SciPy's `nfev`).
    pub evaluations: usize,
}

/// Minimizes `f` from `x0`: repeatedly asks the state for a point, evaluates `f` there (writing the gradient into
/// the buffer it is given and returning the value) and tells the state.
///
/// `x0` becomes the state's iterate; the gradient buffer is allocated once, by cloning it.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, Termination};
///
/// // Rosenbrock's function from the classical start; SciPy's L-BFGS-B stops after 36 iterations.
/// let rosen = |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
///     let (a, b) = (x[0], x[1]);
///     g[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
///     g[1] = 200.0 * (b - a * a);
///     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
/// };
/// let cfg = LbfgsConfig::scipy_default()?;
/// let rep = minimize::minimize(rosen, vec![-1.2, 1.0], &cfg, &mut ());
/// assert!(matches!(rep.termination, Termination::GradTol | Termination::FTol));
/// assert!((rep.x[0] - 1.0).abs() < 1e-3 && (rep.x[1] - 1.0).abs() < 1e-3);
/// # Ok::<(), minimize::lbfgs::ConfigError>(())
/// ```
pub fn minimize<V, F>(mut f: F, x0: V, cfg: &LbfgsConfig, ctx: &mut V::Ctx) -> Report<V>
where
    V: Vector,
    F: FnMut(&mut V::Ctx, &V, &mut V) -> f64,
{
    let mut g = x0.clone();
    let mut state = LbfgsState::new(cfg, ctx, x0);
    let termination = loop {
        let fx = match state.ask() {
            Step::Evaluate { x } => f(ctx, x, &mut g),
            Step::Converged(t) => break t,
        };
        state.tell(cfg, ctx, fx, &g);
    };
    Report {
        x: state.x().clone(),
        f: state.f(),
        grad: state.grad().clone(),
        termination,
        iterations: state.iterations(),
        evaluations: state.evaluations(),
    }
}
