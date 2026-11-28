//! Static dispatch of the top-level `minimize::minimize` (decision 5 on #40): the configuration type selects the
//! algorithm and its report type; there is no runtime method enum.
//!
//! Status in the red stage: the L-BFGS half passes (A's behaviour is unchanged); the BFGS half fails at run time
//! because the BFGS iteration is a stub, and compiles because the dispatch is real.

use minimize::bfgs::{self, BfgsConfig};
use minimize::lbfgs::{self, LbfgsConfig};
use minimize::{minimize, Optimizer};

// the objective signature is `FnMut(&mut V::Ctx, &V, &mut V)` with `V = Vec<f64>`, so `&Vec` is required
#[allow(clippy::ptr_arg)]
fn quad(_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>) -> f64 {
    let c = [1.0, -2.0];
    let mut v = 0.0;
    for i in 0..2 {
        g[i] = x[i] - c[i];
        v += 0.5 * g[i] * g[i];
    }
    v
}

/// Both configurations implement the trait for `Vec<f64>`; the report types differ and are named by the config.
#[test]
fn both_configs_are_optimizers_with_their_own_report() {
    fn takes<C: Optimizer<Vec<f64>>>() {}
    takes::<LbfgsConfig>();
    takes::<BfgsConfig>();
    let _: fn(&LbfgsConfig) -> lbfgs::Report<Vec<f64>> =
        |c| minimize(quad, vec![0.0, 0.0], c, &mut ());
    let _: fn(&BfgsConfig) -> bfgs::Report<Vec<f64>> =
        |c| minimize(quad, vec![0.0, 0.0], c, &mut ());
}

/// The top-level function is the module driver: same result as `lbfgs::minimize` (A's behaviour is unchanged).
#[test]
fn top_level_minimize_with_lbfgs_equals_the_module_driver() {
    let cfg = LbfgsConfig::scipy_default();
    let a = minimize(quad, vec![0.0, 0.0], &cfg, &mut ());
    let b = lbfgs::minimize(quad, vec![0.0, 0.0], &cfg, &mut ());
    let c = cfg.minimize(quad, vec![0.0, 0.0], &mut ());
    assert_eq!(
        (a.iterations, a.evaluations, a.termination),
        (b.iterations, b.evaluations, b.termination)
    );
    assert_eq!(a.x, b.x);
    assert_eq!(a.x, c.x);
    assert!(a.iterations > 0, "the quadratic needs a step");
}

/// The same call with a `BfgsConfig` runs BFGS: same iterations and evaluations as `bfgs::minimize`, and converges.
#[test]
fn top_level_minimize_with_bfgs_equals_the_module_driver_and_converges() {
    let cfg = BfgsConfig::scipy_default(2);
    let a = minimize(quad, vec![0.0, 0.0], &cfg, &mut ());
    let b = bfgs::minimize(quad, vec![0.0, 0.0], &cfg, &mut ());
    assert_eq!(
        (a.iterations, a.evaluations, a.termination),
        (b.iterations, b.evaluations, b.termination)
    );
    assert_eq!(a.termination, bfgs::Termination::GradTol);
    assert!(
        (a.x[0] - 1.0).abs() < 1e-5 && (a.x[1] + 2.0).abs() < 1e-5,
        "{:?}",
        a.x
    );
}
