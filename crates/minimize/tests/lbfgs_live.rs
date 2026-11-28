//! Live runs of `minimize::minimize` on closed-form problems: solutions, not paths.
//!
//! Validates: from the classical starting points the driver ends with `Termination::GradTol`, `‖g‖∞ ≤ gtol`, and an
//! iterate within a derived bound of the known minimizer. For `f` with minimizer `x*` and smallest Hessian eigenvalue
//! `λ` at `x*`, `g ≈ H (x − x*)` gives `‖x − x*‖₂ ≤ ‖g‖₂ / λ` and `f − f* ≈ ½ gᵀH⁻¹g ≤ ‖g‖²/(2λ)`; a factor 2 covers the
//! second-order terms at distances of 1e-8. Quadratics are exact (`λ = 1`). `λ`: Rosenbrock n = 2: 0.3993 (analytic:
//! the 2×2 Hessian [[802, −400], [−400, 200]]), n = 5: 0.4973, Beale at (3, 0.5): 0.3014 (numpy `eigvalsh` of the
//! Hessian, rounded down).
//!
//! Evaluation counts of live runs are *reported* against SciPy's recorded ones (stderr), not asserted: the paths of
//! Rosenbrock-type problems amplify last-digit differences (the replay tests pin the counts instead).
//!
//! Source: the problems are Rosenbrock's function (H. H. Rosenbrock, Comput. J. 3(3), 175–184, 1960), Beale's function as
//! in J. J. Moré, B. S. Garbow and K. E. Hillstrom, ACM Trans. Math. Software 7(1), 17–41, 1981, and the Householder
//! quadratics of the generator. SciPy parity: `LbfgsConfig::scipy_default()`.

mod lbfgs_common;

use lbfgs_common::*;
use minimize::lbfgs::{LbfgsConfig, Termination, H0};

fn tight(gtol: f64) -> LbfgsConfig {
    // ftol = 0 so that only the gradient test (or a failure) can stop the run
    LbfgsConfig::new(10, gtol, 0.0, 15000, 15000, 20, H0::ScaledGamma).expect("valid")
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    norm2(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>())
}

/// SciPy's `(nit, nfev)` of the recorded run `live_<problem>`: the same call as the live test (`maxcor = 10`,
/// `gtol = 1e-8`, `ftol = 0`, SciPy's other defaults), asserted here so that a missing run is a data-coverage failure.
fn scipy_counts(problem: &str) -> (usize, usize) {
    let id = format!("live_{problem}");
    let r = load_runs()
        .into_iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("data coverage: no recorded SciPy run {id}"));
    assert!(
        r.maxcor == 10
            && r.gtol == 1e-8
            && r.ftol == 0.0
            && r.maxiter == 15000
            && r.maxfun == 15000
            && r.maxls == 20,
        "{id}: the recorded options differ from the live test's"
    );
    (r.nit, r.nfev)
}

/// SciPy's `(nit, nfev)` of a recorded run with SciPy's default options (the same call as `LbfgsConfig::scipy_default()`).
fn recorded_default_counts(id: &str) -> (usize, usize) {
    let r = load_runs()
        .into_iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("data coverage: no recorded SciPy run {id}"));
    assert!(
        r.maxcor == 10
            && r.gtol == 1e-5
            && r.ftol == 2.220446049250313e-9
            && r.maxiter == 15000
            && r.maxfun == 15000
            && r.maxls == 20,
        "{id}: not a default-options run"
    );
    (r.nit, r.nfev)
}

#[test]
fn quadratics_reach_the_minimizer_within_the_gradient_bound() {
    for (n, cond, id) in [
        (5, 1.0, "quad5_k1"),
        (5, 1e2, "quad5_k1e2"),
        (5, 1e4, "quad5_k1e4"),
        (10, 1e2, "quad10_k1e2"),
    ] {
        let q = Quadratic::new(n, cond);
        let gtol = 1e-8;
        let rep = minimize::minimize(
            |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| q.eval(x, 0.0, g),
            vec![0.0; n],
            &tight(gtol),
            &mut (),
        );
        assert_eq!(rep.termination, Termination::GradTol, "{id}");
        let ginf = rep.grad.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
        assert!(ginf <= gtol, "{id}: ‖g‖∞ = {ginf:e}");
        // λ_min = 1: ‖x − x*‖ ≤ ‖g‖ (up to the rounding of g: 1e-10 relative)
        let bound = norm2(&rep.grad) * (1.0 + 1e-10);
        assert!(
            dist(&rep.x, &q.xs) <= bound,
            "{id}: |x - x*| = {:e}, bound {bound:e}",
            dist(&rep.x, &q.xs)
        );
        assert!(
            rep.f <= 0.5 * norm2(&rep.grad).powi(2) * (1.0 + 1e-10),
            "{id}: f = {:e}",
            rep.f
        );
        eprintln!(
            "{id}: rust nit {} nfev {}; scipy (same options) {:?}",
            rep.iterations,
            rep.evaluations,
            scipy_counts(id)
        );
    }
}

#[test]
fn rosenbrock_reaches_the_minimizer() {
    for (x0, lambda, id) in [
        (vec![-1.2, 1.0], 0.399, "rosen2"),
        (vec![3.0, -2.0], 0.399, "rosen2_far"),
        (vec![-1.2, 1.0, -1.2, 1.0, -1.2], 0.497, "rosen5"),
    ] {
        let gtol = 1e-8;
        let rep = minimize::minimize(rosen, x0.clone(), &tight(gtol), &mut ());
        assert_eq!(rep.termination, Termination::GradTol, "{id}");
        let ginf = rep.grad.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
        assert!(ginf <= gtol);
        let ones = vec![1.0; x0.len()];
        let bound = 2.0 * norm2(&rep.grad) / lambda;
        assert!(
            dist(&rep.x, &ones) <= bound,
            "{id}: |x - x*| = {:e} > {bound:e}",
            dist(&rep.x, &ones)
        );
        assert!(
            rep.f <= norm2(&rep.grad).powi(2) / lambda,
            "{id}: f = {:e}",
            rep.f
        );
        eprintln!(
            "{id}: rust nit {} nfev {}; scipy {:?}",
            rep.iterations,
            rep.evaluations,
            scipy_counts(id)
        );
    }
}

#[test]
fn beale_reaches_its_minimizer() {
    let gtol = 1e-8;
    let rep = minimize::minimize(beale, vec![1.0, 1.0], &tight(gtol), &mut ());
    assert_eq!(rep.termination, Termination::GradTol);
    let bound = 2.0 * norm2(&rep.grad) / 0.301;
    assert!(
        dist(&rep.x, &[3.0, 0.5]) <= bound,
        "|x - x*| = {:e} > {bound:e}",
        dist(&rep.x, &[3.0, 0.5])
    );
    assert!(rep.f <= norm2(&rep.grad).powi(2) / 0.301);
    eprintln!(
        "beale: rust nit {} nfev {}; scipy {:?}",
        rep.iterations,
        rep.evaluations,
        scipy_counts("beale")
    );
}

#[test]
fn the_scipy_default_configuration_stops_on_gradient_or_function_tolerance() {
    // SciPy's defaults stop Rosenbrock-5 with FTol at ‖g‖∞ ≈ 9e-5 > gtol (recorded: default_rosen5); the run ends
    // on a tolerance, not on a failure, and the point is near the minimizer.
    let cfg = LbfgsConfig::scipy_default();
    let rep = minimize::minimize(rosen, vec![-1.2, 1.0, -1.2, 1.0, -1.2], &cfg, &mut ());
    assert!(
        matches!(rep.termination, Termination::GradTol | Termination::FTol),
        "{:?}",
        rep.termination
    );
    assert!(
        dist(&rep.x, &[1.0; 5]) < 1e-3,
        "|x - x*| = {:e}",
        dist(&rep.x, &[1.0; 5])
    );
    eprintln!(
        "default rosen5: rust {:?} nit {} nfev {}; scipy {:?}",
        rep.termination,
        rep.iterations,
        rep.evaluations,
        recorded_default_counts("default_rosen5")
    );
}

#[test]
fn the_report_is_consistent_with_the_objective() {
    // f and the gradient of the report are the values of the objective at the reported point
    let q = Quadratic::new(5, 1e2);
    let rep = minimize::minimize(
        |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| q.eval(x, 0.0, g),
        vec![0.0; 5],
        &LbfgsConfig::scipy_default(),
        &mut (),
    );
    let mut g = vec![0.0; 5];
    let f = q.eval(&rep.x, 0.0, &mut g);
    assert_eq!(rep.f, f);
    assert_eq!(rep.grad, g);
    assert!(rep.iterations > 0 && rep.evaluations > rep.iterations);
}
