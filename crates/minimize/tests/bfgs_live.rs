//! Live runs of `minimize::bfgs` on closed-form problems: solutions and invariants, not paths.
//!
//! Validates: that BFGS with the default options (SciPy 1.16.3 `_minimize_bfgs`, commit
//! b9105ccc2237f57acb1060202cd77f6dd264fb34) converges on the problems of the reference data and
//! stops with `GradTol`. The objectives are evaluated here (`bfgs_common`), so counts and paths may
//! differ from SciPy in the last bits of Rosenbrock-type valleys; they are therefore **reported**
//! (`cargo test --test bfgs_live -- --nocapture` prints a table next to SciPy's), not asserted,
//! except for the quadratic probe of draft 12 §2.2 row 6 (5 iterations on a 5-dim quadratic), where
//! the path is the exact-ish one and SciPy's recorded counts are the claim.
//!
//! Solution bounds, all consequences of strong convexity with modulus `μ` on the segment `[x, x*]`
//! (`problems.csv`, `mu`; for non-quadratics the README of the data says how it was obtained):
//!
//! - `‖x − x*‖₂ ≤ ‖g‖₂ / μ` (exact inequality; `g` is `∇f` at `x`),
//! - `f(x) − f* ≤ ‖g‖₂² / (2μ)` (Polyak–Łojasiewicz, exact),
//!
//! each with a slack for the rounding of the *evaluated* gradient: `4 n ε` times the absolute
//! sum of the terms of `g` (for a quadratic `|A||x| + |b|`), which this file computes. No test
//! here only checks that the result is finite.
//!
//! Status in the red stage: every live test fails (the stub ends the run before the first
//! evaluation).

mod bfgs_common;

use bfgs_common::*;
use minimize::bfgs::{minimize, BfgsConfig, Termination};

/// `|A||x| + |b|` for a quadratic, `|g|` otherwise: the scale of the rounding of an evaluated gradient.
fn grad_scale(p: &ProblemData, x: &[f64], g: &[f64]) -> Vec<f64> {
    match (&p.a, &p.b) {
        (Some(a), Some(b)) => {
            let n = p.n;
            (0..n)
                .map(|i| (0..n).map(|j| a[i * n + j].abs() * x[j].abs()).sum::<f64>() + b[i].abs())
                .collect()
        }
        _ => g.iter().map(|t| t.abs()).collect(),
    }
}

fn live(id: &str) -> (Run, Trace) {
    let run = run_by_id(id);
    let obj = objective(&run.problem);
    let trace = drive(&run.config(), &run.x0, &*obj, 10 * run.maxiter + 100);
    println!(
        "{id:<14} Rust: nit {:>3} nfev {:>3} {:?}   SciPy: nit {:>3} nfev {:>3} status {} ({})",
        trace.state.iterations(),
        trace.state.nfev(),
        trace.termination,
        run.nit,
        run.nfev,
        run.status,
        run.message
    );
    (run, trace)
}

fn check_solution(id: &str) {
    let (run, trace) = live(id);
    let p = load_problems().remove(&run.problem).expect("problem");
    let mu = p.mu.expect("this problem has a strong-convexity bound");
    let x_star = p.x_star.as_ref().expect("minimizer");
    let n = p.n;
    assert!(!trace.evals.is_empty(), "{id}: nothing was evaluated");
    assert_eq!(trace.termination, Termination::GradTol, "{id}");
    let x = trace.state.x().clone();
    let g = trace.state.g().clone();
    // the reported gradient is the objective's gradient at the reported point
    let (f_check, g_check) = objective(&run.problem)(&x);
    assert_eq!(
        &g, &g_check,
        "{id}: state.g() is not the gradient at state.x()"
    );
    assert_eq!(
        trace.state.f(),
        f_check,
        "{id}: state.f() is not f at state.x()"
    );
    assert!(
        norm_inf(&g) <= run.gtol,
        "{id}: ‖g‖∞ = {:e} > gtol {:e}",
        norm_inf(&g),
        run.gtol
    );
    // solution bound
    let scale = grad_scale(&p, &x, &g);
    let slack = 4.0 * n as f64 * EPS * norm2(&scale);
    let err = x
        .iter()
        .zip(x_star)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let bound = (norm2(&g) + slack) / mu;
    assert!(
        err <= bound,
        "{id}: ‖x − x*‖ = {err:e} > (‖g‖ + slack)/μ = {bound:e}"
    );
    // value bound
    // f* is recorded for the non-quadratics; for a quadratic it is f(x*), whose rounding is of the order of
    // n ε Σ|terms| = n ε Σᵢ |xᵢ| (|A||x| + |b|)ᵢ
    let f_star = if p.f_star.is_nan() {
        objective(&run.problem)(x_star).0
    } else {
        p.f_star
    };
    let fscale = |pt: &[f64]| -> f64 {
        let (_, gp) = objective(&run.problem)(pt);
        pt.iter()
            .zip(grad_scale(&p, pt, &gp))
            .map(|(a, b)| a.abs() * b)
            .sum::<f64>()
    };
    let f_slack = 4.0 * n as f64 * EPS * (fscale(&x) + fscale(x_star) + f_check.abs());
    let gap = f_check - f_star;
    let fbound = (norm2(&g) + slack).powi(2) / (2.0 * mu) + f_slack;
    assert!(gap <= fbound, "{id}: f − f* = {gap:e} > {fbound:e}");
    // bookkeeping
    assert!(trace.state.iterations() <= run.maxiter, "{id}");
    assert_eq!(
        trace.evals.len(),
        trace.state.nfev(),
        "{id}: nfev counts the evaluations told"
    );
    assert!(
        trace.state.nfev() > trace.state.iterations(),
        "{id}: every iteration costs an evaluation"
    );
}

macro_rules! solution_tests {
    ($($name:ident),* $(,)?) => { $( #[test] fn $name() { check_solution(stringify!($name)); } )* };
}

solution_tests!(
    rosen2_a,
    rosen2_b,
    rosen2_c,
    rosen2_tight,
    rosen5,
    himmelblau,
    beale,
    quad5,
    quad10_k1e2,
    quad10_k1e4,
);

/// SciPy's own probe (draft 12 §2.2 row 6): BFGS with its Moré–Thuente search finishes a 5-dim
/// SPD quadratic in n = 5 iterations and 11 evaluations. The claim here is SciPy's recorded counts.
#[test]
fn quadratic_probe_finishes_in_n_iterations() {
    let (run, trace) = live("quad5");
    assert_eq!(trace.termination, Termination::GradTol);
    assert_eq!(trace.state.iterations(), 5, "n = 5 iterations");
    assert_eq!(trace.state.iterations(), run.nit);
    assert_eq!(trace.state.nfev(), run.nfev, "SciPy's 11 evaluations");
}

/// The driver is a thin loop over ask and tell: same iterations, evaluations and result.
#[test]
fn driver_matches_ask_tell() {
    let run = run_by_id("rosen2_a");
    let cfg = run.config();
    let obj = objective("rosen2");
    let trace = drive(&cfg, &run.x0, &*obj, 1000);
    let rep = minimize(
        |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
            let (f, gx) = obj(x);
            g.copy_from_slice(&gx);
            f
        },
        run.x0.clone(),
        &cfg,
        &mut (),
    );
    assert!(!trace.evals.is_empty());
    assert_eq!(rep.termination, trace.termination);
    assert_eq!(
        (rep.iterations, rep.nfev),
        (trace.state.iterations(), trace.state.nfev())
    );
    assert_eq!(&rep.x, trace.state.x());
    assert_eq!(rep.f, trace.state.f());
    assert_eq!(&rep.g, trace.state.g());
}

/// The default configuration for the problem dimension runs and converges: no option needs tuning.
#[test]
fn default_config_converges_on_himmelblau() {
    let cfg = BfgsConfig::scipy_default(2);
    let rep = minimize(
        |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
            let (f, gx) = bfgs_common::himmelblau(x);
            g.copy_from_slice(&gx);
            f
        },
        vec![1.0, 1.0],
        &cfg,
        &mut (),
    );
    assert_eq!(rep.termination, Termination::GradTol);
    // Himmelblau's minimizer nearest to (1, 1) is (3, 2); ‖g‖∞ ≤ 1e-5 and μ ≈ 23 give 1e-6
    assert!(
        (rep.x[0] - 3.0).abs() < 1e-5 && (rep.x[1] - 2.0).abs() < 1e-5,
        "{:?}",
        rep.x
    );
}
