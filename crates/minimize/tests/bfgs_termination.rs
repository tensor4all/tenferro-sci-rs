//! One test per termination variant of `minimize::bfgs`, and the boundary of the `wolfe2` fallback.
//!
//! Validates: the stopping logic of `_minimize_bfgs` (SciPy v1.16.3, commit
//! b9105ccc2237f57acb1060202cd77f6dd264fb34, `_optimize.py`): the loop condition
//! `gnorm > gtol and k < maxiter`, the break order after a step (`gnorm <= gtol`, then
//! `α‖p‖ <= xrtol (xrtol + ‖x‖)`, then `!isfinite(f)`) and the status mapping after the loop.
//!
//! | Variant | Test |
//! |---|---|
//! | `GradTol` (at the start; after k iterations) | `grad_tol_*` |
//! | `StepTol` | `step_tol_*` |
//! | `MaxIter` (and its precedence over `GradTol`) | `max_iter_*` |
//! | `LineSearchFailed` (first line search; a replayed SciPy fallback case) | `line_search_failed_*` |
//! | `NonFiniteValue` (NaN, +inf, NaN gradient only, at the start) | `non_finite_*` |
//!
//! **The wolfe2 boundary** (decision 3 on #40): where SciPy's `wolfe1` returns `None` it tries `wolfe2`;
//! this port stops with `LineSearchFailed` and takes no step. `line_search_failed_replays_scipy_fallback_case`
//! feeds the recorded evaluations of `w2_unbounded` (`f = −x₀`, unbounded below: `wolfe1` never meets the
//! curvature condition and ends at its limit of 100 trials on the first line search, in exact arithmetic, so no
//! rounding can change a decision) and asserts that the state asks for exactly the 101 evaluations SciPy made
//! before it called `wolfe2` (x₀ and 100 trials), then stops: no further request, no step, the starting
//! iterate reported. **SciPy's `wolfe2` returns a step there and the run goes on**: not a Wolfe step but its last
//! trial after ten doublings (it warns, and `_line_search_wolfe12` silences the warning); with `y = 0` after that
//! step SciPy even reaches its `ρ = 1000` branch. This is the recorded place where the port and SciPy part for
//! good. The other fallback runs of the data (`w2_neggrad`, `w2_l1`, `w2_noisy`, `w2_pert`) fail in the rounding
//! regime, where SciPy's `ScalarFunction` skips the evaluation at a point equal to the previous one, so its
//! recorded evaluations are fewer than its trials (`cache_hits` in the data) and cannot be matched to trials one
//! to one: they are data, not asserted. The fallback runs are constructed problems (data README).
//!
//! Open question for the maintainer, deliberately **not** pinned here: what an iteration does with
//! `yᵀs <= 0` (SciPy: `ρ = 1000` at exactly 0, a negative `ρ` otherwise; a skip or a stop are the
//! alternatives). With a strong Wolfe step `yᵀs > 0` (`bfgs_properties.rs`), so SciPy's branch is dead code in
//! exact arithmetic.
//!
//! Status in the red stage: every test here fails (the stub ends every run before the first evaluation).

mod bfgs_common;

use bfgs_common::*;
use minimize::bfgs::{BfgsConfig, BfgsState, Step, Termination};

fn cfg(gtol: f64, xrtol: f64, maxiter: usize) -> BfgsConfig {
    BfgsConfig::new(gtol, xrtol, 1e-4, 0.9, maxiter).expect("valid")
}

// ---- GradTol ----------------------------------------------------------------------------------

#[test]
fn grad_tol_at_the_start_takes_one_evaluation_and_no_step() {
    let x0 = [0.0, 0.0, 0.0];
    let obj = objective("halfsq3");
    let t = drive(&cfg(1e-5, 0.0, 600), &x0, &*obj, 10);
    assert_eq!(t.termination, Termination::GradTol);
    assert_eq!(t.evals.len(), 1, "one evaluation, at x0");
    assert_eq!((t.state.iterations(), t.state.evaluations()), (0, 1));
    assert_eq!(t.state.x(), &x0.to_vec());
    assert_eq!(t.state.f(), 0.0);
    // SciPy: the loop is never entered, H stays the identity
    assert_eq!(
        matrix_of(&t.state),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn grad_tol_after_iterations_is_met_for_the_first_time_at_the_end() {
    let run = run_by_id("quad5");
    let t = drive(&run.config(), &run.x0, &*objective("quad5"), 100);
    assert_eq!(t.termination, Termination::GradTol);
    assert!(t.snaps.len() >= 2, "iterations happened");
    let last = t.snaps.last().expect("snapshot");
    assert!(
        norm_inf(&last.g) <= run.gtol,
        "‖g‖∞ = {:e}",
        norm_inf(&last.g)
    );
    // every earlier iterate had ‖g‖∞ > gtol, else the loop would have ended there
    for s in &t.snaps[..t.snaps.len() - 1] {
        assert!(
            norm_inf(&s.g) > run.gtol,
            "iteration {} already met gtol",
            s.iterations
        );
    }
}

// ---- StepTol ----------------------------------------------------------------------------------

#[test]
fn step_tol_ends_a_run_whose_gradient_tolerance_cannot_be_met() {
    // gtol = 0: ‖g‖∞ <= 0 is not reached; xrtol ends the run (SciPy: status 0, `gnorm > gtol`)
    let run = run_by_id("rosen2_xrtol");
    let t = drive(&run.config(), &run.x0, &*objective("rosen2"), 1000);
    assert_eq!(t.termination, Termination::StepTol);
    assert!(t.snaps.len() >= 3, "iterations happened");
    let n = t.snaps.len();
    let (prev, last) = (&t.snaps[n - 2], &t.snaps[n - 1]);
    assert!(norm_inf(&last.g) > run.gtol, "StepTol, not GradTol");
    // the last step satisfied ‖s‖₂ <= xrtol (xrtol + ‖x‖₂), s = α p
    let s: Vec<f64> = last.x.iter().zip(&prev.x).map(|(a, b)| a - b).collect();
    let rhs = run.xrtol * (run.xrtol + norm2(&last.x));
    assert!(
        norm2(&s) <= rhs * (1.0 + 4.0 * EPS),
        "‖s‖ = {:e} > {:e}",
        norm2(&s),
        rhs
    );
    // and no earlier step did
    for w in t.snaps.windows(2).take(n - 2) {
        let s: Vec<f64> = w[1].x.iter().zip(&w[0].x).map(|(a, b)| a - b).collect();
        assert!(
            norm2(&s) > run.xrtol * (run.xrtol + norm2(&w[1].x)),
            "iteration {} already met the step test",
            w[1].iterations
        );
    }
}

// ---- MaxIter ----------------------------------------------------------------------------------

#[test]
fn max_iter_stops_after_exactly_maxiter_accepted_iterations() {
    for maxiter in [1usize, 5] {
        let x0 = [-1.2, 1.0];
        let t = drive(&cfg(1e-5, 0.0, maxiter), &x0, &*objective("rosen2"), 100);
        assert_eq!(t.termination, Termination::MaxIter, "maxiter = {maxiter}");
        assert_eq!(t.state.iterations(), maxiter);
        let (f0, _) = rosenbrock(&x0);
        assert!(t.state.f() < f0, "the accepted iterate decreased f");
        assert!(norm_inf(t.state.grad()) > 1e-5, "not converged");
        // the update of the last allowed iteration was applied (SciPy updates before testing `k < maxiter`)
        assert_ne!(
            matrix_of(&t.state),
            vec![1.0, 0.0, 0.0, 1.0],
            "maxiter = {maxiter}"
        );
    }
}

/// SciPy's order: `k >= maxiter` is tested after the loop, so a run that converges on its last
/// allowed iteration reports status 1. The data has the recorded case (`rosen2_maxiter_eq_nit`).
#[test]
fn max_iter_wins_over_grad_tol_on_the_last_allowed_iteration() {
    let x0 = [-1.2, 1.0];
    let obj = objective("rosen2");
    let free = drive(&cfg(1e-5, 0.0, 400), &x0, &*obj, 1000);
    assert_eq!(free.termination, Termination::GradTol);
    let nit = free.state.iterations();
    assert!(nit > 3);
    let capped = drive(&cfg(1e-5, 0.0, nit), &x0, &*obj, 1000);
    assert_eq!(capped.termination, Termination::MaxIter);
    assert_eq!(capped.state.iterations(), nit);
    assert!(
        norm_inf(capped.state.grad()) <= 1e-5,
        "the iterate satisfies gtol, the status is MaxIter anyway"
    );
    // one iteration more and the gradient test wins
    let roomy = drive(&cfg(1e-5, 0.0, nit + 1), &x0, &*obj, 1000);
    assert_eq!(roomy.termination, Termination::GradTol);
}

// ---- LineSearchFailed -------------------------------------------------------------------------

/// `wolfe2` boundary, recorded case: see the file docs.
#[test]
fn line_search_failed_replays_scipy_fallback_case() {
    let run = run_by_id("w2_unbounded");
    assert_eq!(run.group, "fallback");
    let n0 = run
        .wolfe2_first_eval
        .expect("the recorded run called wolfe2");
    assert_eq!(
        n0, 101,
        "SciPy made 101 evaluations (x0, 100 trials) before wolfe2 began"
    );
    assert!(
        run.wolfe1_fail
            .contains("did not converge within max iterations"),
        "wolfe1 failed with {:?}",
        run.wolfe1_fail
    );
    assert_eq!(run.nit, 1, "SciPy took a wolfe2 step; the port must not");
    // the step SciPy took is wolfe2's exhausted-maxiter return: found, but with a warning (not a Wolfe step)
    assert_eq!(
        (run.wolfe2_found.as_str(), run.wolfe2_warned.as_str()),
        ("1", "1")
    );
    // and its pair has yᵀs = 0 exactly, so SciPy applied its ρ = 1000 branch
    assert_eq!((run.n_rho1000, run.cache_hits), (1, 0));
    assert!(run.nfev > n0, "SciPy went on to evaluate wolfe2's trials");

    let r = replay(&run, None);
    assert_eq!(
        r.termination,
        Some(Termination::LineSearchFailed),
        "after {} evaluations",
        r.asked.len()
    );
    // exactly the evaluations before wolfe2: no further request, no step
    assert_eq!(
        r.asked.len(),
        n0,
        "the state must stop where SciPy's wolfe1 gives up"
    );
    assert_eq!(r.state.evaluations(), n0);
    assert_eq!(r.state.iterations(), 0, "no step was taken");
    assert_eq!(
        r.state.x(),
        &run.x0,
        "the reported iterate is the starting point"
    );
    assert_eq!(r.state.f(), run.evals[0].f);
    assert_eq!(r.state.grad(), &run.evals[0].g);
    assert!(matches!(
        r.state.ask(),
        Step::Converged(Termination::LineSearchFailed)
    ));
    // the trials were the extrapolation sequence of DCSRCH on a linear function: the 100th is evaluated
    // but never told (SciPy's `for i in range(maxiter)` counts START as an iteration)
    for (i, x) in r.asked.iter().enumerate() {
        // exact arithmetic on both sides; a relative 1e-12 allows a reordered expression without hiding a
        // different step
        for (j, (a, b)) in x.iter().zip(&run.evals[i].x).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                "request {i}[{j}]: {a:e} vs SciPy {b:e}"
            );
        }
    }
}

/// The same on the live objective, where the arithmetic is exact too (`f = −x₀`).
#[test]
fn line_search_failed_live_on_an_unbounded_objective() {
    let x0 = [0.0, 0.0];
    let t = drive(
        &cfg(1e-5, 0.0, 1),
        &x0,
        &*objective("linear_unbounded"),
        500,
    );
    assert_eq!(t.termination, Termination::LineSearchFailed);
    assert_eq!(t.evals.len(), 101, "x0 and the 100 trials of wolfe1");
    assert_eq!((t.state.iterations(), t.state.evaluations()), (0, 101));
    assert_eq!(t.state.x(), &x0.to_vec());
}

/// A search that ends through rounding noise: the number of trials is not pinned, the outcome is.
#[test]
fn line_search_failed_live_with_the_wrong_gradient_sign() {
    let x0 = [-1.2, 1.0];
    let t = drive(
        &cfg(1e-5, 0.0, 400),
        &x0,
        &*objective("rosen2_neggrad"),
        500,
    );
    assert_eq!(t.termination, Termination::LineSearchFailed);
    assert!(
        t.evals.len() > 10,
        "the search tried: {} evaluations",
        t.evals.len()
    );
    assert_eq!(t.state.iterations(), 0);
    assert_eq!(t.state.x(), &x0.to_vec());
}

// ---- NonFiniteValue ---------------------------------------------------------------------------

fn check_non_finite_stops_at_the_last_finite_iterate(problem: &str) {
    let x0 = [-1.2, 1.0];
    let t = drive(&cfg(1e-5, 0.0, 400), &x0, &*objective(problem), 1000);
    assert_eq!(t.termination, Termination::NonFiniteValue, "{problem}");
    assert!(t.evals.len() > 2, "{problem}: the run got going");
    // the last evaluation is the non-finite one; every earlier one was finite
    let (last_x, last_f, last_g) = t.evals.last().expect("evaluations");
    assert!(
        !last_f.is_finite() || last_g.iter().any(|v| !v.is_finite()),
        "{problem}: the last evaluation at {last_x:?} was finite"
    );
    assert!(
        last_x[0] > 0.5,
        "{problem}: the non-finite region is x0 > 0.5"
    );
    for (x, f, g) in &t.evals[..t.evals.len() - 1] {
        assert!(
            f.is_finite() && g.iter().all(|v| v.is_finite()),
            "{problem}: finite before the end, at {x:?}"
        );
    }
    // the reported point is the last accepted iterate, finite, and not the failed trial
    let snap = t.snaps.last().expect("snapshots");
    assert_eq!(t.state.x(), &snap.x);
    assert_eq!(t.state.f(), snap.f);
    assert!(t.state.f().is_finite() && t.state.grad().iter().all(|v| v.is_finite()));
    assert!(
        t.state.x()[0] <= 0.5,
        "{problem}: x is not in the non-finite region"
    );
    let (fx, gx) = objective(problem)(t.state.x());
    assert_eq!(
        (t.state.f(), t.state.grad()),
        (fx, &gx),
        "{problem}: (f, g) belong to x"
    );
    assert_ne!(t.state.x(), last_x);
}

#[test]
fn non_finite_nan_stops_at_the_last_finite_iterate() {
    check_non_finite_stops_at_the_last_finite_iterate("rosen2_nan");
}

#[test]
fn non_finite_inf_stops_at_the_last_finite_iterate() {
    check_non_finite_stops_at_the_last_finite_iterate("rosen2_inf");
}

/// `φ'(α) = g·p` is non-finite when any gradient entry is (NaN·pᵢ = NaN, ±∞·0 = NaN), so a NaN
/// gradient with a finite `f` ends the search too.
#[test]
fn non_finite_gradient_with_finite_value_stops() {
    let x0 = [-1.2, 1.0];
    let obj = |x: &[f64]| {
        let (f, mut g) = rosenbrock(x);
        if x[0] > 0.5 {
            g[1] = f64::NAN;
        }
        (f, g)
    };
    let t = drive(&cfg(1e-5, 0.0, 400), &x0, &obj, 1000);
    assert_eq!(t.termination, Termination::NonFiniteValue);
    assert!(t.state.grad().iter().all(|v| v.is_finite()));
    assert!(t.state.x()[0] <= 0.5);
}

#[test]
fn non_finite_start_is_refused_after_one_evaluation() {
    let obj = |x: &[f64]| (f64::NAN, vec![0.0; x.len()]);
    let t = drive(&cfg(1e-5, 0.0, 400), &[1.0, 2.0], &obj, 10);
    assert_eq!(t.termination, Termination::NonFiniteValue);
    assert_eq!(t.evals.len(), 1);
    assert_eq!((t.state.iterations(), t.state.evaluations()), (0, 1));
    assert_eq!(t.state.x(), &vec![1.0, 2.0]);
}

// ---- ask before any evaluation ----------------------------------------------------------------

#[test]
fn ask_before_the_first_tell_asks_for_x0() {
    let state = BfgsState::new(&cfg(1e-5, 0.0, 10), &mut (), vec![0.25, -4.0]);
    match state.ask() {
        Step::Evaluate { x } => assert_eq!(x, &vec![0.25, -4.0]),
        Step::Converged(t) => panic!("nothing evaluated yet, but Converged({t:?})"),
    }
    assert_eq!(state.termination(), None);
}
