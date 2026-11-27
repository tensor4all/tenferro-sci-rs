//! One test per `Termination` variant, and the counting quirks of SciPy's L-BFGS-B driver.
//!
//! Validates: `GradTol` (at the start: zero iterations and one evaluation; after k iterations), `FTol` (one-sided,
//! exactly on the boundary `f_old − f = ftol · max(|f_old|, |f|, 1)` and one ulp beyond it), `MaxIter` (tested
//! *before* the gradient test), `MaxEval` (strict `>`, tested only after an accepted iteration), `LineSearchFailed`
//! (`maxls` trial evaluations, never `maxls + 1`; the restart path is replayed from SciPy data in
//! `lbfgs_replay.rs`), `NonFiniteValue` (decision (b) of #38: stop at the last finite iterate; a NaN in one gradient
//! component where `dᵢ = 0` still stops, because `NaN · 0 = NaN` makes φ' non-finite; NaN at `x0`).
//!
//! The scenarios are hand-fed `(f, g)` sequences in one dimension, checked on a float64 mirror of the algorithm, and
//! (where SciPy has the same path) against SciPy 1.16.3: `maxls = 2` with a function that rejects every trial gives
//! `ABNORMAL`, `nfev = 3` in both.
//!
//! Source: SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34) `_lbfgsb_py.py` (the driver tests after
//! `NEW_X`) and `__lbfgsb.c` `mainlb` LINE666/LINE777.

mod lbfgs_common;

use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step, Termination, H0};

fn cfg(gtol: f64, ftol: f64, maxiter: usize, maxfun: usize, maxls: usize) -> LbfgsConfig {
    LbfgsConfig::new(10, gtol, ftol, maxiter, maxfun, maxls, H0::ScaledGamma).expect("valid")
}

/// The state of a run on `x0 = [0]`, started with `f0 = 4`, `g0 = [-1]`: the first search starts at `1/‖d‖ = 1`.
fn started(c: &LbfgsConfig) -> LbfgsState<Vec<f64>> {
    let mut s = LbfgsState::new(c, &mut (), vec![0.0]);
    assert!(matches!(s.ask(), Step::Evaluate { x } if *x == vec![0.0]));
    s.tell(c, &mut (), 4.0, &vec![-1.0]);
    match s.ask() {
        Step::Evaluate { x } => {
            assert_eq!(*x, vec![1.0], "first trial: x0 + (1/‖d‖) d with d = −g")
        }
        Step::Converged(t) => panic!("stopped with {t:?} after the first evaluation"),
    }
    s
}

fn converged(s: &LbfgsState<Vec<f64>>) -> Option<Termination> {
    match s.ask() {
        Step::Converged(t) => Some(t),
        Step::Evaluate { .. } => None,
    }
}

#[test]
fn grad_tol_at_the_start_takes_one_evaluation_and_no_iteration() {
    let c = cfg(1e-5, 0.0, 100, 100, 20);
    let rep = minimize::minimize(
        |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| {
            g.copy_from_slice(x);
            0.5 * x.iter().map(|t| t * t).sum::<f64>()
        },
        vec![0.0, 0.0],
        &c,
        &mut (),
    );
    assert_eq!(rep.termination, Termination::GradTol);
    assert_eq!((rep.iterations, rep.evaluations), (0, 1));
    assert_eq!(rep.x, vec![0.0, 0.0]);
}

#[test]
fn grad_tol_after_k_iterations_leaves_the_gradient_inside_the_tolerance() {
    let q = lbfgs_common::Quadratic::new(5, 1e2);
    let rep = minimize::minimize(
        |_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>| q.eval(x, 0.0, g),
        vec![0.0; 5],
        &cfg(1e-6, 0.0, 15000, 15000, 20),
        &mut (),
    );
    assert_eq!(rep.termination, Termination::GradTol);
    assert!(rep.iterations >= 2);
    assert!(rep.grad.iter().all(|t| t.abs() <= 1e-6));
}

#[test]
fn f_tol_is_one_sided_and_exact_on_its_boundary() {
    // After the first step f goes 4 → 3 with the Wolfe conditions satisfied: f_old − f = 1 = 0.25 · max(4, 3, 1).
    // (0.25 and 4 are powers of two: the test (f_old − f) <= tol · max(…) is evaluated without rounding.)
    let c = cfg(1e-12, 0.25, 100, 100, 20);
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0, &vec![-0.5]);
    assert_eq!(
        converged(&s),
        Some(Termination::FTol),
        "the boundary case stops (≤)"
    );
    assert_eq!((s.iterations(), s.evaluations()), (1, 2));
    assert_eq!(
        (s.x(), s.f()),
        (&vec![1.0], 3.0),
        "the accepted point is kept"
    );

    // One ulp (of 3) more decrease than the boundary: f = 3 − 2⁻⁵¹ gives f_old − f = 1 + 2⁻⁵¹ > 0.25 · 4. No stop.
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0 - 4.440892098500626e-16, &vec![-0.5]);
    assert_eq!(
        converged(&s),
        None,
        "a decrease just above ftol · max(…) continues"
    );

    // ftol = 0 switches the test off for every strictly decreasing step.
    let c0 = cfg(1e-12, 0.0, 100, 100, 20);
    let mut s = started(&c0);
    s.tell(&c0, &mut (), 3.0, &vec![-0.5]);
    assert_eq!(
        converged(&s),
        None,
        "ftol = 0: a decrease of 1 does not stop"
    );
}

#[test]
fn a_line_search_warning_accepts_the_last_trial_and_the_one_sided_f_tol_stops() {
    // The scenario of `scripted_checks` in the generator (SciPy 1.16.3 gives the same requests, result and
    // `CONVERGENCE: RELATIVE REDUCTION OF F`, nit 1, nfev 6): x0 = [0], f0 = 4, g0 = [−1], so d = [1] and the first
    // trial is x = 1. The values below make the bracket [1, 5] shrink until dcsrch ends with "WARNING: XTOL TEST
    // SATISFIED" at a trial with f = 10 > f0. SciPy's lnsrlb treats a warning like a convergence: the last trial is
    // the new iterate, and the one-sided test f_old − f = −6 ≤ ftol · max(4, 10, 1) stops the run with FTol. A
    // line search that discarded the warning would restore x0, and a two-sided test (|f_old − f| = 6 > 2.5) would
    // go on. The state machine sees only (f, g), so the values need not come from one function.
    let c = cfg(1e-12, 0.25, 100, 100, 40);
    let mut s = LbfgsState::new(&c, &mut (), vec![0.0]);
    s.tell(&c, &mut (), 4.0, &vec![-1.0]);
    // (f, g) told at the k-th trial and the trial SciPy evaluated there (1.0, 5.0, 1.3552…, 1.0030…, 1.0)
    let script = [
        (3.5, -0.95, 1.0),
        (10.0, 1.0, 5.0),
        (10.0, 1.0, 1.3552244689139858),
        (10.0, 1.0, 1.0030498702469033),
        (10.0, 1.0, 1.0),
    ];
    for (k, (f, g, want)) in script.into_iter().enumerate() {
        match s.ask() {
            Step::Evaluate { x } => {
                assert!(
                    (x[0] - want).abs() <= 1e-12 * want,
                    "request {k}: {} vs {want}",
                    x[0]
                );
            }
            Step::Converged(t) => panic!("stopped with {t:?} before request {k}"),
        }
        s.tell(&c, &mut (), f, &vec![g]);
        if k == 4 {
            // the last answer ends the search with a warning: FTol, with the last trial as the new iterate
            assert_eq!(converged(&s), Some(Termination::FTol));
        } else {
            assert!(
                converged(&s).is_none(),
                "request {k} was answered and the search goes on"
            );
        }
    }
    assert_eq!(
        (s.x(), s.f(), s.grad()),
        (&vec![1.0], 10.0, &vec![1.0]),
        "the last trial is the new iterate"
    );
    assert_eq!((s.iterations(), s.evaluations()), (1, 6));
}

#[test]
fn max_iter_is_tested_before_the_gradient_test() {
    // After the first accepted step ‖g‖∞ = 1e-13 ≤ gtol = 1e-12. With maxiter = 1 SciPy's driver stops on the
    // iteration limit first; with maxiter = 2 the same point is a GradTol.
    for (maxiter, want) in [(1, Termination::MaxIter), (2, Termination::GradTol)] {
        let c = cfg(1e-12, 0.0, maxiter, 100, 20);
        let mut s = started(&c);
        s.tell(&c, &mut (), 3.0, &vec![-1e-13]);
        assert_eq!(converged(&s), Some(want), "maxiter = {maxiter}");
        assert_eq!(s.iterations(), 1);
    }
}

#[test]
fn max_iter_is_tested_before_max_eval() {
    let c = cfg(1e-12, 0.0, 1, 1, 20);
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0, &vec![-0.5]);
    assert_eq!(
        converged(&s),
        Some(Termination::MaxIter),
        "evaluations 2 > maxfun 1 as well"
    );
}

#[test]
fn max_eval_is_a_strict_greater_than_after_an_accepted_iteration() {
    // after the first accepted step there are 2 evaluations: maxfun = 2 continues (2 > 2 is false), maxfun = 1 stops
    let c = cfg(1e-12, 0.0, 100, 2, 20);
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0, &vec![-0.5]);
    assert_eq!(converged(&s), None, "maxfun = 2, evaluations = 2");
    let c = cfg(1e-12, 0.0, 100, 1, 20);
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0, &vec![-0.5]);
    assert_eq!(converged(&s), Some(Termination::MaxEval));
    assert_eq!((s.iterations(), s.evaluations()), (1, 2));
}

#[test]
fn max_eval_is_not_checked_inside_a_line_search() {
    // maxfun = 1: the start evaluation already exceeds nothing (1 > 1 is false); the first trial of the search is
    // asked and evaluated although the budget is used up; only the accepted iteration ends the run.
    let c = cfg(1e-12, 0.0, 100, 1, 20);
    let mut s = LbfgsState::new(&c, &mut (), vec![0.0]);
    s.tell(&c, &mut (), 4.0, &vec![-1.0]);
    assert!(
        converged(&s).is_none(),
        "the search starts although evaluations == maxfun"
    );
    // a rejected first trial: the search goes on and asks again (evaluations 2 > maxfun) without stopping
    s.tell(&c, &mut (), 10.0, &vec![1.0]);
    assert!(
        converged(&s).is_none(),
        "a search in progress is not interrupted by maxfun"
    );
}

#[test]
fn line_search_failure_costs_maxls_evaluations_and_restores_the_start() {
    // f rejects every trial (10 > 4): with maxls = 2 the start plus two trials are evaluated. The third trial is
    // never asked for (SciPy: iback >= maxls aborts before the driver is called): ask() says LineSearchFailed.
    let c = cfg(1e-12, 0.0, 100, 100, 2);
    let mut s = started(&c);
    s.tell(&c, &mut (), 10.0, &vec![1.0]);
    assert!(
        converged(&s).is_none(),
        "one trial evaluated, maxls = 2: one more allowed"
    );
    s.tell(&c, &mut (), 10.0, &vec![1.0]);
    assert_eq!(converged(&s), Some(Termination::LineSearchFailed));
    assert_eq!(s.evaluations(), 3, "1 + maxls, as SciPy's nfev");
    assert_eq!(s.iterations(), 0);
    assert_eq!(
        (s.x(), s.f(), s.grad()),
        (&vec![0.0], 4.0, &vec![-1.0]),
        "restored to the start of the search"
    );
}

#[test]
fn maxls_one_allows_exactly_one_trial() {
    let c = cfg(1e-12, 0.0, 100, 100, 1);
    let mut s = started(&c);
    s.tell(&c, &mut (), 10.0, &vec![1.0]);
    assert_eq!(converged(&s), Some(Termination::LineSearchFailed));
    assert_eq!(s.evaluations(), 2);
}

#[test]
fn nan_at_the_start_stops_at_once() {
    for (f, g) in [
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 1.0),
        (1.0, f64::NEG_INFINITY),
    ] {
        let c = cfg(1e-5, 0.0, 100, 100, 20);
        let mut s = LbfgsState::new(&c, &mut (), vec![0.5]);
        s.tell(&c, &mut (), f, &vec![g]);
        assert_eq!(
            converged(&s),
            Some(Termination::NonFiniteValue),
            "f = {f}, g = {g}"
        );
        assert_eq!((s.iterations(), s.evaluations()), (0, 1));
        assert_eq!(s.x(), &vec![0.5]);
    }
}

#[test]
fn a_non_finite_trial_stops_at_the_last_finite_iterate() {
    for (f, g) in [
        (f64::NAN, -0.5),
        (f64::INFINITY, -0.5),
        (3.0, f64::NAN),
        (3.0, f64::INFINITY),
    ] {
        let c = cfg(1e-12, 0.0, 100, 100, 20);
        let mut s = started(&c);
        s.tell(&c, &mut (), f, &vec![g]);
        assert_eq!(
            converged(&s),
            Some(Termination::NonFiniteValue),
            "f = {f}, g = {g}"
        );
        assert_eq!(
            (s.x(), s.f(), s.grad()),
            (&vec![0.0], 4.0, &vec![-1.0]),
            "the iterate before the search"
        );
        assert_eq!((s.iterations(), s.evaluations()), (0, 2));
    }
}

#[test]
fn a_nan_gradient_component_with_zero_direction_still_stops() {
    // x0 = (1, 0), g0 = (2, 0): d = −g = (−2, 0). The trial gradient (0.1, NaN) has φ' = g·d = −0.2 + NaN·0 = NaN:
    // the component with dᵢ = 0 does not hide the NaN. (∞ · 0 = NaN too.)
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let c = cfg(1e-12, 0.0, 100, 100, 20);
        let mut s = LbfgsState::new(&c, &mut (), vec![1.0, 0.0]);
        s.tell(&c, &mut (), 1.0, &vec![2.0, 0.0]);
        assert!(converged(&s).is_none());
        s.tell(&c, &mut (), 0.5, &vec![0.1, bad]);
        assert_eq!(
            converged(&s),
            Some(Termination::NonFiniteValue),
            "g[1] = {bad}"
        );
        assert_eq!(s.x(), &vec![1.0, 0.0]);
        assert_eq!(s.evaluations(), 2);
    }
}

#[test]
fn a_non_finite_value_after_accepted_iterations_keeps_the_last_accepted_iterate() {
    let c = cfg(1e-12, 0.0, 100, 100, 20);
    let mut s = started(&c);
    s.tell(&c, &mut (), 3.0, &vec![-0.5]); // accepted: x = 1, f = 3
    assert!(converged(&s).is_none());
    assert_eq!(s.iterations(), 1);
    s.tell(&c, &mut (), f64::NAN, &vec![-0.1]);
    assert_eq!(converged(&s), Some(Termination::NonFiniteValue));
    assert_eq!((s.x(), s.f(), s.grad()), (&vec![1.0], 3.0, &vec![-0.5]));
    assert_eq!((s.iterations(), s.evaluations()), (1, 3));
}
