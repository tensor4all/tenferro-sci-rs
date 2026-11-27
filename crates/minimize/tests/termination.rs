//! One pinned case per termination outcome of `minimize::linesearch`: every `Reason`, the driver's
//! `MaxIter`, the driver's start-error outcomes, the priority of CONVERGENCE over the warnings, and
//! priority of CONVERGENCE over the warnings.
//!
//! Validates: the task outcomes of `DCSRCH` in `scipy/optimize/_dcsrch.py` and the driver's
//! `maxiter` counting and `isfinite(stp)` check in `_linesearch.py` (SciPy v1.16.3, commit
//! b9105ccc2237f57acb1060202cd77f6dd264fb34). The inputs are the generator's recorded cases
//! (`scripts/gen_minimize_linesearch_reference.py`); the evaluation counts and trial values asserted
//! here are SciPy's (the STPMIN / XTOL / ROUNDING ERRORS runs were found by the generator's small
//! deterministic search and are recorded, not hand-built).
//!
//! Non-finite phi or phi' ends the search with `Reason::NonFiniteValue` (option (b), decided on
//! #38); that contract, and SciPy's different recorded behaviour, are in `nonfinite.rs`.

mod common;

use common::*;
use minimize::linesearch::{
    scalar_search_wolfe1, Reason, Wolfe1Options, Wolfe1Outcome, Wolfe1Report,
};

/// A recorded `wolfe1` case through the driver, with the closures' call counts checked.
fn via_driver(case: &Case, maxiter: usize) -> Wolfe1Report {
    let (name, p) = (case.func.as_str(), case.p.as_slice());
    let o = Wolfe1Options {
        c1: case.ftol,
        c2: case.gtol,
        amin: case.stpmin,
        amax: case.stpmax,
        xtol: case.xtol,
        maxiter,
    };
    let rep = scalar_search_wolfe1(
        |a| phi_of(name, p, a),
        |a| dphi_of(name, p, a),
        case.phi0_given.then(|| phi_of(name, p, 0.0)),
        case.old_phi0,
        case.dphi0_given.then(|| dphi_of(name, p, 0.0)),
        &o,
    )
    .expect("valid options");
    if !case.is_nonfinite_group() {
        // P12 (Thm 2.1, invariant (2.1), p. 290): `best` satisfies (1.1) on every termination of
        // the driver, warnings and MaxIter included. The non-finite group is
        // checked in `nonfinite.rs`.
        let (phi0, dphi0) = eval(name, p, 0.0);
        assert!(
            sufficient_decrease(phi0, dphi0, case.ftol, rep.best.alpha, rep.best.phi),
            "{}: Wolfe1Report.best {:?} violates (1.1)",
            case.id,
            rep.best
        );
    }
    rep
}

/// A recorded case through `MoreThuente` directly with SciPy's maxiter counting.
fn via_state_machine(case: &Case) -> Run {
    let (name, p) = (case.func.as_str(), case.p.as_slice());
    drive(
        &case.config(),
        |a| phi_of(name, p, a),
        |a| dphi_of(name, p, a),
        case.phi0,
        case.dphi0,
        case.alpha1,
        case.maxiter,
    )
    .expect("starts")
}

fn done(run: Run) -> (minimize::linesearch::Termination, Vec<Trial>) {
    match run.ended {
        Ended::Done(t) => (t, run.trace),
        Ended::MaxIter => panic!("maxiter reached"),
    }
}

#[test]
fn converged() {
    // test_wolfe_terminate: phi = -s + 0.05 s^2, accepted at the first trial s = 1.
    let case = case_by_id("wt.closed");
    let (t, trace) = done(via_state_machine(&Case {
        kind: "dcsrch".into(),
        ..case.clone()
    }));
    assert!(matches!(t.reason, Reason::Converged));
    assert!(t.reason.is_converged());
    assert_eq!((t.evaluations, t.alpha), (1, 1.0));
    assert_eq!(trace, vec![(1.0, -0.95, -0.9)]);
    // phi(1) = -1 + 0.05, phi'(1) = -1 + 0.1; exact strong Wolfe was checked inside `drive`
    assert_eq!((t.phi, t.dphi), (-0.95, -0.9));
    let rep = via_driver(&case, 100);
    assert!(matches!(rep.outcome, Wolfe1Outcome::Converged));
    assert_eq!(
        (rep.alpha, rep.phi_evals, rep.dphi_evals),
        (Some(1.0), 1, 1)
    );
}

#[test]
fn stp_max() {
    // monotone phi = -a is unbounded below: the step grows 1, 5, 21, 50 (= amax) and stops.
    let case = case_by_id("monotone_stpmax");
    let (t, trace) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::StpMax), "{:?}", t.reason);
    assert!(!t.reason.is_converged());
    assert_eq!(t.evaluations, 4);
    assert_eq!(t.alpha, 50.0);
    let alphas: Vec<f64> = trace.iter().map(|t| t.0).collect();
    assert_eq!(alphas, vec![1.0, 5.0, 21.0, 50.0]);
    // SciPy returns None for every warning; the report says so
    let rep = via_driver(&case, 100);
    assert!(matches!(
        rep.outcome,
        Wolfe1Outcome::Warning(Reason::StpMax)
    ));
    assert!(rep.alpha.is_none());
    assert_eq!((rep.phi_evals, rep.dphi_evals), (4, 4));
}

#[test]
fn stp_max_when_alpha0_equals_amax() {
    // (a-5)^2 with amax = alpha0 = 1e-3: one trial, then STPMAX
    let case = case_by_id("amax_eq_alpha0");
    let (t, _) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::StpMax), "{:?}", t.reason);
    assert_eq!((t.evaluations, t.alpha), (1, 1e-3));
}

#[test]
fn stp_min() {
    // phi = -a + 1e6 a^2 rises steeply: the step shrinks onto stpmin = 1e-3 where no decrease holds.
    let case = case_by_id("search_stpmin");
    assert_eq!((case.stpmin, case.stpmax), (1e-3, 50.0));
    let (t, trace) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::StpMin), "{:?}", t.reason);
    assert_eq!(t.evaluations, 2);
    assert_eq!(t.alpha, 1e-3);
    assert_eq!(
        trace.iter().map(|t| t.0).collect::<Vec<_>>(),
        vec![1.0, 1e-3]
    );
}

#[test]
fn xtol_satisfied() {
    // MT2 (mu = 1e-4, eta = 1e-12) with xtol = 0.1: the bracket becomes narrower than xtol * stmax
    // long before strong Wolfe holds (SciPy: 11 trials).
    let case = case_by_id("search_xtol");
    assert_eq!(case.xtol, 0.1);
    let (t, _) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::XtolSatisfied), "{:?}", t.reason);
    assert_eq!(t.evaluations, 11);
    assert_eq!(case.n_trials, 11);
}

#[test]
fn rounding_errors() {
    // MT1 with gtol = 0 and xtol = 0: |phi'| <= 0 is unreachable, the iterates stall at the
    // limit of the arithmetic (SciPy: 10 trials, then ROUNDING ERRORS PREVENT PROGRESS).
    let case = case_by_id("search_rounding");
    assert_eq!((case.gtol, case.xtol), (0.0, 0.0));
    let (t, _) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::RoundingErrors), "{:?}", t.reason);
    assert_eq!(t.evaluations, 10);
    assert_eq!(case.n_trials, 10);
}

#[test]
fn non_finite_step_without_non_finite_phi() {
    // phi(a >= 0.5) = 1e308 is finite, but dcstep's cubic overflows (theta = 3 (fx - fp) / (stp - stx)
    // = -inf, theta / s = NaN), the next step is NaN, and SciPy's `isfinite(stp)` check ends the
    // search with its bare WARN. So the non-finite-step exit is *not* only about non-finite phi,
    // and it does not depend on the non-finite check of option (b).
    let case = case_by_id("overflow_finite");
    let (t, trace) = done(via_state_machine(&Case {
        kind: "dcsrch".into(),
        ..case.clone()
    }));
    assert!(matches!(t.reason, Reason::NonFiniteStep), "{:?}", t.reason);
    assert_eq!(t.evaluations, 1);
    assert!(
        trace.iter().all(|t| t.1.is_finite() && t.2.is_finite()),
        "phi stayed finite"
    );
    // Termination.alpha/phi/dphi are the last evaluated trial (finite), bitwise
    let &(a, f, g) = trace.last().unwrap();
    assert!(a.is_finite());
    assert_eq!(
        (t.alpha.to_bits(), t.phi.to_bits(), t.dphi.to_bits()),
        (a.to_bits(), f.to_bits(), g.to_bits())
    );
    let rep = via_driver(&case, 100);
    assert!(matches!(
        rep.outcome,
        Wolfe1Outcome::Warning(Reason::NonFiniteStep)
    ));
    assert!(rep.alpha.is_none());
    assert_eq!((rep.phi_evals, rep.dphi_evals), (1, 1));
}

#[test]
fn max_iter_off_by_one() {
    // SciPy's `maxiter` counts START: with maxiter = 2 the driver evaluates two trial steps, tests
    // only the first, and reports "did not converge". The point evaluated last is never tested,
    // even though it satisfies strong Wolfe: (a-20)^2 from alpha0 = 1: trial 1 = 1 fails curvature,
    // trial 2 = 5 (= 5 alpha0, the extrapolation bound) satisfies both conditions.
    let case = case_by_id("maxiter2");
    let case3 = case_by_id("maxiter3");
    assert_eq!(
        (case.maxiter, case.n_trials, case.reason.as_str()),
        (2, 2, "maxiter")
    );
    assert_eq!(
        (case3.maxiter, case3.n_trials, case3.reason.as_str()),
        (3, 2, "converged")
    );

    let rep = via_driver(&case, 2);
    assert!(
        matches!(rep.outcome, Wolfe1Outcome::MaxIter),
        "{:?}",
        rep.outcome
    );
    assert!(rep.alpha.is_none());
    assert_eq!((rep.phi_evals, rep.dphi_evals), (2, 2));

    // the same search with one more iteration accepts exactly that second trial
    let rep = via_driver(&case, 3);
    assert!(matches!(rep.outcome, Wolfe1Outcome::Converged));
    assert_eq!(
        (rep.alpha, rep.phi_evals, rep.dphi_evals),
        (Some(5.0), 2, 2)
    );

    // state machine: only maxiter - 1 = 1 tell; the second asked point is evaluated but not told
    let run = via_state_machine(&case);
    assert!(matches!(run.ended, Ended::MaxIter));
    assert_eq!(run.trace.len(), 2);
    let (a, f, g) = run.trace[1];
    assert_eq!(a, 5.0);
    assert!(
        sufficient_decrease(run.phi0, run.dphi0, case.ftol, a, f),
        "untested point is acceptable"
    );
    assert!(
        curvature(run.dphi0, case.gtol, g),
        "untested point is acceptable"
    );
}

#[test]
fn start_errors_are_outcomes_of_the_driver() {
    for (id, want) in [
        ("not_descent", "err_not_descent"), // dphi0 > 0, no trial evaluation
        ("above_stpmax", "err_above_stpmax"), // alpha1 = 1 > amax = 1e-3
    ] {
        let case = case_by_id(id);
        let rep = via_driver(&case, 100);
        assert_eq!(outcome_name(&rep.outcome), want, "{id}");
        assert!(matches!(rep.outcome, Wolfe1Outcome::Error(_)));
        assert!(rep.alpha.is_none());
        assert_eq!((rep.phi_evals, rep.dphi_evals), (0, 0), "{id}");
    }
    // alpha0 < stpmin through DCSRCH directly (the driver's alpha1 is 1 or smaller by its heuristic)
    let case = case_by_id("below_stpmin");
    let e = drive(
        &case.config(),
        |a| phi_of("shiftq", &[5.0], a),
        |a| dphi_of("shiftq", &[5.0], a),
        case.phi0,
        case.dphi0,
        case.alpha1,
        100,
    )
    .unwrap_err();
    assert_eq!(start_error_name(&e), "err_below_stpmin");
}

#[test]
fn convergence_has_priority_over_the_stpmax_warning() {
    // DCSRCH tests ROUNDING < XTOL < STPMAX < STPMIN < CONVERGENCE, each overwriting the last
    // (_dcsrch.py). At alpha = stpmax = 4 on (a-5)^2 both the STPMAX test (f <= ftest, g <= gtest)
    // and strong Wolfe hold: the result is Converged.
    let case = case_by_id("convergence_beats_stpmax");
    let (t, _) = done(via_state_machine(&case));
    assert!(matches!(t.reason, Reason::Converged), "{:?}", t.reason);
    assert_eq!((t.evaluations, t.alpha), (1, 4.0));
}
