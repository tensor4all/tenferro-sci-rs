//! Live parity with SciPy: Rust evaluates phi and phi' in closed form and the search is run
//! through `scalar_search_wolfe1` (and `MoreThuente` directly where SciPy's `DCSRCH` was driven
//! directly), then compared with the recorded SciPy 1.16.3 runs.
//!
//! Validates: `scalar_search_wolfe1` of `scipy/optimize/_linesearch.py` and `DCSRCH` of
//! `_dcsrch.py` (commit b9105ccc2237f57acb1060202cd77f6dd264fb34), on the problems of SciPy's
//! `test_linesearch.py` (`test_scalar_search_wolfe1`, `test_line_search_wolfe1`,
//! `test_wolfe_terminate`), the Moré–Thuente functions and the edge cases. Data:
//! `scripts/gen_minimize_linesearch_reference.py` (rows with run = `closed`).
//!
//! Tolerance policy (mathematical parity): strong-Wolfe inequalities are asserted exactly (the
//! mathematical claim). Steps are compared with a relative 1e-12: the recorded closed-form runs
//! use libm `exp`/`sin`/`cos`/`pow` through Python, this test through Rust's `std`, which may
//! differ by an ulp on another platform; 1e-12 is far above that and far below any change of
//! branch. Evaluation counts are expected to equal SciPy's. The line functions are recorded as
//! quadratics `A a^2 + B a + C` (the coefficients of phi along x + a p), so counts are compared
//! with the closed run; `closed_runs_agree_with_scipy_runs` shows the closed and numpy runs agree,
//! and lists any difference with its reason (divergences.csv).
//!
//! Non-finite phi (group `nonfinite`) is not compared here: the maintainer's decision is open
//! (issue #38, review 9.1); see `termination.rs`.

mod common;

use std::cell::Cell;

use common::*;
use minimize::linesearch::{scalar_search_wolfe1, Wolfe1Options, Wolfe1Outcome};

const STEP_TOL: f64 = 1e-12;

fn closed_cases() -> Vec<Case> {
    load_cases()
        .into_iter()
        .filter(|c| c.run == "closed" && !c.is_nonfinite_group())
        .collect()
}

fn check_converged(case: &Case, alpha: f64, phi_alpha: f64) {
    let stp = case.stp.expect("SciPy returned a step");
    assert!(
        rel_close(alpha, stp, STEP_TOL),
        "{}: alpha {alpha:e} vs SciPy {stp:e}",
        case.id
    );
    // exact strong Wolfe at the returned point, in SciPy's literal form and in DCSRCH's form
    let (name, p) = (case.func.as_str(), case.p.as_slice());
    let phi = |a: f64| phi_of(name, p, a);
    let dphi = |a: f64| dphi_of(name, p, a);
    assert_wolfe_scipy(alpha, &phi, &dphi, case.ftol, case.gtol, &case.id);
    assert_strong_wolfe(
        phi(0.0),
        dphi(0.0),
        case.ftol,
        case.gtol,
        alpha,
        phi(alpha),
        dphi(alpha),
        &case.id,
    );
    assert_fp_equal(phi_alpha, phi(alpha), &case.id);
}

/// Through `scalar_search_wolfe1`: the driver's report against the recorded SciPy run.
fn live_wolfe1(case: &Case) {
    let (name, p) = (case.func.as_str(), case.p.as_slice());
    let (nphi, ndphi) = (Cell::new(0usize), Cell::new(0usize));
    let mut opts = Wolfe1Options::default();
    opts.c1 = case.ftol;
    opts.c2 = case.gtol;
    opts.amin = case.stpmin;
    opts.amax = case.stpmax;
    opts.xtol = case.xtol;
    opts.maxiter = case.maxiter;
    let rep = scalar_search_wolfe1(
        |a| {
            nphi.set(nphi.get() + 1);
            phi_of(name, p, a)
        },
        |a| {
            ndphi.set(ndphi.get() + 1);
            dphi_of(name, p, a)
        },
        case.phi0_given.then(|| phi_of(name, p, 0.0)),
        case.old_phi0,
        case.dphi0_given.then(|| dphi_of(name, p, 0.0)),
        &opts,
    )
    .unwrap_or_else(|e| panic!("{}: driver refused the options: {e:?}", case.id));

    // outcome and `alpha is None` exactly where SciPy's step is None
    assert_eq!(
        outcome_name(&rep.outcome),
        case.reason,
        "{}: outcome ({})",
        case.id,
        case.task
    );
    assert_eq!(
        rep.alpha.is_none(),
        case.stp.is_none(),
        "{}: alpha None-ness",
        case.id
    );
    // P12 (Thm 2.1, invariant (2.1), p. 290): the report's `best` satisfies (1.1), also on warnings,
    // MaxIter and start errors (where it is the start point), although `alpha` is None there.
    assert!(
        sufficient_decrease(
            phi_of(name, p, 0.0),
            dphi_of(name, p, 0.0),
            case.ftol,
            rep.best.alpha,
            rep.best.phi
        ),
        "{}: Wolfe1Report.best {:?} violates (1.1)",
        case.id,
        rep.best
    );
    // phi(0) as in SciPy's third return value
    assert_fp_equal(rep.phi0, phi_of(name, p, 0.0), &case.id);
    if let Some(alpha) = rep.alpha {
        assert!(matches!(rep.outcome, Wolfe1Outcome::Converged));
        check_converged(case, alpha, rep.phi);
    }
    // evaluation counts, SciPy's way (they include the alpha = 0 calls the driver makes itself;
    // here phi0 and dphi0 are passed in except in `none_inputs`, where they are not)
    let given_phi0 = usize::from(!case.phi0_given);
    let given_dphi0 = usize::from(!case.dphi0_given);
    assert_eq!(
        rep.phi_evals, case.phi_evals,
        "{}: phi evaluations reported",
        case.id
    );
    assert_eq!(
        rep.dphi_evals, case.dphi_evals,
        "{}: dphi evaluations reported",
        case.id
    );
    // the report must also be what the closures were actually asked (the calls made here to
    // compute the optional phi0/dphi0 above happen before the driver and are not counted)
    assert_eq!(
        nphi.get(),
        case.n_trials + given_phi0,
        "{}: phi calls made",
        case.id
    );
    assert_eq!(
        ndphi.get(),
        case.n_trials + given_dphi0,
        "{}: dphi calls made",
        case.id
    );
}

/// `DCSRCH` driven directly (the paper runs, the maxiter and searched runs).
fn live_dcsrch(case: &Case) {
    let (name, p) = (case.func.as_str(), case.p.as_slice());
    let cfg = case.config();
    let run = drive(
        &cfg,
        |a| phi_of(name, p, a),
        |a| dphi_of(name, p, a),
        case.phi0,
        case.dphi0,
        case.alpha1,
        case.maxiter,
    );
    if case.reason.starts_with("err_") {
        let e = run.expect_err("expected a StartError");
        assert_eq!(start_error_name(&e), case.reason, "{}", case.id);
        return;
    }
    let run = run.unwrap_or_else(|e| panic!("{}: StartError {}", case.id, start_error_name(&e)));
    assert_eq!(
        run.trace.len(),
        case.n_trials,
        "{}: evaluations (SciPy {})",
        case.id,
        case.task
    );
    match run.ended {
        Ended::MaxIter => assert_eq!(case.reason, "maxiter", "{}", case.id),
        Ended::Done(t) => {
            assert_eq!(
                reason_name(&t.reason),
                case.reason,
                "{}: ({})",
                case.id,
                case.task
            );
            assert_eq!(
                case.stp.is_some(),
                case.reason == "converged",
                "{}",
                case.id
            );
            if case.reason == "converged" {
                check_converged(case, t.alpha, t.phi);
            }
        }
    }
}

#[test]
fn closed_form_runs_match_scipy() {
    let cases = closed_cases();
    assert!(
        cases.len() > 80,
        "the data set shrank: {} runs",
        cases.len()
    );
    for case in &cases {
        match case.kind.as_str() {
            "wolfe1" => live_wolfe1(case),
            "dcsrch" => live_dcsrch(case),
            other => panic!("{}: unknown kind {other}", case.id),
        }
    }
}

#[test]
fn scipy_scalar_and_line_problems() {
    // test_scalar_search_wolfe1 (3 x 3) + test_line_search_wolfe1 (2 x 9 as quadratics) +
    // test_wolfe_terminate, all through the driver with SciPy's c1 = 1e-4, c2 = 0.9.
    let cases: Vec<Case> = closed_cases()
        .into_iter()
        .filter(|c| c.group.starts_with("scipy_"))
        .collect();
    assert_eq!(cases.len(), 9 + 18 + 1);
    for case in &cases {
        live_wolfe1(case);
    }
}

#[test]
fn wolfe_terminate_counts() {
    // test_wolfe_terminate: phi = -s + 0.05 s^2 is accepted at the first trial (s = 1): the
    // driver evaluates phi and phi' once each (SciPy's own assertion is count <= 2 + 2 with the
    // test's phi(0) and derphi(0) calls included).
    let case = case_by_id("wt.closed");
    assert_eq!(case.n_trials, 1);
    live_wolfe1(&case);
}

#[test]
fn closed_runs_agree_with_scipy_runs() {
    // The `closed` run of each SciPy test problem (plain-float functions, quadratics for the line
    // problems) against the `scipy` run (numpy functions, 20-d dot products). Differences, if
    // any, are in divergences.csv with the reason; they are tolerated only when listed there.
    let cases = load_cases();
    let listed: Vec<String> = read_csv("divergences.csv")
        .iter()
        .map(|r| r["id"].clone())
        .collect();
    let mut checked = 0;
    for scipy in cases.iter().filter(|c| c.run == "scipy") {
        let base = scipy.id.trim_end_matches(".scipy");
        let closed = cases
            .iter()
            .find(|c| c.id == format!("{base}.closed"))
            .unwrap_or_else(|| panic!("{base}: no closed run"));
        checked += 1;
        if listed.iter().any(|l| l == base) {
            // documented divergence: the live test above compares with the closed run, which is
            // the same arithmetic as this test's; nothing more to assert here.
            continue;
        }
        assert_eq!(scipy.reason, closed.reason, "{base}: reason");
        assert_eq!(scipy.phi_evals, closed.phi_evals, "{base}: phi evaluations");
        assert_eq!(
            scipy.dphi_evals, closed.dphi_evals,
            "{base}: dphi evaluations"
        );
        assert_eq!(
            scipy.stp.is_none(),
            closed.stp.is_none(),
            "{base}: None-ness"
        );
        if let (Some(a), Some(b)) = (scipy.stp, closed.stp) {
            assert!(rel_close(a, b, STEP_TOL), "{base}: step {a:e} vs {b:e}");
        }
    }
    assert_eq!(checked, 9 + 18);
}
