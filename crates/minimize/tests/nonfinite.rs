//! The contract of option (b), decided on issue #38: a non-finite φ or φ' at a trial ends the
//! search with `Reason::NonFiniteValue`, and the termination carries the last *finite* best point.
//!
//! The finite-path contract: on finite inputs the trajectory is unchanged by this option, so the
//! parity with SciPy (steps within 1e-12 relative, equal evaluation counts) stays the contract.
//! The evidence is `scipy_replay.rs` and `scipy_live.rs`, which pass unchanged with and without
//! the option-(b) commit (they never see a non-finite value: the group `nonfinite` is excluded
//! from both). Here the departure itself is pinned.
//!
//! What is asserted for every run that meets a non-finite value:
//!
//! * the reason is `NonFiniteValue` and `evaluations` is the index of the first non-finite trial
//!   plus one (the search stops at the first one, before it can enter the interval);
//! * `Termination::best` equals, bitwise, `best()` read just before that tell, satisfies (1.1)
//!   and is the start point `(0, φ(0), φ'(0))` or a traced finite trial;
//! * `alpha`, `phi`, `dphi` are the offending trial (bitwise);
//! * through the driver: `Warning(NonFiniteValue)`, no step, and as many calls as trials.
//!
//! SciPy 1.16.3 (the recorded data) behaves differently, and each test says how: it never
//! looks at φ, so +inf gives a bare WARN from a NaN step, NaN burns all `maxiter` iterations.

mod common;

use common::*;
use minimize::linesearch::{
    scalar_search_wolfe1, MoreThuenteConfig, Reason, Wolfe1Options, Wolfe1Outcome,
};

/// A run on a closure pair, with SciPy's recorded outcome for the same problem if there is one.
struct Problem {
    id: &'static str,
    phi: Box<dyn Fn(f64) -> f64>,
    dphi: Box<dyn Fn(f64) -> f64>,
    /// SciPy's recorded reason and number of evaluated trials (cases.csv).
    scipy: (&'static str, usize),
    /// Index of the first non-finite trial in our run.
    first: usize,
}

fn named(
    id: &'static str,
    name: &'static str,
    scipy: (&'static str, usize),
    first: usize,
) -> Problem {
    Problem {
        id,
        phi: Box::new(move |a| phi_of(name, &[], a)),
        dphi: Box::new(move |a| dphi_of(name, &[], a)),
        scipy,
        first,
    }
}

fn problems() -> Vec<Problem> {
    vec![
        // +inf from the first trial on: SciPy's dcstep gives inf/inf = NaN, a bare WARN
        named("inf_half", "inf_half", ("nonfinite_step", 1), 0),
        // NaN from the first trial on: every comparison is false, SciPy burns all 100 iterations
        named("nan_half", "nan_half", ("maxiter", 100), 0),
        // finite phi, NaN phi' at every trial: SciPy stops after 2 trials with a bare WARN
        named("nan_dphi_only", "nan_dphi", ("nonfinite_step", 2), 0),
        // finite and monotone below 10, NaN from there on: trials 1, 5 are finite, 21 is NaN
        Problem {
            id: "nan_late",
            phi: Box::new(|a| if a < 10.0 { -a } else { f64::NAN }),
            dphi: Box::new(|a| if a < 10.0 { -1.0 } else { f64::NAN }),
            scipy: ("maxiter", 100),
            first: 2,
        },
    ]
}

fn drive_case(p: &Problem) -> (Case, Run) {
    let case = case_by_id(p.id);
    assert!(case.is_nonfinite_group(), "{}", p.id);
    let run = drive(
        &case.config(),
        &p.phi,
        &p.dphi,
        case.phi0,
        case.dphi0,
        case.alpha1,
        case.maxiter,
    )
    .unwrap_or_else(|e| panic!("{}: start error {}", p.id, start_error_name(&e)));
    (case, run)
}

#[test]
fn termination_carries_the_last_finite_best_point() {
    for p in problems() {
        let (case, run) = drive_case(&p);
        let Ended::Done(t) = &run.ended else {
            panic!("{}: expected a termination", p.id)
        };
        assert_eq!(t.reason, Reason::NonFiniteValue, "{}", p.id);

        // the first non-finite trial of the trace, and that nothing before it is non-finite
        let first = run
            .trace
            .iter()
            .position(|&(_, f, g)| !f.is_finite() || !g.is_finite())
            .unwrap_or_else(|| panic!("{}: no non-finite trial", p.id));
        assert_eq!(
            first, p.first,
            "{}: index of the first non-finite trial",
            p.id
        );
        assert_eq!(t.evaluations, first + 1, "{}: evaluations", p.id);
        assert_eq!(
            run.trace.len(),
            first + 1,
            "{}: the search stops at it",
            p.id
        );

        // best is the state's best() read just before the non-finite tell, bitwise
        let before = if first == 0 {
            run.start.best
        } else {
            run.snaps[first - 1].best
        };
        assert!(
            same_point(t.best, before),
            "{}: {:?} vs {before:?}",
            p.id,
            t.best
        );
        // ... satisfies (1.1) ...
        assert!(
            sufficient_decrease(run.phi0, run.dphi0, case.ftol, t.best.alpha, t.best.phi),
            "{}: best {:?} violates (1.1)",
            p.id,
            t.best
        );
        // ... and is the start point or a traced finite trial
        let start = minimize::linesearch::Point {
            alpha: 0.0,
            phi: run.phi0,
            dphi: run.dphi0,
        };
        assert!(
            t.best == start
                || run.trace[..first].iter().any(|&(a, f, g)| t.best
                    == minimize::linesearch::Point {
                        alpha: a,
                        phi: f,
                        dphi: g
                    }),
            "{}: best {:?} is neither the start nor a finite trial",
            p.id,
            t.best
        );
        // alpha/phi/dphi are the offending trial, bitwise
        let (a, f, g) = run.trace[first];
        assert_eq!(
            (t.alpha.to_bits(), t.phi.to_bits(), t.dphi.to_bits()),
            (a.to_bits(), f.to_bits(), g.to_bits()),
            "{}: the offending trial",
            p.id
        );
    }
}

#[test]
fn the_late_run_has_finite_trials_before_the_first_non_finite_one() {
    // the best point is a traced finite trial here, not only the start point: trials 1 and 5
    let p = problems().pop().unwrap();
    assert_eq!(p.id, "nan_late");
    let (_, run) = drive_case(&p);
    let alphas: Vec<f64> = run.trace.iter().map(|t| t.0).collect();
    assert_eq!(alphas, vec![1.0, 5.0, 21.0]);
    let Ended::Done(t) = run.ended else { panic!() };
    // phi = -a: each finite trial improves on the last, so best is the last finite one
    assert_eq!((t.best.alpha, t.best.phi, t.best.dphi), (5.0, -5.0, -1.0));
}

/// Records the deliberate departure from SciPy under option (b), decided on #38: SciPy's recorded
/// outcome on non-finite values (it never inspects φ) next to ours.
#[test]
fn option_b_departs_from_scipy_on_non_finite_values() {
    for p in problems() {
        let case = case_by_id(p.id);
        // SciPy: what the data records
        assert_eq!(
            (case.reason.as_str(), case.n_trials),
            p.scipy,
            "{}: SciPy's recorded outcome",
            p.id
        );
        assert!(case.stp.is_none(), "{}: SciPy returns no step", p.id);

        // ours: stops at the first non-finite trial, with a valid best, through the driver too
        let (_, run) = drive_case(&p);
        let Ended::Done(t) = run.ended else {
            panic!("{}", p.id)
        };
        assert_eq!(t.reason, Reason::NonFiniteValue, "{}", p.id);
        assert_eq!(t.evaluations, p.first + 1, "{}", p.id);
        // the departure: never more trials than SciPy, and strictly fewer where SciPy burns
        // maxiter on NaN
        assert!(t.evaluations <= case.n_trials, "{}", p.id);
        if case.reason == "maxiter" {
            assert!(t.evaluations < case.n_trials, "{}", p.id);
        }

        let opts = Wolfe1Options {
            c1: case.ftol,
            c2: case.gtol,
            amin: case.stpmin,
            amax: case.stpmax,
            xtol: case.xtol,
            maxiter: case.maxiter,
        };
        let rep = scalar_search_wolfe1(
            &p.phi,
            &p.dphi,
            Some(case.phi0),
            None,
            Some(case.dphi0),
            &opts,
        )
        .expect("valid options");
        assert!(
            matches!(rep.outcome, Wolfe1Outcome::Warning(Reason::NonFiniteValue)),
            "{}: {:?}",
            p.id,
            rep.outcome
        );
        assert!(rep.alpha.is_none(), "{}", p.id);
        assert_eq!(
            (rep.phi_evals, rep.dphi_evals),
            (p.first + 1, p.first + 1),
            "{}",
            p.id
        );
        assert!(
            sufficient_decrease(
                case.phi0,
                case.dphi0,
                case.ftol,
                rep.best.alpha,
                rep.best.phi
            ),
            "{}: report best {:?}",
            p.id,
            rep.best
        );
        assert!(
            rep.best.phi.is_finite() && rep.best.dphi.is_finite(),
            "{}",
            p.id
        );
    }
}

#[test]
fn finite_values_are_untouched_by_the_check() {
    // A huge but finite phi (1e308) is not "non-finite": it takes the finite path, and its
    // dcstep overflow is NonFiniteStep, as in SciPy (recorded: nonfinite_step after 1 trial).
    let case = case_by_id("overflow_finite");
    assert_eq!(case.reason, "nonfinite_step");
    let cfg: MoreThuenteConfig = case.config();
    let run = drive(
        &cfg,
        |a| phi_of("big_half", &[], a),
        |a| dphi_of("big_half", &[], a),
        case.phi0,
        case.dphi0,
        case.alpha1,
        case.maxiter,
    )
    .expect("starts");
    let Ended::Done(t) = run.ended else { panic!() };
    assert_eq!(t.reason, Reason::NonFiniteStep);
    assert_eq!(run.trace.len(), case.n_trials);
}
