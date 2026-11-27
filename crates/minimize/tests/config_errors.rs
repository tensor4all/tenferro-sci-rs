//! Every configuration and start error of `minimize::linesearch`, each reached by a single
//! violation, plus the layer split between `MoreThuenteConfig` and the SciPy-style driver.
//!
//! Validates: the argument checks of `DCSRCH` (`scipy/optimize/_dcsrch.py`, SciPy v1.16.3, commit
//! b9105ccc2237f57acb1060202cd77f6dd264fb34, the `task == b"START"` block: STP < STPMIN, STP >
//! STPMAX, INITIAL G >= ZERO, FTOL/GTOL/XTOL < 0, STPMIN < 0, STPMAX < STPMIN) and the
//! `_check_c1_c2` of `scalar_search_wolfe1` in `_linesearch.py` (`0 < c1 < c2 < 1`).
//!
//! Layering: `MoreThuenteConfig` holds DCSRCH's own checks only, so ftol == gtol (the paper's
//! functions 2-6 use mu = eta) is allowed; the stricter `0 < c1 < c2 < 1` lives in the driver.
//! DCSRCH reports the *last* failing START check (later checks overwrite earlier ones); with the
//! validation split in two, the start checks keep that relative order (below < above < not descent).

mod common;

use common::*;
use minimize::linesearch::{
    scalar_search_wolfe1, ConfigError, MoreThuente, MoreThuenteConfig, StartError, Wolfe1Options,
    Wolfe1Outcome,
};

fn good() -> MoreThuenteConfig {
    MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0).expect("valid")
}

#[test]
fn each_config_error_by_a_single_violation() {
    let e = MoreThuenteConfig::new(-1e-4, 0.9, 1e-14, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NegativeFtol), "{e:?}");
    let e = MoreThuenteConfig::new(1e-4, -0.9, 1e-14, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NegativeGtol), "{e:?}");
    let e = MoreThuenteConfig::new(1e-4, 0.9, -1e-14, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NegativeXtol), "{e:?}");
    let e = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, -1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NegativeStpmin), "{e:?}");
    // stpmin >= 0 here, so only stpmax < stpmin is violated
    let e = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 2.0, 1.0).unwrap_err();
    assert!(matches!(e, ConfigError::StpmaxBelowStpmin), "{e:?}");
}

#[test]
fn config_accepts_what_dcsrch_accepts() {
    // ftol == gtol: the paper's functions 2-6 (mu = eta) need it
    assert!(MoreThuenteConfig::new(0.1, 0.1, 1e-14, 0.0, 1e10).is_ok());
    // ftol > gtol (mu > eta): the case of the paper's theorem 3.3
    assert!(MoreThuenteConfig::new(0.03, 0.001, 1e-14, 0.0, 1e10).is_ok());
    // zeros are allowed (>= 0, not > 0), and stpmin == stpmax is a valid (degenerate) interval
    assert!(MoreThuenteConfig::new(0.0, 0.0, 0.0, 0.0, 1.0).is_ok());
    assert!(MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1.0, 1.0).is_ok());
}

#[test]
fn each_start_error_by_a_single_violation() {
    let cfg = good();
    // alpha0 < stpmin
    let e = MoreThuente::start(&cfg, 1.0, -1.0, 1e-9).unwrap_err();
    assert!(
        matches!(e, minimize::linesearch::StartError::StepBelowStpmin),
        "{e:?}"
    );
    // alpha0 > stpmax
    let e = MoreThuente::start(&cfg, 1.0, -1.0, 51.0).unwrap_err();
    assert!(
        matches!(e, minimize::linesearch::StartError::StepAboveStpmax),
        "{e:?}"
    );
    // dphi0 >= 0: positive and exactly zero (SciPy: `g >= 0`)
    for g in [1.0, 0.0] {
        let e = MoreThuente::start(&cfg, 1.0, g, 1.0).unwrap_err();
        assert!(
            matches!(e, minimize::linesearch::StartError::NotDescent),
            "g = {g}: {e:?}"
        );
    }
    // the bounds are inclusive: alpha0 == stpmin and alpha0 == stpmax start fine
    assert!(MoreThuente::start(&cfg, 1.0, -1.0, 1e-8).is_ok());
    assert!(MoreThuente::start(&cfg, 1.0, -1.0, 50.0).is_ok());
}

#[test]
fn start_error_priority_follows_dcsrch() {
    // DCSRCH overwrites `task` with each failing check in the order below < above < not descent,
    // so the last failing one is reported (_dcsrch.py, `if task[:5] == b"START"`).
    let cfg = good();
    let e = MoreThuente::start(&cfg, 1.0, 1.0, 1e-9).unwrap_err(); // below stpmin AND not descent
    assert!(
        matches!(e, minimize::linesearch::StartError::NotDescent),
        "{e:?}"
    );
    let e = MoreThuente::start(&cfg, 1.0, 1.0, 51.0).unwrap_err(); // above stpmax AND not descent
    assert!(
        matches!(e, minimize::linesearch::StartError::NotDescent),
        "{e:?}"
    );
}

#[test]
fn driver_requires_c1_below_c2() {
    // `_check_c1_c2`: 0 < c1 < c2 < 1, so the config-valid mu == eta is rejected here.
    let phi = |a: f64| (a - 5.0) * (a - 5.0);
    let dphi = |a: f64| 2.0 * (a - 5.0);
    let bad = [
        (0.9, 0.9),  // c1 == c2
        (0.5, 0.1),  // c1 > c2
        (0.0, 0.9),  // c1 == 0
        (-0.1, 0.9), // c1 < 0
        (1e-4, 1.0), // c2 == 1
        (1e-4, 1.5), // c2 > 1
    ];
    for (c1, c2) in bad {
        let o = Wolfe1Options {
            c1,
            c2,
            ..Wolfe1Options::default()
        };
        let r = scalar_search_wolfe1(phi, dphi, Some(25.0), None, Some(-10.0), &o);
        assert!(r.is_err(), "c1 = {c1}, c2 = {c2} must be rejected");
    }
    // and the defaults are accepted
    let r = scalar_search_wolfe1(
        phi,
        dphi,
        Some(25.0),
        None,
        Some(-10.0),
        &Wolfe1Options::default(),
    );
    assert!(r.is_ok());
}

#[test]
fn driver_rejects_an_invalid_config() {
    // The driver builds a MoreThuenteConfig from amin, amax, xtol; their violations are errors
    // of the driver (not a Wolfe1Outcome), like SciPy's ValueError for c1/c2 is an exception.
    let phi = |a: f64| (a - 5.0) * (a - 5.0);
    let dphi = |a: f64| 2.0 * (a - 5.0);
    for (amin, amax, xtol) in [
        (-1e-8, 50.0, 1e-14),
        (10.0, 1.0, 1e-14),
        (1e-8, 50.0, -1e-14),
    ] {
        let o = Wolfe1Options {
            amin,
            amax,
            xtol,
            ..Wolfe1Options::default()
        };
        let r = scalar_search_wolfe1(phi, dphi, Some(25.0), None, Some(-10.0), &o);
        assert!(r.is_err(), "amin = {amin}, amax = {amax}, xtol = {xtol}");
    }
}

#[test]
fn driver_reports_start_errors_as_outcomes() {
    // SciPy returns (None, ...) with task "ERROR: ..." for these, it does not raise: a start error
    // is an outcome of the report, with alpha None and no trial evaluated.
    let o = Wolfe1Options::default();
    let rep =
        scalar_search_wolfe1(|a| a, |_| 1.0, Some(0.0), None, Some(1.0), &o).expect("options ok");
    assert!(rep.alpha.is_none());
    assert_eq!(outcome_name(&rep.outcome), "err_not_descent");
    assert!(matches!(rep.outcome, Wolfe1Outcome::Error(_)));
    assert_eq!((rep.phi_evals, rep.dphi_evals), (0, 0));
}

#[test]
fn non_finite_start_values() {
    // A documented addition to SciPy: DCSRCH has no check for non-finite phi(0) or phi'(0)
    // (NaN phi'(0) passes its `g >= 0` test and the search proceeds; LineSearches.jl throws).
    // `MoreThuente::start` returns `StartError::NonFiniteStart` when phi0 or dphi0 is NaN or +-inf.
    let cfg = good();
    for (phi0, dphi0) in [
        (f64::NAN, -1.0),
        (f64::INFINITY, -1.0),
        (f64::NEG_INFINITY, -1.0),
        (1.0, f64::NAN),
        (1.0, f64::NEG_INFINITY),
    ] {
        let e = MoreThuente::start(&cfg, phi0, dphi0, 1.0).unwrap_err();
        assert!(
            matches!(e, StartError::NonFiniteStart),
            "phi0 = {phi0}, dphi0 = {dphi0}: {e:?}"
        );
    }
}

#[test]
fn non_finite_is_checked_before_the_descent_test() {
    // dphi0 = +inf and dphi0 = NaN also fail `dphi0 < 0`; non-finiteness wins. dphi0 = -inf is a
    // (not finite) descent value and is NonFiniteStart, not an accepted start.
    let cfg = good();
    for (phi0, dphi0) in [
        (1.0, f64::INFINITY),
        (f64::NAN, 1.0),
        (f64::NAN, 0.0),
        (1.0, f64::NAN),
    ] {
        let e = MoreThuente::start(&cfg, phi0, dphi0, 1.0).unwrap_err();
        assert!(
            matches!(e, StartError::NonFiniteStart),
            "phi0 = {phi0}, dphi0 = {dphi0}: {e:?}"
        );
    }
    // finite non-descent values are still NotDescent
    let e = MoreThuente::start(&cfg, 1.0, 1.0, 1.0).unwrap_err();
    assert!(matches!(e, StartError::NotDescent), "{e:?}");
}

#[test]
fn driver_reports_non_finite_start_as_an_outcome() {
    // `Wolfe1Outcome::Error(StartError::NonFiniteStart)` with `alpha: None`, no trial evaluated.
    let o = Wolfe1Options::default();
    for (phi0, dphi0) in [(f64::NAN, -1.0), (f64::INFINITY, -1.0), (1.0, f64::NAN)] {
        let rep = scalar_search_wolfe1(|a| a, |_| -1.0, Some(phi0), None, Some(dphi0), &o)
            .expect("options ok");
        assert!(rep.alpha.is_none());
        assert_eq!(
            outcome_name(&rep.outcome),
            "err_non_finite_start",
            "phi0 = {phi0}, dphi0 = {dphi0}"
        );
        assert!(matches!(
            rep.outcome,
            Wolfe1Outcome::Error(StartError::NonFiniteStart)
        ));
        assert_eq!((rep.phi_evals, rep.dphi_evals), (0, 0));
    }
}

#[test]
fn non_finite_config_values_are_rejected() {
    // A documented addition to SciPy (whose DCSRCH compares NaN as "not negative" and goes on):
    // any of ftol, gtol, xtol, stpmin, stpmax that is NaN or +-inf is `ConfigError::NonFinite`,
    // except stpmax = +inf, which means "no upper bound" and is allowed.
    let e = MoreThuenteConfig::new(f64::NAN, 0.9, 1e-14, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NonFinite), "ftol NaN: {e:?}");
    let e = MoreThuenteConfig::new(1e-4, f64::INFINITY, 1e-14, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NonFinite), "gtol +inf: {e:?}");
    let e = MoreThuenteConfig::new(1e-4, 0.9, f64::NAN, 1e-8, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NonFinite), "xtol NaN: {e:?}");
    let e = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, f64::NEG_INFINITY, 50.0).unwrap_err();
    assert!(matches!(e, ConfigError::NonFinite), "stpmin -inf: {e:?}");
    let e = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, f64::NAN).unwrap_err();
    assert!(matches!(e, ConfigError::NonFinite), "stpmax NaN: {e:?}");
}

#[test]
fn stpmax_infinity_is_allowed_and_runs() {
    // stpmax = +inf is accepted, and a search with it converges: MT1 (Table I's mu, eta) from
    // alpha0 = 1e-1 (Table I: 3 evaluations, alpha_m = 1.4414).
    let c = cfg_with(1e-3, 1e-1, 1e-14, 0.0, f64::INFINITY);
    let (t, trace) = run_named(&c, "mt1", &[], 1e-1);
    assert!(t.reason.is_converged(), "{:?}", t.reason);
    assert_eq!(t.evaluations, 3);
    assert_eq!(trace.len(), 3);
    assert!((t.alpha - 1.4414).abs() < 1e-3, "{}", t.alpha);
}

#[test]
fn non_finite_alpha0_is_a_start_error_before_the_bounds() {
    // StartError::NonFiniteStart also covers alpha0 = NaN or +-inf, checked before the bound and
    // descent checks: NaN is NonFiniteStart, not a bound error (every comparison with NaN is false,
    // so a bound check could not report it); +inf with a finite stpmax is NonFiniteStart, not
    // StepAboveStpmax; -inf is NonFiniteStart, not StepBelowStpmin.
    let cfg = good();
    for a0 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let e = MoreThuente::start(&cfg, 1.0, -1.0, a0).unwrap_err();
        assert!(
            matches!(e, StartError::NonFiniteStart),
            "alpha0 = {a0}: {e:?}"
        );
        // ... and also before the descent test
        let e = MoreThuente::start(&cfg, 1.0, 1.0, a0).unwrap_err();
        assert!(
            matches!(e, StartError::NonFiniteStart),
            "alpha0 = {a0}, dphi0 = 1: {e:?}"
        );
    }
    // alpha0 = +inf with stpmax = +inf: inside [stpmin, stpmax] but still not a usable step
    let unbounded = cfg_with(1e-4, 0.9, 1e-14, 0.0, f64::INFINITY);
    let e = MoreThuente::start(&unbounded, 1.0, -1.0, f64::INFINITY).unwrap_err();
    assert!(matches!(e, StartError::NonFiniteStart), "{e:?}");
}
