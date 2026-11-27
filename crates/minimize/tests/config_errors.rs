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
    scalar_search_wolfe1, ConfigError, MoreThuente, MoreThuenteConfig, Wolfe1Options, Wolfe1Outcome,
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
        let mut o = Wolfe1Options::default();
        o.c1 = c1;
        o.c2 = c2;
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
        let mut o = Wolfe1Options::default();
        o.amin = amin;
        o.amax = amax;
        o.xtol = xtol;
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
