//! The split between the core and the driver, pinned.
//!
//! `MoreThuenteConfig` validates DCSRCH's own conditions only. The `0 < c1 < c2 < 1` check, the
//! defaults of `scalar_search_wolfe1` and the initial-step heuristic belong to the thin driver, so
//! that L-BFGS-B can pass its own numbers (see `lbfgsb_params.rs`).
//!
//! * `c1 >= c2`, `c1 <= 0` and `c2 >= 1` are rejected by the driver, and `ftol == gtol` is
//!   accepted by the core: `config_errors.rs` (`driver_requires_c1_below_c2`,
//!   `config_accepts_what_dcsrch_accepts`) already covers both; not repeated here.
//! * The defaults and the initial step are pinned below.

use minimize::linesearch::{
    scalar_search_wolfe1, MoreThuenteConfig, StartError, Wolfe1Options, Wolfe1Outcome,
};
use std::cell::Cell;

#[test]
fn defaults_are_scipys() {
    // scipy/optimize/_linesearch.py (v1.16.3): scalar_search_wolfe1, lines 100-102:
    //   c1=1e-4, c2=0.9, amax=50, amin=1e-8, xtol=1e-14; line 167: maxiter = 100
    let o = Wolfe1Options::default();
    assert_eq!(o.c1, 1e-4);
    assert_eq!(o.c2, 0.9);
    assert_eq!(o.amax, 50.0);
    assert_eq!(o.amin, 1e-8);
    assert_eq!(o.xtol, 1e-14);
    assert_eq!(o.maxiter, 100);
}

#[test]
fn the_core_has_no_defaults_and_does_not_apply_the_driver_checks() {
    // a configuration the driver would refuse (c1 == c2, c2 == 1) is a valid core config
    assert!(MoreThuenteConfig::new(0.9, 0.9, 1e-14, 0.0, 1e10).is_ok());
    assert!(MoreThuenteConfig::new(0.0, 1.0, 0.1, 0.0, 1e10).is_ok());
}

/// The first trial the driver asks for, with φ(0) = `phi0`, φ'(0) = `dphi0` and the given
/// `old_phi0`: the initial step of `_linesearch.py` lines 160-165,
/// `min(1, 1.01 * 2 * (phi0 - old_phi0) / derphi0)`, or 1 if that is negative.
fn first_trial(phi0: f64, old_phi0: Option<f64>, dphi0: f64) -> f64 {
    let first = Cell::new(None);
    // amax large enough for any step; maxiter = 1 evaluates exactly the first trial
    let o = Wolfe1Options {
        maxiter: 1,
        ..Wolfe1Options::default()
    };
    let rep = scalar_search_wolfe1(
        |a| {
            if first.get().is_none() {
                first.set(Some(a));
            }
            phi0 + a * dphi0
        },
        |_| dphi0,
        Some(phi0),
        old_phi0,
        Some(dphi0),
        &o,
    )
    .expect("valid options");
    assert!(
        matches!(rep.outcome, Wolfe1Outcome::MaxIter),
        "{:?}",
        rep.outcome
    );
    first.get().expect("one trial was evaluated")
}

#[test]
fn initial_step_heuristic() {
    // the formula, computed in the same operation order
    let f = |phi0: f64, old: f64, d: f64| 1.01 * 2.0 * (phi0 - old) / d;
    // below 1: used as is
    assert_eq!(first_trial(1.0, Some(1.1), -1.0), f(1.0, 1.1, -1.0)); // 0.202
    assert!(f(1.0, 1.1, -1.0) < 1.0);
    // capped at 1
    assert_eq!(first_trial(10.0, Some(1.0), -1.0), 1.0);
    // negative (the function went up since the last point): 1
    assert!(f(1.0, 0.0, -1.0) < 0.0);
    assert_eq!(first_trial(1.0, Some(0.0), -1.0), 1.0);
    // no previous value: 1
    assert_eq!(first_trial(1.0, None, -1.0), 1.0);
}

#[test]
fn initial_step_with_zero_slope_is_one_and_a_start_error() {
    // dphi0 == 0 skips the heuristic (alpha1 = 1, no division by zero); the search then refuses
    // the start for the slope, not for a non-finite step.
    let o = Wolfe1Options::default();
    let rep = scalar_search_wolfe1(|a| a, |_| 0.0, Some(1.0), Some(2.0), Some(0.0), &o)
        .expect("valid options");
    assert!(matches!(
        rep.outcome,
        Wolfe1Outcome::Error(StartError::NotDescent)
    ));
    assert_eq!((rep.phi_evals, rep.dphi_evals), (0, 0));
}
