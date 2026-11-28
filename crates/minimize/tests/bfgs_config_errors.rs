//! Every `BfgsConfig` error, each reached by a single violation, the order of reporting, and the
//! SciPy defaults.
//!
//! Validates: the options of `scipy.optimize._optimize._minimize_bfgs` (SciPy v1.16.3, commit
//! b9105ccc2237f57acb1060202cd77f6dd264fb34): `gtol = 1e-5`, `xrtol = 0`, `c1 = 1e-4`, `c2 = 0.9`,
//! `maxiter = 200 n`, and `_check_c1_c2` (`0 < c1 < c2 < 1`, `_linesearch.py`).
//!
//! Deviations from SciPy, documented in `src/bfgs/config.rs`: NaN and infinite tolerances are
//! refused (SciPy compares them as "not negative"), and `maxiter = 0` is refused.
//!
//! Status in the red stage: the configuration is real code, so every test here passes.

use minimize::bfgs::{BfgsConfig, ConfigError};

const OK: (f64, f64, f64, f64, usize) = (1e-5, 0.0, 1e-4, 0.9, 400);

fn with(
    gtol: f64,
    xrtol: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
) -> Result<BfgsConfig, ConfigError> {
    BfgsConfig::new(gtol, xrtol, c1, c2, maxiter)
}

#[test]
fn each_error_by_a_single_violation() {
    let (g, x, c1, c2, m) = OK;
    assert_eq!(with(-1e-9, x, c1, c2, m), Err(ConfigError::NegativeGtol));
    assert_eq!(with(g, -1e-9, c1, c2, m), Err(ConfigError::NegativeXrtol));
    assert_eq!(
        with(g, x, 0.0, c2, m),
        Err(ConfigError::InvalidWolfeParameters)
    );
    assert_eq!(
        with(g, x, -1e-4, c2, m),
        Err(ConfigError::InvalidWolfeParameters)
    );
    assert_eq!(
        with(g, x, c1, 1.0, m),
        Err(ConfigError::InvalidWolfeParameters)
    );
    assert_eq!(
        with(g, x, 0.9, 0.9, m),
        Err(ConfigError::InvalidWolfeParameters)
    );
    assert_eq!(
        with(g, x, 0.95, 0.9, m),
        Err(ConfigError::InvalidWolfeParameters)
    );
    assert_eq!(with(g, x, c1, c2, 0), Err(ConfigError::ZeroMaxIter));
}

#[test]
fn nan_and_infinity_are_non_finite_in_every_field() {
    let (g, x, c1, c2, m) = OK;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            with(bad, x, c1, c2, m),
            Err(ConfigError::NonFinite),
            "gtol = {bad}"
        );
        assert_eq!(
            with(g, bad, c1, c2, m),
            Err(ConfigError::NonFinite),
            "xrtol = {bad}"
        );
        assert_eq!(
            with(g, x, bad, c2, m),
            Err(ConfigError::NonFinite),
            "c1 = {bad}"
        );
        assert_eq!(
            with(g, x, c1, bad, m),
            Err(ConfigError::NonFinite),
            "c2 = {bad}"
        );
    }
}

#[test]
fn several_violations_report_the_first_in_variant_order() {
    // non-finite before everything
    assert_eq!(
        with(f64::NAN, -1.0, 2.0, 3.0, 0),
        Err(ConfigError::NonFinite)
    );
    // negative gtol before negative xrtol before the Wolfe parameters before maxiter
    assert_eq!(
        with(-1.0, -1.0, 2.0, 3.0, 0),
        Err(ConfigError::NegativeGtol)
    );
    assert_eq!(
        with(1e-5, -1.0, 2.0, 3.0, 0),
        Err(ConfigError::NegativeXrtol)
    );
    assert_eq!(
        with(1e-5, 0.0, 2.0, 3.0, 0),
        Err(ConfigError::InvalidWolfeParameters)
    );
}

#[test]
fn accepts_the_boundaries_scipy_accepts() {
    // gtol = 0 (never met by a nonzero gradient: the xrtol or iteration limit ends the run) and xrtol = 0
    assert!(with(0.0, 0.0, 1e-4, 0.9, 1).is_ok());
    // c1 and c2 arbitrarily close to the open bounds
    assert!(with(1e-5, 0.0, f64::MIN_POSITIVE, 1.0 - f64::EPSILON, 10).is_ok());
    assert!(with(1e-5, 1e-8, 0.1, 0.1 + 1e-12, usize::MAX).is_ok());
}

#[test]
fn scipy_defaults_and_getters() {
    for (n, maxiter) in [(1, 200), (2, 400), (5, 1000), (10, 2000)] {
        let c = BfgsConfig::scipy_default(n);
        assert_eq!(
            (c.gtol(), c.xrtol(), c.c1(), c.c2(), c.maxiter()),
            (1e-5, 0.0, 1e-4, 0.9, maxiter),
            "n = {n}"
        );
    }
    // n = 0 would give maxiter = 0, which is refused; the default stays valid
    assert_eq!(BfgsConfig::scipy_default(0).maxiter(), 200);
    let c = with(1e-3, 1e-7, 0.01, 0.5, 17).expect("valid");
    assert_eq!(
        (c.gtol(), c.xrtol(), c.c1(), c.c2(), c.maxiter()),
        (1e-3, 1e-7, 0.01, 0.5, 17)
    );
}

#[test]
fn the_defaults_are_a_valid_configuration() {
    for n in [0, 1, 7, 100] {
        let d = BfgsConfig::scipy_default(n);
        assert_eq!(
            BfgsConfig::new(d.gtol(), d.xrtol(), d.c1(), d.c2(), d.maxiter()),
            Ok(d)
        );
    }
}

#[test]
fn error_messages_name_the_violation() {
    assert_eq!(
        ConfigError::NegativeGtol.to_string(),
        "gtol must be nonnegative"
    );
    assert!(ConfigError::InvalidWolfeParameters
        .to_string()
        .contains("0 < c1 < c2 < 1"));
}

// ---- the departures from SciPy, one named test each (listed in the `minimize::bfgs` docs) ------------------------

/// Departure 1: `maxiter = 0` is refused; SciPy accepts it and reports "maximum number of iterations".
#[test]
fn departure_zero_maxiter_is_refused() {
    assert_eq!(with(1e-5, 0.0, 1e-4, 0.9, 0), Err(ConfigError::ZeroMaxIter));
    assert!(with(1e-5, 0.0, 1e-4, 0.9, 1).is_ok());
}

/// Departure 2: NaN or infinite `gtol`, `xrtol`, `c1`, `c2` are refused; SciPy compares them as "not negative".
#[test]
fn departure_non_finite_tolerances_are_refused() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(with(bad, 0.0, 1e-4, 0.9, 10), Err(ConfigError::NonFinite));
        assert_eq!(with(1e-5, bad, 1e-4, 0.9, 10), Err(ConfigError::NonFinite));
        assert_eq!(with(1e-5, 0.0, bad, 0.9, 10), Err(ConfigError::NonFinite));
        assert_eq!(with(1e-5, 0.0, 1e-4, bad, 10), Err(ConfigError::NonFinite));
    }
}

/// Departure 3: a negative `gtol` or `xrtol` is refused; SciPy runs with it (the gradient test never fires).
#[test]
fn departure_negative_tolerances_are_refused() {
    assert_eq!(
        with(-1e-12, 0.0, 1e-4, 0.9, 10),
        Err(ConfigError::NegativeGtol)
    );
    assert_eq!(
        with(1e-5, -1e-12, 1e-4, 0.9, 10),
        Err(ConfigError::NegativeXrtol)
    );
}

/// The defaults cannot fail: they are a `const`, validated when the crate is built (as `lbfgs`'s line search),
/// and carry the `wolfe1` line search of `_line_search_wolfe12`.
#[test]
fn scipy_default_is_a_const_and_carries_the_wolfe1_line_search() {
    const D: BfgsConfig = BfgsConfig::scipy_default(3);
    assert_eq!(D.maxiter(), 600);
    let ls = D.line_search();
    assert_eq!(
        (ls.ftol(), ls.gtol(), ls.xtol(), ls.stpmin(), ls.stpmax()),
        (1e-4, 0.9, 1e-14, 1e-100, 1e100)
    );
    let custom = with(1e-8, 1e-9, 0.01, 0.5, 7).expect("valid");
    assert_eq!(
        (custom.line_search().ftol(), custom.line_search().gtol()),
        (0.01, 0.5)
    );
}
