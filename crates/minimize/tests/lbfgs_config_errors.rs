//! `LbfgsConfig` validation: one test per violation, the order of the checks, the boundaries that are valid, and the
//! values of `scipy_default()`.
//!
//! Validates the errors of `minimize::lbfgs::ConfigError`. SciPy facts behind the choices (SciPy 1.16.3, probed):
//! `maxcor = 0` is `ERROR: M <= 0`, `ftol < 0` is `ERROR: FACTR < 0`, `maxls = 0` is a `ValueError`; `maxiter = 0` and
//! `maxfun = 0` run one iteration (their tests come after an increment); a negative or NaN `gtol`/`ftol` and an
//! infinite one are accepted silently. This crate rejects the last group and the two zeros (documented departures).
//!
//! Source: SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34) `_lbfgsb_py.py`, `__lbfgsb.c` `errclb`.

use minimize::lbfgs::{ConfigError, LbfgsConfig, H0};

fn ok() -> (usize, f64, f64, usize, usize, usize, H0) {
    (10, 1e-5, 1e-9, 100, 100, 20, H0::ScaledGamma)
}

fn make(t: (usize, f64, f64, usize, usize, usize, H0)) -> Result<LbfgsConfig, ConfigError> {
    LbfgsConfig::new(t.0, t.1, t.2, t.3, t.4, t.5, t.6)
}

#[test]
fn zero_memory_is_refused() {
    assert_eq!(
        make((0, ok().1, ok().2, ok().3, ok().4, ok().5, ok().6)),
        Err(ConfigError::ZeroMemory)
    );
}

#[test]
fn zero_maxiter_is_refused() {
    assert_eq!(
        make((ok().0, ok().1, ok().2, 0, ok().4, ok().5, ok().6)),
        Err(ConfigError::ZeroMaxiter)
    );
}

#[test]
fn zero_maxfun_is_refused() {
    assert_eq!(
        make((ok().0, ok().1, ok().2, ok().3, 0, ok().5, ok().6)),
        Err(ConfigError::ZeroMaxfun)
    );
}

#[test]
fn zero_maxls_is_refused() {
    assert_eq!(
        make((ok().0, ok().1, ok().2, ok().3, ok().4, 0, ok().6)),
        Err(ConfigError::ZeroMaxls)
    );
}

#[test]
fn non_finite_tolerances_are_refused() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            make((ok().0, bad, ok().2, ok().3, ok().4, ok().5, ok().6)),
            Err(ConfigError::NonFinite),
            "gtol {bad}"
        );
        assert_eq!(
            make((ok().0, ok().1, bad, ok().3, ok().4, ok().5, ok().6)),
            Err(ConfigError::NonFinite),
            "ftol {bad}"
        );
    }
}

#[test]
fn negative_gtol_is_refused() {
    assert_eq!(
        make((ok().0, -1e-12, ok().2, ok().3, ok().4, ok().5, ok().6)),
        Err(ConfigError::NegativeGtol)
    );
}

#[test]
fn negative_ftol_is_refused() {
    assert_eq!(
        make((ok().0, ok().1, -1e-12, ok().3, ok().4, ok().5, ok().6)),
        Err(ConfigError::NegativeFtol)
    );
}

#[test]
fn the_checks_run_in_the_documented_order() {
    // counts before tolerances; non-finite before negative; gtol before ftol
    assert_eq!(
        make((0, f64::NAN, -1.0, 0, 0, 0, H0::ScaledGamma)),
        Err(ConfigError::ZeroMemory)
    );
    assert_eq!(
        make((1, f64::NAN, -1.0, 1, 1, 0, H0::ScaledGamma)),
        Err(ConfigError::ZeroMaxls)
    );
    assert_eq!(
        make((1, -1.0, f64::NAN, 1, 1, 1, H0::ScaledGamma)),
        Err(ConfigError::NonFinite)
    );
    assert_eq!(
        make((1, -1.0, -1.0, 1, 1, 1, H0::ScaledGamma)),
        Err(ConfigError::NegativeGtol)
    );
}

#[test]
fn the_boundaries_are_valid() {
    // zero tolerances switch a test off; m = 1, maxls = 1, maxiter = 1, maxfun = 1 are the smallest counts
    assert!(make((1, 0.0, 0.0, 1, 1, 1, H0::FixedIdentity)).is_ok());
    let cfg = make((3, 0.0, 0.0, 7, 9, 2, H0::FixedIdentity)).expect("valid");
    assert_eq!(
        (
            cfg.m(),
            cfg.gtol(),
            cfg.ftol(),
            cfg.maxiter(),
            cfg.maxfun(),
            cfg.maxls(),
            cfg.h0()
        ),
        (3, 0.0, 0.0, 7, 9, 2, H0::FixedIdentity)
    );
}

#[test]
fn scipy_default_has_scipys_options_and_lnsrlbs_line_search() {
    let cfg = LbfgsConfig::scipy_default();
    assert_eq!(
        (
            cfg.m(),
            cfg.gtol(),
            cfg.maxiter(),
            cfg.maxfun(),
            cfg.maxls(),
            cfg.h0()
        ),
        (10, 1e-5, 15000, 15000, 20, H0::ScaledGamma)
    );
    assert_eq!(cfg.ftol(), 2.220446049250313e-9_f64);
    assert_eq!(cfg.ftol(), 1e7 * f64::EPSILON);
    // __lbfgsb.c lnsrlb: ftol = 1e-3, gtol = 0.9, xtol = 0.1, stpmin = 0, stpmax = 1e10 (unconstrained)
    let ls = cfg.line_search();
    assert_eq!(
        (ls.ftol(), ls.gtol(), ls.xtol(), ls.stpmin(), ls.stpmax()),
        (1e-3, 0.9, 0.1, 0.0, 1e10)
    );
    // and the same line search whatever the other options are
    let other = make((3, 1e-8, 0.0, 7, 9, 2, H0::FixedIdentity)).expect("valid");
    assert_eq!(other.line_search(), ls);
}

#[test]
fn the_infallible_default_equals_the_validated_path() {
    // scipy_default() is built from constants; new() validates caller input. They must agree field by field, and the
    // line search must equal the validated MoreThuenteConfig::new of the lnsrlb constants.
    let validated = LbfgsConfig::new(
        10,
        1e-5,
        2.220446049250313e-9,
        15000,
        15000,
        20,
        H0::ScaledGamma,
    )
    .expect("valid");
    assert_eq!(LbfgsConfig::scipy_default(), validated);
    let ls =
        minimize::linesearch::MoreThuenteConfig::new(1e-3, 0.9, 0.1, 0.0, 1e10).expect("valid");
    assert_eq!(LbfgsConfig::scipy_default().line_search(), &ls);
}

#[test]
fn h0_defaults_to_scaled_gamma() {
    assert_eq!(H0::default(), H0::ScaledGamma);
}

#[test]
fn errors_have_readable_messages() {
    assert_eq!(
        ConfigError::ZeroMemory.to_string(),
        "the memory size m must be at least 1"
    );
    assert!(ConfigError::NonFinite.to_string().contains("finite"));
}
