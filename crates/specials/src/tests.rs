//! The precision-adaptive algorithms evaluated in `f64`.
//!
//! The public API sends `f64` to the xsf kernels, so these tests check the
//! generic algorithms' logic (branches, recurrences, truncation) at `f64`
//! precision against the same mpmath reference points. Their double-double
//! accuracy is checked through the public API in `tests/reference.rs`.

use crate::generic;
use crate::refdata::{error, rows};

/// (function, relative tolerance, absolute floor)
const CASES: [(&str, f64, f64); 7] = [
    ("gamma", 4e-15, 0.0),
    ("lgamma", 4e-15, 0.0),
    ("rgamma", 4e-15, 0.0),
    // The f64 rounding of the tabulated zero of ψ shifts the result at the
    // zero itself by ~1e-16, so the criterion is absolute there.
    ("digamma", 4e-15, 1.0),
    ("zeta", 8e-15, 0.0),
    // exp(ln B) for arguments beyond Γ's range: error ~ |ln B| ε.
    ("beta", 5e-13, 0.0),
    ("lbeta", 4e-15, 0.0),
];

fn eval(name: &str, a: &[f64]) -> f64 {
    match name {
        "gamma" => generic::gamma::gamma(a[0]),
        "lgamma" => generic::gamma::lgamma(a[0]),
        "rgamma" => generic::gamma::rgamma(a[0]),
        "digamma" => generic::digamma::digamma(a[0]),
        "zeta" => generic::zeta::zeta(a[0], a[1]),
        "beta" => generic::beta::beta(a[0], a[1]),
        "lbeta" => generic::beta::lbeta(a[0], a[1]),
        _ => unreachable!("{name}"),
    }
}

#[test]
fn generic_algorithms_in_f64_match_mpmath() {
    for (name, tol, floor) in CASES {
        for r in rows(name).iter().filter(|r| r.is_f64()) {
            let a: Vec<f64> = r.args_as();
            let got = eval(name, &a);
            let e = error(got, r.reference, floor);
            assert!(
                e <= tol,
                "{name}{a:?} (line {}): got {got:e}, ref {:?}, err {e:e} > {tol:e}",
                r.line,
                r.reference
            );
        }
    }
}

#[test]
fn sinpi_is_exact_at_half_integers_and_odd_in_x() {
    for (x, s) in [
        (0.5, 1.0),
        (1.5, -1.0),
        (-0.5, -1.0),
        (-2.5, -1.0),
        (1e10 + 0.5, 1.0),
    ] {
        assert_eq!(generic::sinpi(x), s, "sinpi({x})");
    }
    assert_eq!(generic::sinpi(3.0_f64), 0.0);
}
