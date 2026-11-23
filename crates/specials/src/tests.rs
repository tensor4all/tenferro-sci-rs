//! The precision-adaptive algorithms evaluated in `f64`.
//!
//! The public API sends `f64` to the xsf kernels, so these tests check the
//! generic algorithms' logic (branches, recurrences, truncation) at `f64`
//! precision against the same mpmath reference points. Their double-double
//! accuracy is checked through the public API in `tests/reference.rs`.

use crate::generic;
use crate::refdata::{error, rows};

/// (function, relative tolerance, absolute floor)
const CASES: [(&str, f64, f64); 15] = [
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
    // The rest of the gamma family (#8). poch, binom and gen_harmonic pay
    // |ln Γ| ε in f64 (exp of a difference of ln Γ values, or of -a ln i).
    ("poch", 1e-12, 0.0),
    // x = -3.75 is next to a zero of ln|Γ(1+x)| (x ≈ -3.74): absolute accuracy
    // eps there gives 3e-14 relative.
    ("lgam1p", 5e-14, 0.0),
    ("binom", 1e-12, 0.0),
    ("zetac", 1e-12, 0.0),
    ("trigamma", 4e-15, 0.0),
    ("digammainv", 4e-15, 0.0),
    ("gen_harmonic", 4e-13, 0.0),
    // 6e5 additions at n = 2e5: the f64 recurrence accumulates ~1e-14.
    ("stirling2", 1e-13, 0.0),
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
        "poch" => generic::poch::poch(a[0], a[1]),
        "lgam1p" => generic::lgam1p::lgam1p(a[0]),
        "binom" => generic::binom::binom(a[0], a[1]),
        "zetac" => generic::zeta::zetac(a[0]),
        "trigamma" => generic::trigamma::trigamma(a[0]),
        "digammainv" => generic::digammainv::digammainv(a[0]),
        "gen_harmonic" => generic::gen_harmonic::gen_harmonic(a[0], a[1]),
        "stirling2" => match generic::stirling2::prepare(a[0], a[1]) {
            generic::stirling2::Prepared::Value(v) => v,
            generic::stirling2::Prepared::Kernel(n, k) => generic::stirling2::stirling2(n, k),
        },
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
