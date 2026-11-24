//! Carlson's R_C, R_D, R_F, R_G and R_J at arguments near 2^±520 and 2^±600
//! (tensor4all/tenferro-sci-rs#25): ratios of 1, finite values, NaN before
//! the arguments were scaled ahead of the products that overflow.
//!
//! Both strategies: `f64` takes `Strategy::F64Kernels` (the xsf port), and
//! `xprec::Df64` takes `Strategy::Adaptive` (the duplication algorithms).
//!
//! Each function `R`, homogeneous of degree `d` (DLMF 19.16.1-19.16.5:
//! `d = -1/2` for R_C and R_F, `-3/2` for R_D and R_J, `1/2` for R_G), is
//! evaluated at `2^e a` and the result multiplied by the exact power of two
//! `2^(-d e)` is compared with mpmath's value at the base point `a`, printed
//! by `scripts/gen_carlson_scaling.py` (mpmath 1.3.0, at 200 and 300 bits,
//! which agree to 1e-40, rounded to a double-double `(hi, lo)`). For `p < 0`
//! that value is the principal value of DLMF 19.20.14.
//!
//! Tolerances (relative): those of `tests/reference.rs` for the function at
//! ordinary arguments, [`f64_tol`] for `f64` and 1e-30 for `Df64`. Worst
//! measured over [`CASES`] and [`SCALES`] (f64 / Df64): elliprc 1.8e-16 /
//! 2.4e-32, elliprd 2.1e-16 / 3.2e-32, elliprf 0 / 9.4e-33, elliprg 1.6e-16 /
//! 2.6e-32, elliprj 3.5e-16 / 5.6e-32. Before the fix, every R_J case and
//! R_G at 2^520, 2^600 (NaN) and 2^-600 (0) failed for `f64`, and R_J for
//! `p < 0` failed for `Df64` (NaN, or a relative error up to 1e-11 at 2^-520).

// These tests exercise `xprec::Df64`.
#![cfg(feature = "xprec")]

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use specials::Real;
use xprec::Df64;

/// Exponents `e` of the scale `2^e`: each makes at least one strategy of
/// R_G or R_J return NaN without the scaling, and keeps every result normal.
const SCALES: [i32; 4] = [-600, -520, 520, 600];

/// The `f64` tolerance of `tests/reference.rs` for each function.
fn f64_tol(name: &str) -> f64 {
    if name == "elliprj" {
        2e-14
    } else {
        1e-15
    }
}

const DF64_TOL: f64 = 1e-30;

/// `(name, base point, mpmath value at the base point)`.
type Case = (&'static str, &'static [f64], (f64, f64));

// mpmath 1.3.0, Python 3.12.3, 2026-11-24
const CASES: &[Case] = &[
    (
        "elliprj",
        &[1.0, 1.0, 1.0, -1.0],
        (-0.5651621397896542, 8.788410726402706e-18),
    ),
    (
        "elliprj",
        &[1.0, 2.0, 3.0, 4.0],
        (0.23984809974956775, 1.1926293550006813e-17),
    ),
    (
        "elliprj",
        &[1.0, 2.0, 3.0, -4.0],
        (-0.23786769472998162, -6.990746507619842e-18),
    ),
    (
        "elliprj",
        &[0.0, 1.0, 2.0, -1.0],
        (-1.633978807894258, 2.8154055542643115e-17),
    ),
    (
        "elliprj",
        &[0.0, 1.0, 2.0, 3.0],
        (0.7768862377858233, -1.0206575239892958e-17),
    ),
    (
        "elliprj",
        &[2.0, 3.0, 4.0, -0.5],
        (0.24723819703051564, 1.0564156708793774e-17),
    ),
    ("elliprf", &[1.0, 1.0, 1.0], (1.0, 0.0)),
    (
        "elliprf",
        &[1.0, 2.0, 3.0],
        (0.7269459354689082, 2.0949362428803622e-17),
    ),
    (
        "elliprf",
        &[0.0, 1.0, 2.0],
        (1.3110287771460598, 9.58215479734899e-17),
    ),
    ("elliprd", &[1.0, 1.0, 1.0], (1.0, 0.0)),
    (
        "elliprd",
        &[1.0, 2.0, 3.0],
        (0.29046028102899063, 1.417372539482769e-17),
    ),
    (
        "elliprd",
        &[0.0, 1.0, 2.0],
        (1.0679379896673957, -2.393230569886225e-17),
    ),
    ("elliprg", &[1.0, 1.0, 1.0], (1.0, 0.0)),
    (
        "elliprg",
        &[1.0, 2.0, 3.0],
        (1.4018470999908952, -8.518665947003724e-17),
    ),
    (
        "elliprg",
        &[0.0, 1.0, 2.0],
        (0.955049447256928, 2.9784114898100217e-17),
    ),
    ("elliprc", &[1.0, 1.0], (1.0, 0.0)),
    ("elliprc", &[1.0, 2.0], (FRAC_PI_4, 3.061616997868383e-17)),
    ("elliprc", &[0.0, 1.0], (FRAC_PI_2, 6.123233995736766e-17)),
    (
        "elliprc",
        &[1.0, -1.0],
        (0.6232252401402305, 5.858940484268471e-18),
    ),
    (
        "elliprc",
        &[2.0, -1.0],
        (0.6617680207599845, 4.6847250842651736e-17),
    ),
];

fn eval<T: Real>(name: &str, a: &[T]) -> T {
    match name {
        "elliprc" => specials::elliprc(a[0], a[1]),
        "elliprd" => specials::elliprd(a[0], a[1], a[2]),
        "elliprf" => specials::elliprf(a[0], a[1], a[2]),
        "elliprg" => specials::elliprg(a[0], a[1], a[2]),
        "elliprj" => specials::elliprj(a[0], a[1], a[2], a[3]),
        _ => unreachable!("{name}"),
    }
}

/// Twice the degree of homogeneity.
fn twice_degree(name: &str) -> i32 {
    match name {
        "elliprc" | "elliprf" => -1,
        "elliprd" | "elliprj" => -3,
        "elliprg" => 1,
        _ => unreachable!("{name}"),
    }
}

/// `2^e` as an `f64` (exact for the exponents used here).
fn pow2(e: i32) -> f64 {
    assert!((-1022..=1023).contains(&e), "2^{e}");
    2.0_f64.powi(e)
}

/// The relative errors of `f64` and `Df64` at `2^e a`, rescaled to `a`.
fn errors(name: &str, a: &[f64], want: (f64, f64), e: i32) -> (f64, f64) {
    let scaled: Vec<f64> = a.iter().map(|v| v * pow2(e)).collect();
    // R(2^e a) 2^(-d e) = R(a); `-d e` is an integer since `e` is even.
    let back = pow2(-twice_degree(name) * e / 2);
    let w = Df64::from_dd(want.0, want.1);
    let f = eval(name, &scaled) * back;
    let d: Vec<Df64> = scaled.iter().map(|&v| Df64::from(v)).collect();
    let g = eval(name, &d) * Df64::from(back);
    ((f - want.0).abs() / want.0.abs(), ((g - w) / w).abs().hi())
}

/// The issue's example: R_J(a, a, a, -a) at a = 2^520 is
/// a^(-3/2) R_J(1, 1, 1, -1) = -8.887435110666537e-236, not NaN.
#[test]
fn elliprj_principal_value_at_2_520() {
    let a = pow2(520);
    let want = -0.5651621397896542 * pow2(-780);
    let f = specials::elliprj(a, a, a, -a);
    assert!(
        ((f - want) / want).abs() <= f64_tol("elliprj"),
        "f64 (F64Kernels): {f:e}, want {want:e}"
    );
    let d = Df64::from(a);
    let g = specials::elliprj(d, d, d, -d) * Df64::from(pow2(780));
    let w = Df64::from_dd(-0.5651621397896542, 8.788410726402706e-18);
    let e = ((g - w) / w).abs().hi();
    assert!(e <= DF64_TOL, "Df64 (Adaptive): {g:?}, error {e:e}");
}

/// Every function, both strategies, at `2^e` times each base point.
#[test]
fn carlson_forms_at_2_pow_pm_520_and_pm_600() {
    let mut failures = Vec::new();
    for &(name, a, want) in CASES {
        for e in SCALES {
            let (ef, ed) = errors(name, a, want, e);
            if ef.is_nan() || ef > f64_tol(name) {
                failures.push(format!("f64 {name}{a:?} at 2^{e}: error {ef:e}"));
            }
            if ed.is_nan() || ed > DF64_TOL {
                failures.push(format!("Df64 {name}{a:?} at 2^{e}: error {ed:e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A ratio of exactly 2^600 (the limit of the `Df64` R_J) centered at 2^99:
/// the `Df64` R_J scaled only from a center of 2^100 on, so the largest
/// argument, 2^399, entered `p (p + λ)²` unscaled.
#[test]
fn elliprj_at_the_ratio_limit_centered_near_2_99() {
    let (x, y, z) = (pow2(-201), 1.0, pow2(399));
    // mpmath 1.3.0, scripts/gen_carlson_scaling.py.
    let cases = [
        (z, (2.835624531459568e-178, 2.0260347664832208e-194)),
        (-z, (-2.843329141308721e-178, 2.482035430738333e-195)),
    ];
    for (p, want) in cases {
        let w = Df64::from_dd(want.0, want.1);
        let f = specials::elliprj(x, y, z, p);
        let ef = ((f - want.0) / want.0).abs();
        assert!(ef <= f64_tol("elliprj"), "f64 p={p:e}: {f:e}, error {ef:e}");
        let d = Df64::from;
        let g = specials::elliprj(d(x), d(y), d(z), d(p));
        let ed = ((g - w) / w).abs().hi();
        assert!(ed <= DF64_TOL, "Df64 p={p:e}: {g:?}, error {ed:e}");
    }
}

/// The `f64` scaling leaves xsf's treatment of a subnormal argument as zero
/// alone: with a subnormal argument the arguments are not scaled, and the
/// results are SciPy 1.18.1's (`scipy.special.elliprg`, `elliprj`).
#[test]
fn f64_keeps_xsf_subnormal_as_zero() {
    // 2^e for the subnormal and normal exponents used here.
    let p = |e: i32| 2.0_f64.powi(e + 64) / 2.0_f64.powi(64);
    // R_G(0, 0, z) = √z / 2 with z = 2^-1023, itself subnormal: 2^-512.5.
    let half_sqrt = 5.2738433074315e-155;
    assert_eq!(specials::elliprg(p(-1060), p(-1023), p(-1023)), half_sqrt);
    assert_eq!(specials::elliprg(p(-1030), p(-1030), p(-1023)), half_sqrt);
    // Two arguments of R_J zero: the pole.
    assert_eq!(
        specials::elliprj(p(-1030), p(-1030), p(-400), p(-400)),
        f64::INFINITY
    );
    assert!(specials::elliprj(p(-1060), p(-1060), p(-1023), p(-1023)).is_nan());
}

/// The two wide-ratio arms of the `f64` scaling (ratios beyond 2^600, where
/// the arguments cannot be centered within 2^±300): the largest scaled down
/// from 2^381 as far as 2^-900 stays above 2^-1000, and the largest scaled up
/// from 2^-350.4 to 2^-300. NaN on `main`. `f64` only: these arms are in the
/// xsf wrappers, and the `Df64` R_J returns NaN beyond a ratio of 2^600 by
/// design. References: mpmath 1.3.0, `scripts/gen_carlson_scaling.py` (for
/// `p < 0`, DLMF 19.20.14 checked against Carlson 1995 Eq. 2.26).
#[test]
fn f64_wide_ratio_arms() {
    let h = pow2(380);
    let t = pow2(-352);
    let rj: [([f64; 4], f64); 3] = [
        ([pow2(-900), h, 2.0 * h, 3.0 * h], 2.010295496314046e-172),
        ([pow2(-900), h, 2.0 * h, -3.0 * h], -2.2977297286893855e-172),
        ([pow2(-1000), t, 2.0 * t, 3.0 * t], 6.826453538994934e+158),
    ];
    for (a, want) in rj {
        let got = specials::elliprj(a[0], a[1], a[2], a[3]);
        let err = ((got - want) / want).abs();
        assert!(
            err <= f64_tol("elliprj"),
            "elliprj{a:?} = {got:e}, error {err:e}"
        );
    }
    let a = [pow2(-900), pow2(560), pow2(561)];
    let want = 1.8553448517234111e+84;
    let got = specials::elliprg(a[0], a[1], a[2]);
    let err = ((got - want) / want).abs();
    assert!(
        err <= f64_tol("elliprg"),
        "elliprg{a:?} = {got:e}, error {err:e}"
    );
}

/// Where xsf's unscaled `f64` value is right although an argument lies far
/// beyond 2^300, the wrappers return it unchanged: scaling 2^578 down would
/// push 2^-362 so low that the products of the small arguments underflow
/// (NaN at 4b828a6, found in review). Reference: mpmath 1.3.0,
/// `scripts/gen_carlson_scaling.py` (DLMF 19.20.14 against Carlson 1995 Eq. 2.26).
#[test]
fn f64_keeps_xsf_value_where_it_is_finite() {
    let a = [pow2(-168), pow2(-362), pow2(-129), -pow2(578)];
    let want = -1.1789147522121987e-153;
    let got = specials::elliprj(a[0], a[1], a[2], a[3]);
    let err = ((got - want) / want).abs();
    assert!(
        err <= f64_tol("elliprj"),
        "elliprj{a:?} = {got:e}, error {err:e}"
    );
}
