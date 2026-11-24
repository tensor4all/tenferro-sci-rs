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
//! Those worst errors hold for these base points; beyond 2^±300 the `f64`
//! path can keep xsf's unscaled value, whose error can reach about 1e-10
//! (`f64_kept_xsf_value_is_within_1e_10`).

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

/// Spreads beyond 2^600 at scales where xsf's products leave the range: the
/// `f64` path does not scale these (no power of four brings such a spread
/// within range), so it returns xsf's value, NaN here as on `main` and in
/// SciPy. Each of these points must be NaN or accurate: a retry that moved
/// only the largest argument into range gave 7.2e54 for 1.4e41, 6.4e39 for
/// 4.1e39, 5.1e9 for 1.0e-4 and 8.2231e-53 for 8.2310e-53 on the last four
/// (review of 470803a). Elsewhere at such spreads xsf can return a wrong
/// finite value itself, as on `main` (`docs/upstream-issues.md`). `f64` only: the `Df64` R_J returns
/// NaN beyond a ratio of 2^600 by design. References: mpmath 1.3.0,
/// `scripts/gen_carlson_scaling.py` (for `p < 0`, DLMF 19.20.14 checked
/// against Carlson 1995 Eq. 2.26).
#[test]
fn f64_wide_spread_is_nan_or_accurate() {
    let h = pow2(380);
    let t = pow2(-352);
    let rj: [([f64; 4], f64); 6] = [
        ([pow2(-900), h, 2.0 * h, 3.0 * h], 2.010295496314046e-172),
        ([pow2(-900), h, 2.0 * h, -3.0 * h], -2.2977297286893855e-172),
        ([pow2(-1000), t, 2.0 * t, 3.0 * t], 6.826453538994934e+158),
        (
            [pow2(-400), pow2(-360), pow2(500), pow2(-500)],
            1.4434984319979335e+41,
        ),
        (
            [pow2(-400), pow2(-360), pow2(500), pow2(-400)],
            4.0833845088323315e+39,
        ),
        (
            [pow2(-300), pow2(-260), pow2(600), pow2(-400)],
            0.00010113860680069928,
        ),
    ];
    for (a, want) in rj {
        let got = specials::elliprj(a[0], a[1], a[2], a[3]);
        assert!(
            got.is_nan() || ((got - want) / want).abs() <= f64_tol("elliprj"),
            "elliprj{a:?} = {got:e}, want {want:e} or NaN"
        );
    }
    let rg: [([f64; 3], f64); 2] = [
        ([pow2(-900), pow2(560), pow2(561)], 1.8553448517234111e+84),
        ([1.78e-245, 2.71e-104, 2.85e-306], 8.231038816577164e-53),
    ];
    for (a, want) in rg {
        let got = specials::elliprg(a[0], a[1], a[2]);
        assert!(
            got.is_nan() || ((got - want) / want).abs() <= f64_tol("elliprg"),
            "elliprg{a:?} = {got:e}, want {want:e} or NaN"
        );
    }
}

/// Where xsf's unscaled `f64` value is right although an argument lies far
/// beyond 2^300, the wrappers return it unchanged. Here: scaling 2^578 down would
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
    // Review round 6: arguments 2^396 apart, moved up by 2^389 into 2^±300,
    // where xsf no longer takes its small-`p` expansion and is off by 1e-11;
    // the unscaled value agrees with it to 1e-10 and is kept. Reference at the
    // arguments times 4^200, scaled back exactly.
    let a = [
        2.53438683249964e-208,
        2.917058701014637e-114,
        3.916777135234618e-89,
        -4.5006477943934846e-101,
    ];
    let want = -1.6910016572016675e+146;
    let got = specials::elliprj(a[0], a[1], a[2], a[3]);
    let err = ((got - want) / want).abs();
    assert!(
        err <= f64_tol("elliprj"),
        "elliprj{a:?} = {got:e}, error {err:e}"
    );
}

/// Values xsf gets wrong but finite beyond 2^±300 (on `main`: -2.1e137 for a
/// true 3.72e157, and 1.632e-76 for 1.609e-76): the arguments are at most
/// 2^600 apart, so the `f64` wrappers move them into 2^±300. Each is
/// compared, rescaled exactly, with mpmath's value at the base point
/// (`scripts/gen_carlson_scaling.py`).
#[test]
fn f64_scaling_corrects_wrong_finite_values() {
    let (e, a) = (-392, [3e-100, 1.0, 2e80, -1e-70]);
    let want = 3.6742346141747673e-20;
    let s: Vec<f64> = a.iter().map(|v| v * pow2(e)).collect();
    let got = specials::elliprj(s[0], s[1], s[2], s[3]) * pow2(3 * e / 2);
    let err = ((got - want) / want).abs();
    assert!(
        err <= f64_tol("elliprj"),
        "elliprj{a:?} at 2^{e}: error {err:e}"
    );

    let (e, a) = (-568, [1e-20, 1.0, 1e20]);
    let want = 5000000000.0;
    let s: Vec<f64> = a.iter().map(|v| v * pow2(e)).collect();
    let got = specials::elliprg(s[0], s[1], s[2]) * pow2(-e / 2);
    let err = ((got - want) / want).abs();
    assert!(
        err <= f64_tol("elliprg"),
        "elliprg{a:?} at 2^{e}: error {err:e}"
    );
}

/// Principal values whose transformation (DLMF 19.20.14 in the `Df64` path,
/// Carlson 1995 Eq. 2.26 in xsf) cancels completely, with condition numbers
/// of 1e45 to 1e91: `|p|` far below `y`, spreads of 150 to 500 bits, at
/// scales where both paths scale their arguments (found in review, where the
/// scaled evaluation returned +2.7e154 for -1.6e81). Both strategies must give NaN
/// or the value, never a wrong finite number. References: mpmath 1.3.0 with
/// 400 extra bits, `scripts/gen_carlson_scaling.py`.
#[test]
fn ill_conditioned_principal_value_is_nan_not_wrong() {
    let cases = [
        ((-675, -475, -175, -575), -1.6484213442199829e+81),
        ((-675, -625, -475, -650), -2.4165269490410304e+216),
        ((-625, -575, -425, -600), -6.396493574267708e+193),
        // Scaled by the `f64` window, where the unscaled and the scaled value
        // agree (the condition number does not depend on the scale), so the
        // check must not depend on which one is returned (review of 62f0191:
        // 2.26e-72 for -2.78e-145).
        ((-175, 25, 325, -75), -2.7833734563365376e-145),
    ];
    for ((ex, ey, ez, ep), want) in cases {
        let a = [pow2(ex), pow2(ey), pow2(ez), -pow2(ep)];
        let f = specials::elliprj(a[0], a[1], a[2], a[3]);
        assert!(
            f.is_nan() || ((f - want) / want).abs() <= f64_tol("elliprj"),
            "f64 elliprj{a:?} = {f:e}, want {want:e} or NaN"
        );
        let d: Vec<Df64> = a.iter().map(|&v| Df64::from(v)).collect();
        let g = specials::elliprj(d[0], d[1], d[2], d[3]).hi();
        assert!(
            g.is_nan() || ((g - want) / want).abs() <= DF64_TOL,
            "Df64 elliprj{a:?} = {g:e}, want {want:e} or NaN"
        );
    }
}

/// Beyond 2^±300 the `f64` wrappers keep xsf's unscaled value where it agrees
/// with the scaled one to 1e-10, so there the accuracy is xsf's own at those
/// arguments, which can be about 1e-10 rather than [`f64_tol`]: these two
/// values (the same as on `main`) are 5.8e-11 and 7.4e-11 off (review of
/// b6d45b0). This pins that ceiling. References: mpmath 1.3.0 at the
/// arguments times 4^k, scaled back exactly (`scripts/gen_carlson_scaling.py`).
#[test]
fn f64_kept_xsf_value_is_within_1e_10() {
    let got = specials::elliprg(pow2(-763), pow2(-368), pow2(-695));
    let want = 2.039157646249539e-56;
    let err = ((got - want) / want).abs();
    assert!(err <= 1e-10, "elliprg: {got:e}, error {err:e}");
    let got = specials::elliprj(pow2(-431), pow2(-326), pow2(-330), pow2(-365));
    let want = 2.2338413140293353e+154;
    let err = ((got - want) / want).abs();
    assert!(err <= 1e-10, "elliprj: {got:e}, error {err:e}");
}

/// The pole (two of `x`, `y`, `z` zero, `p != 0`) is `+∞` at every scale and
/// in both strategies, also for `p < 0` beyond 2^300, where the `f64`
/// principal-value check must not run (review of 5e63e30: NaN there).
#[test]
fn pole_is_infinite_at_every_scale() {
    for (z, p) in [
        (pow2(299), -1.0),
        (pow2(301), -1.0),
        (pow2(400), -1.0),
        (pow2(400), -pow2(400)),
        (pow2(400), 1.0),
        (pow2(-400), -pow2(-400)),
    ] {
        assert_eq!(
            specials::elliprj(0.0, 0.0, z, p),
            f64::INFINITY,
            "f64 z={z:e} p={p:e}"
        );
        let d = Df64::from;
        assert_eq!(
            specials::elliprj(d(0.0), d(0.0), d(z), d(p)).hi(),
            f64::INFINITY,
            "Df64 z={z:e} p={p:e}"
        );
    }
}
