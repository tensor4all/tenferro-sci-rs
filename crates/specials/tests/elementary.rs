//! The elementary helpers of #19 (`cosm1`, `log1pmx`, `xlogy`, `xlog1py`,
//! `logit`, `expit`, `log_expit`, `sinpi`, `cospi`, `sindg`, `cosdg`, `tandg`,
//! `cotdg`, `radian`, `exp2`, `exp10`, `cbrt`, `expm1`, `log1p`).
//!
//! * `f64` against xsf's own reference tables (scipy/xsref, imported by
//!   `scripts/import_xsref.py`) with xsf's criterion, against mpmath at the
//!   points of `scripts/gen_specials_reference.py`, and against SciPy's exact
//!   results (signed zeros, infinities, NaN).
//! * `Df64` against mpmath double-double values at the xsref points and at
//!   points chosen for the double-double code, and against the same exact
//!   results.
//! * xsf's hand-written real-valued regression cases for these functions
//!   (`tests/xsf_tests/test_trig.cpp` at commit
//!   `31177fd9a458539cf05c04d215341245d0245e53`, BSD-3-Clause, see
//!   `LICENSE-BSD-3-Clause-xsf`), for both types.
//!
//! Only the `Df64` items are gated on `xprec`, so the `f64` tests also run
//! with `--no-default-features` (#30).

// Reference values keep the digits xsf's tests give them.
#![allow(clippy::excessive_precision)]

#[path = "../src/refdata.rs"]
mod refdata;

use refdata::{
    adjust_tolerance, elementary_special_values, error, extended_relative_error, rows, same,
    xsref_rows,
};
use specials::Real;

const NAMES: [&str; 19] = [
    "cosm1",
    "log1pmx",
    "xlogy",
    "xlog1py",
    "logit",
    "expit",
    "log_expit",
    "sinpi",
    "cospi",
    "sindg",
    "cosdg",
    "tandg",
    "cotdg",
    "radian",
    "exp2",
    "exp10",
    "cbrt",
    "expm1",
    "log1p",
];

fn eval<T: Real>(name: &str, a: &[T]) -> T {
    match name {
        "cosm1" => specials::cosm1(a[0]),
        "log1pmx" => specials::log1pmx(a[0]),
        "xlogy" => specials::xlogy(a[0], a[1]),
        "xlog1py" => specials::xlog1py(a[0], a[1]),
        "logit" => specials::logit(a[0]),
        "expit" => specials::expit(a[0]),
        "log_expit" => specials::log_expit(a[0]),
        "sinpi" => specials::sinpi(a[0]),
        "cospi" => specials::cospi(a[0]),
        "sindg" => specials::sindg(a[0]),
        "cosdg" => specials::cosdg(a[0]),
        "tandg" => specials::tandg(a[0]),
        "cotdg" => specials::cotdg(a[0]),
        "radian" => specials::radian(a[0], a[1], a[2]),
        "exp2" => specials::exp2(a[0]),
        "exp10" => specials::exp10(a[0]),
        "cbrt" => specials::cbrt(a[0]),
        "expm1" => specials::expm1(a[0]),
        "log1p" => specials::log1p(a[0]),
        _ => unreachable!("{name}"),
    }
}

/// Worst error of one function, for the summary printed by each test.
#[derive(Default)]
struct Worst {
    error: f64,
    tol: f64,
    args: Vec<f64>,
    rows: usize,
}

impl Worst {
    fn add(&mut self, e: f64, tol: f64, args: &[f64]) {
        self.rows += 1;
        if e.is_finite() && e > self.error {
            self.error = e;
            self.tol = tol;
            self.args = args.to_vec();
        }
    }
}

/// Asserts that `fails` is empty, and prints the worst error per function
/// (`cargo test --test elementary -- --nocapture`).
fn finish(what: &str, worst: &[(&str, Worst)], fails: &[String]) {
    println!("{what}: worst error per function (rows, worst, its tolerance, at)");
    for (name, w) in worst {
        println!(
            "  {name:10} {:4} {:9.2e} {:9.2e} {:?}",
            w.rows, w.error, w.tol, w.args
        );
    }
    assert!(
        fails.is_empty(),
        "{what}: {} failures\n{}",
        fails.len(),
        fails.join("\n")
    );
}

// ------------------------------------------------------------------- f64 vs xsref

/// xsf's criterion on every row of every table: `extended_relative_error <=
/// adjust_tolerance(tol)` with xsref's `gcc-linux-x86_64` tolerance.
///
/// Rows with a wrong reference (see [`BAD_REFERENCE`]) are checked with their
/// recorded tolerance but left out of the reported worst errors.
#[test]
fn f64_matches_xsref() {
    let mut worst = vec![];
    let mut fails = vec![];
    let mut flagged = vec![];
    for name in NAMES {
        let mut w = Worst::default();
        for r in xsref_rows(name) {
            let got = eval::<f64>(name, &r.args);
            let e = extended_relative_error(got, r.desired);
            let tol = adjust_tolerance(r.tol);
            if r.tol < BAD_REFERENCE {
                w.add(e, tol, &r.args);
            } else {
                flagged.push(format!("{name}{:?} (xsref tolerance {:e})", r.args, r.tol));
            }
            if e.is_nan() || e > tol {
                fails.push(format!(
                    "{name} line {} {:?}: got {got:e}, desired {:e}, error {e:e} > {tol:e}",
                    r.line, r.args, r.desired
                ));
            }
        }
        worst.push((name, w));
    }
    println!("rows with a wrong xsref reference, not in the worst errors: {flagged:?}");
    assert_eq!(flagged.len(), 7, "{flagged:?}");
    finish("f64 against xsref", &worst, &fails);
}

/// xsref tolerances at or above this mark a row where xsref's reference and
/// xsf disagree completely (`docs/upstream-issues.md`): `expit(-710)` (the
/// reference 4.5e-309 is right; xsf returns 0 because `e^710` overflows),
/// `log_expit` at 500, 710, 725 and 735 (the references are 0; the values are
/// `-e^-x`, which xsf returns), and `cbrt` at `-10` and `-1` (the references
/// are NaN; the cube root is real). They still pass xsf's criterion with
/// their recorded tolerance.
const BAD_REFERENCE: f64 = 1e-3;

/// `round` is not ported: `f64::round_ties_even` passes xsref's table
/// exactly. The only difference from SciPy (Cephes) is the sign of a zero
/// result: Cephes computes `floor(x) + 1 = +0` for `-1/2 <= x < 0`, where
/// `round_ties_even` keeps `-0`.
#[test]
fn round_is_std_round_ties_even() {
    let rs = xsref_rows("round");
    assert_eq!(rs.len(), 14);
    for r in rs {
        assert_eq!(r.args[0].round_ties_even(), r.desired, "line {}", r.line);
    }
    assert!((-0.4_f64).round_ties_even().is_sign_negative());
    assert_eq!((2.5_f64).round_ties_even(), 2.0);
    assert!(f64::NAN.round_ties_even().is_nan());
}

// --------------------------------------------------------- f64 vs mpmath and SciPy

const EPS: f64 = f64::EPSILON;

/// Relative tolerance of the `f64` result against mpmath, and the absolute
/// floor (see `refdata::error`).
///
/// * `tandg`, `cotdg`: xsf computes `tan(t π/180)` with `t` reduced to
///   `[0°, 90°)`; rounding `t π/180` costs `z (|v| + 1/|v|)` ulp, with
///   `v` the value and `z = atan |v|` the reduced angle in radians (the
///   condition number of tan), so the tolerance is `4 ε (1 + z (|v| + 1/|v|))`.
/// * `cosm1`: beyond `|x| = π/4` xsf computes `cos x - 1`, accurate in
///   absolute terms (floor 1).
/// * `log1pmx`: `log1p(x) - x` for `|x| >= 1/2` cancels up to a factor 5.
fn f64_tolerance(name: &str, a: &[f64], v: f64) -> (f64, f64) {
    match name {
        "tandg" | "cotdg" => {
            let v = v.abs();
            let z = v.atan();
            (4.0 * EPS * (1.0 + z * (v + 1.0 / v)), 0.0)
        }
        "cosm1" if a[0].abs() > core::f64::consts::FRAC_PI_4 => (4.0 * EPS, 1.0),
        "log1pmx" => (8.0 * EPS, 0.0),
        _ => (2.0 * EPS, 0.0),
    }
}

/// Points where the `f64` result is not expected to equal SciPy's within
/// `4 ε`: `exp2` and `exp10` are `std`'s, which are subnormal where Cephes
/// returns 0 (`x < -1024`, `x < -308.25`); `tandg`/`cotdg` and `cosm1` near
/// their poles/zeros, where both are only as accurate as their tolerance.
fn scipy_comparable(name: &str, a: &[f64]) -> bool {
    match name {
        "exp2" => a[0] >= -1024.0,
        "exp10" => a[0] >= -308.25,
        _ => true,
    }
}

/// xsf's `cotdg` (`tandg` at `90° - x`) reduces `x` to `90 - x`, which
/// rounds to `90` for `0 < |x| < 90 ε / 2`, so the `f64` result is `±∞` there
/// (SciPy's too) while the exact value is finite (`1/(x π/180)`). Such rows
/// are checked against SciPy only; the `Df64` path is exact there.
fn f64_known_limit(name: &str, a: &[f64]) -> bool {
    match name {
        "cotdg" => a[0] != 0.0 && a[0].abs() < 45.0 * EPS,
        "tandg" => (a[0].abs() - 90.0).abs() < 45.0 * EPS * 2.0 && a[0].abs() != 90.0,
        _ => false,
    }
}

#[test]
fn f64_matches_mpmath_and_scipy() {
    let mut worst = vec![];
    let mut fails = vec![];
    for name in NAMES {
        let mut w = Worst::default();
        for r in rows(name).into_iter().filter(|r| r.is_f64()) {
            let a: Vec<f64> = r.args.iter().map(|p| p.0).collect();
            let got = eval::<f64>(name, &a);
            let (tol, floor) = f64_tolerance(name, &a, r.reference.0);
            let e = error(got, r.reference, floor);
            let limit = f64_known_limit(name, &a);
            if !limit {
                w.add(e, tol, &a);
            }
            if !limit && (e.is_nan() || e > tol) {
                fails.push(format!(
                    "{name} line {} {a:?}: got {got:e}, mpmath {:e}, error {e:e} > {tol:e}",
                    r.line, r.reference.0
                ));
            }
            if let Some(s) = r.scipy.filter(|_| scipy_comparable(name, &a)) {
                let es = extended_relative_error(got, s);
                let tol_s = tol.max(4.0 * EPS);
                // At a pole of tandg/cotdg or a zero of cosm1 SciPy's value is
                // as inaccurate as ours; compare within the mpmath tolerance.
                let tol_s = if floor > 0.0 {
                    tol_s / s.abs().max(f64::MIN_POSITIVE)
                } else {
                    tol_s
                };
                if es > tol_s {
                    fails
                        .push(format!(
                        "{name} line {} {a:?}: got {got:e}, SciPy {s:e}, error {es:e} > {tol_s:e}"
                    , r.line));
                }
            }
        }
        worst.push((name, w));
    }
    finish("f64 against mpmath", &worst, &fails);
}

// --------------------------------------------------------------- special values

fn check_special_values<T: Real>(conv: impl Fn(f64) -> T, back: impl Fn(T) -> f64) {
    let mut fails = vec![];
    let mut n = 0;
    for (name, args, expected, _scipy) in elementary_special_values() {
        let a: Vec<T> = args.iter().map(|&x| conv(x)).collect();
        let got = eval::<T>(&name, &a);
        n += 1;
        if !same(got, expected) {
            fails.push(format!(
                "{name}{args:?}: got {:?}, expected {expected:?}",
                back(got)
            ));
        }
    }
    assert!(n > 150, "only {n} special values");
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// SciPy's exact results, with the sign of zero: undefined domains
/// (`logit(-0.5)`, `log1p(-2)`), the `0 ln 0 = 0` convention of `xlogy`,
/// zeros of `sinpi` at integers, poles of `tandg`/`cotdg`, ±∞ and NaN.
#[test]
fn f64_special_values() {
    check_special_values::<f64>(|x| x, |x| x);
}

/// The undefined domains named in #19, spelled out.
#[test]
fn f64_undefined_domains() {
    use specials::{log1p, log1pmx, logit, sinpi, xlog1py, xlogy};
    assert_eq!(logit(0.0_f64), f64::NEG_INFINITY);
    assert_eq!(logit(1.0_f64), f64::INFINITY);
    assert!(logit(-1e-300_f64).is_nan() && logit(1.0 + EPS).is_nan());
    assert_eq!(xlogy(0.0_f64, 0.0), 0.0);
    assert!(xlogy(0.0_f64, 0.0).is_sign_positive());
    assert_eq!(xlogy(2.0_f64, 0.0), f64::NEG_INFINITY);
    assert!(xlogy(2.0_f64, -1.0).is_nan());
    assert_eq!(xlog1py(0.0_f64, -1.0), 0.0);
    assert_eq!(xlog1py(-2.0_f64, -1.0), f64::INFINITY);
    assert_eq!(log1p(-1.0_f64), f64::NEG_INFINITY);
    assert!(log1p(-1.0 - EPS).is_nan());
    assert_eq!(log1pmx(-1.0_f64), f64::NEG_INFINITY);
    for n in -6..=6 {
        let s = sinpi(n as f64);
        assert_eq!(s, 0.0, "sinpi({n})");
    }
}

// --------------------------------------------------- xsf's test_trig.cpp (real)

/// `check_zeros`: `f(±0) = ±0`.
fn check_zeros<T: Real>(f: impl Fn(T) -> T, name: &str) {
    for x in [-0.0, 0.0] {
        let r = f(T::from_dd(x, 0.0)).approx_f64();
        assert_eq!(r, 0.0, "{name}({x})");
        assert_eq!(r.is_sign_negative(), x.is_sign_negative(), "{name}({x})");
    }
}

fn xsf_trig<T: Real>() {
    let t = |x: f64| T::from_dd(x, 0.0);
    // "sindg/sinpi/tandg IEEE pm zero scipy/20731"
    check_zeros::<T>(specials::sindg, "sindg");
    check_zeros::<T>(specials::sinpi, "sinpi");
    check_zeros::<T>(specials::tandg, "tandg");
    // "tandg IEEE zero sign at multiples of 180 scipy/20731"
    for n in [1, 2] {
        let r = specials::tandg(t((2 * n + 1) as f64 * 180.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_negative(), "tandg n={n}: {r}");
        let r = specials::tandg(t((2 * n + 2) as f64 * 180.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_positive(), "tandg n={n}: {r}");
    }
    for n in [-1, -2] {
        let r = specials::tandg(t((2 * n - 1) as f64 * 180.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_positive(), "tandg n={n}: {r}");
        let r = specials::tandg(t((2 * n - 2) as f64 * 180.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_negative(), "tandg n={n}: {r}");
    }
    // "cotdg IEEE zero sign at multiples of 90 scipy/20731"
    for n in [1, 2] {
        let r = specials::cotdg(t((4 * n - 3) as f64 * 90.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_positive(), "cotdg n={n}: {r}");
        let r = specials::cotdg(t((4 * n - 1) as f64 * 90.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_negative(), "cotdg n={n}: {r}");
    }
    for n in [-1, -2] {
        let r = specials::cotdg(t((4 * n + 3) as f64 * 90.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_negative(), "cotdg n={n}: {r}");
        let r = specials::cotdg(t((4 * n + 1) as f64 * 90.0)).approx_f64();
        assert!(r == 0.0 && r.is_sign_positive(), "cotdg n={n}: {r}");
    }
    // "tandg IEEE infinity sign scipy/20731"
    for n in [-2, -1, 0, 1, 2] {
        let r = specials::tandg(t((2 * n + 1) as f64 * 90.0)).approx_f64();
        assert!(r.is_infinite(), "tandg n={n}: {r}");
        assert_eq!(r.is_sign_negative(), n % 2 != 0, "tandg n={n}: {r}");
    }
    // "cotdg IEEE infinity sign scipy/20731"
    for n in [-2.0_f64, -1.0, -0.0, 0.0, 1.0, 2.0] {
        let r = specials::cotdg(t(n * 180.0)).approx_f64();
        assert!(r.is_infinite(), "cotdg n={n}: {r}");
        let odd = (n as i64) % 2 != 0;
        let expect_neg = if n.is_sign_negative() { !odd } else { odd };
        assert_eq!(r.is_sign_negative(), expect_neg, "cotdg n={n}: {r}");
    }
}

#[test]
fn f64_xsf_trig_regressions() {
    xsf_trig::<f64>();
}

// ----------------------------------------------------------------- Df64

#[cfg(feature = "xprec")]
mod df64 {
    use super::*;
    use xprec::Df64;

    /// Relative tolerance of the `Df64` result against mpmath.
    ///
    /// Worst errors measured over the xsref points and the points of
    /// `gen_specials_reference.py` are listed in the PR and in the mapping
    /// table of `lib.rs`; each tolerance is at least 3x the worst.
    fn tolerance(name: &str) -> f64 {
        let _ = name;
        1e-30
    }

    fn check(what: &str, points: impl Fn(&str) -> Vec<(Vec<Df64>, (f64, f64), usize)>) {
        let mut worst = vec![];
        let mut fails = vec![];
        for name in NAMES {
            let mut w = Worst::default();
            let tol = tolerance(name);
            for (a, d, line) in points(name) {
                let got = eval::<Df64>(name, &a);
                let e = error(got, d, 0.0);
                let args: Vec<f64> = a.iter().map(|x| x.hi()).collect();
                w.add(e, tol, &args);
                if e.is_nan() || e > tol {
                    fails.push(format!(
                        "{name} line {line} {a:?}: got {got:?}, mpmath {d:?}, error {e:e} > {tol:e}"
                    ));
                }
            }
            worst.push((name, w));
        }
        finish(what, &worst, &fails);
    }

    #[test]
    fn df64_matches_mpmath_on_xsref_points() {
        check("Df64 against mpmath at the xsref points", |name| {
            xsref_rows(name)
                .into_iter()
                .filter_map(|r| {
                    let d = r.dd?;
                    let a = r.args.iter().map(|&x| Df64::from(x)).collect();
                    Some((a, d, r.line))
                })
                .collect()
        });
    }

    #[test]
    fn df64_matches_mpmath() {
        check("Df64 against mpmath at the double-double points", |name| {
            rows(name)
                .into_iter()
                .map(|r| (r.args_as::<Df64>(), r.reference, r.line))
                .collect()
        });
    }

    /// Where mpmath has no value (zeros and poles, outside the domain,
    /// non-finite inputs), the `Df64` result rounded to `f64` satisfies
    /// xsf's criterion against SciPy's value.
    #[test]
    fn df64_matches_xsref_where_mpmath_is_undefined() {
        let mut fails = vec![];
        let mut n = 0;
        for name in NAMES {
            for r in xsref_rows(name) {
                if r.dd.is_some() || r.tol.is_infinite() {
                    continue;
                }
                n += 1;
                let a: Vec<Df64> = r.args.iter().map(|&x| Df64::from(x)).collect();
                let got = eval::<Df64>(name, &a).hi();
                let e = extended_relative_error(got, r.desired);
                let tol = adjust_tolerance(r.tol);
                if e.is_nan() || e > tol {
                    fails.push(format!(
                        "{name} line {} {:?}: got {got:e}, desired {:e}",
                        r.line, r.args, r.desired
                    ));
                }
            }
        }
        assert!(n > 50, "only {n} rows");
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    #[test]
    fn df64_special_values() {
        check_special_values::<Df64>(Df64::from, |x| x.hi());
    }

    #[test]
    fn df64_xsf_trig_regressions() {
        xsf_trig::<Df64>();
    }

    /// The xprec 0.2.3 defects the double-double path works around
    /// (`generic/elementary.rs`, `docs/upstream-issues.md`).
    #[test]
    fn xprec_defects_are_handled() {
        let z = Df64::from(-0.0);
        // xprec: exp_m1(-0) = ln_1p(-0) = +0.
        assert!(specials::expm1(z).hi() == 0.0 && specials::expm1(z).hi().is_sign_negative());
        assert!(specials::log1p(z).hi() == 0.0 && specials::log1p(z).hi().is_sign_negative());
        // xprec: ln_1p of -1 + 1e-25 (high word -1) is -inf.
        let x = Df64::new_full(-1.0, 1e-25);
        let expected = Df64::from(1e-25).ln();
        let got = specials::log1p(x);
        assert!(((got - expected) / expected).abs().hi() < 1e-30, "{got:?}");
        // ... and with 1 + x below 2^-900, where xprec's ln is off by 1.2e-27.
        // mpmath: ln(1e-300) = -690.7755278982137 - 2.3670096176709832e-14.
        let got = specials::log1p(Df64::new_full(-1.0, 1e-300));
        let expected = Df64::new_full(-690.7755278982137, -2.3670096176709832e-14);
        assert!(((got - expected) / expected).abs().hi() < 1e-30, "{got:?}");
        // xprec: exp2(±inf) = NaN.
        assert_eq!(
            specials::exp2(Df64::from(f64::INFINITY)).hi(),
            f64::INFINITY
        );
        assert_eq!(specials::exp2(Df64::from(f64::NEG_INFINITY)).hi(), 0.0);
        // xprec: Float::cbrt is `todo!()`.
        assert_eq!(specials::cbrt(Df64::from(-27.0)).hi(), -3.0);
    }

    /// Beyond `|x| = 2^50` `Df64` `cosm1` falls back to `cos(hi + lo) - 1`
    /// in `f64`. References: mpmath 1.3.0 at 4000 bits, `cos(x) - 1`.
    #[test]
    fn df64_cosm1_beyond_the_reduction_limit() {
        for (x, v) in [
            (1e300, -1.575386111957549),
            (3.802951800684688e30, -1.999302459991256),
            (1e29, -1.7641618474297996),
        ] {
            let got = specials::cosm1(Df64::from(x)).hi();
            assert!(
                (got - v).abs() < 4.0 * f64::EPSILON,
                "cosm1({x}) = {got}, mpmath {v}"
            );
        }
        // A low word that matters: 2^50 (1 + 2^-70) = 2^50 + 2^-20.
        // mpmath: cos(2^50 + 2^-20) - 1 = -0.13190456874045147.
        let got = specials::cosm1(Df64::new_full(2.0_f64.powi(50), 2.0_f64.powi(-20))).hi();
        assert!(
            (got + 0.13190456874045147).abs() < 4.0 * f64::EPSILON,
            "{got}"
        );
    }

    /// On a grid of extreme arguments both strategies classify the result
    /// alike (NaN, the same infinity, the same signed zero) and, where it is
    /// finite, agree to `1e-14` relative or `1e-300` absolute (`expit(-720)`:
    /// `f64` flushes to 0 as SciPy, `Df64` is subnormal), except at the
    /// `f64` known limits (`f64_known_limit`).
    #[test]
    fn strategies_agree_on_extreme_arguments() {
        let grid = [
            0.0,
            -0.0,
            5e-324,
            -5e-324,
            1e-300,
            -1e-300,
            1e-20,
            -1e-20,
            0.5,
            -0.5,
            1.0,
            -1.0,
            7.5,
            -7.5,
            720.0,
            -720.0,
            1e15,
            -1e15,
            1e300,
            -1e300,
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut fails = vec![];
        for name in NAMES {
            let nargs = match name {
                "xlogy" | "xlog1py" => 2,
                "radian" => 3,
                _ => 1,
            };
            let mut args = vec![vec![]];
            for _ in 0..nargs {
                args = args
                    .into_iter()
                    .flat_map(|a: Vec<f64>| {
                        grid.iter().map(move |&x| {
                            let mut b = a.clone();
                            b.push(x);
                            b
                        })
                    })
                    .collect();
            }
            for a in args {
                if f64_known_limit(name, &a) {
                    continue;
                }
                let f = eval::<f64>(name, &a);
                let ad: Vec<Df64> = a.iter().map(|&x| Df64::from(x)).collect();
                let d = eval::<Df64>(name, &ad).hi();
                let ok = if f.is_nan() || d.is_nan() || f.is_infinite() || f == 0.0 && d == 0.0 {
                    same(Df64::from(d), f)
                } else {
                    (f - d).abs() <= 1e-14 * f.abs() || (f - d).abs() <= 1e-300
                };
                if !ok {
                    fails.push(format!("{name}{a:?}: f64 {f:e}, Df64 {d:e}"));
                }
            }
        }
        assert!(
            fails.is_empty(),
            "{} disagreements\n{}",
            fails.len(),
            fails.join("\n")
        );
    }

    /// On every xsref row both strategies classify the result alike (NaN,
    /// the same infinity, the same signed zero) and otherwise agree to `1e-14`
    /// relative or `1e-300` absolute (`expit(-710)`: `f64` 0, `Df64` the
    /// subnormal value).
    #[test]
    fn strategies_agree_on_every_xsref_row() {
        let mut fails = vec![];
        for name in NAMES {
            for r in xsref_rows(name) {
                let f = eval::<f64>(name, &r.args);
                let a: Vec<Df64> = r.args.iter().map(|&x| Df64::from(x)).collect();
                let d = eval::<Df64>(name, &a).hi();
                let ok = if f.is_nan() || d.is_nan() || f.is_infinite() || f == 0.0 && d == 0.0 {
                    same(Df64::from(d), f)
                } else {
                    (f - d).abs() <= 1e-14 * f.abs() || (f - d).abs() <= 1e-300
                };
                if !ok {
                    fails.push(format!(
                        "{name} line {} {:?}: f64 {f:e}, Df64 {d:e}",
                        r.line, r.args
                    ));
                }
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    }

    /// Between the underflow of the type's `exp` (about ±709.8) and the end
    /// of the subnormal range, `expit` and `log_expit` return the subnormal
    /// value, as the `f64` path of `log_expit` (`-e^-x`) does.
    #[test]
    fn df64_expit_and_log_expit_subnormal_results() {
        for x in [709.0_f64, 710.0, 720.0, 740.0, 745.0] {
            let want = -(-x).exp();
            assert!(want != 0.0 || x >= 745.0);
            let got = specials::log_expit(Df64::from(x)).hi();
            assert_eq!(got, want, "log_expit({x})");
            assert_eq!(specials::log_expit(x), want, "f64 log_expit({x})");
            let got = specials::expit(Df64::from(-x)).hi();
            assert_eq!(got, -want, "expit({})", -x);
        }
    }
}
