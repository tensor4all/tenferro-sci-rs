//! A regression ratchet for Carlson's R_C, R_D, R_F, R_G and R_J in the
//! regimes that the argument scaling of tensor4all/tenferro-sci-rs#25 (PR
//! #26) touched (tensor4all/tenferro-sci-rs#27). Each of the three review
//! rounds that #26 needed found a failure no committed test covered; each is
//! a row here:
//!
//! * a retry arm that returned any finite value at spreads beyond 2^600 (off by
//!   13 orders of magnitude): `spread_over_600` rows and
//!   [`f64_homogeneity_beyond_spread_600`] accept NaN there, never a wrong value;
//! * a window that moved the arguments across xsf's absolute threshold
//!   `y <= 1e-26` and discarded the accurate unscaled value
//!   (`elliprj(2^-322, 2^-110, 2^-92, 2^-50)`): `threshold_y` rows;
//! * statistics from uncommitted probes: the table and its generator
//!   `scripts/gen_carlson_ratchet.py` (mpmath 1.3.0 pinned) are committed.
//!
//! Regimes of `tests/data/carlson_ratchet.csv` (column `regime`): `ordinary`;
//! `ratio_near_one` (arguments equal or within 2^-7 of each other at 2^±520
//! and 2^±600); `spread_to_600` and `spread_over_600` (exponent spreads of 596
//! to 600 and 601 to 640 bits around `RJ_RATIO_BITS = 600`, at centers inside
//! and outside the `f64` window 2^±300, plus a real ratio of 2^600.58 at an
//! exponent spread of 600); `xsf_one_huge` (one argument 2^500 to 2^600 above
//! the others); `threshold_p` and `threshold_y` (xsf's absolute thresholds
//! `p <= 1e-9` and `y <= 1e-26`); `xsf_tiny_y` (a known xsf error, see
//! [`f64_tolerance`]); `principal_value` (`p < 0`, well conditioned up to
//! 2^±520, and cancelling ones beyond 2^±300); `kept_xsf` (values the `f64`
//! window keeps from xsf); `pv_ill_conditioned` (NaN or the value).
//!
//! Both strategies: `f64` takes `Strategy::F64Kernels` (the xsf port, where the
//! threshold band lives) and runs in every configuration; `xprec::Df64` takes
//! `Strategy::Adaptive` (the duplication algorithms, where the generic overflow
//! lived) and needs the `xprec` feature (on by default). Only the `Df64` tests
//! are gated, so `cargo test --no-default-features` still runs the four `f64`
//! tests.
//!
//! Checks:
//!
//! * [`f64_reference_table`], `df64_reference_table`: every row against mpmath
//!   (256 and 512 bits plus the argument spread), with the tolerance of its
//!   regime ([`f64_tolerance`], `df64_tolerance`). A row may allow NaN for one
//!   strategy where the crate documents NaN; a finite value must then still be
//!   within tolerance.
//! * [`f64_matches_scipy`]: the `f64` path against SciPy 1.16.3 wherever it
//!   does not scale the arguments.
//! * [`f64_homogeneity`], `df64_homogeneity` and the `beyond_spread_600`
//!   pair: `R(λ a) = λ^d R(a)` (`d = -1/2` for R_C and R_F, `-3/2` for R_D and
//!   R_J, `1/2` for R_G; DLMF 19.16.1-19.16.5) for `λ = 2^e` over the whole
//!   exponent range, up to the documented NaN edges, with no reference value.
//!
//! The suite runs in about 0.5 s (debug build).

use specials::Real;
#[cfg(feature = "xprec")]
use xprec::Df64;

/// One row of the reference table.
struct Row {
    name: String,
    regime: String,
    args: Vec<f64>,
    reference: (f64, f64),
    scipy: f64,
    f64_nan_ok: bool,
    #[cfg_attr(not(feature = "xprec"), allow(dead_code))]
    df64_nan_ok: bool,
    /// For R_J with `p < 0`: the condition numbers of DLMF 19.20.14 (the `Df64`
    /// path) and of Carlson (1995) Eq. 2.26 (xsf); 1 elsewhere.
    #[cfg_attr(not(feature = "xprec"), allow(dead_code))]
    cond_dlmf: f64,
    cond_carlson: f64,
    line: usize,
}

fn num(s: &str) -> f64 {
    s.parse::<f64>()
        .unwrap_or_else(|e| panic!("bad number {s:?}: {e}"))
}

fn table() -> Vec<Row> {
    include_str!("data/carlson_ratchet.csv")
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.starts_with('#') && !l.trim().is_empty())
        .map(|(i, l)| {
            let f: Vec<&str> = l.split(',').collect();
            assert_eq!(f.len(), 13, "line {}: {l}", i + 1);
            let cond = |s: &str| if s.is_empty() { 1.0 } else { num(s) };
            Row {
                name: f[0].to_string(),
                regime: f[1].to_string(),
                args: f[2..6]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .map(|s| num(s))
                    .collect(),
                reference: (num(f[6]), num(f[7])),
                scipy: num(f[8]),
                f64_nan_ok: f[9] == "1",
                df64_nan_ok: f[10] == "1",
                cond_dlmf: cond(f[11]),
                cond_carlson: cond(f[12]),
                line: i + 1,
            }
        })
        .collect()
}

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

/// Relative error of `got` against the double-double `want`.
fn rel<T: Real>(got: T, want: (f64, f64)) -> f64 {
    let w = T::from_dd(want.0, want.1);
    ((got - w) / w).abs().approx_f64()
}

/// Whether the `f64` R_G or R_J scales `args` (`Scaling::Window` in
/// `xsf::carlson`, as documented on [`specials::elliprj`]): a nonzero
/// argument beyond 2^±300, none subnormal, and exponents at most 600 apart.
/// The result is then xsf's value at one of two scales, kept from the
/// unscaled one unless the independent evaluation shows it off by more than
/// 1e-10: its documented accuracy is 1e-10.
fn f64_window(name: &str, args: &[f64]) -> bool {
    if !matches!(name, "elliprg" | "elliprj") || args.iter().any(|v| v.is_subnormal()) {
        return false;
    }
    let exps: Vec<i32> = args
        .iter()
        .filter(|v| **v != 0.0 && v.is_finite())
        .map(|v| v.abs().log2().floor() as i32)
        .collect();
    match (exps.iter().max(), exps.iter().min()) {
        (Some(&max), Some(&min)) => (max > 300 || min < -300) && max - min <= 600,
        _ => false,
    }
}

/// `f64` epsilon, and the double-double epsilon `2^-105`.
const EPS_F64: f64 = f64::EPSILON;
#[cfg(feature = "xprec")]
const EPS_DF64: f64 = 2.465_190_328_815_662e-32;

/// The relative tolerance of a row against mpmath, `f64`.
///
/// * Every regime starts from the tolerances of `tests/reference.rs` at
///   ordinary arguments: `f64` 2e-14 for R_J and 1e-15 for the others,
///   `Df64` 1e-30 (the documented accuracy of the crate).
/// * `f64` R_G at spreads of 2^596 and more: 4e-15. xsf's R_G there is 8 ulp
///   off (1.8e-15 measured), the same value as SciPy (checked by
///   [`f64_matches_scipy`]).
/// * `f64` R_J in `threshold_p` (`p` from 5e-10 to 1e-8 with `x, y, z` up to
///   1): 1e-12. Just above xsf's switch `p <= 1e-9` (an absolute threshold)
///   to Gustafson's identity, which is exact, xsf's duplication for `p << x`
///   is up to 2.6e-13 off (`(1, 1, 1, 1.0000000000000002e-9)`; SciPy gives the
///   same value).
/// * `f64` R_J in `xsf_tiny_y`: `y / z` (sorted `x <= y <= z`). xsf's
///   asymptotic expansion for `y <= 1e-26` (Carlson & Gustafson 1994,
///   Proposition J3) has a truncation error of order `y / z`, and its selection
///   bound lets `y / z` be about `1 / |ln(p / y)|`: 1.9e-6 for `y / z = 2^-18`
///   (`docs/upstream-issues.md`; SciPy gives the same value). These rows pin
///   that bound and the parity with SciPy, not an accuracy the code can reach.
/// * `f64` R_G and R_J where the `f64` path scales ([`f64_window`]): 1e-10,
///   the documented accuracy of a value kept from xsf's unscaled evaluation.
/// * R_J with `p < 0`: at least `4 cond ε`, with `cond` the condition number
///   of the transformation the strategy evaluates (Eq. 2.26 in `f64`, DLMF
///   19.20.14 in `Df64`) and `ε` its epsilon: the three terms cancel, and no
///   evaluation of them does better.
fn f64_tolerance(r: &Row) -> f64 {
    let (name, regime, a) = (r.name.as_str(), r.regime.as_str(), &r.args);
    let mut tf = if name == "elliprj" { 2e-14 } else { 1e-15 };
    if name == "elliprg" && regime.starts_with("spread") {
        tf = 4e-15;
    }
    if regime == "threshold_p" {
        tf = 1e-12;
    }
    if regime == "xsf_tiny_y" {
        let mut v = [a[0], a[1], a[2]];
        v.sort_by(f64::total_cmp);
        tf = v[1] / v[2];
    }
    if f64_window(name, a) {
        tf = tf.max(1e-10);
    }
    if name == "elliprj" && a[3] < 0.0 {
        tf = tf.max(4.0 * r.cond_carlson * EPS_F64);
    }
    tf
}

/// The `Df64` counterpart of [`f64_tolerance`]: 1e-30, and `4 cond ε` for a
/// principal value.
#[cfg(feature = "xprec")]
fn df64_tolerance(r: &Row) -> f64 {
    if r.name == "elliprj" && r.args[3] < 0.0 {
        1e-30_f64.max(4.0 * r.cond_dlmf * EPS_DF64)
    } else {
        1e-30
    }
}

/// `e <= tol`, false for a NaN error.
fn within(e: f64, tol: f64) -> bool {
    e <= tol
}

/// Every row against mpmath in one strategy (`T` = `f64` or `Df64`): NaN only
/// where the row allows it for that strategy; otherwise, and for any finite
/// value, the error within `tolerance` ([`f64_tolerance`] or `df64_tolerance`). Prints the worst
/// error per regime and function.
fn check_table<T: Real>(label: &str, tolerance: fn(&Row) -> f64, nan_ok: fn(&Row) -> bool) {
    let rows = table();
    assert!(rows.len() > 600, "{} rows", rows.len());
    let mut failures = Vec::new();
    // (regime, function) -> (worst error over non-NaN results, NaN results, rows)
    let mut worst: std::collections::BTreeMap<(&str, &str), (f64, usize, usize)> =
        Default::default();
    for r in &rows {
        let (tol, nan_ok) = (tolerance(r), nan_ok(r));
        let a: Vec<T> = r.args.iter().map(|&v| T::from_dd(v, 0.0)).collect();
        let got = eval(&r.name, &a);
        let e = rel(got, r.reference);
        let w = worst
            .entry((r.regime.as_str(), r.name.as_str()))
            .or_default();
        w.2 += 1;
        if got.is_nan() {
            w.1 += 1;
        } else {
            w.0 = w.0.max(e);
        }
        if !(got.is_nan() && nan_ok) && !within(e, tol) {
            failures.push(format!(
                "line {} {} {label} {}{:?} = {:e}, mpmath {:e}, error {e:.2e} > {tol:.1e}",
                r.line,
                r.regime,
                r.name,
                r.args,
                got.approx_f64(),
                r.reference.0
            ));
        }
    }
    println!("{label}: worst relative error against mpmath (NaN results / rows):");
    for ((regime, name), (e, nans, n)) in &worst {
        println!("  {regime:18} {name:8} {e:8.1e} ({nans:2}/{n:3})");
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every row against mpmath, `Strategy::F64Kernels`.
#[test]
fn f64_reference_table() {
    check_table::<f64>("f64", f64_tolerance, |r| r.f64_nan_ok);
}

/// Every row against mpmath, `Strategy::Adaptive`.
#[cfg(feature = "xprec")]
#[test]
fn df64_reference_table() {
    check_table::<Df64>("Df64", df64_tolerance, |r| r.df64_nan_ok);
}

/// The `f64` path against SciPy 1.16.3 where it does not scale the arguments
/// (there it is xsf's code, unchanged): 2e-15 for R_J and 1e-15 for the
/// others, as in `tests/reference.rs`, and NaN exactly where SciPy gives NaN.
/// Where it scales ([`f64_window`]) SciPy has no counterpart (NaN, 0 or
/// xsf's unscaled value), so those rows are checked only against mpmath.
#[test]
fn f64_matches_scipy() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for r in table().iter().filter(|r| !f64_window(&r.name, &r.args)) {
        let f = eval(&r.name, &r.args);
        let tol = if r.name == "elliprj" { 2e-15 } else { 1e-15 };
        let ok = if r.scipy.is_nan() {
            f.is_nan()
        } else {
            ((f - r.scipy) / r.scipy).abs() <= tol
        };
        checked += 1;
        if !ok {
            failures.push(format!(
                "line {} {} {}{:?}: f64 {f:e}, SciPy {:e}",
                r.line, r.regime, r.name, r.args, r.scipy
            ));
        }
    }
    println!("{checked} rows compared with SciPy");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// --- homogeneity ---------------------------------------------------------------

/// Twice the degree of homogeneity `d` (DLMF 19.16.1-19.16.5).
fn twice_degree(name: &str) -> i32 {
    match name {
        "elliprc" | "elliprf" => -1,
        "elliprd" | "elliprj" => -3,
        "elliprg" => 1,
        _ => unreachable!("{name}"),
    }
}

/// `v 2^e` for any `e`, exact while the result is normal.
fn ldexp<T: Real>(mut v: T, mut e: i32) -> T {
    while e != 0 {
        let k = e.clamp(-1000, 1000);
        v = v * T::from_dd(2.0_f64.powi(k), 0.0);
        e -= k;
    }
    v
}

/// `2^e`, exact for normal powers of two.
fn pow2(e: i32) -> f64 {
    ldexp(1.0, e)
}

/// The `f64` path is checked for homogeneity while its largest argument is
/// below `2^F64_TOP_BITS`. From there on xsf's R_C and R_F overflow in a sum
/// of their arguments and return NaN or 0 (SciPy 1.16.3 gives the same values;
/// `docs/upstream-issues.md`, "`elliprj` is NaN for one huge ..."): for example
/// `elliprc(2^1022, -2^1022)` is NaN and `elliprf(2^1022, 2^1023, 3 2^1022)`
/// is NaN. There [`f64_homogeneity`] accepts NaN or 0 from `f64`, never a wrong
/// nonzero value. The `Df64` path is checked up to `f64::MAX`.
const F64_TOP_BITS: f64 = 1022.0;

/// Exponent spread, in bits, of the nonzero arguments.
fn spread_bits(a: &[f64]) -> f64 {
    let l: Vec<f64> = a
        .iter()
        .filter(|v| **v != 0.0)
        .map(|v| v.abs().log2())
        .collect();
    l.iter().copied().fold(f64::MIN, f64::max) - l.iter().copied().fold(f64::MAX, f64::min)
}

/// The exponents `e` (even, so that `λ^d = 2^(d e)` is a power of two) at
/// which every nonzero argument of `2^e a` is a normal `f64` and the value
/// `2^(d e) R(a)` lies between 2^-960 (above which the double-double low word
/// is normal too) and 2^1000: the whole exponent range, up to where the
/// arguments or the result leave it.
fn exponents(name: &str, a: &[f64], value: f64) -> Vec<i32> {
    let td = twice_degree(name);
    let lg = |v: f64| v.abs().log2();
    (-1100..=1100)
        .step_by(2)
        .filter(|&e| {
            let e_f = f64::from(e);
            a.iter()
                .filter(|v| **v != 0.0)
                .all(|&v| lg(v) + e_f >= -1022.0 && lg(v) + e_f < 1024.0)
                && (-960.0..1000.0).contains(&(lg(value) + f64::from(td * e / 2)))
        })
        .collect()
}

/// What a strategy may return at `2^e a` besides the homogeneous value.
#[derive(Clone, Copy, PartialEq)]
enum Allow {
    /// The homogeneous value only.
    Value,
    /// NaN or the homogeneous value (a documented NaN edge).
    NanOrValue,
    /// NaN, 0 or the homogeneous value: where xsf's documented failures
    /// include 0 (`docs/upstream-issues.md`).
    NanZeroOrValue,
}

/// Checks `R(2^e a) 2^(-d e) = base` over [`exponents`]; `allow` and `tol`
/// take the scaled arguments. Returns the worst error over the finite nonzero
/// results, the number of exponents, the number of NaN or 0 results, and the
/// failures.
fn check_homogeneity<T: Real>(
    name: &str,
    a: &[f64],
    base: T,
    allow: impl Fn(&[f64]) -> Allow,
    tol: impl Fn(&[f64]) -> f64,
) -> (f64, usize, usize, Vec<String>) {
    let es = exponents(name, a, base.approx_f64());
    let (mut worst, mut nans, mut bad) = (0.0_f64, 0, Vec::new());
    for &e in &es {
        let scaled: Vec<f64> = a.iter().map(|&v| pow2(e) * v).collect();
        let s: Vec<T> = scaled.iter().map(|&v| T::from_dd(v, 0.0)).collect();
        let got = eval(name, &s);
        if got.is_nan() || got == T::zero() {
            nans += 1;
            let ok = match allow(&scaled) {
                Allow::Value => false,
                Allow::NanOrValue => got.is_nan(),
                Allow::NanZeroOrValue => true,
            };
            if !ok {
                bad.push(format!("2^{e}: {:e}", got.approx_f64()));
            }
            continue;
        }
        let back = ldexp(got, -twice_degree(name) * e / 2);
        let err = ((back - base) / base).abs().approx_f64();
        if err <= tol(&scaled) {
            worst = worst.max(err);
        } else {
            bad.push(format!("2^{e}: {:e}, error {err:.1e}", got.approx_f64()));
        }
    }
    (worst, es.len(), nans, bad)
}

/// The `f64` tolerance of the homogeneity check at the scaled arguments `s`:
/// twice that of `tests/reference.rs` (both values carry an error), and the
/// documented 1e-10 where the `f64` path scales ([`f64_window`]).
fn f64_homogeneity_tol(name: &str, s: &[f64]) -> f64 {
    if f64_window(name, s) {
        1e-10
    } else if name == "elliprj" {
        4e-14
    } else {
        2e-15
    }
}

/// `Df64`: 4e-30, twice the documented 1e-30 (the principal values among the
/// base points are conditioned below 10).
#[cfg(feature = "xprec")]
const DF64_HOMOGENEITY_TOL: f64 = 4e-30;

/// Base points of [`f64_homogeneity`] and [`df64_homogeneity`]: every
/// function, `p` of both signs, a zero argument, and spreads of 2^40, 2^598
/// and 2^600 (the largest ratio of the `Df64` R_J), where no NaN is
/// documented.
fn homogeneity_bases() -> Vec<(&'static str, Vec<f64>)> {
    let mut bases: Vec<(&str, Vec<f64>)> = vec![
        ("elliprc", vec![1.0, 2.0]),
        ("elliprc", vec![2.0, 1.0]),
        ("elliprc", vec![1.0, -1.0]),
        ("elliprc", vec![0.25, 1e3]),
        ("elliprc", vec![0.25, -1.0]),
        ("elliprf", vec![1.0, 2.0, 3.0]),
        ("elliprf", vec![0.0, 1.0, 2.0]),
        ("elliprf", vec![1e-6, 1.0, 1e6]),
        ("elliprd", vec![1.0, 2.0, 3.0]),
        ("elliprd", vec![0.0, 2.0, 1.0]),
        ("elliprd", vec![1e6, 1.0, 1e-6]),
        ("elliprg", vec![1.0, 2.0, 3.0]),
        ("elliprg", vec![0.0, 1.0, 2.0]),
        ("elliprg", vec![1e-6, 1.0, 1e6]),
        ("elliprj", vec![1.0, 2.0, 3.0, 4.0]),
        ("elliprj", vec![0.0, 1.0, 2.0, 3.0]),
        ("elliprj", vec![1.0, 1.0, 1.0, -1.0]),
        ("elliprj", vec![1.0, 2.0, 3.0, -4.0]),
        ("elliprj", vec![2.0, 3.0, 4.0, -0.5]),
        ("elliprj", vec![1e-6, 1.0, 1e6, 1e-3]),
        ("elliprj", vec![pow2(-20), 1.0, pow2(20), -pow2(10)]),
    ];
    for (s, p) in [(598, 1.0), (600, 1.0), (600, -1.0)] {
        let half = s / 2;
        bases.push(("elliprj", vec![pow2(-half), 1.0, pow2(s - half), p]));
    }
    for s in [598, 600] {
        let half = s / 2;
        bases.push(("elliprg", vec![pow2(-half), 1.0, pow2(s - half)]));
    }
    bases
}

/// Base points with spreads of more than 2^600, beyond the documented NaN
/// edge ([`f64_homogeneity_beyond_spread_600`]).
fn beyond_600_bases() -> Vec<(&'static str, Vec<f64>)> {
    let bases: Vec<(&str, Vec<f64>)> = vec![
        ("elliprj", vec![pow2(-301), 1.0, pow2(301), 1.0]),
        ("elliprj", vec![pow2(-301), 1.0, pow2(301), -pow2(10)]),
        (
            "elliprj",
            vec![pow2(-301), pow2(-100), pow2(301), pow2(200)],
        ),
        ("elliprj", vec![pow2(-320), 1.0, pow2(300), pow2(-5)]),
        ("elliprg", vec![pow2(-301), 1.0, pow2(301)]),
        ("elliprg", vec![pow2(-320), pow2(-100), pow2(300)]),
    ];
    for (name, a) in &bases {
        assert!(spread_bits(a) > 600.0, "{name}{a:?}");
    }
    bases
}

fn assert_no_failures(failures: Vec<String>) {
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `R(λ x, ...) = λ^d R(x, ...)` for `λ = 2^e` over the whole exponent range,
/// `Strategy::F64Kernels`, with no reference value: a scaling change that
/// breaks the exponent of the power of two, or a range where the path returns
/// NaN or a wrong value, fails here. Tolerance [`f64_homogeneity_tol`]; at and
/// above `2^F64_TOP_BITS`, NaN or 0 is accepted ([`F64_TOP_BITS`]).
#[test]
fn f64_homogeneity() {
    let mut failures = Vec::new();
    for (name, a) in homogeneity_bases() {
        let base = eval(name, &a);
        assert!(base.is_finite(), "{name}{a:?} = {base:e}");
        let top = |s: &[f64]| s.iter().any(|v| v.abs().log2() >= F64_TOP_BITS);
        let (w, n, nans, bad) = check_homogeneity(
            name,
            &a,
            base,
            |s| {
                if top(s) {
                    Allow::NanZeroOrValue
                } else {
                    Allow::Value
                }
            },
            |s| f64_homogeneity_tol(name, s),
        );
        println!("{name}{a:?}: {w:.1e} over {n} exponents ({nans} NaN or 0 at the top)");
        failures.extend(bad.into_iter().map(|b| format!("{name}{a:?} at {b}")));
    }
    assert_no_failures(failures);
}

/// The same for `Strategy::Adaptive`, up to `f64::MAX`, tolerance
/// [`DF64_HOMOGENEITY_TOL`], no NaN anywhere.
#[cfg(feature = "xprec")]
#[test]
fn df64_homogeneity() {
    let mut failures = Vec::new();
    for (name, a) in homogeneity_bases() {
        let d: Vec<Df64> = a.iter().map(|&v| Df64::from(v)).collect();
        let base = eval(name, &d);
        assert!(base.is_finite(), "{name}{a:?} = {base:?}");
        let (w, n, _, bad) =
            check_homogeneity(name, &a, base, |_| Allow::Value, |_| DF64_HOMOGENEITY_TOL);
        println!("{name}{a:?}: {w:.1e} over {n} exponents");
        failures.extend(bad.into_iter().map(|b| format!("{name}{a:?} at {b}")));
    }
    assert_no_failures(failures);
}

/// Beyond the documented NaN edge: spreads of more than 2^600, which the
/// `f64` R_J and R_G do not scale; they get xsf's value, NaN where its
/// products leave the range (and, for R_G, 0 where they underflow:
/// `docs/upstream-issues.md`, "`elliprj` and `elliprg` are NaN or 0 ...").
/// At every exponent the result must be one of those or the homogeneous
/// value, never a wrong nonzero finite one (the retry arm of the review of
/// 470803a returned values off by 13 orders of magnitude here).
#[test]
fn f64_homogeneity_beyond_spread_600() {
    let mut failures = Vec::new();
    for (name, a) in beyond_600_bases() {
        let base = eval(name, &a);
        assert!(base.is_finite(), "{name}{a:?} = {base:e}");
        let allow = if name == "elliprg" {
            Allow::NanZeroOrValue
        } else {
            Allow::NanOrValue
        };
        let (w, n, nans, bad) =
            check_homogeneity(name, &a, base, |_| allow, |s| f64_homogeneity_tol(name, s));
        println!("{name}{a:?}: {w:.1e} over {n} exponents ({nans} NaN or 0)");
        failures.extend(bad.into_iter().map(|b| format!("{name}{a:?} at {b}")));
    }
    assert_no_failures(failures);
}

/// The `Df64` R_J returns NaN beyond an argument ratio of 2^600
/// (`RJ_RATIO_BITS`) by design, at every scale; the `Df64` R_G has no ratio
/// limit and must be homogeneous there.
#[cfg(feature = "xprec")]
#[test]
fn df64_homogeneity_beyond_spread_600() {
    let mut failures = Vec::new();
    for (name, a) in beyond_600_bases() {
        let d: Vec<Df64> = a.iter().map(|&v| Df64::from(v)).collect();
        if name == "elliprj" {
            // The exponents of the f64 value: the range of the arguments and
            // the result is the same.
            let es = exponents(name, &a, eval(name, &a));
            assert!(!es.is_empty(), "{name}{a:?}");
            for e in es {
                let s: Vec<Df64> = d.iter().map(|&v| v * Df64::from(pow2(e))).collect();
                let g = eval(name, &s);
                if !g.is_nan() {
                    failures.push(format!("{name}{a:?} at 2^{e}: {:e}, want NaN", g.hi()));
                }
            }
            println!("{name}{a:?}: NaN at every exponent");
        } else {
            let base = eval(name, &d);
            let (w, n, _, bad) =
                check_homogeneity(name, &a, base, |_| Allow::Value, |_| DF64_HOMOGENEITY_TOL);
            println!("{name}{a:?}: {w:.1e} over {n} exponents");
            failures.extend(bad.into_iter().map(|b| format!("{name}{a:?} at {b}")));
        }
    }
    assert_no_failures(failures);
}
