//! xsf's own test tables (scipy/xsref), imported by `scripts/import_xsref.py`.
//!
//! Three tests check every row of every table:
//!
//! * `f64_matches_xsref`: the `f64` result satisfies xsf's own criterion from
//!   its test suite (`tests/testing_utils.h`, `xsf/fp_error_metrics.h`):
//!   `extended_relative_error(got, desired) <= adjust_tolerance(tol)`, where
//!   `tol` is xsref's per-row tolerance (the error xsf itself shows) and
//!   `adjust_tolerance` is `4 max(eps, tol)`.
//! * `df64_matches_mpmath_on_xsref_points`: where xsref provides an mpmath
//!   value as a double-double (`dd`), the `Df64` result must agree with it to
//!   a per-function relative tolerance (`1e-29` for gamma, lgamma, rgamma,
//!   digamma, zeta; `2e-30` for lbeta; `5e-30 max(1, |ln B|)` for beta; see
//!   `dd_tolerance` for the rest, including the elliptic family).
//! * `df64_matches_xsref_where_mpmath_is_undefined`: where mpmath has no value
//!   (poles, outside the domain, non-finite inputs), the `Df64` result rounded
//!   to `f64` must satisfy xsf's criterion against SciPy's value, i.e. the
//!   double-double path returns the same special values as SciPy.
//!
//! `f64_matches_xsref` and `tolerance_is_the_gcc_column_on_every_platform` run
//! in every configuration; the two `Df64` tests need the `xprec` feature (on by
//! default).

#[path = "../src/refdata.rs"]
mod refdata;

#[cfg(feature = "xprec")]
use refdata::error;
use refdata::{adjust_tolerance, extended_relative_error, xsref_rows};
use specials::Real;
#[cfg(feature = "xprec")]
use xprec::Df64;

fn eval<T: Real>(name: &str, a: &[T]) -> T {
    match name {
        "gamma" => specials::gamma(a[0]),
        "lgamma" => specials::lgamma(a[0]),
        "rgamma" => specials::rgamma(a[0]),
        "gammasgn" => specials::gammasgn(a[0]),
        "digamma" => specials::digamma(a[0]),
        "zeta" => specials::zeta(a[0], a[1]),
        "beta" => specials::beta(a[0], a[1]),
        "lbeta" => specials::lbeta(a[0], a[1]),
        "poch" => specials::poch(a[0], a[1]),
        "lgam1p" => specials::lgam1p(a[0]),
        "binom" => specials::binom(a[0], a[1]),
        "zetac" => specials::zetac(a[0]),
        "ellipk" => specials::ellipk(a[0]),
        "ellipkm1" => specials::ellipkm1(a[0]),
        "ellipe" => specials::ellipe(a[0]),
        "ellipkinc" => specials::ellipkinc(a[0], a[1]),
        "ellipeinc" => specials::ellipeinc(a[0], a[1]),
        "ellipj_sn" => specials::ellipj(a[0], a[1]).0,
        "ellipj_cn" => specials::ellipj(a[0], a[1]).1,
        "ellipj_dn" => specials::ellipj(a[0], a[1]).2,
        "ellipj_ph" => specials::ellipj(a[0], a[1]).3,
        "pmv" => specials::pmv(a[0], a[1], a[2]),
        _ => unreachable!("{name}"),
    }
}

const NAMES: [&str; 22] = [
    "gamma",
    "lgamma",
    "rgamma",
    "gammasgn",
    "digamma",
    "zeta",
    "beta",
    "lbeta",
    "poch",
    "lgam1p",
    "binom",
    "zetac",
    "ellipk",
    "ellipkm1",
    "ellipe",
    "ellipkinc",
    "ellipeinc",
    "ellipj_sn",
    "ellipj_cn",
    "ellipj_dn",
    "ellipj_ph",
    "pmv",
];

/// One failing row.
struct Failure {
    line: usize,
    args: Vec<f64>,
    got: f64,
    desired: f64,
    error: f64,
    tol: f64,
}

/// Collects failures per function and reports them all at the end.
#[derive(Default)]
struct Report {
    lines: Vec<String>,
    count: usize,
}

impl Report {
    fn add(&mut self, name: &str, fails: &[Failure]) {
        if fails.is_empty() {
            return;
        }
        self.count += fails.len();
        self.lines.push(format!("{name}: {} failures", fails.len()));
        for f in fails.iter().take(10) {
            self.lines.push(format!(
                "  line {} args {:?}: got {:e}, desired {:e}, error {:e}, tolerance {:e}",
                f.line, f.args, f.got, f.desired, f.error, f.tol
            ));
        }
    }

    fn assert_empty(&self, what: &str) {
        assert!(
            self.count == 0,
            "{what}: {} failures in total\n{}",
            self.count,
            self.lines.join("\n")
        );
    }
}

#[test]
fn f64_matches_xsref() {
    let mut report = Report::default();
    for name in NAMES {
        let mut fails = vec![];
        for r in xsref_rows(name) {
            let got = eval::<f64>(name, &r.args);
            let e = extended_relative_error(got, r.desired);
            let tol = adjust_tolerance(r.tol);
            if e.is_nan() || e > tol {
                fails.push(Failure {
                    line: r.line,
                    args: r.args.clone(),
                    got,
                    desired: r.desired,
                    error: e,
                    tol,
                });
            }
        }
        report.add(name, &fails);
    }
    report.assert_empty("f64 against xsref");
}

/// Tolerance of the `Df64` result against the mpmath value.
#[cfg(feature = "xprec")]
fn dd_tolerance(name: &str, args: &[f64]) -> f64 {
    match name {
        "gamma" | "lgamma" | "rgamma" | "digamma" | "zeta" => 1e-29,
        "lbeta" => 2e-30,
        // Worst measured over the xsref points: poch 1.1e-30, lgam1p 9.2e-31,
        // zetac 2.1e-30 (a 5x margin or more).
        "poch" | "lgam1p" | "zetac" => 1e-29,
        // binom: 1.4e-29 where B(a, b) of large negative a goes through ln Γ
        // (error ~ |ln Γ| eps); 5e-29 is 3.6x the worst.
        "binom" => 5e-29,
        // beta is computed as exp(ln B): the relative error of B grows like
        // |ln B| eps, so the tolerance scales with max(1, |ln B|).
        "beta" => 5e-30 * specials::lbeta(args[0], args[1]).abs().max(1.0),
        // Elliptic family (#11). Worst measured over the xsref points: ellipk 1.1e-30,
        // ellipkm1 6.5e-32, ellipe 4.6e-32, ellipkinc 1.1e-31, ellipeinc 1.1e-31,
        // ellipj_ph 4.7e-32. ellipj_sn, _cn and _dn also allow an absolute error
        // of 24 eps |u| sqrt(max(1, 1 - m)) (see `refdata::jacobi_scale`).
        "ellipk" => 5e-30,
        "ellipkm1" | "ellipe" | "ellipkinc" | "ellipeinc" => 1e-30,
        "ellipj_sn" | "ellipj_cn" | "ellipj_dn" => 1e-29,
        "ellipj_ph" => 5e-30,
        // pmv (#16). An integer degree is evaluated by the double-double
        // recurrence; a non-integer degree by the f64 kernel, whose series
        // stops at 1e-14.
        "pmv" if args[1] == args[1].floor() => 1e-28,
        "pmv" => 1e-9,
        _ => unreachable!("{name}"),
    }
}

#[cfg(feature = "xprec")]
#[test]
fn df64_matches_mpmath_on_xsref_points() {
    let mut report = Report::default();
    // gammasgn has no double-double values.
    for name in NAMES.into_iter().filter(|&n| n != "gammasgn") {
        let mut fails = vec![];
        for r in xsref_rows(name) {
            let Some(d) = r.dd else { continue };
            let a: Vec<Df64> = r.args.iter().map(|&x| Df64::from(x)).collect();
            let got = eval::<Df64>(name, &a);
            let tol = dd_tolerance(name, &r.args);
            // Absolute error 24 ε scale, ε = 2^-105 (measured worst 1.6 ε scale).
            let scale =
                refdata::jacobi_scale(name, r.args[0], r.args.get(1).copied().unwrap_or(0.0), d.0);
            let e = error(
                got,
                d,
                refdata::jacobi_floor(scale, 24.0, 2.0_f64.powi(-105), tol),
            );
            if e.is_nan() || e > tol {
                fails.push(Failure {
                    line: r.line,
                    args: r.args.clone(),
                    got: got.approx_f64(),
                    desired: d.0,
                    error: e,
                    tol,
                });
            }
        }
        report.add(name, &fails);
    }
    report.assert_empty("Df64 against mpmath");
}

#[cfg(feature = "xprec")]
#[test]
fn df64_matches_xsref_where_mpmath_is_undefined() {
    let mut report = Report::default();
    for name in NAMES {
        let mut fails = vec![];
        for r in xsref_rows(name) {
            // Rows with an infinite tolerance are known disagreements with
            // SciPy marked by xsref.
            if r.dd.is_some() || r.tol.is_infinite() {
                continue;
            }
            let a: Vec<Df64> = r.args.iter().map(|&x| Df64::from(x)).collect();
            let got = eval::<Df64>(name, &a).approx_f64();
            let e = extended_relative_error(got, r.desired);
            let tol = adjust_tolerance(r.tol);
            if e.is_nan() || e > tol {
                fails.push(Failure {
                    line: r.line,
                    args: r.args.clone(),
                    got,
                    desired: r.desired,
                    error: e,
                    tol,
                });
            }
        }
        report.add(name, &fails);
    }
    report.assert_empty("Df64 against xsref where mpmath is undefined");
}

/// The tolerance is xsref's `gcc-linux-x86_64` column on every platform (issue
/// #8, `lib.rs`). A macOS-only `max` with the `clang-darwin-aarch64` column used
/// to loosen 1,259 rows across all twelve tables.
#[test]
fn tolerance_is_the_gcc_column_on_every_platform() {
    let row = xsref_rows("binom")
        .into_iter()
        .find(|r| r.line == 2030)
        .expect("binom.csv line 2030");
    // Columns: clang-darwin-aarch64 4.3956158062363253e-13, gcc-linux-x86_64 1.5231444905425505e-14.
    assert_eq!(row.tol, 1.5231444905425505e-14);
}
