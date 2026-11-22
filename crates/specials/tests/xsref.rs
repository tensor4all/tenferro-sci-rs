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
//!   digamma, zeta; `2e-30` for lbeta; `5e-30 max(1, |ln B|)` for beta).
//! * `df64_matches_xsref_where_mpmath_is_undefined`: where mpmath has no value
//!   (poles, outside the domain, non-finite inputs), the `Df64` result rounded
//!   to `f64` must satisfy xsf's criterion against SciPy's value, i.e. the
//!   double-double path returns the same special values as SciPy.

#[path = "../src/refdata.rs"]
mod refdata;

use refdata::{adjust_tolerance, error, extended_relative_error, xsref_rows};
use specials::Real;
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
        _ => unreachable!("{name}"),
    }
}

const NAMES: [&str; 8] = [
    "gamma", "lgamma", "rgamma", "gammasgn", "digamma", "zeta", "beta", "lbeta",
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
fn dd_tolerance(name: &str, args: &[f64]) -> f64 {
    match name {
        "gamma" | "lgamma" | "rgamma" | "digamma" | "zeta" => 1e-29,
        "lbeta" => 2e-30,
        // beta is computed as exp(ln B): the relative error of B grows like
        // |ln B| eps, so the tolerance scales with max(1, |ln B|).
        "beta" => 5e-30 * specials::lbeta(args[0], args[1]).abs().max(1.0),
        _ => unreachable!("{name}"),
    }
}

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
            let e = error(got, d, 0.0);
            let tol = dd_tolerance(name, &r.args);
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
