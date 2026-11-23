//! Reference data of `tests/data` and the comparison used by the tests.
//!
//! Compiled only for tests: as a module of the library (`cfg(test)`) and,
//! through `#[path]`, in the integration tests.

#![allow(dead_code)]

use crate::Real;

/// One reference point.
#[derive(Debug, Clone)]
pub struct Row {
    /// Arguments as double-double pairs.
    pub args: Vec<(f64, f64)>,
    /// mpmath value as a double-double pair.
    pub reference: (f64, f64),
    /// SciPy's value, present for f64 rows.
    pub scipy: Option<f64>,
    /// 1-based line in the CSV file, for messages.
    pub line: usize,
}

impl Row {
    /// True when every argument is an `f64`.
    pub fn is_f64(&self) -> bool {
        self.args.iter().all(|&(_, lo)| lo == 0.0)
    }

    /// The arguments as `T`.
    pub fn args_as<T: Real>(&self) -> Vec<T> {
        self.args.iter().map(|&(h, l)| T::from_dd(h, l)).collect()
    }
}

fn num(s: &str) -> f64 {
    s.trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("bad number {s:?}: {e}"))
}

/// Parse a reference CSV with `nargs` arguments.
pub fn parse(text: &str, nargs: usize) -> Vec<Row> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.starts_with('#') && !l.trim().is_empty())
        .map(|(i, l)| {
            let f: Vec<&str> = l.split(',').collect();
            assert_eq!(f.len(), 2 * nargs + 3, "line {}: {l}", i + 1);
            let args = (0..nargs)
                .map(|j| (num(f[2 * j]), num(f[2 * j + 1])))
                .collect();
            let r = 2 * nargs;
            Row {
                args,
                reference: (num(f[r]), num(f[r + 1])),
                scipy: (!f[r + 2].trim().is_empty()).then(|| num(f[r + 2])),
                line: i + 1,
            }
        })
        .collect()
}

/// A reference file by function name.
pub fn data(name: &str) -> (&'static str, usize) {
    match name {
        "gamma" => (include_str!("../tests/data/gamma.csv"), 1),
        "lgamma" => (include_str!("../tests/data/lgamma.csv"), 1),
        "rgamma" => (include_str!("../tests/data/rgamma.csv"), 1),
        "digamma" => (include_str!("../tests/data/digamma.csv"), 1),
        "zeta" => (include_str!("../tests/data/zeta.csv"), 2),
        "beta" => (include_str!("../tests/data/beta.csv"), 2),
        "lbeta" => (include_str!("../tests/data/lbeta.csv"), 2),
        "poch" => (include_str!("../tests/data/poch.csv"), 2),
        "lgam1p" => (include_str!("../tests/data/lgam1p.csv"), 1),
        "binom" => (include_str!("../tests/data/binom.csv"), 2),
        "zetac" => (include_str!("../tests/data/zetac.csv"), 1),
        "trigamma" => (include_str!("../tests/data/trigamma.csv"), 1),
        "digammainv" => (include_str!("../tests/data/digammainv.csv"), 1),
        "gen_harmonic" => (include_str!("../tests/data/gen_harmonic.csv"), 2),
        "stirling2" => (include_str!("../tests/data/stirling2.csv"), 2),
        _ => panic!("no reference data for {name}"),
    }
}

/// The rows of a function.
pub fn rows(name: &str) -> Vec<Row> {
    let (text, nargs) = data(name);
    parse(text, nargs)
}

/// SciPy's special values: `(function, args, value)`.
pub fn special_values() -> Vec<(String, Vec<f64>, f64)> {
    include_str!("../tests/data/special_values.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let args = f[1..3]
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| num(s))
                .collect();
            (f[0].to_string(), args, num(f[3]))
        })
        .collect()
}

/// Error of `got` against `reference`: `|got - ref| / max(|ref|, floor)`.
///
/// Non-finite references must be matched exactly (same infinity, or NaN), and
/// references below `f64::MIN_POSITIVE` only require `|got| <= MIN_POSITIVE`
/// (underflow); both return 0 on success and +∞ on failure. `floor` turns the
/// criterion absolute near zeros of the function.
pub fn error<T: Real>(got: T, reference: (f64, f64), floor: f64) -> f64 {
    let r = T::from_dd(reference.0, reference.1);
    if !reference.0.is_finite() {
        let ok = (reference.0.is_nan() && got.is_nan()) || got == r;
        return if ok { 0.0 } else { f64::INFINITY };
    }
    if reference.0.abs() < f64::MIN_POSITIVE {
        return if got.abs().approx_f64() <= f64::MIN_POSITIVE {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if !got.is_finite() {
        return f64::INFINITY;
    }
    // Below ~1e-292 the low word of a double-double is subnormal, so its
    // precision is limited by the subnormal spacing 2^-1074: accept a
    // difference of a few such units.
    let diff = (got - r).abs();
    if diff.approx_f64() <= 4.0 * f64::from_bits(1) {
        return 0.0;
    }
    let denom = r.abs().max(T::from_dd(floor, 0.0));
    (diff / denom).approx_f64()
}

/// Same-value check for special values: NaN matches NaN, zeros must have the same sign.
pub fn same<T: Real>(got: T, expected: f64) -> bool {
    if expected.is_nan() {
        return got.is_nan();
    }
    let g = got.approx_f64();
    g == expected && g.is_sign_negative() == expected.is_sign_negative()
}

/// One row of an xsref table (`tests/data/xsref`).
#[derive(Debug, Clone)]
pub struct XsrefRow {
    /// Arguments (f64).
    pub args: Vec<f64>,
    /// xsref's reference value, rounded to f64.
    pub desired: f64,
    /// True where xsref used an older SciPy instead of an arbitrary-precision reference.
    pub fallback: bool,
    /// xsref's tolerance for this platform.
    pub tol: f64,
    /// mpmath value as a double-double pair, if defined.
    pub dd: Option<(f64, f64)>,
    /// 1-based line in the CSV file.
    pub line: usize,
}

/// Column of the xsref tolerance to use: `gcc-linux-x86_64` on every platform.
///
/// xsref's tolerances record the error xsf itself showed on each platform.
/// Clang on aarch64 contracts `a*b + c` into FMA; gcc on x86-64 (no `-mfma`)
/// does not, and neither does Rust. The port therefore rounds like xsf built
/// by gcc on x86-64, and its errors match that column (for example
/// beta(2154.43…, 7.5): 2.2870e-12 here and in the column).
fn platform_column() -> usize {
    1
}

/// xsref's tolerance of a row (`cols`: the `clang-darwin-aarch64`,
/// `gcc-linux-x86_64` and `other` columns).
///
/// The `gcc-linux-x86_64` column is authoritative (and what CI on Linux runs).
/// Some rows depend on the platform's libm, not only on the algorithm: for
/// example `binom(458.139…, 56.00000000000006)` evaluates `exp` of a
/// difference of ln Γ values of size 1.5e3, so one ulp of difference in a libm
/// `log`/`pow` moves the result by 2e-13. On macOS the libm differs from
/// glibc, and the port's error on such rows equals xsf's own error on
/// `clang-darwin-aarch64`; there the larger of the two columns applies.
fn tolerance(cols: &[&str], col: usize) -> f64 {
    let t = num(cols[col]);
    if cfg!(target_os = "macos") {
        t.max(num(cols[0]))
    } else {
        t
    }
}

/// The xsref rows of a function.
pub fn xsref_rows(name: &str) -> Vec<XsrefRow> {
    let (text, nargs) = match name {
        "gamma" => (include_str!("../tests/data/xsref/gamma.csv"), 1),
        "lgamma" => (include_str!("../tests/data/xsref/lgamma.csv"), 1),
        "rgamma" => (include_str!("../tests/data/xsref/rgamma.csv"), 1),
        "gammasgn" => (include_str!("../tests/data/xsref/gammasgn.csv"), 1),
        "digamma" => (include_str!("../tests/data/xsref/digamma.csv"), 1),
        "zeta" => (include_str!("../tests/data/xsref/zeta.csv"), 2),
        "beta" => (include_str!("../tests/data/xsref/beta.csv"), 2),
        "lbeta" => (include_str!("../tests/data/xsref/lbeta.csv"), 2),
        "poch" => (include_str!("../tests/data/xsref/poch.csv"), 2),
        "lgam1p" => (include_str!("../tests/data/xsref/lgam1p.csv"), 1),
        "binom" => (include_str!("../tests/data/xsref/binom.csv"), 2),
        "zetac" => (include_str!("../tests/data/xsref/zetac.csv"), 1),
        _ => panic!("no xsref table for {name}"),
    };
    let col = platform_column();
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.starts_with('#') && !l.trim().is_empty())
        .map(|(i, l)| {
            let f: Vec<&str> = l.split(',').collect();
            assert_eq!(f.len(), nargs + 7, "line {}: {l}", i + 1);
            let n = nargs;
            XsrefRow {
                args: f[..n].iter().map(|s| num(s)).collect(),
                desired: num(f[n]),
                fallback: f[n + 1] == "1",
                tol: tolerance(&f[n + 2..n + 5], col),
                dd: (!f[n + 5].is_empty()).then(|| (num(f[n + 5]), num(f[n + 6]))),
                line: i + 1,
            }
        })
        .collect()
}

/// xsf's `extended_relative_error` (ported from xsf
/// `include/xsf/fp_error_metrics.h`, BSD-3-Clause): NaN matches NaN, an
/// infinite value is compared with the "next float after MAX", and a zero
/// reference is normalized by the smallest subnormal.
pub fn extended_relative_error(actual: f64, desired: f64) -> f64 {
    let abs_err = if actual == desired || (actual.is_nan() && desired.is_nan()) {
        0.0
    } else if actual.is_nan() || desired.is_nan() {
        f64::INFINITY
    } else if actual.is_infinite() {
        let sgn = 1.0_f64.copysign(actual);
        let ulp = 2.0_f64.powi(-(f64::MANTISSA_DIGITS as i32)) * f64::MAX;
        ((sgn * f64::MAX - desired) + sgn * ulp).abs()
    } else if desired.is_infinite() {
        let sgn = 1.0_f64.copysign(desired);
        let ulp = 2.0_f64.powi(-(f64::MANTISSA_DIGITS as i32)) * f64::MAX;
        ((sgn * f64::MAX - actual) + sgn * ulp).abs()
    } else {
        (actual - desired).abs()
    };
    let abs_desired = if desired == 0.0 {
        f64::from_bits(1)
    } else if desired.is_infinite() {
        f64::MAX
    } else if desired.is_nan() {
        1.0
    } else {
        desired.abs()
    };
    abs_err / abs_desired
}

/// xsf's `adjust_tolerance` (`tests/testing_utils.h`): `4 max(eps, tol)`.
pub fn adjust_tolerance(tol: f64) -> f64 {
    4.0 * f64::EPSILON.max(tol)
}
