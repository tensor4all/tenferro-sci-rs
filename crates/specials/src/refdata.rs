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
