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
        "ellipk" => (include_str!("../tests/data/ellipk.csv"), 1),
        "ellipkm1" => (include_str!("../tests/data/ellipkm1.csv"), 1),
        "ellipe" => (include_str!("../tests/data/ellipe.csv"), 1),
        "ellipkinc" => (include_str!("../tests/data/ellipkinc.csv"), 2),
        "ellipeinc" => (include_str!("../tests/data/ellipeinc.csv"), 2),
        "ellipj_sn" => (include_str!("../tests/data/ellipj_sn.csv"), 2),
        "ellipj_cn" => (include_str!("../tests/data/ellipj_cn.csv"), 2),
        "ellipj_dn" => (include_str!("../tests/data/ellipj_dn.csv"), 2),
        "ellipj_ph" => (include_str!("../tests/data/ellipj_ph.csv"), 2),
        "elliprc" => (include_str!("../tests/data/elliprc.csv"), 2),
        "elliprf" => (include_str!("../tests/data/elliprf.csv"), 3),
        "elliprd" => (include_str!("../tests/data/elliprd.csv"), 3),
        "elliprj" => (include_str!("../tests/data/elliprj.csv"), 4),
        "elliprg" => (include_str!("../tests/data/elliprg.csv"), 3),
        "agm" => (include_str!("../tests/data/agm.csv"), 2),
        "cosm1" => (include_str!("../tests/data/cosm1.csv"), 1),
        "log1pmx" => (include_str!("../tests/data/log1pmx.csv"), 1),
        "xlogy" => (include_str!("../tests/data/xlogy.csv"), 2),
        "xlog1py" => (include_str!("../tests/data/xlog1py.csv"), 2),
        "logit" => (include_str!("../tests/data/logit.csv"), 1),
        "expit" => (include_str!("../tests/data/expit.csv"), 1),
        "log_expit" => (include_str!("../tests/data/log_expit.csv"), 1),
        "sinpi" => (include_str!("../tests/data/sinpi.csv"), 1),
        "cospi" => (include_str!("../tests/data/cospi.csv"), 1),
        "sindg" => (include_str!("../tests/data/sindg.csv"), 1),
        "cosdg" => (include_str!("../tests/data/cosdg.csv"), 1),
        "tandg" => (include_str!("../tests/data/tandg.csv"), 1),
        "cotdg" => (include_str!("../tests/data/cotdg.csv"), 1),
        "radian" => (include_str!("../tests/data/radian.csv"), 3),
        "exp2" => (include_str!("../tests/data/exp2.csv"), 1),
        "exp10" => (include_str!("../tests/data/exp10.csv"), 1),
        "cbrt" => (include_str!("../tests/data/cbrt.csv"), 1),
        "expm1" => (include_str!("../tests/data/expm1.csv"), 1),
        "log1p" => (include_str!("../tests/data/log1p.csv"), 1),
        "eval_jacobi_int" => (include_str!("../tests/data/eval_jacobi_int.csv"), 5),
        "eval_sh_jacobi_int" => (include_str!("../tests/data/eval_sh_jacobi_int.csv"), 5),
        "eval_gegenbauer_int" => (include_str!("../tests/data/eval_gegenbauer_int.csv"), 4),
        "eval_chebyt_int" => (include_str!("../tests/data/eval_chebyt_int.csv"), 3),
        "eval_chebyu_int" => (include_str!("../tests/data/eval_chebyu_int.csv"), 3),
        "eval_chebyc_int" => (include_str!("../tests/data/eval_chebyc_int.csv"), 3),
        "eval_chebys_int" => (include_str!("../tests/data/eval_chebys_int.csv"), 3),
        "eval_sh_chebyt_int" => (include_str!("../tests/data/eval_sh_chebyt_int.csv"), 3),
        "eval_sh_chebyu_int" => (include_str!("../tests/data/eval_sh_chebyu_int.csv"), 3),
        "eval_legendre_int" => (include_str!("../tests/data/eval_legendre_int.csv"), 3),
        "eval_sh_legendre_int" => (include_str!("../tests/data/eval_sh_legendre_int.csv"), 3),
        "eval_laguerre_int" => (include_str!("../tests/data/eval_laguerre_int.csv"), 3),
        "eval_genlaguerre_int" => (include_str!("../tests/data/eval_genlaguerre_int.csv"), 4),
        "eval_hermite" => (include_str!("../tests/data/eval_hermite.csv"), 3),
        "eval_hermitenorm" => (include_str!("../tests/data/eval_hermitenorm.csv"), 3),
        "eval_jacobi" => (include_str!("../tests/data/eval_jacobi.csv"), 4),
        "eval_sh_jacobi" => (include_str!("../tests/data/eval_sh_jacobi.csv"), 4),
        "eval_gegenbauer" => (include_str!("../tests/data/eval_gegenbauer.csv"), 3),
        "eval_chebyt" => (include_str!("../tests/data/eval_chebyt.csv"), 2),
        "eval_chebyu" => (include_str!("../tests/data/eval_chebyu.csv"), 2),
        "eval_chebyc" => (include_str!("../tests/data/eval_chebyc.csv"), 2),
        "eval_chebys" => (include_str!("../tests/data/eval_chebys.csv"), 2),
        "eval_sh_chebyt" => (include_str!("../tests/data/eval_sh_chebyt.csv"), 2),
        "eval_sh_chebyu" => (include_str!("../tests/data/eval_sh_chebyu.csv"), 2),
        "eval_legendre" => (include_str!("../tests/data/eval_legendre.csv"), 2),
        "eval_sh_legendre" => (include_str!("../tests/data/eval_sh_legendre.csv"), 2),
        "eval_laguerre" => (include_str!("../tests/data/eval_laguerre.csv"), 2),
        "eval_genlaguerre" => (include_str!("../tests/data/eval_genlaguerre.csv"), 3),
        "pmv_int" => (include_str!("../tests/data/pmv_int.csv"), 4),
        "pmv" => (include_str!("../tests/data/pmv.csv"), 3),
        "assoc_legendre_p" => (include_str!("../tests/data/assoc_legendre_p.csv"), 4),
        "assoc_legendre_p_t3" => (include_str!("../tests/data/assoc_legendre_p_t3.csv"), 4),
        "assoc_legendre_p_norm" => (include_str!("../tests/data/assoc_legendre_p_norm.csv"), 4),
        "assoc_legendre_p_norm_t3" => (
            include_str!("../tests/data/assoc_legendre_p_norm_t3.csv"),
            4,
        ),
        "sph_legendre_p" => (include_str!("../tests/data/sph_legendre_p.csv"), 4),
        "lqn" => (include_str!("../tests/data/lqn.csv"), 2),
        "lqn_d" => (include_str!("../tests/data/lqn_d.csv"), 2),
        "lqmn" => (include_str!("../tests/data/lqmn.csv"), 3),
        "lqmn_d" => (include_str!("../tests/data/lqmn_d.csv"), 3),
        _ => panic!("no reference data for {name}"),
    }
}

/// The rows of a function.
pub fn rows(name: &str) -> Vec<Row> {
    let (text, nargs) = data(name);
    parse(text, nargs)
}

/// SciPy's special values: `(function, args, value)`; a row has up to four arguments.
pub fn special_values() -> Vec<(String, Vec<f64>, f64)> {
    include_str!("../tests/data/special_values.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let args = f[1..5]
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| num(s))
                .collect();
            (f[0].to_string(), args, num(f[5]))
        })
        .collect()
}

/// The exact results of the elementary helpers (#19):
/// `(function, args, expected, scipy)`, with signed zeros. `expected` is
/// SciPy's value except where xsf's own tests fix a sign after SciPy 1.16.3
/// (see `scripts/gen_specials_reference.py`).
pub fn elementary_special_values() -> Vec<(String, Vec<f64>, f64, f64)> {
    include_str!("../tests/data/elementary_special_values.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let args = f[1..4]
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| num(s))
                .collect();
            (f[0].to_string(), args, num(f[4]), num(f[5]))
        })
        .collect()
}

/// The exact results of the orthogonal polynomials and Legendre functions
/// (#16): `(function, args, expected, scipy)`.
pub fn orthogonal_special_values() -> Vec<(String, Vec<f64>, f64, f64)> {
    include_str!("../tests/data/orthogonal_special_values.csv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let args = f[1..5]
                .iter()
                .filter(|s| !s.is_empty())
                .map(|s| num(s))
                .collect();
            (f[0].to_string(), args, num(f[5]), num(f[6]))
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

/// Size of the absolute error that the phase of `sn`, `cn` and `dn` allows:
/// `max(1, |u| √max(1, 1 - m)) max(1, |ref|)` for the Jacobi functions, 0 for
/// every other function.
///
/// The phase of `sn` and `cn` is `φ = a_N 2^N u` (`u √(1 - m)` after the
/// transformation of a negative `m`), computed with a relative error `ε`, so
/// near a zero of `sn` or `cn` the result has an absolute rather than a
/// relative error of about `ε` times this scale; `dn` carries the factor
/// `max(1, |dn|)`. Callers turn `C ε scale` into the `floor` of [`error`]
/// for their relative tolerance `tol` with [`jacobi_floor`].
pub fn jacobi_scale(name: &str, u: f64, m: f64, reference: f64) -> f64 {
    if !matches!(name, "ellipj_sn" | "ellipj_cn" | "ellipj_dn") {
        return 0.0;
    }
    (u.abs() * (1.0 - m).max(1.0).sqrt()).max(1.0) * reference.abs().max(1.0)
}

/// The `floor` argument of [`error`] that allows an absolute error of `c eps
/// scale` next to the relative tolerance `tol`: `error <= tol` then means
/// `|got - ref| <= tol max(|ref|, floor)`.
pub fn jacobi_floor(scale: f64, c: f64, eps: f64, tol: f64) -> f64 {
    c * eps * scale / tol
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

/// Column of the xsref tolerance to use: `gcc-linux-x86_64`, on every platform.
///
/// xsref's tolerances record the error xsf itself showed on each platform.
/// Clang on aarch64 contracts `a*b + c` into FMA; gcc on x86-64 (no `-mfma`)
/// does not, and neither does Rust. The port therefore rounds like xsf built
/// by gcc on x86-64, and its errors match that column (for example
/// beta(2154.43…, 7.5): 2.2870e-12 here and in the column). A row that fails it
/// on a platform whose libm is not correctly rounded is a known deviation
/// (`docs/upstream-issues.md`), not a reason to loosen the bound.
const TOL_COLUMN: usize = 1;

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
        "ellipk" => (include_str!("../tests/data/xsref/ellipk.csv"), 1),
        "ellipkm1" => (include_str!("../tests/data/xsref/ellipkm1.csv"), 1),
        "ellipe" => (include_str!("../tests/data/xsref/ellipe.csv"), 1),
        "ellipkinc" => (include_str!("../tests/data/xsref/ellipkinc.csv"), 2),
        "ellipeinc" => (include_str!("../tests/data/xsref/ellipeinc.csv"), 2),
        "ellipj_sn" => (include_str!("../tests/data/xsref/ellipj_sn.csv"), 2),
        "ellipj_cn" => (include_str!("../tests/data/xsref/ellipj_cn.csv"), 2),
        "ellipj_dn" => (include_str!("../tests/data/xsref/ellipj_dn.csv"), 2),
        "ellipj_ph" => (include_str!("../tests/data/xsref/ellipj_ph.csv"), 2),
        "cosm1" => (include_str!("../tests/data/xsref/cosm1.csv"), 1),
        "log1pmx" => (include_str!("../tests/data/xsref/log1pmx.csv"), 1),
        "xlogy" => (include_str!("../tests/data/xsref/xlogy.csv"), 2),
        "xlog1py" => (include_str!("../tests/data/xsref/xlog1py.csv"), 2),
        "logit" => (include_str!("../tests/data/xsref/logit.csv"), 1),
        "expit" => (include_str!("../tests/data/xsref/expit.csv"), 1),
        "log_expit" => (include_str!("../tests/data/xsref/log_expit.csv"), 1),
        "sinpi" => (include_str!("../tests/data/xsref/sinpi.csv"), 1),
        "cospi" => (include_str!("../tests/data/xsref/cospi.csv"), 1),
        "sindg" => (include_str!("../tests/data/xsref/sindg.csv"), 1),
        "cosdg" => (include_str!("../tests/data/xsref/cosdg.csv"), 1),
        "tandg" => (include_str!("../tests/data/xsref/tandg.csv"), 1),
        "cotdg" => (include_str!("../tests/data/xsref/cotdg.csv"), 1),
        "radian" => (include_str!("../tests/data/xsref/radian.csv"), 3),
        "round" => (include_str!("../tests/data/xsref/round.csv"), 1),
        "exp2" => (include_str!("../tests/data/xsref/exp2.csv"), 1),
        "exp10" => (include_str!("../tests/data/xsref/exp10.csv"), 1),
        "cbrt" => (include_str!("../tests/data/xsref/cbrt.csv"), 1),
        "expm1" => (include_str!("../tests/data/xsref/expm1.csv"), 1),
        "log1p" => (include_str!("../tests/data/xsref/log1p.csv"), 1),
        "pmv" => (include_str!("../tests/data/xsref/pmv.csv"), 3),
        _ => panic!("no xsref table for {name}"),
    };
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
                // Columns n+2..n+5: clang-darwin-aarch64, gcc-linux-x86_64, other.
                tol: num(f[n + 2 + TOL_COLUMN]),
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
