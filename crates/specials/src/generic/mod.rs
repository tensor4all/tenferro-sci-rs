//! Precision-adaptive algorithms, generic over [`Real`](crate::Real).
//!
//! These are written independently from the formulas in the NIST Digital
//! Library of Mathematical Functions (DLMF); no code was translated. They
//! truncate every series at `Float::epsilon()` of the scalar type, and their
//! constants come from double-double tables generated with mpmath
//! (`scripts/gen_specials_reference.py`), so the same code serves `f64` and
//! double-double. The structure of the special cases (poles, signed zeros,
//! negative integers) follows SciPy's conventions so that both strategies
//! return the same non-finite values.

pub(crate) mod agm;
pub(crate) mod beta;
pub(crate) mod binom;
pub(crate) mod carlson;
pub(crate) mod digamma;
pub(crate) mod digammainv;
pub(crate) mod elementary;
pub(crate) mod ellip;
pub(crate) mod gamma;
pub(crate) mod gen_harmonic;
pub(crate) mod legendre;
pub(crate) mod lgam1p;
pub(crate) mod orthogonal;
pub(crate) mod poch;
pub(crate) mod stirling2;
#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub(crate) mod tables;
pub(crate) mod trigamma;
pub(crate) mod zeta;

use crate::real::{k, Real};

/// `sin(πx)` with exact reduction modulo 2.
pub(crate) fn sinpi<T: Real>(x: T) -> T {
    let (r, flip) = reduce_half(x);
    let s = (T::PI() * r).sin();
    if flip {
        -s
    } else {
        s
    }
}

/// `cot(πx)`.
pub(crate) fn cotpi<T: Real>(x: T) -> T {
    // cot has period 1: reduce to [-1/2, 1/2].
    let r = x - x.round();
    let pr = T::PI() * r;
    pr.cos() / pr.sin()
}

/// Write `x = n + r` with `n` an integer and `|r| <= 1/2`, and return
/// `(r, n odd)`, so that `sin(πx) = ± sin(πr)`. `x - n` is exact.
fn reduce_half<T: Real>(x: T) -> (T, bool) {
    let n = x.round();
    let r = x - n;
    (r, (n / k::<T>(2.0)).fract() != T::zero())
}

/// `a * b` that follows IEEE rules for infinite operands and overflow.
/// Double-double multiplication computes an error term `fma(a, b, -ab)`,
/// which is `inf - inf = NaN` when the product is infinite.
pub(crate) fn mul<T: Real>(a: T, b: T) -> T {
    let r = a * b;
    if r.is_nan() && !a.is_nan() && !b.is_nan() {
        // An infinite operand, or a result that overflowed: the error term is NaN.
        T::from_dd(a.approx_f64() * b.approx_f64(), 0.0)
    } else {
        r
    }
}

/// `a / b` with IEEE rules for infinite operands (see [`mul`]).
///
/// Double-double division goes through `1/b`; for `|b| > 2^900` that
/// reciprocal has a subnormal low word and loses digits, so both operands are
/// first scaled by the same exact power of two.
pub(crate) fn div<T: Real>(a: T, b: T) -> T {
    let bh = b.approx_f64().abs();
    let (a, b) = if bh.is_finite() && bh > 2.0_f64.powi(900) {
        let s = k::<T>(2.0_f64.powi(-600));
        (a * s, b * s)
    } else {
        (a, b)
    };
    let r = a / b;
    if r.is_nan() && !a.is_nan() && !b.is_nan() {
        // An infinite operand, or a result that overflowed: the error term is NaN.
        T::from_dd(a.approx_f64() / b.approx_f64(), 0.0)
    } else {
        r
    }
}

/// `a + b` with IEEE rules for infinite operands (see [`mul`]).
pub(crate) fn add<T: Real>(a: T, b: T) -> T {
    let r = a + b;
    if r.is_nan() && !a.is_nan() && !b.is_nan() {
        // An infinite operand, or a result that overflowed: the error term is NaN.
        T::from_dd(a.approx_f64() + b.approx_f64(), 0.0)
    } else {
        r
    }
}

/// `ln x` that is also correct for `x < 2^-900` and for finite `x >= 2^1023`,
/// where the double-double logarithm is not (xprec 0.2.3 returns NaN from
/// 2^1023 up, and below about 1e-290 its Newton step evaluates `exp` with a
/// subnormal low word: `ln(1e-300)` is off by 1.2e-27; see
/// `docs/upstream-issues.md`): scale into the normal range by an exact power
/// of two.
pub(crate) fn ln<T: Real>(x: T) -> T {
    let xf = x.abs().approx_f64();
    if x > T::zero() && xf < 2.0_f64.powi(-900) {
        let scale = k::<T>(2.0_f64.powi(600));
        (x * scale).ln() - k::<T>(600.0) * T::LN_2()
    } else if x > T::zero() && xf.is_finite() && xf > 2.0_f64.powi(1000) {
        let scale = k::<T>(2.0_f64.powi(-600));
        (x * scale).ln() + k::<T>(600.0) * T::LN_2()
    } else {
        x.ln()
    }
}

/// `x 2^e`, for any `e` (in steps that are exact powers of two).
pub(crate) fn ldexp<T: Real>(mut x: T, mut e: i32) -> T {
    while e != 0 {
        let step = e.clamp(-1000, 1000);
        x = mul(x, k::<T>(2.0_f64.powi(step)));
        e -= step;
    }
    x
}

/// `√x`, also accurate below 2^-900, where the error term of the
/// double-double square root underflows: scale by an exact power of four.
pub(crate) fn sqrt<T: Real>(x: T) -> T {
    let h = x.approx_f64();
    if h > 0.0 && h < 2.0_f64.powi(-900) {
        ldexp(ldexp(x, 1000).sqrt(), -500)
    } else {
        x.sqrt()
    }
}

/// Sum `Σ coef[i] z^(i+1)` until a term drops below `eps |sum|`.
pub(crate) fn power_series<T: Real>(coef: &[(f64, f64)], z: T, init: T) -> T {
    let eps = T::epsilon();
    let mut sum = init;
    let mut zn = z;
    for &cf in coef {
        let term = crate::real::c::<T>(cf) * zn;
        sum = sum + term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
        zn = zn * z;
    }
    sum
}
