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

pub(crate) mod beta;
pub(crate) mod digamma;
pub(crate) mod gamma;
#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
pub(crate) mod tables;
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

/// `a * b` that follows IEEE rules for infinite operands. Double-double
/// multiplication computes an error term `fma(a, b, -ab)`, which is
/// `inf - inf = NaN` when the product is infinite.
pub(crate) fn mul<T: Real>(a: T, b: T) -> T {
    if a.is_infinite() || b.is_infinite() {
        T::from_dd(a.approx_f64() * b.approx_f64(), 0.0)
    } else {
        a * b
    }
}

/// `a / b` with IEEE rules for infinite operands (see [`mul`]).
pub(crate) fn div<T: Real>(a: T, b: T) -> T {
    if a.is_infinite() || b.is_infinite() {
        T::from_dd(a.approx_f64() / b.approx_f64(), 0.0)
    } else {
        a / b
    }
}

/// `a + b` with IEEE rules for infinite operands (see [`mul`]).
pub(crate) fn add<T: Real>(a: T, b: T) -> T {
    if a.is_infinite() || b.is_infinite() {
        T::from_dd(a.approx_f64() + b.approx_f64(), 0.0)
    } else {
        a + b
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
