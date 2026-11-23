//! Arithmetic-geometric mean, precision-adaptive.
//!
//! Written from the iteration `a' = (a + b)/2`, `b' = √(ab)` (DLMF 19.8.1),
//! which converges quadratically (DLMF 19.8.2): the loop stops when the two
//! means agree to the epsilon of the scalar type, at which point their
//! arithmetic mean has an error of order `ε²`. The conventions for zeros,
//! infinities and signs are those of `scipy.special.agm`.

use super::{add, div, ldexp, mul, sqrt};
use crate::real::{k, Real};

/// The AGM of `a, b > 0`, both finite.
///
/// The arithmetic mean is `a/2 + b/2` and the geometric mean `√a √b`, so that
/// neither overflows or underflows unless the result does.
pub(crate) fn agm_positive<T: Real>(a: T, b: T) -> T {
    // agm(2^-s a, 2^-s b) = 2^-s agm(a, b): center the geometric mean at 1, so
    // that both values keep a normal low word for ratios up to about 2^1000.
    let e = ((a.approx_f64().log2() + b.approx_f64().log2()) / 2.0).round() as i32;
    if e.abs() >= 200 {
        return ldexp(agm_positive(ldexp(a, -e), ldexp(b, -e)), e);
    }
    let eps = T::epsilon();
    let half = k::<T>(0.5);
    let (mut a, mut b) = (a, b);
    // About log2(ln(a/b)) + log2(1/ε) steps; 200 is far beyond that.
    for _ in 0..200 {
        let mean = add(a * half, b * half);
        if div((a - b).abs(), mean) <= eps {
            return mean;
        }
        let geo = mul(sqrt(a), sqrt(b));
        a = mean;
        b = geo;
    }
    T::nan()
}

pub(crate) fn agm<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        return T::nan();
    }
    let zero = T::zero();
    if (a < zero && b > zero) || (a > zero && b < zero) {
        return T::nan();
    }
    if (a.is_infinite() || b.is_infinite()) && (a == zero || b == zero) {
        return T::nan();
    }
    if a == zero || b == zero {
        return zero;
    }
    if a == b {
        return a;
    }
    if a < zero {
        return -agm(-a, -b);
    }
    // a and b are positive; an infinite one gives an infinite mean.
    if a.is_infinite() || b.is_infinite() {
        return T::infinity();
    }
    agm_positive(a, b)
}
