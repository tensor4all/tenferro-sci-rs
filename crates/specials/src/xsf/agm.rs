//! Arithmetic-geometric mean for `f64`.
//!
//! Ported from xsf `include/xsf/agm.h` (`agm` and `detail::agm_iter`).
//! Translated from Cython to C++ by the xsf developers in 2026.

use core::f64::consts::PI;

use super::ellip::ellpk;

/// Iterative AGM; `a` and `b` must be positive (not zero, not NaN).
fn agm_iter(mut a: f64, mut b: f64) -> f64 {
    let mut amean = 0.5 * a + 0.5 * b;
    let mut count = 20;
    while count > 0 && (amean != a && amean != b) {
        let gmean = a.sqrt() * b.sqrt();
        a = amean;
        b = gmean;
        amean = 0.5 * a + 0.5 * b;
        count -= 1;
    }
    amean
}

/// The arithmetic-geometric mean of `a` and `b`.
pub(crate) fn agm(mut a: f64, mut b: f64) -> f64 {
    // sqrt(f64::MAX / 2) and its reciprocal
    const SQRTHALFMAX: f64 = 9.480751908109176e+153;
    const INVSQRTHALFMAX: f64 = 1.0547686614863e-154;

    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }

    if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
        // a and b have opposite sign.
        return f64::NAN;
    }

    if (a.is_infinite() || b.is_infinite()) && (a == 0.0 || b == 0.0) {
        // One value is inf and the other is 0.
        return f64::NAN;
    }

    if a == 0.0 || b == 0.0 {
        // At least one of the arguments is 0.
        return 0.0;
    }

    if a == b {
        return a;
    }

    let mut sgn = 1.0;
    if a < 0.0 {
        sgn = -1.0;
        a = -a;
        b = -b;
    }

    // At this point, a and b are both positive and not NaN.

    if (INVSQRTHALFMAX < a && a < SQRTHALFMAX) && (INVSQRTHALFMAX < b && b < SQRTHALFMAX) {
        let e = 4.0 * a * b / ((a + b) * (a + b));
        return sgn * (PI / 4.0) * (a + b) / ellpk(e);
    }

    // At least one value is "extreme" (very big or very small).
    // Use the iteration to avoid overflow or underflow.

    sgn * agm_iter(a, b)
}
