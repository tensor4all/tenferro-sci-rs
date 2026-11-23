//! Inverse of the digamma function, precision-adaptive.
//!
//! ψ is increasing on `x > 0`, from -∞ to +∞, so ψ(x) = y has one solution
//! there. The `f64` kernel (Newton's method from the starting point of Minka,
//! 2000) provides a starting value with a relative error of about 1e-15, and
//! Newton's method on ψ(x) - y with ψ' = trigamma (quadratic convergence)
//! finishes at the working precision, in two steps on every test point (the
//! loop is capped at six). When ψ(x) and y are both huge (x tiny) the residual
//! has only a few correct digits, but the step it gives is correct to that
//! relative accuracy, which is enough. Below x = 1e-100 the step is taken in
//! the scaled form `(ψ(x) - y) x² / (1 + x² ζ(2, 1 + x))`, because ψ'(x)
//! overflows below 7.5e-155. Below x ≈ 1e-292 (y below about -1e292) the low
//! word of a double-double is subnormal, and the accuracy falls towards that of
//! `f64` at the bottom of the normal range.

use super::digamma::digamma;
use super::trigamma::trigamma;
use super::zeta::zeta;
use super::{div, mul};
use crate::real::{k, Real};
use crate::xsf;

pub(crate) fn digammainv<T: Real>(y: T) -> T {
    if y.is_nan() {
        return y;
    }
    // Beyond ln(DBL_MAX) the result overflows; y = -∞ gives zero.
    if y.approx_f64() > xsf::consts::MAXLOG {
        return T::infinity();
    }
    if y == T::neg_infinity() {
        return T::zero();
    }
    let guess = xsf::digammainv::digammainv(y.approx_f64());
    if !guess.is_finite() || guess <= 0.0 {
        return T::from_dd(guess, 0.0);
    }
    let mut x = k::<T>(guess);
    for _ in 0..6 {
        let step = if x < k(1e-100) {
            // ψ'(x) = (1 + x² ζ(2, 1 + x)) / x² overflows below 7.5e-155 and the
            // plain step would be 0: divide by it in scaled form instead.
            let res = digamma(x) - y;
            let x2z = mul(mul(x, x), zeta(k(2.0), T::one() + x));
            div(mul(mul(res, x), x), T::one() + x2z)
        } else {
            div(digamma(x) - y, trigamma(x))
        };
        if !step.is_finite() {
            break;
        }
        x = x - step;
        if step.abs() <= k::<T>(4.0) * T::epsilon() * x.abs() {
            break;
        }
    }
    x
}
