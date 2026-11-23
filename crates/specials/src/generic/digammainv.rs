//! Inverse of the digamma function, precision-adaptive.
//!
//! ψ is increasing on `x > 0`, from -∞ to +∞, so ψ(x) = y has one solution
//! there. The `f64` kernel (Newton's method from the starting point of Minka,
//! 2000) provides a starting value with a relative error of about 1e-15, and
//! Newton's method on ψ(x) - y with ψ' = trigamma (quadratic convergence)
//! finishes at the working precision in at most two steps. When ψ(x) and y are
//! both huge (x tiny) the residual has only a few correct digits, but the
//! step it gives is correct to that relative accuracy, which is enough.

use super::digamma::digamma;
use super::div;
use super::trigamma::trigamma;
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
        let step = div(digamma(x) - y, trigamma(x));
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
