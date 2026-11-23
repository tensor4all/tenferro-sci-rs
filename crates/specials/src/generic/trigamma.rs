//! ψ'(x), the trigamma function, precision-adaptive.
//!
//! Written from ψ'(x) = ζ(2, x) (DLMF 25.11.1 and 5.15.1) for `x > 0` and the
//! reflection formula ψ'(1-x) + ψ'(x) = π²/sin²(πx) (DLMF 5.15.6) for
//! `x <= 0`. The reflection does not cancel: π²/sin²(πx) >= π² while
//! ψ'(1-x) <= ψ'(1) < 1.65 for `x <= 0`.

use super::zeta::zeta;
use super::{add, div, mul, sinpi};
use crate::real::{k, Real};

/// ψ'(x). Poles at zero and the negative integers give `+∞`; NaN at `-∞`.
pub(crate) fn trigamma<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x == T::neg_infinity() {
        return T::nan();
    }
    if x == T::infinity() {
        return T::zero();
    }
    if x > T::zero() {
        return zeta(k(2.0), x);
    }
    if x == x.floor() {
        return T::infinity();
    }
    let c = div(T::PI(), sinpi(x));
    // `add`, not `-`: the subtrahend can overflow to infinity for tiny `|x|`, and a raw
    // subtraction would turn the error term into NaN where the f64 path returns +inf.
    add(mul(c, c), -zeta(k(2.0), T::one() - x))
}
