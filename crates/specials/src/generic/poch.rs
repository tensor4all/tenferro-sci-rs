//! Pochhammer symbol `(a)_m = Γ(a + m) / Γ(a)`, precision-adaptive.
//!
//! Written from the recurrence Γ(z+1) = zΓ(z) (DLMF 5.5.1), which reduces
//! `|m|` below 1 by an exact product, and then:
//!
//! * for `a >= 21`, the difference of Stirling series
//!   `ln Γ(a+m) - ln Γ(a)` without the cancellation of the two large
//!   logarithms ([`lgamma_diff_large`](super::gamma::lgamma_diff_large));
//! * for `a + m < 0` and `1 - a - m >= 21`, the reflection formula
//!   (DLMF 5.5.3) `(a)_m = sin(πa) / sin(π(a+m)) · (1-a-m)_m`, which turns it
//!   into the previous case;
//! * for `0 < |a| < 1e-200 <= |a+m|`, `a · exp(ln|Γ(a+m)| - ln Γ(1+a))`, because
//!   `ln Γ(a) ≈ -ln|a|` reaches 745 for subnormal `a` and the exponential
//!   would underflow before the reduction product brings it back;
//! * otherwise `exp(ln|Γ(a+m)| - ln|Γ(a)|)` with the signs of Γ, where
//!   `|ln Γ|` is below about 100 (`-21 < a < 21`) and the error is a few
//!   `|ln Γ| ε`.
//!
//! The zeros and poles follow xsf's `poch`: `(a)_m = 0` when `a` is a
//! non-positive integer and `a + m` is not, and `+∞` the other way round.

use super::gamma::{gammasgn, lgamma, lgamma_diff_large, STIRLING_MIN};
use super::{add, div, mul, sinpi};
use crate::real::{k, Real};
use crate::xsf;

/// Below this `|a|` the factor `a` of `(a)_m` is taken out before `exp`.
const TINY_A: f64 = 1e-200;

/// A non-positive integer of magnitude below 1e13 (xsf's `is_nonpos_int`).
fn is_nonpos_int<T: Real>(x: T) -> bool {
    x <= T::zero() && x == x.ceil() && x.abs() < k(1e13)
}

/// `sin(π(a + m))` without rounding `a + m` first: the sum is not exact in
/// double-double, and next to an integer the rounding error would dominate the
/// reduced argument. Splitting off the integer part of `a` leaves the exactly
/// representable `(a - n) + m` of moderate size.
fn sinpi_sum<T: Real>(a: T, m: T) -> T {
    let n = a.round();
    let s = sinpi(a - n + m);
    if (n / k(2.0)).fract() != T::zero() {
        -s
    } else {
        s
    }
}

/// `exp(l)` that returns `+∞` instead of NaN on overflow (double-double
/// reports the overflow of `exp` as NaN).
fn exp_sat<T: Real>(l: T) -> T {
    if l.approx_f64() > 709.7 {
        T::infinity()
    } else if l.approx_f64() < -745.0 {
        T::zero()
    } else {
        l.exp()
    }
}

/// The result when the product of the reduction over- or underflowed.
///
/// The value is then 0 or ±∞ (the remaining factor `Γ(a+m)/Γ(a)` of the
/// reduced `m` is moderate), decided from the integrality of `a` and `m`
/// (`a + m` is an integer exactly when both are, and a huge `m` is an
/// integer) and the sign of `a + m0` for the original `m0`:
///
/// * underflow (`r = 0`): `+∞` at a pole (`a` a positive integer, `m` an
///   integer, `a + m0 <= 0`), else 0;
/// * overflow: for `a` a non-positive integer, 0 unless `m` is an integer
///   with `a + m0 <= 0`, where the product of negative integers has the sign
///   `(-1)^m0` and magnitude beyond the range; otherwise `±∞` with the sign of
///   `r` and of the remaining factor.
///
/// These are the exact limits; they agree with mpmath where xsf's `0 * ∞`
/// gives NaN or a wrong infinity.
fn extreme<T: Real>(a: T, m_cur: T, m0: T, r: T) -> T {
    let a_int = a == a.floor();
    let m_int = m0 == m0.floor();
    let sum_positive = (a + m0).approx_f64() > 0.0;
    if r == T::zero() {
        if a_int && a > T::zero() && m_int && !sum_positive {
            T::infinity()
        } else {
            T::zero()
        }
    } else if a_int && a <= T::zero() {
        if !m_int || sum_positive {
            T::zero()
        } else {
            let odd = (m0 / k(2.0)).fract() != T::zero();
            T::infinity().copysign(if odd { -T::one() } else { T::one() })
        }
    } else {
        let sign = mul(gammasgn(a + m_cur), gammasgn(a)) * T::one().copysign(r);
        mul(T::infinity(), sign)
    }
}

pub(crate) fn poch<T: Real>(a: T, m: T) -> T {
    if a.is_nan() || m.is_nan() {
        return T::nan();
    }
    if !a.is_finite() || !m.is_finite() {
        // The value is a special one (0, 1, ±∞ or NaN): take xsf's.
        return T::from_dd(xsf::poch::poch(a.approx_f64(), m.approx_f64()), 0.0);
    }
    let one = T::one();
    let mut r = one;
    let m0 = m;
    let mut m = m;

    // Reduce |m| below 1 with exact products, as xsf does.
    while m >= one {
        if a + m == one {
            break;
        }
        m = m - one;
        r = mul(r, a + m);
        if !r.is_finite() || r == T::zero() {
            break;
        }
    }
    while m <= -one {
        if a + m == T::zero() {
            break;
        }
        r = div(r, a + m);
        m = m + one;
        if !r.is_finite() || r == T::zero() {
            break;
        }
    }

    if !r.is_finite() || r == T::zero() {
        return extreme(a, m, m0, r);
    }
    if m == T::zero() {
        return r;
    }

    let small_m = m.abs() <= one;
    if a >= k(STIRLING_MIN + 1.0) && small_m {
        // a + m >= 20 as well.
        return mul(r, exp_sat(lgamma_diff_large(a, m)));
    }

    if is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m {
        return T::infinity();
    }
    if !is_nonpos_int(a + m) && is_nonpos_int(a) {
        return T::zero();
    }
    if small_m && a < T::zero() && a + m < T::zero() {
        let b = one - a - m;
        if b >= k(STIRLING_MIN + 1.0) {
            let ratio = div(sinpi(a), sinpi_sum(a, m));
            return mul(mul(r, ratio), exp_sat(lgamma_diff_large(b, m)));
        }
    }
    if a != T::zero() && a.abs() < k(TINY_A) && (a + m).abs() >= k(TINY_A) {
        // Γ(a) ≈ 1/a overflows its logarithm's range, and exp(ln Γ(a+m) - ln Γ(a))
        // underflows before the product `r` restores the value. Take the factor
        // `a` out: (a)_m = a Γ(a+m) / Γ(1+a), with Γ(1+a) > 0 for |a| < 1.
        let e = exp_sat(add(lgamma(a + m), -lgamma(one + a))) * gammasgn(a + m);
        let g = mul(r, e);
        return if g.approx_f64().is_finite() {
            mul(g, a)
        } else {
            mul(r, mul(e, a))
        };
    }
    let e = exp_sat(add(lgamma(a + m), -lgamma(a)));
    mul(mul(r, e), mul(gammasgn(a + m), gammasgn(a)))
}
