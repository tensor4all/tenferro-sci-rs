//! Hurwitz ζ(s, q), precision-adaptive.
//!
//! Written from the Euler–Maclaurin formula (DLMF 2.10.1 applied to
//! Σ (q+k)^-s; see also DLMF 25.11.5). The domain conventions (s = 1, s < 1,
//! q a non-positive integer, q < 0 with non-integer s) follow xsf's `zeta`.

use super::tables::EULER_MACLAURIN;
use crate::real::{c, k, Real};

/// `y^-s` for `y != 0`; a negative `y` requires an integer `s`.
fn pow_neg<T: Real>(y: T, s: T, s_int: Option<i32>) -> T {
    match s_int {
        Some(n) => y.powi(n).recip(),
        None => (-s * y.ln()).exp(),
    }
}

pub(crate) fn zeta<T: Real>(s: T, q: T) -> T {
    if s.is_nan() || q.is_nan() {
        return T::nan();
    }
    if s == T::one() {
        return T::infinity();
    }
    if s < T::one() {
        return T::nan();
    }
    let s_is_int = s == s.floor();
    if q <= T::zero() {
        if q == q.floor() {
            return T::infinity();
        }
        if !s_is_int {
            return T::nan();
        }
    }
    if s.is_infinite() {
        // ζ(∞, q) = lim q^-s.
        return if q > T::one() {
            T::zero()
        } else if q == T::one() {
            T::one()
        } else {
            T::infinity()
        };
    }
    let s_int = (s_is_int && s < k(1e9)).then(|| s.approx_f64() as i32);

    // Direct sum up to a = q + N with a >= max(25, s), so that the
    // Euler–Maclaurin tail, whose terms shrink like ((s+2j)/(2πa))², converges
    // fast.
    let eps = T::epsilon();
    let target = s.max(k(25.0));
    let mut sum = T::zero();
    let mut a = q;
    while a < target {
        let t = pow_neg(a, s, s_int);
        sum = sum + t;
        a = a + T::one();
        if t.abs() <= eps * sum.abs() && a > T::zero() {
            // The remaining terms, including the tail ~ a^(1-s)/(s-1), are below eps.
            let tail = pow_neg(a, s, s_int) * a / (s - T::one());
            if tail.abs() <= eps * sum.abs() {
                return sum;
            }
        }
    }

    // Tail: a^(1-s)/(s-1) + a^-s/2 + Σ_j B_{2j}/(2j)! (s)_{2j-1} a^(-s-2j+1).
    let b = pow_neg(a, s, s_int);
    sum = sum + b * a / (s - T::one()) + b / k(2.0);
    let ai2 = (a * a).recip();
    let mut fac = s * b / a; // (s)_1 a^(-s-1)
    let mut j = 0.0;
    for &cf in &EULER_MACLAURIN {
        let term = c::<T>(cf) * fac;
        sum = sum + term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
        // (s)_{2j+1} = (s)_{2j-1} (s + 2j - 1)(s + 2j).
        j += 1.0;
        fac = fac * (s + k(2.0 * j - 1.0)) * (s + k(2.0 * j)) * ai2;
    }
    sum
}
