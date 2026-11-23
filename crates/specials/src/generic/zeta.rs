//! Hurwitz ζ(s, q) and Riemann ζ(s), precision-adaptive.
//!
//! Written from the Euler–Maclaurin formula (DLMF 2.10.1 applied to
//! Σ (q+k)^-s; see also DLMF 25.11.5) and, for the Riemann function at
//! s < 0, the functional equation (DLMF 25.4.1). The domain conventions
//! follow xsf's `zeta(x, q)`: q = 1 is the Riemann function for every real s;
//! otherwise s = 1 gives +∞, s < 1 NaN, q a non-positive integer +∞, and q < 0
//! needs an integer s. For `|s| < 0.1` the Riemann function uses its Taylor
//! series about 0 (coefficients ζ⁽ⁿ⁾(0)/n! from mpmath).

use super::gamma::{gamma, lgamma};
use super::tables::{EULER_MACLAURIN, ZETA_TAYLOR0};
use super::{mul, power_series, sinpi};
use crate::real::{c, k, Real};

/// `y^-s` for `y != 0`; a negative `y` requires an integer `s`.
fn pow_neg<T: Real>(y: T, s: T, s_int: Option<i32>) -> T {
    match s_int {
        Some(n) => {
            // For |y| < 1, invert first: y^n would lose the low word to
            // subnormal range long before 1/y^n overflows.
            if y.abs() < T::one() {
                return y.recip().powi(n);
            }
            let p = y.powi(n);
            if p.is_finite() {
                p.recip()
            } else if y.abs() > T::one() {
                // |y|^n overflowed (double-double reports this as NaN).
                T::zero()
            } else {
                T::infinity()
            }
        }
        None => (-s * y.ln()).exp(),
    }
}

pub(crate) fn zeta<T: Real>(s: T, q: T) -> T {
    if s.is_nan() || q.is_nan() {
        return T::nan();
    }
    if q == T::one() {
        return riemann(s);
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
    let first = pow_neg(q, s, s_int);
    if first.is_nan() || first.is_infinite() {
        // q^-s alone overflows (or is undefined): so does the sum.
        return first.abs();
    }
    // Sum directly up to a = q + N with a >= max(25, s), so that the
    // Euler–Maclaurin tail, whose terms shrink like ((s+2j)/(2πa))², converges
    // fast. For large s the direct terms vanish long before that, and the
    // loop stops early.
    euler_maclaurin(s, q, s.max(k(25.0)), s_int)
}

/// Σ_{k>=0} (q+k)^-s: direct sum while q + k < `target` (stopping early when
/// the rest is below epsilon), then the Euler–Maclaurin tail.
fn euler_maclaurin<T: Real>(s: T, q: T, target: T, s_int: Option<i32>) -> T {
    let eps = T::epsilon();
    let mut sum = T::zero();
    let mut a = q;
    while a < target {
        let t = pow_neg(a, s, s_int);
        if t.is_nan() {
            return t;
        }
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
    // The leading term a^(1-s)/(s-1) is computed directly: a^-s alone can
    // underflow while it does not.
    let sm1 = s - T::one();
    let lead = pow_neg(a, sm1, s_int.map(|n| n - 1)) / sm1;
    let b = lead * sm1 / a; // a^-s
    sum = sum + lead + b / k(2.0);
    let ai = a.recip();
    let ai2 = ai * ai;
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

/// Below this, ζ(s) for s < 0 uses the functional equation in log form.
const REFLECT_LOG: f64 = -160.0;

/// The Riemann zeta function ζ(s) for every real s.
fn riemann<T: Real>(s: T) -> T {
    let one = T::one();
    if s == one {
        return T::infinity();
    }
    if s.is_infinite() {
        // ζ(+∞) = 1; ζ(s) oscillates without limit as s → -∞.
        return if s > T::zero() { one } else { T::nan() };
    }
    if s.abs() < k(0.1) {
        // Taylor series about 0 (radius 1). Near 0 the reflection formula
        // would need 1 - s, which cannot hold the low word of a tiny `s`
        // next to the pole of ζ at 1.
        return power_series(&ZETA_TAYLOR0[1..], s, c(ZETA_TAYLOR0[0]));
    }
    if s > one {
        return euler_maclaurin(s, one, s.max(k(25.0)), None);
    }
    if s > T::zero() {
        // 0 < s < 1: Euler–Maclaurin is valid here too (DLMF 25.2.8 with N
        // terms); with N = 13 the tail reaches double-double epsilon, and
        // the cancellation between the direct sum and N^(1-s)/(s-1) costs
        // about one digit.
        return euler_maclaurin(s, one, k(13.0), None);
    }
    // s < 0: ζ(s) = 2 (2π)^(s-1) sin(πs/2) Γ(1-s) ζ(1-s) (DLMF 25.4.1).
    let half = s / k(2.0);
    if half == half.floor() {
        // Trivial zeros at the negative even integers.
        return T::zero();
    }
    let t = one - s;
    let two_pi = k::<T>(2.0) * T::PI();
    let z = riemann(t);
    let sn = sinpi(half);
    if s > k(REFLECT_LOG) {
        let f = k::<T>(2.0) * ((s - one) * two_pi.ln()).exp();
        return mul(mul(f, sn), mul(gamma(t), z));
    }
    // Γ(1-s) overflows: combine in logarithms (relative error ~ |log| ε).
    let l = k::<T>(2.0).ln() + (s - one) * two_pi.ln() + lgamma(t);
    let v = l.exp();
    let v = if v.is_finite() { v } else { T::infinity() };
    mul(mul(v, sn), z)
}

/// Above this, ζ(s) - 1 < 1e-292 and equals 2^-s to 1e-170.
const LOW_WORD_UNDERFLOW: f64 = 970.0;

/// ζ(s) - 1 (SciPy's `zetac`).
///
/// For `s > 1` this is the Hurwitz function ζ(s, 2) (DLMF 25.11.1 with the
/// first term removed), which has no cancellation. For `s < 1` the Riemann
/// function is far from 1 except near the isolated points where ζ(s) = 1
/// (the first one at s ≈ -17.2); there the result is accurate in absolute
/// terms only.
pub(crate) fn zetac<T: Real>(s: T) -> T {
    if s.is_nan() {
        return s;
    }
    if s == T::one() {
        return T::infinity();
    }
    if s == T::infinity() {
        return T::zero();
    }
    if s == T::neg_infinity() {
        return T::nan();
    }
    if s > T::one() {
        // 2^-s underflows to zero beyond s = 1075.
        if s > k(1080.0) {
            return T::zero();
        }
        if s > k(LOW_WORD_UNDERFLOW) {
            // ζ(s, 2) = 2^-s (1 + (2/3)^s + ...) = 2^-s to better than 1e-170,
            // and below 1e-292 the low word of a double-double is subnormal:
            // evaluate 2^-(s-600) in the type and scale by the exact 2^-600.
            let v = (-(s - k(600.0)) * T::LN_2()).exp();
            return v * k(2.0_f64.powi(-600));
        }
        return zeta(s, k(2.0));
    }
    // `add`, not `-`: for s < about -260 `riemann` returns (±∞, 0) and a raw
    // subtraction makes the error term NaN where the f64 path returns ±∞.
    super::add(riemann(s), -T::one())
}
