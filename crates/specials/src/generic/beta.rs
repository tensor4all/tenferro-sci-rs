//! B(a, b) and ln|B(a, b)|, precision-adaptive.
//!
//! Written from DLMF 5.12.1 (B = Γ(a)Γ(b)/Γ(a+b)). The handling of negative
//! integer arguments follows xsf's `beta` / `lbeta` (`beta_negint`), so that
//! both strategies agree at SciPy's special values. For large arguments,
//! ln Γ(a+b) - ln Γ(a) is evaluated from the Stirling series directly to
//! avoid the cancellation of two large logarithms.

use super::gamma::{gamma, gammasgn, lbeta_large, lgamma, lgamma_diff_large, rgamma, STIRLING_MIN};
use super::{add, div, ln, mul, sinpi};
use crate::real::{k, Real};

/// Γ arguments up to here are evaluated directly.
const DIRECT_MAX: f64 = 170.0;

/// `Some(n)` when `x` is an integer that fits in `i32`.
fn as_int<T: Real>(x: T) -> Option<i32> {
    let f = x.approx_f64();
    let i = f as i32;
    (x == x.floor() && f64::from(i) == f).then_some(i)
}

enum Head<T> {
    Done(T),
    Go(T, T),
}

/// The negative-integer cases shared by `beta` and `lbeta`, then |a| >= |b|.
fn head<T: Real>(a: T, b: T, negint: fn(i32, T) -> T) -> Head<T> {
    if a <= T::zero() && a == a.floor() {
        return Head::Done(as_int(a).map_or(T::infinity(), |ai| negint(ai, b)));
    }
    if b <= T::zero() && b == b.floor() {
        return Head::Done(as_int(b).map_or(T::infinity(), |bi| negint(bi, a)));
    }
    if a.abs() < b.abs() {
        Head::Go(b, a)
    } else {
        Head::Go(a, b)
    }
}

fn beta_negint<T: Real>(a: i32, b: T) -> T {
    match as_int(b) {
        Some(bi) if T::one() - k::<T>(f64::from(a)) - b > T::zero() => {
            let sgn = if bi % 2 == 0 { T::one() } else { -T::one() };
            mul(sgn, beta(T::one() - k::<T>(f64::from(a)) - b, b))
        }
        _ => T::infinity(),
    }
}

fn lbeta_negint<T: Real>(a: i32, b: T) -> T {
    match as_int(b) {
        Some(_) if T::one() - k::<T>(f64::from(a)) - b > T::zero() => {
            lbeta(T::one() - k::<T>(f64::from(a)) - b, b)
        }
        _ => T::infinity(),
    }
}

/// `(ln|B(a, b)|, sign)` for |a| >= |b| outside the direct range.
fn lbeta_signed<T: Real>(a: T, b: T) -> (T, T) {
    if a >= k(STIRLING_MIN) && b >= k(STIRLING_MIN) {
        // Both large (a >= b): no cancellation, and a + b may overflow.
        return (lbeta_large(a, b), T::one());
    }
    if a >= k(STIRLING_MIN) && a + b >= k(STIRLING_MIN) {
        // ln Γ(a+b) - ln Γ(a) from the Stirling series, without cancellation;
        // valid for either sign of b.
        return (lgamma(b) - lgamma_diff_large(a, b), gammasgn(b));
    }
    if a <= -k::<T>(STIRLING_MIN) && T::one() - a - b >= k(STIRLING_MIN) {
        // Large negative a: by reflection (DLMF 5.5.3),
        // Γ(a)/Γ(a+b) = [sin π(a+b) / sin πa] Γ(1-a-b)/Γ(1-a),
        // and the gamma ratio has large positive arguments.
        let c = T::one() - a - b;
        let (sab, sa) = (sinpi(a + b), sinpi(a));
        if sab == T::zero() {
            // a + b is a pole of Γ: B = 0.
            return (T::neg_infinity(), T::one());
        }
        let l = lgamma(b) - lgamma_diff_large(c, b) + ln(sab.abs()) - ln(sa.abs());
        return (l, gammasgn(b) * sab.signum() * sa.signum());
    }
    let ab = a + b;
    let sign = gammasgn(a) * gammasgn(b) * gammasgn(ab);
    (add(add(lgamma(a), lgamma(b)), -lgamma(ab)), sign)
}

fn in_direct_range<T: Real>(a: T, b: T) -> bool {
    let m = k::<T>(DIRECT_MAX);
    a.abs() <= m && b.abs() <= m && (a + b).abs() <= m
}

/// Γ(a) Γ(b) / Γ(a + b), ordered to stay in range. The quotients are formed
/// by division rather than through 1/Γ(a+b): near the ends of the range that
/// reciprocal is below 1e-292, where the low word of a double-double is
/// subnormal and the result loses digits.
fn gamma_ratio<T: Real>(a: T, b: T) -> T {
    let ga = gamma(a);
    let gb = gamma(b);
    let gab = gamma(a + b);
    if !gab.is_finite() || gab == T::zero() {
        // a + b at a pole of Γ (1/Γ = 0) or beyond its range.
        let y = rgamma(a + b);
        return mul(mul(ga, y), gb);
    }
    let (ta, tb) = (div(ga, gab), div(gb, gab));
    if (ta.abs() - T::one()).abs() > (tb.abs() - T::one()).abs() {
        mul(tb, ga)
    } else {
        mul(ta, gb)
    }
}

pub(crate) fn beta<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        return T::nan();
    }
    let (a, b) = match head(a, b, beta_negint) {
        Head::Done(v) => return v,
        Head::Go(a, b) => (a, b),
    };
    if in_direct_range(a, b) {
        return gamma_ratio(a, b);
    }
    let (l, sign) = lbeta_signed(a, b);
    if l.is_nan() {
        return l;
    }
    let e = if l == T::neg_infinity() {
        T::zero()
    } else {
        l.exp()
    };
    mul(sign, e)
}

pub(crate) fn lbeta<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        return T::nan();
    }
    let (a, b) = match head(a, b, lbeta_negint) {
        Head::Done(v) => return v,
        Head::Go(a, b) => (a, b),
    };
    if in_direct_range(a, b) && b.abs() >= k(1e-300) {
        // Γ(b) for tiny b would overflow; that case goes through ln Γ.
        let v = gamma_ratio(a, b);
        if v.is_finite() && v != T::zero() {
            return ln(v.abs());
        }
    }
    lbeta_signed(a, b).0
}
