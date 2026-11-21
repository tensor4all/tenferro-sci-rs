//! B(a, b) and ln|B(a, b)|, precision-adaptive.
//!
//! Written from DLMF 5.12.1 (B = Γ(a)Γ(b)/Γ(a+b)). The handling of negative
//! integer arguments follows xsf's `beta` / `lbeta` (`beta_negint`), so that
//! both strategies agree at SciPy's special values. For large arguments,
//! ln Γ(a+b) - ln Γ(a) is evaluated from the Stirling series directly to
//! avoid the cancellation of two large logarithms.

use super::gamma::{gamma, gammasgn, lgamma, lgamma_diff_large, lgamma_pos, rgamma, STIRLING_MIN};
use super::{add, mul};
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
            sgn * beta(T::one() - k::<T>(f64::from(a)) - b, b)
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
    let zero = T::zero();
    if b > zero && a >= k(STIRLING_MIN) {
        // a is the larger, so a + b >= STIRLING_MIN too.
        return (lgamma_pos(b) - lgamma_diff_large(a, b), T::one());
    }
    let ab = a + b;
    let sign = gammasgn(a) * gammasgn(b) * gammasgn(ab);
    (add(add(lgamma(a), lgamma(b)), -lgamma(ab)), sign)
}

fn in_direct_range<T: Real>(a: T, b: T) -> bool {
    let m = k::<T>(DIRECT_MAX);
    a.abs() <= m && b.abs() <= m && (a + b).abs() <= m
}

/// Γ(a) Γ(b) / Γ(a + b) as `(Γ(a) / Γ(a+b)) Γ(b)`, ordered to stay in range.
fn gamma_ratio<T: Real>(a: T, b: T) -> T {
    let y = rgamma(a + b);
    let ga = gamma(a);
    let gb = gamma(b);
    let (gay, gby) = (mul(ga, y), mul(gb, y));
    if (gay.abs() - T::one()).abs() > (gby.abs() - T::one()).abs() {
        mul(gby, ga)
    } else {
        mul(gay, gb)
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
            return v.abs().ln();
        }
    }
    lbeta_signed(a, b).0
}
