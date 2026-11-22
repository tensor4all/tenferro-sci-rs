//! B(a, b) and ln|B(a, b)| for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/beta.h`.
//! Cephes Math Library, Copyright 1984, 1987 by Stephen L. Moshier.

use super::consts::{MAXGAM, MAXLOG};
use super::gamma::{gamma, lgam_sgn, rgamma};

const ASYMP_FACTOR: f64 = 1e6;

/// Asymptotic expansion of ln|B(a, b)| for a > ASYMP_FACTOR max(|b|, 1).
fn lbeta_asymp(a: f64, b: f64) -> (f64, f64) {
    let (mut r, sgn) = lgam_sgn(b);
    r -= b * a.ln();
    // Evaluate the corrections through b/a and (1-b)/a, which do not overflow.
    let t = b / a;
    let u = (1.0 - b) / a;
    r += t * (1.0 - b) / 2.0;
    r += t * u * (1.0 - 2.0 * b) / 12.0;
    r += -t * t * u * (1.0 - b) / 12.0;
    (r, sgn)
}

/// `Some(n)` when `x` is a value of the C++ `static_cast<int>` round trip.
fn as_int(x: f64) -> Option<i32> {
    let i = x as i32;
    (f64::from(i) == x).then_some(i)
}

fn beta_negint(a: i32, b: f64) -> f64 {
    match as_int(b) {
        Some(bi) if 1.0 - f64::from(a) - b > 0.0 => {
            let sgn = if bi % 2 == 0 { 1.0 } else { -1.0 };
            sgn * beta(1.0 - f64::from(a) - b, b)
        }
        _ => f64::INFINITY,
    }
}

fn lbeta_negint(a: i32, b: f64) -> f64 {
    match as_int(b) {
        Some(_) if 1.0 - f64::from(a) - b > 0.0 => lbeta(1.0 - f64::from(a) - b, b),
        _ => f64::INFINITY,
    }
}

/// Shared head of `beta` and `lbeta`: the negative-integer cases.
enum Head {
    Done(f64),
    Go(f64, f64),
}

fn head(a: f64, b: f64, negint: fn(i32, f64) -> f64) -> Head {
    if a <= 0.0 && a == a.floor() {
        return Head::Done(as_int(a).map_or(f64::INFINITY, |ai| negint(ai, b)));
    }
    if b <= 0.0 && b == b.floor() {
        return Head::Done(as_int(b).map_or(f64::INFINITY, |bi| negint(bi, a)));
    }
    if a.abs() < b.abs() {
        Head::Go(b, a)
    } else {
        Head::Go(a, b)
    }
}

/// Γ(a) Γ(b) / Γ(a + b) in the order that keeps the product in range.
fn gamma_ratio(a: f64, b: f64) -> Option<f64> {
    let y = rgamma(a + b);
    let ga = gamma(a);
    let gb = gamma(b);
    if y.is_infinite() {
        return None;
    }
    Some(
        if ((ga * y).abs() - 1.0).abs() > ((gb * y).abs() - 1.0).abs() {
            (gb * y) * ga
        } else {
            (ga * y) * gb
        },
    )
}

fn lgam_sum(a: f64, b: f64) -> (f64, f64) {
    let (y, s1) = lgam_sgn(a + b);
    let (lb, s2) = lgam_sgn(b);
    let (la, s3) = lgam_sgn(a);
    (la + (lb - y), s1 * s2 * s3)
}

pub(crate) fn beta(a: f64, b: f64) -> f64 {
    let (a, b) = match head(a, b, beta_negint) {
        Head::Done(v) => return v,
        Head::Go(a, b) => (a, b),
    };
    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        // Avoids the loss of precision in lgam(a + b) - lgam(a).
        let (y, sign) = lbeta_asymp(a, b);
        return sign * y.exp();
    }
    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let (y, sign) = lgam_sum(a, b);
        if y > MAXLOG {
            return sign * f64::INFINITY;
        }
        return sign * y.exp();
    }
    gamma_ratio(a, b).unwrap_or(f64::INFINITY)
}

pub(crate) fn lbeta(a: f64, b: f64) -> f64 {
    let (a, b) = match head(a, b, lbeta_negint) {
        Head::Done(v) => return v,
        Head::Go(a, b) => (a, b),
    };
    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        return lbeta_asymp(a, b).0;
    }
    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM || b.abs() < f64::MIN_POSITIVE {
        return lgam_sum(a, b).0;
    }
    match gamma_ratio(a, b) {
        Some(y) => y.abs().ln(),
        None => f64::INFINITY,
    }
}
