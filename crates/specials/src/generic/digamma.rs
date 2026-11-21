//! ψ (digamma), precision-adaptive.
//!
//! Written from DLMF 5.5.2 (recurrence), 5.5.4 (reflection), 5.11.2
//! (asymptotic series) and the Taylor series about the zeros of ψ
//! (coefficients ψ⁽ⁿ⁾(r)/n!, DLMF 5.15.1, tabulated with mpmath). Using a
//! series about the zero near -0.504 follows xsf's `digamma`; the series about
//! the positive zero replaces xsf's rational approximation on [1, 2].

use super::cotpi;
use super::power_series;
use super::tables::{
    DIGAMMA_ASY, DIGAMMA_NEGROOT, DIGAMMA_NEGROOT_VAL, DIGAMMA_NEG_TAYLOR, DIGAMMA_POSROOT,
    DIGAMMA_POSROOT_VAL, DIGAMMA_POS_TAYLOR,
};
use crate::real::{c, k, Real};

/// The asymptotic series is used from here on.
const ASY_MIN: f64 = 20.0;
/// Radii of the Taylor series about the zeros. The coefficients grow like
/// 1.46^-n and 0.496^-n (distance to the nearest pole), so the tables of 120
/// coefficients reach the double-double epsilon at these radii.
const POS_RADIUS: f64 = 0.5;
const NEG_RADIUS: f64 = 0.25;

/// ψ(x) for x >= ASY_MIN: `ln x - 1/(2x) - Σ B_{2k} / (2k x^{2k})`.
fn digamma_asy<T: Real>(x: T) -> T {
    let eps = T::epsilon();
    let xi2 = (x * x).recip();
    let mut sum = x.ln() - k::<T>(0.5) / x;
    let mut p = xi2;
    for &cf in &DIGAMMA_ASY {
        let term = c::<T>(cf) * p;
        sum = sum - term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
        p = p * xi2;
    }
    sum
}

/// ψ(x) for x > 0 outside the positive zero's disc.
fn digamma_pos<T: Real>(mut x: T) -> T {
    let mut acc = T::zero();
    while x < k(ASY_MIN) {
        acc = acc + x.recip();
        x = x + T::one();
    }
    digamma_asy(x) - acc
}

/// ψ(x).
pub(crate) fn digamma<T: Real>(x: T) -> T {
    if x.is_nan() || x == T::infinity() {
        return x;
    }
    if x == T::neg_infinity() {
        return T::nan();
    }
    if x == T::zero() {
        return T::infinity().copysign(-x);
    }
    if x < T::zero() && x == x.floor() {
        return T::nan();
    }

    let pos = c::<T>(DIGAMMA_POSROOT);
    if (x - pos).abs() < k(POS_RADIUS) {
        return power_series(&DIGAMMA_POS_TAYLOR, x - pos, c(DIGAMMA_POSROOT_VAL));
    }
    let neg = c::<T>(DIGAMMA_NEGROOT);
    if (x - neg).abs() < k(NEG_RADIUS) {
        return power_series(&DIGAMMA_NEG_TAYLOR, x - neg, c(DIGAMMA_NEGROOT_VAL));
    }
    if x > T::zero() {
        return digamma_pos(x);
    }
    // Reflection, DLMF 5.5.4: ψ(x) = ψ(1-x) - π cot(πx).
    digamma_pos(T::one() - x) - T::PI() * cotpi(x)
}
