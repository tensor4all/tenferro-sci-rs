//! Γ, ln|Γ|, sign of Γ and 1/Γ, precision-adaptive.
//!
//! Written from DLMF 5.5.3 (reflection), 5.5.1 (recurrence), 5.7.3 (Maclaurin
//! series of ln Γ(1+z)) and 5.11.1 / 5.11.3 (Stirling series).

use super::tables::{EULER_GAMMA, LN_PI, LN_SQRT_2PI, SQRT_2PI, STIRLING, ZETA_M1};
use super::{div, mul, sinpi};
use crate::real::{c, k, Real};

/// Below this, Stirling's series is not used.
pub(crate) const STIRLING_MIN: f64 = 20.0;

/// True when `x` is zero or a negative integer (a pole of Γ).
fn is_pole<T: Real>(x: T) -> bool {
    x <= T::zero() && x == x.floor()
}

/// `Σ_k STIRLING[k] / x^(2k-1)`, the correction in ln Γ(x) = (x-½)ln x - x + ln√(2π) + S(x).
pub(crate) fn stirling_tail<T: Real>(x: T) -> T {
    let eps = T::epsilon();
    let xi = x.recip();
    let xi2 = xi * xi;
    let mut p = xi;
    let mut sum = T::zero();
    for &cf in &STIRLING {
        let term = c::<T>(cf) * p;
        sum = sum + term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
        p = p * xi2;
    }
    sum
}

/// ln Γ(2 + z) for |z| <= 1/2 (DLMF 5.7.3 with ln Γ(2+z) = ln Γ(1+z) + ln(1+z)):
/// `(1-γ) z + Σ_{k>=2} (-1)^k (ζ(k)-1) z^k / k`.
fn lgamma2p<T: Real>(z: T) -> T {
    let eps = T::epsilon();
    let mut sum = (T::one() - c::<T>(EULER_GAMMA)) * z;
    let mz = -z;
    let mut zk = mz;
    for (i, &cf) in ZETA_M1.iter().enumerate() {
        zk = zk * mz; // (-z)^k for k = i + 2
        let term = c::<T>(cf) * zk / k::<T>((i + 2) as f64);
        sum = sum + term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
    }
    sum
}

/// ln Γ(x) for x >= STIRLING_MIN.
fn lgamma_stirling<T: Real>(x: T) -> T {
    // (x-½)ln x - x written as x(ln x - 1) - ½ln x, which overflows only when
    // the result does.
    let l = x.ln();
    let lead = x * (l - T::one());
    if !lead.is_finite() {
        return T::infinity();
    }
    lead - k::<T>(0.5) * l + c::<T>(LN_SQRT_2PI) + stirling_tail(x)
}

/// ln Γ(x) for x > 0.
pub(crate) fn lgamma_pos<T: Real>(x: T) -> T {
    let one = T::one();
    let half = k::<T>(0.5);
    if x < k(1e-20) {
        // DLMF 5.7.3: ln Γ(x) = -ln x - γx + O(x²).
        return -x.ln() - c::<T>(EULER_GAMMA) * x;
    }
    if x < half {
        // ln Γ(x) = ln Γ(1+x) - ln x = ln Γ(2+x) - ln(1+x) - ln x.
        return lgamma2p(x) - x.ln_1p() - x.ln();
    }
    if x < k(1.5) {
        let z = x - one;
        return lgamma2p(z) - z.ln_1p();
    }
    if x < k(2.5) {
        return lgamma2p(x - k(2.0));
    }
    if x < k(STIRLING_MIN) {
        // Downward recurrence onto [1.5, 2.5): ln Γ(x) = ln Γ(x-n) + ln Π (x-j).
        let mut y = x;
        let mut prod = one;
        while y >= k(2.5) {
            y = y - one;
            prod = prod * y;
        }
        return lgamma2p(y - k(2.0)) + prod.ln();
    }
    lgamma_stirling(x)
}

/// ln|Γ(x)|.
pub(crate) fn lgamma<T: Real>(x: T) -> T {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if is_pole(x) {
        return T::infinity();
    }
    if x > T::zero() {
        return lgamma_pos(x);
    }
    if x > k(-1e-20) {
        return -(-x).ln() - c::<T>(EULER_GAMMA) * x;
    }
    // Reflection, DLMF 5.5.3: ln|Γ(x)| = ln π - ln|sin πx| - ln Γ(1-x).
    c::<T>(LN_PI) - sinpi(x).abs().ln() - lgamma_pos(T::one() - x)
}

/// The sign of Γ(x), with SciPy's conventions at zeros, poles and infinities.
pub(crate) fn gammasgn<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x > T::zero() {
        return T::one();
    }
    if x == T::zero() {
        return T::one().copysign(x);
    }
    if x.is_infinite() || x == x.floor() {
        return T::nan();
    }
    // The sign on (-n, -n+1) is (-1)^n.
    let n = x.floor();
    if (n / k(2.0)).fract() == T::zero() {
        T::one()
    } else {
        -T::one()
    }
}

/// Γ(x) for x >= STIRLING_MIN, avoiding the absolute error of exp(ln Γ):
/// `√(2π) x^(x-½) e^(-x) e^(S(x))` with the power split into integer powers.
fn gamma_stirling<T: Real>(x: T) -> T {
    if x >= k(172.0) {
        return T::infinity();
    }
    let e = x - k(0.5);
    let n = e.floor();
    let f = e - n;
    let ni = n.approx_f64() as i32;
    let n1 = ni / 2;
    let n2 = ni - n1;
    let xf = if f == T::zero() {
        T::one()
    } else {
        (f * x.ln()).exp()
    };
    let pow = (x.powi(n1) * (-x).exp()) * x.powi(n2);
    let g = c::<T>(SQRT_2PI) * pow * xf * stirling_tail(x).exp();
    // Overflow between Γ⁻¹(MAX) ≈ 171.62 and 172 leaves NaN in the error term.
    if g.is_finite() {
        g
    } else {
        T::infinity()
    }
}

/// Γ(x) for x >= 1/2.
fn gamma_pos<T: Real>(x: T) -> T {
    let one = T::one();
    if x < k(1.5) {
        // Γ(x) = Γ(x+1) / x.
        return (lgamma2p(x - one)).exp() / x;
    }
    if x < k(STIRLING_MIN) {
        let mut y = x;
        let mut prod = one;
        while y >= k(2.5) {
            y = y - one;
            prod = prod * y;
        }
        return lgamma2p(y - k(2.0)).exp() * prod;
    }
    gamma_stirling(x)
}

/// Γ(x).
pub(crate) fn gamma<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return if x > T::zero() { x } else { T::nan() };
    }
    if x == T::zero() {
        return T::infinity().copysign(x);
    }
    if is_pole(x) {
        return T::nan();
    }
    if x.abs() < k(1e-20) {
        // DLMF 5.7.1: Γ(x) = 1/x - γ + O(x). Avoids sin(πx), whose low word
        // underflows for tiny x.
        return x.recip() - c::<T>(EULER_GAMMA);
    }
    if x >= k(0.5) {
        return gamma_pos(x);
    }
    // Reflection, DLMF 5.5.3: Γ(x) = π / (sin(πx) Γ(1-x)).
    div(T::PI(), mul(sinpi(x), gamma_pos(T::one() - x)))
}

/// 1/Γ(x), zero at the poles.
pub(crate) fn rgamma<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x == T::zero() {
        return x;
    }
    if x.is_infinite() {
        return if x > T::zero() { T::zero() } else { T::nan() };
    }
    if is_pole(x) {
        return T::zero();
    }
    if x.abs() < k(1e-20) {
        // DLMF 5.7.1: 1/Γ(x) = x + γx² + O(x³).
        return x + c::<T>(EULER_GAMMA) * x * x;
    }
    if x >= k(0.5) {
        return div(T::one(), gamma_pos(x));
    }
    // 1/Γ(x) = sin(πx) Γ(1-x) / π.
    div(mul(sinpi(x), gamma_pos(T::one() - x)), T::PI())
}

/// `ln Γ(a + b) - ln Γ(a)` for `a >= STIRLING_MIN` and `a + b >= STIRLING_MIN`,
/// without the cancellation of the two large logarithms.
pub(crate) fn lgamma_diff_large<T: Real>(a: T, b: T) -> T {
    // (a+b-½)ln(a+b) - (a-½)ln a - b = (a-½) ln(1 + b/a) + b ln(a+b) - b.
    let half = k::<T>(0.5);
    let ab = a + b;
    (a - half) * (b / a).ln_1p() + b * ab.ln() - b + (stirling_tail(ab) - stirling_tail(a))
}
