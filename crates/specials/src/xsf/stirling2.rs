//! Stirling numbers of the second kind `S(n, k)` for `f64`.
//!
//! Ported from xsf `include/xsf/stirling2.h` (`stirling2`, `stirling2_dp`,
//! `stirling2_temme`). Original implementation by Lucas Roberts; ported to xsf
//! by SciPy developers.
//! Reference: N. M. Temme, "Asymptotic estimates of Stirling numbers", Studies
//! in Applied Mathematics, 89(3):233-243, 1993.
//!
//! Deviation: the public `stirling2` applies one argument contract to both
//! strategies (`generic::stirling2::prepare`: truncation towards zero, `n < 0` or
//! `k < 0` gives 0, `S(+∞, k) = +∞`) before calling this kernel, so it only sees
//! integers `2 <= k < n`. xsf compares the raw reals and evaluates Temme at
//! non-integer `n > 219`. Two further deviations keep the `f64` path from
//! returning NaN where the value overflows: `S(n, 2) = 2^(n-1) - 1` for
//! `n > 219`, and the Temme product taken in logarithms when its factors over-
//! or underflow separately.

use super::binom::binom;
use super::gamma::lgam_sgn;
use super::lambertw::lambertw0_neg;

/// Dynamic programming, valid for `n <= 219` (the stack array of xsf holds
/// at most 111 entries).
fn stirling2_dp(n: f64, k: f64) -> f64 {
    if (n == 0.0 && k == 0.0) || (n == 1.0 && k == 1.0) {
        return 1.0;
    }
    if k < 1.0 || k > n || n < 0.0 {
        return 0.0;
    }
    if k.is_nan() {
        return f64::NAN;
    }

    let ni = n as i32;
    let ki = k as i32;
    let array_size = if ki <= ni - ki + 1 { ki } else { ni - ki + 1 } as usize;

    const MAX_ARRAY_SIZE: usize = 111;
    if array_size > MAX_ARRAY_SIZE {
        return f64::NAN;
    }

    let mut curr = [1.0_f64; MAX_ARRAY_SIZE];
    if ki <= ni - ki + 1 {
        for _ in 1..ni - ki + 1 {
            for j in 1..ki as usize {
                curr[j] = (j + 1) as f64 * curr[j] + curr[j - 1];
                if curr[j].is_infinite() {
                    return f64::INFINITY;
                }
            }
        }
    } else {
        for i in 1..ki {
            for j in 1..(ni - ki + 1) as usize {
                curr[j] += f64::from(i + 1) * curr[j - 1];
                if curr[j].is_infinite() {
                    return f64::INFINITY;
                }
            }
        }
    }
    curr[array_size - 1]
}

/// Second-order Temme asymptotic approximation for large `n`.
pub(crate) fn stirling2_temme(n: f64, k: f64) -> f64 {
    if (n == k && n >= 0.0) || (n > 0.0 && k == 1.0) {
        return 1.0;
    }
    if k <= 0.0 || k > n || n < 0.0 {
        return 0.0;
    }

    let mu = k / n;
    let d = (-1.0 / mu).exp() / mu;
    // The real part of lambertw(-d, 0, 1e-8).
    let lwv = lambertw0_neg(-d, 1e-8);
    let x0 = lwv + 1.0 / mu;
    let t0 = 1.0 / mu - 1.0;
    let f = (t0 / ((1.0 + t0) * (x0 - t0))).sqrt();
    let a = -n * x0.ln() + k * (x0.exp() - 1.0).ln() - k * t0 + (n - k) * t0.ln();

    // F1 correction term (Horner scheme applied to the numerator).
    let xt = x0 * t0;
    let t0power3 = t0 * t0 * t0;
    let mut num = -2.0 * x0 * x0 * x0;
    num += ((t0 + 2.0) * t0 + 2.0) * (2.0 * t0power3);
    num +=
        (-6.0 * t0power3 + (8.0 * t0 - 6.0 * x0 - 5.0) * xt + ((2.0 * x0 + 1.0) * x0 + 3.0) * x0)
            * xt;
    let denom = 24.0 * f * (1.0 + t0) * (1.0 + t0) * (x0 - t0) * (x0 - t0) * (x0 - t0) * (x0 - t0);
    let f1 = num / denom;

    let v = a.exp() * k.powf(n - k) * binom(n, k) * (f - f1 / k);
    if v.is_nan() {
        // Deviation from xsf: the product of the factors is `0 * ∞` (NaN) when
        // they over- or underflow separately. Take it in logarithms instead.
        let lbinom = lgam_sgn(n + 1.0).0 - lgam_sgn(k + 1.0).0 - lgam_sgn(n - k + 1.0).0;
        let l = a + (n - k) * k.ln() + lbinom + (f - f1 / k).ln();
        return l.exp();
    }
    v
}

pub(crate) fn stirling2(n: f64, k: f64) -> f64 {
    // 219 is the largest n for which no S(n, k) overflows.
    if n <= 219.0 {
        return stirling2_dp(n, k);
    }
    if k == 2.0 {
        // Deviation from xsf: S(n, 2) = 2^(n-1) - 1 exactly (xsf's Temme gives NaN once
        // e^(n/2) overflows), the same closed form as the double-double path.
        return if n >= 1025.0 {
            f64::INFINITY
        } else {
            2.0_f64.powi(n as i32 - 1) - 1.0
        };
    }
    stirling2_temme(n, k)
}
