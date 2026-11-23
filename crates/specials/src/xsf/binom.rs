//! Binomial coefficient for `f64`.
//!
//! Ported from xsf `include/xsf/binom.h` (`binom`). Original authors: Pauli
//! Virtanen, Eric Moore; C++ translation by SciPy developers (2024).

use super::beta::{beta, lbeta};
use super::gamma::gamma;
use super::sinpi;

/// `static_cast<int>(x) == x` without the undefined behavior of the C++ cast.
fn is_i32(x: f64) -> bool {
    f64::from(x as i32) == x
}

pub(crate) fn binom(n: f64, k: f64) -> f64 {
    if n < 0.0 {
        let nx = n.floor();
        if n == nx {
            // Undefined.
            return f64::NAN;
        }
    }

    let mut kx = k.floor();
    if k == kx && (n.abs() > 1e-8 || n == 0.0) {
        // Integer case: the multiplication formula has less rounding error
        // for results that are integers. It cannot be used for small nonzero
        // n because of loss of precision.
        let nx = n.floor();
        if nx == n && kx > nx / 2.0 && nx > 0.0 {
            // Reduce kx by symmetry.
            kx = nx - kx;
        }

        if (0.0..20.0).contains(&kx) {
            let mut num = 1.0;
            let mut den = 1.0;
            for i in 1..1 + kx as i32 {
                num *= (f64::from(i) + n) - kx;
                den *= f64::from(i);
                if num.abs() > 1e50 {
                    num /= den;
                    den = 1.0;
                }
            }
            return num / den;
        }
    }

    // General case.
    if n >= 1e10 * k && k > 0.0 {
        // Avoid under/overflows in intermediate results.
        return (-lbeta(1.0 + n - k, 1.0 + k) - (n + 1.0).ln()).exp();
    }
    if k > 1e8 * n.abs() {
        // Avoid loss of precision (DLMF 5.11.13).
        let mut num = gamma(1.0 + n) / k.abs() * (1.0 + n * (n + 1.0) / (2.0 * k)); // + ...
                                                                                    // Split k^n, which can overflow when the result does not.
        let den = k.abs().powf(n / 2.0);
        num = num / (core::f64::consts::PI * den) / den;
        if k > 0.0 {
            kx = k.floor();
            let dk = k - kx;
            let sgn = if kx % 2.0 == 0.0 { 1.0 } else { -1.0 };
            return num * sinpi(dk - n) * sgn;
        }
        kx = k.floor();
        if is_i32(kx) {
            return 0.0;
        }
        return num * (k * core::f64::consts::PI).sin();
    }
    1.0 / (n + 1.0) / beta(1.0 + n - k, 1.0 + k)
}
