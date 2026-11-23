//! Pochhammer symbol `(a)_m = Γ(a + m) / Γ(a)` for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/poch.h` (`poch`).
//! Cephes Math Library, Copyright 1984, 1987, 1989 by Stephen L. Moshier;
//! C++ translation by SciPy developers (2024).

use super::gamma::{gammasgn, lgam_sgn};

/// `xsf::cephes::detail::is_nonpos_int`.
fn is_nonpos_int(x: f64) -> bool {
    x <= 0.0 && x == x.ceil() && x.abs() < 1e13
}

pub(crate) fn poch(a: f64, mut m: f64) -> f64 {
    let mut r = 1.0;

    // 1. Reduce the magnitude of `m` to |m| < 1 by the recurrence relations.
    // This may over/underflow, but then the function itself diverges or goes to
    // zero; if the remainder goes the other way we return 0 * INF = NaN.

    // Recurse down.
    while m >= 1.0 {
        if a + m == 1.0 {
            break;
        }
        m -= 1.0;
        r *= a + m;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }

    // Recurse up.
    while m <= -1.0 {
        if a + m == 0.0 {
            break;
        }
        r /= a + m;
        m += 1.0;
        if !r.is_finite() || r == 0.0 {
            break;
        }
    }

    // 2. Evaluate the function with the reduced `m`.
    if m == 0.0 {
        return r;
    } else if a > 1e4 && m.abs() <= 1.0 {
        // Avoid loss of precision.
        return r
            * a.powf(m)
            * (1.0
                + m * (m - 1.0) / (2.0 * a)
                + m * (m - 1.0) * (m - 2.0) * (3.0 * m - 1.0) / (24.0 * a * a)
                + m * m * (m - 1.0) * (m - 1.0) * (m - 2.0) * (m - 3.0) / (48.0 * a * a * a));
    }

    // Check for infinity.
    if is_nonpos_int(a + m) && !is_nonpos_int(a) && a + m != m {
        return f64::INFINITY;
    }

    // Check for zero.
    if !is_nonpos_int(a + m) && is_nonpos_int(a) {
        return 0.0;
    }

    r * (lgam_sgn(a + m).0 - lgam_sgn(a).0).exp() * gammasgn(a + m) * gammasgn(a)
}
