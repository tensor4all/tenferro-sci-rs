//! Circular tangent and cotangent of an argument in degrees, for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/tandg.h` (`tandg`, `cotdg`,
//! `detail::tancot`). Cephes Math Library Release 2.0: April, 1987. Copyright
//! 1984, 1987 by Stephen L. Moshier. C++ translation by SciPy developers
//! (2024).
//!
//! Deviation from upstream: xsf reports `SF_ERROR_NO_RESULT` (|x| > 1e14,
//! result 0) and `SF_ERROR_SINGULAR` (poles, result ±∞); this port returns the
//! same values without the flag.

use super::sindg::{LOSSTH, PI180};

/// tan(x°) (`cot = false`) or cot(x°) (`cot = true`).
pub(crate) fn tancot(xx: f64, cot: bool) -> f64 {
    // make argument positive but save the sign
    let (mut x, mut sign) = if xx.is_sign_negative() {
        (-xx, -1.0)
    } else {
        (xx, 1.0)
    };
    if x > LOSSTH {
        return 0.0;
    }
    // modulo 180
    let k = (x / 180.0).floor();
    x -= 180.0 * k;
    if cot {
        if x <= 90.0 {
            x = 90.0 - x;
        } else {
            x -= 90.0;
            sign = -sign;
        }
    } else if x > 90.0 {
        x = 180.0 - x;
        sign = -sign;
    }
    // k <= 1e14 / 180, so the cast is exact.
    let k_odd = (k as i64) & 1 != 0;
    if x == 0.0 {
        if k_odd {
            sign = -sign;
        }
        return sign * 0.0;
    } else if x == 45.0 {
        return sign * 1.0;
    } else if x == 90.0 {
        if k_odd {
            sign = -sign;
        }
        return sign * f64::INFINITY;
    }
    // x is now transformed into [0, 90)
    sign * (x * PI180).tan()
}

/// tan(x°).
pub(crate) fn tandg(x: f64) -> f64 {
    tancot(x, false)
}

/// cot(x°).
pub(crate) fn cotdg(x: f64) -> f64 {
    tancot(x, true)
}
