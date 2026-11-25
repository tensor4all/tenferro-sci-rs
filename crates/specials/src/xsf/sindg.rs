//! Circular sine and cosine of an argument in degrees, and degrees, minutes,
//! seconds to radians, for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/sindg.h` (`sindg`, `cosdg`, `radian`).
//! Cephes Math Library Release 2.0: April, 1987. Copyright 1985, 1987 by
//! Stephen L. Moshier. C++ translation by SciPy developers (2024).
//!
//! Deviation from upstream: for `|x| > 1e14` xsf reports `SF_ERROR_NO_RESULT`
//! and returns 0; this port returns 0 without the flag.

use super::polevl::polevl;

const SINCOF: [f64; 6] = [
    1.58962301572218447952E-10,
    -2.50507477628503540135E-8,
    2.75573136213856773549E-6,
    -1.98412698295895384658E-4,
    8.33333333332211858862E-3,
    -1.66666666666666307295E-1,
];

const COSCOF: [f64; 7] = [
    1.13678171382044553091E-11,
    -2.08758833757683644217E-9,
    2.75573155429816611547E-7,
    -2.48015872936186303776E-5,
    1.38888888888806666760E-3,
    -4.16666666666666348141E-2,
    4.99999999999999999798E-1,
];

/// `xsf::cephes::detail::PI180` (`cephes/const.h`), π/180.
pub(crate) const PI180: f64 = 1.745329251994329576923690768488612713443E-2;

/// `sindg_lossth`: beyond this the argument has no fractional degrees left.
pub(crate) const LOSSTH: f64 = 1.0e14;

/// Octant reduction shared by `sindg` and `cosdg`: `(y, j)` with `y` the
/// octant index (rounded up to even) and `j` the octant modulo 8.
fn octant(x: f64) -> (f64, i32) {
    let mut y = (x / 45.0).floor(); // integer part of x / 45
                                    // strip high bits of the integer part to prevent integer overflow
    let mut z = ldexp(y, -4).floor(); // integer part of y / 16
    z = y - ldexp(z, 4); // y - 16 (y / 16)
    let mut j = z as i32;
    // map zeros to origin
    if j & 1 != 0 {
        j += 1;
        y += 1.0;
    }
    (y, j & 0o7)
}

/// `ldexp` for the powers of two used here (`2^±4`; exact).
fn ldexp(x: f64, e: i32) -> f64 {
    x * 2.0_f64.powi(e)
}

/// sin(x°).
pub(crate) fn sindg(mut x: f64) -> f64 {
    // make argument positive but save the sign
    let mut sign = 1.0;
    if x.is_sign_negative() {
        x = -x;
        sign = -1.0;
    }
    if x > LOSSTH {
        return 0.0;
    }
    let (y, mut j) = octant(x);
    // reflect in x axis
    if j > 3 {
        sign = -sign;
        j -= 4;
    }
    let z = (x - y * 45.0) * PI180; // x mod 45 degrees, in radians
    let zz = z * z;
    let r = if j == 1 || j == 2 {
        1.0 - zz * polevl(zz, &COSCOF)
    } else {
        z + z * (zz * polevl(zz, &SINCOF))
    };
    if sign < 0.0 {
        -r
    } else {
        r
    }
}

/// cos(x°).
pub(crate) fn cosdg(mut x: f64) -> f64 {
    // make argument positive
    let mut sign = 1.0;
    if x < 0.0 {
        x = -x;
    }
    if x > LOSSTH {
        return 0.0;
    }
    let (y, mut j) = octant(x);
    if j > 3 {
        j -= 4;
        sign = -sign;
    }
    if j > 1 {
        sign = -sign;
    }
    let z = (x - y * 45.0) * PI180; // x mod 45 degrees, in radians
    let zz = z * z;
    let r = if j == 1 || j == 2 {
        z + z * (zz * polevl(zz, &SINCOF))
    } else {
        1.0 - zz * polevl(zz, &COSCOF)
    };
    if sign < 0.0 {
        -r
    } else {
        r
    }
}

/// One arc second in radians.
const P64800: f64 = 4.848136811095359935899141023579479759563533023727e-6;

/// Degrees, minutes and seconds to radians.
pub(crate) fn radian(d: f64, m: f64, s: f64) -> f64 {
    ((d * 60.0 + m) * 60.0 + s) * P64800
}
