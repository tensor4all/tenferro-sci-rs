//! ψ (digamma) for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/psi.h` (`psi`, with the rational
//! approximation on [1, 2] that xsf took from Boost) and
//! `include/xsf/digamma.h` (`digamma`, the series around the negative zero).

use super::consts::SCIPY_EULER;
use super::polevl::polevl;
use super::zeta::zeta;

const PSI_A: [f64; 7] = [
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2,
];

const PSI_Y: f64 = 0.99558162689208984_f32 as f64;
const PSI_ROOT1: f64 = 1569415565.0 / 1073741824.0;
const PSI_ROOT2: f64 = (381566830.0 / 1073741824.0) / 1073741824.0;
const PSI_ROOT3: f64 = 0.9016312093258695918615325266959189453125e-19;

const PSI_P: [f64; 6] = [
    -0.0020713321167745952,
    -0.045251321448739056,
    -0.28919126444774784,
    -0.65031853770896507,
    -0.32555031186804491,
    0.25479851061131551,
];
const PSI_Q: [f64; 7] = [
    -0.55789841321675513e-6,
    0.0021284987017821144,
    0.054151797245674225,
    0.43593529692665969,
    1.4606242909763515,
    2.0767117023730469,
    1.0,
];

/// ψ on [1, 2] as `(x - root) (Y + R(x - 1))`.
fn digamma_imp_1_2(x: f64) -> f64 {
    let mut g = x - PSI_ROOT1;
    g -= PSI_ROOT2;
    g -= PSI_ROOT3;
    let r = polevl(x - 1.0, &PSI_P) / polevl(x - 1.0, &PSI_Q);
    g * PSI_Y + g * r
}

fn psi_asy(x: f64) -> f64 {
    let y = if x < 1.0e17 {
        let z = 1.0 / (x * x);
        z * polevl(z, &PSI_A)
    } else {
        0.0
    };
    x.ln() - (0.5 / x) - y
}

pub(crate) fn psi(mut x: f64) -> f64 {
    let mut y = 0.0;
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(-x);
    }
    if x < 0.0 {
        // Argument reduction before evaluating tan(πx).
        let r = x.fract();
        if r == 0.0 {
            return f64::NAN;
        }
        y = -core::f64::consts::PI / (core::f64::consts::PI * r).tan();
        x = 1.0 - x;
    }

    // Positive integers up to 10.
    if x <= 10.0 && x == x.floor() {
        let n = x as i32;
        for i in 1..n {
            y += 1.0 / f64::from(i);
        }
        return y - SCIPY_EULER;
    }

    // Move x into [1, 2] by the recurrence.
    if x < 1.0 {
        y -= 1.0 / x;
        x += 1.0;
    } else if x < 10.0 {
        while x > 2.0 {
            x -= 1.0;
            y += 1.0 / x;
        }
    }
    if (1.0..=2.0).contains(&x) {
        return y + digamma_imp_1_2(x);
    }
    y + psi_asy(x)
}

/// Location of the negative zero near -0.504, and ψ there (computed with mpmath upstream).
const DIGAMMA_NEGROOT: f64 = -0.504083008264455409;
const DIGAMMA_NEGROOTVAL: f64 = 7.2897639029768949e-17;

fn digamma_zeta_series(z: f64, root: f64, rootval: f64) -> f64 {
    let mut res = rootval;
    let mut coeff = -1.0;
    let z = z - root;
    for n in 1..100 {
        coeff *= -z;
        let term = coeff * zeta(f64::from(n + 1), root);
        res += term;
        if term.abs() < f64::EPSILON * res.abs() {
            break;
        }
    }
    res
}

pub(crate) fn digamma(z: f64) -> f64 {
    if (z - DIGAMMA_NEGROOT).abs() < 0.3 {
        return digamma_zeta_series(z, DIGAMMA_NEGROOT, DIGAMMA_NEGROOTVAL);
    }
    psi(z)
}
