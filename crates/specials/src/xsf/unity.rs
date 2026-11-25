//! ln Γ(1 + x), cos(x) − 1, ln(1 + x) and ln(1 + x) − x for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/unity.h` (`lgam1p`, `lgam1p_taylor`,
//! `cosm1`, `log1p`, `log1pmx`). `expm1` of that header is not ported: the
//! public `expm1` uses `f64::exp_m1`, which passes xsref's `expm1` table.
//! Cephes Math Library, Copyright 1984, 1987, 1989, 1992 by Stephen L.
//! Moshier; `lgam1p` added by SciPy developers (2016); C++ translation by
//! SciPy developers (2024).

use super::consts::{MACHEP, SCIPY_EULER};
use super::gamma::lgam_sgn;
use super::polevl::{p1evl, polevl};
use super::zeta::zeta;

/// `xsf::cephes::detail::MAXITER`.
const MAXITER: u64 = 500;

/// ln Γ(x + 1) around x = 0 from its Taylor series.
fn lgam1p_taylor(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut res = -SCIPY_EULER * x;
    let mut xfac = -x;
    for n in 2..MAXITER {
        xfac *= -x;
        let coeff = zeta(n as f64, 1.0) * xfac / n as f64;
        res += coeff;
        if coeff.abs() <= MACHEP * res.abs() {
            break;
        }
    }
    res
}

/// ln Γ(x + 1).
pub(crate) fn lgam1p(x: f64) -> f64 {
    if x.abs() <= 0.5 {
        lgam1p_taylor(x)
    } else if (x - 1.0).abs() < 0.5 {
        x.ln() + lgam1p_taylor(x - 1.0)
    } else {
        lgam_sgn(x + 1.0).0
    }
}

const UNITY_COSCOF: [f64; 7] = [
    4.7377507964246204691685E-14,
    -1.1470284843425359765671E-11,
    2.0876754287081521758361E-9,
    -2.7557319214999787979814E-7,
    2.4801587301570552304991E-5,
    -1.3888888888888872993737E-3,
    4.1666666666666666609054E-2,
];

/// cos(x) - 1.
pub(crate) fn cosm1(x: f64) -> f64 {
    if !(-core::f64::consts::FRAC_PI_4..=core::f64::consts::FRAC_PI_4).contains(&x) {
        return x.cos() - 1.0;
    }
    let xx = x * x;
    -0.5 * xx + xx * xx * polevl(xx, &UNITY_COSCOF)
}

const UNITY_LP: [f64; 7] = [
    4.5270000862445199635215E-5,
    4.9854102823193375972212E-1,
    6.5787325942061044846969E0,
    2.9911919328553073277375E1,
    6.0949667980987787057556E1,
    5.7112963590585538103336E1,
    2.0039553499201281259648E1,
];

const UNITY_LQ: [f64; 6] = [
    1.5062909083469192043167E1,
    8.3047565967967209469434E1,
    2.2176239823732856465394E2,
    3.0909872225312059774938E2,
    2.1642788614495947685003E2,
    6.0118660497603843919306E1,
];

/// ln(1 + x), Cephes' `log1p` (xsf's `xsf::log1p` for real arguments).
pub(crate) fn log1p(x: f64) -> f64 {
    let z = 1.0 + x;
    if !(core::f64::consts::FRAC_1_SQRT_2..=core::f64::consts::SQRT_2).contains(&z) {
        return z.ln();
    }
    let zz = x * x;
    let zz = -0.5 * zz + x * (zz * polevl(x, &UNITY_LP) / p1evl(x, &UNITY_LQ));
    x + zz
}

/// ln(1 + x) - x.
pub(crate) fn log1pmx(x: f64) -> f64 {
    if x.abs() < 0.5 {
        let mut xfac = x;
        let mut res = 0.0;
        for n in 2..MAXITER {
            xfac *= -x;
            let term = xfac / n as f64;
            res += term;
            if term.abs() < MACHEP * res.abs() {
                break;
            }
        }
        res
    } else {
        log1p(x) - x
    }
}
