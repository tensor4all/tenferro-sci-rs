//! ln Γ(1 + x) for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/unity.h` (`lgam1p`, `lgam1p_taylor`;
//! the other functions of that header are not part of this crate yet).
//! Cephes Math Library, Copyright 1984, 1987, 1989, 1992 by Stephen L.
//! Moshier; `lgam1p` added by SciPy developers (2016); C++ translation by
//! SciPy developers (2024).

use super::consts::{MACHEP, SCIPY_EULER};
use super::gamma::lgam_sgn;
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
