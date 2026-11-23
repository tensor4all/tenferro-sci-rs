//! Inverse of the digamma function for `f64`.
//!
//! Ported from xsf `include/xsf/digammainv.h` (`digammainv`) and
//! `include/xsf/tools.h` (`find_root_newton`).
//! Reference: T. Minka, "Estimating a Dirichlet distribution", 2000,
//! Appendix C.

use super::consts::{MAXLOG, SCIPY_EULER};
use super::psi::psi;
use super::zeta::zeta;

/// `xsf::detail::NewtonRootFinderStatus` (the variants xsf's callers inspect).
#[derive(PartialEq, Eq, Clone, Copy)]
enum Status {
    Success,
    MaxIterationsExceeded,
    DerivativeZero,
    ObjectiveReturnedNan,
    ObjectiveReturnedInf,
    InitialGuessReturnedInf,
}

/// `xsf::detail::find_root_newton` with xsf's default tolerances.
fn find_root_newton(func: impl Fn(f64) -> (f64, f64), mut x: f64) -> (f64, Status) {
    let rtol = 4.0 * f64::EPSILON;
    let atol = 0.0;
    let maxiter = 100;
    if x.is_infinite() {
        return (x, Status::InitialGuessReturnedInf);
    }
    let mut x_previous = x;
    for i in 0..maxiter {
        let (f, df) = func(x);
        if f.is_nan() || df.is_nan() {
            return (f64::NAN, Status::ObjectiveReturnedNan);
        }
        if f == 0.0 {
            return (x, Status::Success);
        }
        if !f.is_finite() || !df.is_finite() {
            return (f64::NAN, Status::ObjectiveReturnedInf);
        }
        if df == 0.0 {
            return (f64::NAN, Status::DerivativeZero);
        }
        let step = f / df;
        let x_next = x - step;
        if !x_next.is_finite() {
            return (f64::NAN, Status::ObjectiveReturnedInf);
        }
        if x == x_next || step.abs() <= rtol * x.abs() + atol {
            return (x_next, Status::Success);
        }
        if i > 0 && x_next == x_previous {
            // xsf: sf_error LOSS, then success.
            return (x, Status::Success);
        }
        x_previous = x;
        x = x_next;
    }
    (x, Status::MaxIterationsExceeded)
}

/// Initial guess of Minka (2000): `exp(y) + 1/2` for `y >= -2.22`, else `-1/(y + γ)`.
fn initial_guess(y: f64) -> f64 {
    if y >= -2.22 {
        return y.exp() + 0.5;
    }
    -1.0 / (y + SCIPY_EULER)
}

/// Newton–Raphson on `ψ(x) - y` with `ψ'(x) = ζ(2, x)`.
fn newton_raphson(x: f64, y: f64) -> f64 {
    let (res, status) = find_root_newton(|x| (psi(x) - y, zeta(2.0, x)), x);
    match status {
        Status::MaxIterationsExceeded | Status::DerivativeZero | Status::ObjectiveReturnedNan => {
            f64::NAN
        }
        // `res` is NaN for ObjectiveReturnedInf; the initial guess for the other.
        Status::ObjectiveReturnedInf | Status::InitialGuessReturnedInf | Status::Success => res,
    }
}

/// The `x > 0` with `ψ(x) = y`.
pub(crate) fn digammainv(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    // ψ(x) ≈ ln x for large x, so the inverse overflows for y > ln(DBL_MAX).
    if y > MAXLOG {
        return f64::INFINITY;
    }
    // ψ(x) → -∞ for x → 0+.
    if y.is_infinite() && y < 0.0 {
        return 0.0;
    }
    newton_raphson(initial_guess(y), y)
}
