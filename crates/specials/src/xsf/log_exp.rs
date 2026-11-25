//! The logistic sigmoid, its inverse and its logarithm for `f64`.
//!
//! Ported from xsf `include/xsf/log_exp.h` (`expit`, `logit`, `log_expit`).
//! xsf calls `std::log1p`; this port calls `f64::ln_1p`.

/// 1/(1 + e^-x).
pub(crate) fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ln(x/(1 - x)).
pub(crate) fn logit(x: f64) -> f64 {
    // The standard formula is log(x/(1 - x)), but this expression loses
    // precision near x = 0.5, as does log(x) - log1p(-x). Use the standard
    // formula away from 0.5, and log1p(2(x - 0.5)) - log1p(-2(x - 0.5)) around
    // it, which provides very good precision in this interval.
    if !(0.3..=0.65).contains(&x) {
        (x / (1.0 - x)).ln()
    } else {
        let s = 2.0 * (x - 0.5);
        s.ln_1p() - (-s).ln_1p()
    }
}

/// ln(1/(1 + e^-x)): `x - log1p(e^x)` for `x < 0`, `-log1p(e^-x)` otherwise.
pub(crate) fn log_expit(x: f64) -> f64 {
    if x < 0.0 {
        return x - x.exp().ln_1p();
    }
    -(-x).exp().ln_1p()
}
