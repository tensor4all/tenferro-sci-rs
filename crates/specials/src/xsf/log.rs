//! x ln y and x ln(1 + y) for `f64`.
//!
//! Ported from xsf `include/xsf/log.h` (`xlogy` and `xlog1py`, real
//! template). The complex `log1p` of that header, the only code there that
//! uses `cephes/dd_real.h` (the qd library, BSD-3-Clause-LBNL), is not ported.

use super::unity::log1p;

/// x ln y, with `0 ln y = 0` for every `y` that is not NaN.
pub(crate) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    x * y.ln()
}

/// x ln(1 + y), with `0 ln(1 + y) = 0` for every `y` that is not NaN.
pub(crate) fn xlog1py(x: f64, y: f64) -> f64 {
    if x == 0.0 && !y.is_nan() {
        return 0.0;
    }
    x * log1p(y)
}
