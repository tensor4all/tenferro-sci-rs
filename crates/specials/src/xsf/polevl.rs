//! Polynomial and Chebyshev evaluation.
//!
//! Ported from xsf `include/xsf/cephes/polevl.h` and `chbevl.h`.

/// `coef[0] x^N + ... + coef[N]`.
pub(crate) fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef[1..].iter().fold(coef[0], |ans, &c| ans * x + c)
}

/// Like [`polevl`] with an implicit leading coefficient 1.
pub(crate) fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().fold(1.0, |ans, &c| ans * x + c)
}

/// Chebyshev series `Σ' coef[i] T_i(x/2)` with coefficients in reverse order.
pub(crate) fn chbevl(x: f64, coef: &[f64]) -> f64 {
    let mut b0 = coef[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in &coef[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}
