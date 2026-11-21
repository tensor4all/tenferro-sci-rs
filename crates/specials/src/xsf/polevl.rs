//! Polynomial and Chebyshev evaluation.
//!
//! Ported from xsf `include/xsf/cephes/polevl.h` (`polevl`, `p1evl`, `ratevl`)
//! and `chbevl.h`.

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

/// Rational function `num(x) / denom(x)` with coefficients in descending
/// order, both of degree `num.len() - 1` and `denom.len() - 1`; evaluated as a
/// polynomial in `1/x` for `|x| > 1` (`ratevl` in `polevl.h`).
pub(crate) fn ratevl(x: f64, num: &[f64], denom: &[f64]) -> f64 {
    let m = num.len() as i32 - 1;
    let n = denom.len() as i32 - 1;
    let absx = x.abs();
    if absx > 1.0 {
        let y = 1.0 / x;
        let num_ans = num
            .iter()
            .rev()
            .skip(1)
            .fold(num[num.len() - 1], |a, &c| a * y + c);
        let denom_ans = denom
            .iter()
            .rev()
            .skip(1)
            .fold(denom[denom.len() - 1], |a, &c| a * y + c);
        x.powi(m - n) * num_ans / denom_ans
    } else {
        let num_ans = num[1..].iter().fold(num[0], |a, &c| a * x + c);
        let denom_ans = denom[1..].iter().fold(denom[0], |a, &c| a * x + c);
        num_ans / denom_ans
    }
}
