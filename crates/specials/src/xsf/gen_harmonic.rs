//! Generalized harmonic number `H_{n,a} = Σ_{i=1}^{n} i^-a` for `f64`.
//!
//! Ported from xsf `include/xsf/gen_harmonic.h` (`gen_harmonic` and
//! `detail::sum_powers`; `normalized_gen_harmonic` is not part of this crate).
//!
//! Deviation: xsf documents that `n` must be a positive integer and leaves the
//! check to SciPy's Python wrapper; here a non-integer `n` gives NaN.

use super::zeta::zeta;

/// If ζ(a, n + 1) / ζ(a, 1) exceeds this, the subtraction ζ(a, 1) - ζ(a, n + 1)
/// loses precision and the direct sum of powers is used instead.
const ZETA_RATIO_THRESHOLD: f64 = 0.9;

/// `Σ_{i=m}^{n} i^-a` for `1 <= m <= n` and finite `a`, summed from the
/// smaller terms to the larger.
fn sum_powers(m: f64, n: f64, a: f64) -> f64 {
    let mut sum = 0.0;
    if a >= 0.0 {
        let mut i = n;
        while i >= m {
            sum += i.powf(-a);
            i -= 1.0;
        }
    } else {
        let mut i = m;
        while i <= n {
            sum += i.powf(-a);
            i += 1.0;
        }
    }
    sum
}

pub(crate) fn gen_harmonic(n: f64, a: f64) -> f64 {
    if n.is_nan() {
        return f64::NAN;
    }
    if n < 1.0 || n != n.floor() {
        return f64::NAN;
    }
    if n == 1.0 {
        // IEEE: pow(1.0, _) is 1.0.
        return 1.0;
    }
    if a.is_nan() {
        return f64::NAN;
    }
    if a.is_infinite() {
        return if a > 0.0 { 1.0 } else { f64::INFINITY };
    }
    if a == 0.0 {
        return n;
    }
    if a <= 1.0 {
        return sum_powers(1.0, n, a);
    }
    // a is finite, a > 1 and n > 1: ζ(a, 1) - ζ(a, n + 1), unless the two are close.
    let z1 = zeta(a, 1.0);
    let znp1 = zeta(a, n + 1.0);
    if znp1 / z1 <= ZETA_RATIO_THRESHOLD {
        z1 - znp1
    } else {
        sum_powers(1.0, n, a)
    }
}
