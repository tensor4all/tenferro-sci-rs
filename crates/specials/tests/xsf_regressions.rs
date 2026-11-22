//! Regression tests for real arguments, ported from SciPy's xsf
//! (`tests/xsf_tests/test_gammaln.cpp`, `test_rgamma.cpp`, `test_zeta.cpp` and
//! `test_betaln.cpp` at commit `31177fd9a458539cf05c04d215341245d0245e53`).
//!
//! The xsf sources are BSD-3-Clause licensed; see `LICENSE-BSD-3-Clause-xsf`
//! in this crate. Reference values, tolerances and regression-issue comments
//! are kept as in xsf. Every case runs for `f64` and for `xprec::Df64`; the
//! `Df64` result is rounded to its high word (`hi()`) and compared against the
//! same f64-rounded reference with the same tolerance. Complex-valued and
//! float-template cases are not ported.

use specials::Real;
use xprec::Df64;

/// Test-only bridge between `f64` references and the generic `Real` types.
trait Num: Real {
    const NAME: &'static str;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Num for f64 {
    const NAME: &'static str = "f64";
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl Num for Df64 {
    const NAME: &'static str = "Df64";
    fn from_f64(x: f64) -> Self {
        Df64::from(x)
    }
    fn to_f64(self) -> f64 {
        self.hi()
    }
}

/// xsf's `extended_absolute_error` for real `T = f64`.
fn extended_absolute_error(actual: f64, desired: f64) -> f64 {
    if actual == desired || (actual.is_nan() && desired.is_nan()) {
        return 0.0;
    }
    if desired.is_nan() || actual.is_nan() {
        // Expected NaN but got non-NaN (or vice versa): infinite error.
        return f64::INFINITY;
    }
    // max_float * 2**-digits = ulp(max_float)
    let ulp = 2.0_f64.powf(-(f64::MANTISSA_DIGITS as f64)) * f64::MAX;
    if actual.is_infinite() {
        // Do not penalize early overflow too harshly: compare with the
        // mythical value nextafter(max_float).
        let sgn = 1.0_f64.copysign(actual);
        return ((sgn * f64::MAX - desired) + sgn * ulp).abs();
    }
    if desired.is_infinite() {
        let sgn = 1.0_f64.copysign(desired);
        return ((sgn * f64::MAX - actual) + sgn * ulp).abs();
    }
    (actual - desired).abs()
}

/// xsf's `extended_relative_error` for real `T = f64`.
fn extended_relative_error(actual: f64, desired: f64) -> f64 {
    let abs_error = extended_absolute_error(actual, desired);
    let abs_desired = if desired == 0.0 {
        // Normalize by the smallest subnormal instead of zero.
        f64::from_bits(1)
    } else if desired.is_infinite() {
        f64::MAX
    } else if desired.is_nan() {
        // extended_relative_error(nan, nan) = 0, infinite if only one is NaN.
        1.0
    } else {
        desired.abs()
    };
    abs_error / abs_desired
}

/// `numpy.linspace(start, end, n)` as in xsf's `linspace`.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f64;
            let mut xs: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
            xs[n - 1] = end;
            xs
        }
    }
}

macro_rules! for_both {
    ($f:ident) => {{
        $f::<f64>();
        $f::<Df64>();
    }};
}

// ---------------------------------------------------------------- gammaln

fn gammaln_subnormal_arguments<T: Num>() {
    // Regression test for gh-540: the x < 13 branch divided by x, overflowed, and returned inf.
    // Reference values: float(mpmath.log(abs(mpmath.gamma(x)))) with mpmath.mp.dps = 60.
    let cases = [(5e-309, 709.889355822726), (-5e-309, 709.889355822726)];
    for (x, desired) in cases {
        let out = specials::lgamma(T::from_f64(x)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 5e-16;
        assert!(
            error <= rtol,
            "[{}] lgamma(x={x:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn gammaln_subnormal_arguments_test() {
    for_both!(gammaln_subnormal_arguments);
}

// ----------------------------------------------------------------- rgamma

fn rgamma_neg178_to_neg170<T: Num>() {
    // Regression test for gh-377: this returned inf because Gamma(x) underflows.
    // Reference values: float(mpmath.rgamma(x)) with mpmath.mp.dps = 50.
    let cases = [
        (-170.91716291792244, -6.637711907120738e+307),
        (-175.99999999999997, 5.624744700044757e+306),
        (-176.00000000000003, -5.624744700046411e+306),
    ];
    for (x, desired) in cases {
        let out = specials::rgamma(T::from_f64(x)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 1e-15;
        assert!(
            error <= rtol,
            "[{}] rgamma(x={x:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn rgamma_neg178_to_neg170_test() {
    for_both!(rgamma_neg178_to_neg170);
}

fn rgamma_tiny_subnormal_x<T: Num>() {
    // Regression test for gh-417: negative x returned -0.0 because 1/x overflows.
    // Reference values: float(mpmath.rgamma(x)) with mpmath.mp.dps = 50.
    // Exact equality, including the sign (for a subnormal input the result
    // must not be a zero, so `==` against the nonzero input also rules out +-0).
    for x in [-1e-310, -5e-324, 1e-310, 5e-324] {
        let out = specials::rgamma(T::from_f64(x)).to_f64();
        assert!(
            out == x,
            "[{}] rgamma(x={x:e}): out={out:e} (bits {:#x}), expected exactly {x:e}",
            T::NAME,
            out.to_bits()
        );
    }
}

#[test]
fn rgamma_tiny_subnormal_x_test() {
    for_both!(rgamma_tiny_subnormal_x);
}

// ------------------------------------------------------------------- zeta

fn zeta_near_one_tiny_q<T: Num>() {
    // Regression test for gh-345: the direct sum stopped early, dropping the
    // tail of about (q + n)^(1 - x) / (x - 1).
    // Reference values: float(mpmath.zeta(x, q)) with mpmath.mp.dps = 50.
    let cases = [
        (1.0000000000000002, 1e-15, 5503599627370504.0),
        (1.000000000001, 1e-17, 1.0000099991502206e+17),
    ];
    for (x, q, desired) in cases {
        let out = specials::zeta(T::from_f64(x), T::from_f64(q)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 5e-16;
        assert!(
            error <= rtol,
            "[{}] zeta(x={x:e}, q={q:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn zeta_near_one_tiny_q_test() {
    for_both!(zeta_near_one_tiny_q);
}

// xsf checks `zeta(x, q=1) == riemann_zeta(x)` for double, float and complex.
// This crate has no separate `riemann_zeta`, so the double case is adapted:
// for x in linspace(-20, 20, 81), `zeta(x, 1)` must not be NaN, except that
// x == 1 (the pole) must give +inf. The float and complex cases are skipped.
fn zeta_q1_is_riemann_zeta<T: Num>() {
    for x in linspace(-20.0, 20.0, 81) {
        let out = specials::zeta(T::from_f64(x), T::from_f64(1.0)).to_f64();
        if x == 1.0 {
            assert!(
                out == f64::INFINITY,
                "[{}] zeta(x={x:e}, q=1): out={out:e}, expected +inf",
                T::NAME
            );
        } else {
            assert!(
                !out.is_nan(),
                "[{}] zeta(x={x:e}, q=1): out={out:e}, expected non-NaN",
                T::NAME
            );
        }
    }
}

#[test]
fn zeta_q1_is_riemann_zeta_test() {
    for_both!(zeta_q1_is_riemann_zeta);
}

fn zeta_q_above_1e8<T: Num>() {
    // Regression test for gh-345: the asymptotic expansion omitted the B_2 term.
    // Reference values: float(mpmath.zeta(x, q)) with mpmath.mp.dps = 50.
    let cases = [
        (20.0, 1.0000001e8, 5.263148394745859e-154),
        (30.0, 1.0000001e8, 3.4482663620824903e-234),
        (35.0, 1.2e8, 5.975287124820555e-277),
    ];
    for (x, q, desired) in cases {
        let out = specials::zeta(T::from_f64(x), T::from_f64(q)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 5e-16;
        assert!(
            error <= rtol,
            "[{}] zeta(x={x:e}, q={q:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn zeta_q_above_1e8_test() {
    for_both!(zeta_q_above_1e8);
}

// ----------------------------------------------------------------- betaln

fn betaln_large_arguments<T: Num>() {
    // Before the fix lbeta_asymp overflowed b*(1-b), b*b*(1-b)*(1-b) or a*a*a
    // and returned NaN (inf/inf) for every case in the first group.
    let rtol = 1e-13;
    // Reference values: lgamma(a) + lgamma(b) - lgamma(a + b) with mpmath 1.3.0,
    // computed at a precision where a + b is exact, e.g. for the first case
    //     mp.dps = 80
    //     mp.loggamma(mpf(1e100)) + mp.loggamma(mpf(1e107)) - mp.loggamma(mpf(1e100) + mpf(1e107))
    //     = -1.711809570095832...e+101
    // (dps = 50 + log10(max(a, b) / min(a, b)) + 10 for each case), then rounded
    // to double.  The inputs are powers of ten that are exact in double.
    let cases = [
        (1e100, 1e107, -1.711809570095832e+101),
        (1e150, 1e157, -1.711809570095832e+151),
        (1e200, 1e230, -7.007755278982136e+201),
        (1e7, 1e100, -2151404143.624572),
        // ratio below the threshold: the non-asymptotic path, unchanged by the fix
        (1e10, 1e20, -240258509310.49857),
        (1e20, 1e30, -2.4025850929990457e+21),
        (2.0, 3.0, -2.4849066497880004),
    ];
    for (a, b, expected) in cases {
        let output = specials::lbeta(T::from_f64(a), T::from_f64(b)).to_f64();
        let rel_error = extended_relative_error(output, expected);
        assert!(
            output.is_finite(),
            "[{}] betaln(a={a:e}, b={b:e}): output={output:e} is not finite (expected={expected:e})",
            T::NAME
        );
        assert!(
            rel_error <= rtol,
            "[{}] betaln(a={a:e}, b={b:e}): output={output:e}, expected={expected:e}, rtol={rtol:e}, rel_error={rel_error:e}",
            T::NAME
        );
    }
}

#[test]
fn betaln_large_arguments_test() {
    for_both!(betaln_large_arguments);
}

fn betaln_subnormal_arguments<T: Num>() {
    // Regression test for gh-540: Gamma(b) overflowed and betaln returned inf.
    // Reference values: float(mpmath.log(abs(mpmath.beta(a, b)))) with mpmath.mp.dps = 60.
    let cases = [
        (5e-309, 1.0, 709.889355822726),
        (1e-320, 0.5, 736.8272408909739),
        (3e-310, -2e-310, 712.0096193589261),
    ];
    for (a, b, desired) in cases {
        let out = specials::lbeta(T::from_f64(a), T::from_f64(b)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 5e-16;
        assert!(
            error <= rtol,
            "[{}] betaln(a={a:e}, b={b:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn betaln_subnormal_arguments_test() {
    for_both!(betaln_subnormal_arguments);
}

fn betaln_subnormal_a_plus_b_zero<T: Num>() {
    // Regression test for gh-540: Gamma(b) overflowed, and multiplying by rgamma(0) gave NaN.
    let out = specials::lbeta(T::from_f64(5e-309), T::from_f64(-5e-309)).to_f64();
    assert!(
        out == f64::NEG_INFINITY,
        "[{}] betaln(a=5e-309, b=-5e-309): out={out:e}, expected exactly -inf",
        T::NAME
    );
}

#[test]
fn betaln_subnormal_a_plus_b_zero_test() {
    for_both!(betaln_subnormal_a_plus_b_zero);
}
