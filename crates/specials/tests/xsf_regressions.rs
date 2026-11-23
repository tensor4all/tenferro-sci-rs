//! Regression tests for real arguments, ported from SciPy's xsf
//! (`tests/xsf_tests/test_gammaln.cpp`, `test_rgamma.cpp`, `test_zeta.cpp`,
//! `test_betaln.cpp`, `test_binom.cpp`, `test_trigamma.cpp`,
//! `test_digammainv.cpp`, `test_gen_harmonic.cpp`, `test_lgam1p.cpp`,
//! `test_zetac.cpp`, `test_stirling2.cpp`, `test_ellip.cpp`, `test_ellpj.cpp`,
//! `test_agm.cpp` and `test_ellint_carlson.cpp` at commit
//! `31177fd9a458539cf05c04d215341245d0245e53`).
//!
//! The xsf sources are BSD-3-Clause licensed; see `LICENSE-BSD-3-Clause-xsf`
//! in this crate. Reference values, tolerances and regression-issue comments
//! are kept as in xsf. Every case runs for `f64` and for `xprec::Df64`; the
//! `Df64` result is rounded to its high word (`hi()`) and compared against the
//! same f64-rounded reference with the same tolerance. Complex-valued and
//! float-template cases are not ported.

// These tests exercise `xprec::Df64`.
#![cfg(feature = "xprec")]
// Reference values keep the digits xsf's tests give them.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

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

/// `logspace<double>(start, end, n)` of xsf's tests: `10^linspace(start, end, n)`.
fn logspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    linspace(start, end, n)
        .into_iter()
        .map(|e| 10.0_f64.powf(e))
        .collect()
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

// ------------------------------------------------------------------ binom

fn binom_k_above_1e8_abs_n<T: Num>() {
    // Regression test for gh-352: the asymptotic branch for large k missed the
    // n + 1 factor, overflowed an int cast for k >= 2**31, used sin instead of
    // sinpi, and returned 0 when k**n overflowed.
    // Reference values: float(mpmath.binomial(n, k)) with mpmath.mp.dps = 50.
    let cases = [
        (20.5, 2100000000.25, -9.316864069123419e-183),
        (-10.5, 500000000000.25, 8.617145704224517e+104),
        (1.0, 2147483647.0, 0.0),
        (
            31.68727107709636,
            10283293173.796093,
            5.0927883742793725e-294,
        ),
    ];
    for (n, k, desired) in cases {
        let out = specials::binom(T::from_f64(n), T::from_f64(k)).to_f64();
        let error = extended_relative_error(out, desired);
        // limited by the two-term expansion and the rounding of k - floor(k) - n
        let rtol = 2e-14;
        assert!(
            error <= rtol,
            "[{}] binom(n={n:e}, k={k:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn binom_k_above_1e8_abs_n_test() {
    for_both!(binom_k_above_1e8_abs_n);
}

// -------------------------------------------------------------- trigamma

fn trigamma_real<T: Num>() {
    let rtol = 1e-14;
    let pi = core::f64::consts::PI;
    // Reference values from
    // https://github.com/JuliaMath/SpecialFunctions.jl/blob/c9ff36085c9008b9bdaa457e745f459a1827e52e/test/gamma.jl#L30-L44
    // and from mpmath (the last two).
    let cases = [
        (0.1, 101.433299150792758817),
        (0.5, pi * pi / 2.0),
        (1.0, pi * pi / 6.0),
        (2.0, pi * pi / 6.0 - 1.0),
        (3.0, pi * pi / 6.0 - 5.0 / 4.0),
        (4.0, pi * pi / 6.0 - 49.0 / 36.0),
        (5.0, pi * pi / 6.0 - 205.0 / 144.0),
        (10.0, 0.10516633568168565),
        (-7.5, 9.744766282170433),
        (-3.2, 28.29818640219408),
    ];
    for (z, desired) in cases {
        let out = specials::trigamma(T::from_f64(z)).to_f64();
        let error = extended_relative_error(out, desired);
        assert!(
            error <= rtol,
            "[{}] trigamma(z={z:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn trigamma_real_test() {
    for_both!(trigamma_real);
}

fn trigamma_real_poles<T: Num>() {
    for z in [-1.0, 0.0] {
        let out = specials::trigamma(T::from_f64(z)).to_f64();
        assert!(
            out.is_infinite(),
            "[{}] trigamma(z={z:e}): out={out:e}",
            T::NAME
        );
    }
}

#[test]
fn trigamma_real_poles_test() {
    for_both!(trigamma_real_poles);
}

fn trigamma_gamma_zeta_consistency<T: Num>() {
    // trigamma(x) = gamma(2) zeta(2, x), with the crate's own zeta and gamma
    // (zeta(2, x) for x < 0 is the Hurwitz sum, defined for integer s).
    let rtol = 1e-14;
    for x in [-7.5, -3.2, -0.5, 1e-6, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e6] {
        let out = specials::trigamma(T::from_f64(x)).to_f64();
        let two = T::from_f64(2.0);
        let reference = (specials::gamma(two) * specials::zeta(two, T::from_f64(x))).to_f64();
        let error = extended_relative_error(out, reference);
        assert!(
            error <= rtol,
            "[{}] trigamma(x={x:e}): out={out:e}, gamma(2) zeta(2, x)={reference:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn trigamma_gamma_zeta_consistency_test() {
    for_both!(trigamma_gamma_zeta_consistency);
}

// ------------------------------------------------------------ digammainv

fn digammainv_round_trip<T: Num>() {
    let rtol = 50.0 * f64::EPSILON;
    for x in logspace(-20.0, 20.0, 500) {
        let y = specials::digamma(T::from_f64(x));
        let out = specials::digammainv(y).to_f64();
        let error = extended_relative_error(out, x);
        assert!(
            error <= rtol,
            "[{}] digammainv(digamma({x:e})): out={out:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn digammainv_round_trip_test() {
    for_both!(digammainv_round_trip);
}

fn digammainv_edge_cases<T: Num>() {
    let inf = f64::INFINITY;
    let log_max_double = f64::MAX.ln();
    let next_after = f64::from_bits(log_max_double.to_bits() + 1);
    assert!(specials::digammainv(T::from_f64(f64::NAN))
        .to_f64()
        .is_nan());
    assert_eq!(specials::digammainv(T::from_f64(inf)).to_f64(), inf);
    assert_eq!(specials::digammainv(T::from_f64(-inf)).to_f64(), 0.0);
    assert_eq!(specials::digammainv(T::from_f64(next_after)).to_f64(), inf);
}

#[test]
fn digammainv_edge_cases_test() {
    for_both!(digammainv_edge_cases);
}

fn digammainv_precision<T: Num>() {
    // Some test cases taken from
    // https://en.wikipedia.org/wiki/Digamma_function#Special_values
    // Reference values for digamma were computed with mpmath.
    let cases = [
        (1e-100, -1e+100),
        (1e-50, -1e+50),
        (1e-20, -1e+20),
        (1e-10, -10000000000.577215),
        (1e-3, -1000.5755719318103),
        (1e-1, -10.423754940411076),
        (1. / 4., -4.2274535333762655),
        (1. / 3., -3.1320337800208065),
        (1. / 2., -1.9635100260214235),
        (1.0, -0.5772156649015329),
        (1.461632144968362341, 0.0), // positive root of digamma
        (10.0, 2.251752589066721),
        (1e5, 11.512920464961896),
        (1e50, 115.12925464970229),
        (1e200, 460.51701859880916),
    ];
    for (x, ref_digamma) in cases {
        let out = specials::digammainv(T::from_f64(ref_digamma)).to_f64();
        let error = extended_relative_error(out, x);
        assert!(
            error <= 1e-13,
            "[{}] digammainv(y={ref_digamma:e}): out={out:e}, expected={x:e}, error={error:e}",
            T::NAME
        );
    }
}

#[test]
fn digammainv_precision_test() {
    for_both!(digammainv_precision);
}

/// xorshift64*, standing in for xsf's `std::mt19937` (the stream differs; the
/// test is a round trip, so any uniform sample serves).
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (r >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn digammainv_random_inputs<T: Num>() {
    // Round trip with random y in [0, 40]; xsf draws 10000 values from
    // std::mt19937(42088764423867).
    let mut rng = Rng(42088764423867);
    for _ in 0..2000 {
        let y = 40.0 * rng.next_f64();
        let x = specials::digammainv(T::from_f64(y));
        let y_check = specials::digamma(x).to_f64();
        let error = extended_relative_error(y_check, y);
        assert!(
            error <= 1e-13,
            "[{}] digamma(digammainv(y={y:e})): x={:e}, y_check={y_check:e}, error={error:e}",
            T::NAME,
            x.to_f64()
        );
    }
}

#[test]
fn digammainv_random_inputs_test() {
    for_both!(digammainv_random_inputs);
}

fn digammainv_gh_361<T: Num>() {
    // gh-361 reported that digammainv yielded nan for certain inputs.
    let y = 9.220027678716601;
    let x = specials::digammainv(T::from_f64(y));
    let y_check = specials::digamma(x).to_f64();
    let error = extended_relative_error(y_check, y);
    assert!(
        error <= 1e-14,
        "[{}] digamma(digammainv(y={y:e})): y_check={y_check:e}, error={error:e}",
        T::NAME
    );
}

#[test]
fn digammainv_gh_361_test() {
    for_both!(digammainv_gh_361);
}

// ------------------------------------------------------------ gen_harmonic

fn gen_harmonic_values<T: Num>() {
    // https://github.com/scipy/scipy/blob/a125578782dd3213fb57fda0f4b97c70dd054b1d/scipy/special/tests/test_gen_harmonic.py#L11-L39
    let rtol = 5e-15;
    let cases = [
        (8.0, 9.0, 1.0020083884212339),
        (1000.0, 2.5, 1.3414661912046497),
        (10.0, 1.5, 1.9953364933456017),
        (10000.0, 1.25, 4.1951168257387765),
        (10000.0, 1.00001, 9.787182620770265),
        (80.0, 1.000002, 4.965460167788836),
        (75.0, 1.0 + 1e-12, 4.901355630543771),
        (100.0, 1.0 + 1e-14, 5.187377517639515),
        (100.0, 1.0 + 8e-16, 5.187377517639611),
        (100.0, 1.0, 5.187377517639621),
        (7.0, 1.0, 2.592857142857143),
        (8000.0, 1.0, 9.564474984261423),
        (5.0, 1.0 - 1e-12, 2.2833333333347143),
        (25000.0, 1.0 - 1e-12, 10.703866768669737),
        (1000.0, 0.995, 7.6058022857089975),
        (1000.0, 0.75, 19.055178975831392),
        (10000.0, 0.25, 1332.5700547197382),
        (5.0, 1e-8, 4.999999952125083),
        (15.0, 1e-16, 14.999999999999996),
        (100.0, 0.0, 100.0),
        (4.0, -1.0, 10.0),
        (75.0, -1.5, 19811.38815892374),
    ];
    for (n, a, expected) in cases {
        let out = specials::gen_harmonic(T::from_f64(n), T::from_f64(a)).to_f64();
        let error = extended_relative_error(out, expected);
        assert!(
            error <= rtol,
            "[{}] gen_harmonic(n={n}, a={a:e}): out={out:e}, expected={expected:e}, error={error:e}",
            T::NAME
        );
    }
}

#[test]
fn gen_harmonic_values_test() {
    for_both!(gen_harmonic_values);
}

fn gen_harmonic_exact_and_nan_cases<T: Num>() {
    // test_gen_harmonic.py#L42-L58 (the n = nan case takes a float n here).
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    let cases = [
        (10.0, inf, 1.0),
        (1.0, nan, 1.0),
        (1.0, -inf, 1.0),
        (3.0, nan, nan),
        (-3.0, 1.0, nan),
        (nan, 0.75, nan),
    ];
    for (n, a, expected) in cases {
        let out = specials::gen_harmonic(T::from_f64(n), T::from_f64(a)).to_f64();
        let error = extended_relative_error(out, expected);
        assert!(
            error <= 5e-15,
            "[{}] gen_harmonic(n={n}, a={a}): out={out:e}, expected={expected:e}",
            T::NAME
        );
    }
}

#[test]
fn gen_harmonic_exact_and_nan_cases_test() {
    for_both!(gen_harmonic_exact_and_nan_cases);
}

// ----------------------------------------------------------------- lgam1p

fn lgam1p_near_abs_x_half<T: Num>() {
    // Regression test for gh-348: the Taylor series stopped after 41 terms.
    // Reference values: float(mpmath.loggamma(1 + mpmath.mpf(x))) with mpmath.mp.dps = 50.
    let cases = [
        (-0.5, 0.5723649429247001),
        (0.5, -0.12078223763524522),
        (1.499, 0.2839799590505287),
    ];
    for (x, desired) in cases {
        let out = specials::lgam1p(T::from_f64(x)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 1e-15;
        assert!(
            error <= rtol,
            "[{}] lgam1p(x={x:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn lgam1p_near_abs_x_half_test() {
    for_both!(lgam1p_near_abs_x_half);
}

// ------------------------------------------------------------------ zetac

fn zetac_large_x<T: Num>() {
    // Regression test for gh-346: this returned 0 for 127 <= x < 1075.
    // Reference values: float(mpmath.zeta(x, 2)) with mpmath.mp.dps = 50.
    let cases = [
        (127.0, 5.877471754111438e-39),
        (1000.0, 9.332636185032189e-302),
        (1074.0, 5e-324),
    ];
    for (x, desired) in cases {
        let out = specials::zetac(T::from_f64(x)).to_f64();
        let error = extended_relative_error(out, desired);
        let rtol = 2e-16;
        assert!(
            error <= rtol,
            "[{}] zetac(x={x:e}): out={out:e}, desired={desired:e}, error={error:e}, rtol={rtol:e}",
            T::NAME
        );
    }
}

#[test]
fn zetac_large_x_test() {
    for_both!(zetac_large_x);
}

fn zetac_underflows_to_zero<T: Num>() {
    for x in [1076.0, 1e300] {
        let out = specials::zetac(T::from_f64(x)).to_f64();
        assert!(out == 0.0, "[{}] zetac(x={x:e}): out={out:e}", T::NAME);
    }
}

#[test]
fn zetac_underflows_to_zero_test() {
    for_both!(zetac_underflows_to_zero);
}

// -------------------------------------------------------------- stirling2

/// Row n of S(n, k) for n = 0..=10 (xsf `stirling2_table`).
const STIRLING2_TABLE: [&[f64]; 11] = [
    &[1.0],
    &[0.0, 1.0],
    &[0.0, 1.0, 1.0],
    &[0.0, 1.0, 3.0, 1.0],
    &[0.0, 1.0, 7.0, 6.0, 1.0],
    &[0.0, 1.0, 15.0, 25.0, 10.0, 1.0],
    &[0.0, 1.0, 31.0, 90.0, 65.0, 15.0, 1.0],
    &[0.0, 1.0, 63.0, 301.0, 350.0, 140.0, 21.0, 1.0],
    &[0.0, 1.0, 127.0, 966.0, 1701.0, 1050.0, 266.0, 28.0, 1.0],
    &[
        0.0, 1.0, 255.0, 3025.0, 7770.0, 6951.0, 2646.0, 462.0, 36.0, 1.0,
    ],
    &[
        0.0, 1.0, 511.0, 9330.0, 34105.0, 42525.0, 22827.0, 5880.0, 750.0, 45.0, 1.0,
    ],
];

fn stirling2_table_values<T: Num>() {
    let rtol = 1e-12;
    for (n, row) in STIRLING2_TABLE.iter().enumerate() {
        for (k, &expected) in row.iter().enumerate() {
            let out = specials::stirling2(T::from_f64(n as f64), T::from_f64(k as f64)).to_f64();
            let error = extended_relative_error(out, expected);
            assert!(
                error <= rtol,
                "[{}] stirling2(n={n}, k={k}): out={out:e}, expected={expected:e}, error={error:e}",
                T::NAME
            );
        }
    }
}

#[test]
fn stirling2_table_values_test() {
    for_both!(stirling2_table_values);
}

fn stirling2_single_and_mixed_values<T: Num>() {
    let cases = [
        (0.0, 0.0, 1.0),
        (4.0, 2.0, 7.0),
        (5.0, 3.0, 25.0),
        (-1.0, -2.0, 0.0),
        (3.0, 2.0, 3.0),
        (8.0, 5.0, 1050.0),
        (10.0, 7.0, 5880.0),
        (10.0, 3.0, 9330.0),
        // negative inputs
        (-1.0, -1.0, 0.0),
        (-1.0, 2.0, 0.0),
        (2.0, -1.0, 0.0),
        // k out of range
        (3.0, 4.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ];
    for (n, k, expected) in cases {
        let out = specials::stirling2(T::from_f64(n), T::from_f64(k)).to_f64();
        let error = extended_relative_error(out, expected);
        assert!(
            error <= 1e-13,
            "[{}] stirling2(n={n}, k={k}): out={out:e}, expected={expected:e}, error={error:e}",
            T::NAME
        );
    }
}

#[test]
fn stirling2_single_and_mixed_values_test() {
    for_both!(stirling2_single_and_mixed_values);
}

fn stirling2_identities_for_n_above_50<T: Num>() {
    // S(n, 1) = 1, S(n, n) = 1, S(n, n-1) = n(n-1)/2, S(n, 2) = 2^(n-1) - 1
    // (exact for n <= 53 in double).
    let max_rtol = 2e-5;
    for n in [51.0, 55.0, 60.0, 75.0, 100.0] {
        let checks = [(1.0, 1.0), (n, 1.0), (n - 1.0, n * (n - 1.0) / 2.0)];
        for (k, expected) in checks {
            let out = specials::stirling2(T::from_f64(n), T::from_f64(k)).to_f64();
            let error = extended_relative_error(out, expected);
            assert!(
                error <= max_rtol,
                "[{}] stirling2(n={n}, k={k}): out={out:e}, expected={expected:e}, error={error:e}",
                T::NAME
            );
        }
        if n <= 53.0 {
            let expected = 2.0_f64.powf(n - 1.0) - 1.0;
            let out = specials::stirling2(T::from_f64(n), T::from_f64(2.0)).to_f64();
            assert!(extended_relative_error(out, expected) <= max_rtol);
        }
    }
}

#[test]
fn stirling2_identities_for_n_above_50_test() {
    for_both!(stirling2_identities_for_n_above_50);
}

fn stirling2_gh_378<T: Num>() {
    // Regression test for gh-378: relative error up to 2e-5 for n > 50.
    // Exact references from Python integers:
    //     S(n, k) = sum((-1)**j * comb(k, j) * (k - j)**n for j in range(k + 1)) // factorial(k)
    let cases = [
        (51.0, 7.0, 2.4911342878123612e+39),
        (60.0, 30.0, 9.563528855094403e+53),
        (100.0, 50.0, 4.3098323700936636e+101),
        (219.0, 110.0, 2.071041889974642e+260),
    ];
    for (n, k, expected) in cases {
        let out = specials::stirling2(T::from_f64(n), T::from_f64(k)).to_f64();
        let error = extended_relative_error(out, expected);
        assert!(
            error <= 1e-14,
            "[{}] stirling2(n={n}, k={k}): out={out:e}, expected={expected:e}, error={error:e}",
            T::NAME
        );
    }
}

#[test]
fn stirling2_gh_378_test() {
    for_both!(stirling2_gh_378);
}

fn stirling2_gh_354<T: Num>() {
    // Regression test for gh-354: out-of-bounds read for 0 < k < 1 and NaN k.
    let nan = T::from_f64(f64::NAN);
    assert_eq!(
        specials::stirling2(T::from_f64(5.0), T::from_f64(0.5)).to_f64(),
        0.0
    );
    assert_eq!(
        specials::stirling2(T::from_f64(100.0), T::from_f64(0.5)).to_f64(),
        0.0
    );
    assert!(specials::stirling2(T::from_f64(5.0), nan).to_f64().is_nan());
    assert!(specials::stirling2(T::from_f64(300.0), nan)
        .to_f64()
        .is_nan());
}

#[test]
fn stirling2_gh_354_test() {
    for_both!(stirling2_gh_354);
}

// ------------------------------------------------- elliptic integrals and functions

fn ellipkinc_ellipeinc_huge_phi<T: Num>() {
    // Regression test for gh-359: both returned NaN.
    // Reference values: float(mpmath.ellipf(phi, m)), float(mpmath.ellipe(phi, m)) with
    // mpmath.mp.dps = 50.
    let (phi, m) = (-1.7885758944711641e19, -1.2443495040003017e35);
    let rtol = 2e-16;
    let (p, mm) = (T::from_f64(phi), T::from_f64(m));
    let outk = specials::ellipkinc(p, mm).to_f64();
    let errk = extended_relative_error(outk, -1348.954827564235);
    assert!(
        errk <= rtol,
        "[{}] ellipkinc: out={outk:e}, error={errk:e}, rtol={rtol:e}",
        T::NAME
    );
    let oute = specials::ellipeinc(p, mm).to_f64();
    let erre = extended_relative_error(oute, -4.0166009415816355e+36);
    assert!(
        erre <= rtol,
        "[{}] ellipeinc: out={oute:e}, error={erre:e}, rtol={rtol:e}",
        T::NAME
    );

    // Here the phi reduction loses the residual, which ends up as 4.
    let (phi2, m2) = (3.560108682609496e16, -1.24e35);
    let (p2, m2) = (T::from_f64(phi2), T::from_f64(m2));
    let e = extended_relative_error(specials::ellipkinc(p2, m2).to_f64(), 2.689647972067577);
    assert!(e <= rtol, "[{}] ellipkinc(phi2): error={e:e}", T::NAME);
    let e = extended_relative_error(specials::ellipeinc(p2, m2).to_f64(), 7.980943234086402e33);
    assert!(e <= rtol, "[{}] ellipeinc(phi2): error={e:e}", T::NAME);

    // n * E(m) overflows here, so this must stay -inf, not NaN.
    let out = specials::ellipeinc(
        T::from_f64(-7.49757919668992e261),
        T::from_f64(-2.6578197111983853e169),
    );
    assert_eq!(
        out.to_f64(),
        f64::NEG_INFINITY,
        "[{}] ellipeinc overflow",
        T::NAME
    );
}

#[test]
fn ellipkinc_ellipeinc_huge_phi_test() {
    for_both!(ellipkinc_ellipeinc_huge_phi);
}

/// `(u, m, sn, cn, dn)` for negative `m`, mpmath values of xsf's `test_ellpj.cpp`:
/// `mpmath.re(mpmath.ellipfun(kind, u, m))` with `u` in `linspace(-100, 100, 6)` and `m` in
/// `-logspace(-4, 4, 11)`.
const ELLPJ_NEGATIVE_M: [(f64, f64, f64, f64, f64); 66] = [
    // m = -0.0001
    (
        -100.0,
        -0.0001,
        0.50419893107942393,
        0.86358753922133813,
        1.0000127107473236,
    ),
    (
        -60.0,
        -0.0001,
        0.30623191490155632,
        -0.95195693930751191,
        1.0000046888882923,
    ),
    (
        -20.0,
        -0.0001,
        -0.91314537820787445,
        0.40763404943355497,
        1.0000416908550234,
    ),
    (
        20.0,
        -0.0001,
        0.91314537820787445,
        0.40763404943355497,
        1.0000416908550234,
    ),
    (
        60.0,
        -0.0001,
        -0.30623191490155632,
        -0.95195693930751191,
        1.0000046888882923,
    ),
    (
        100.0,
        -0.0001,
        -0.50419893107942393,
        0.86358753922133813,
        1.0000127107473236,
    ),
    // m = -0.00063095734448019298
    (
        -100.0,
        -0.00063095734448019298,
        0.49264492185198283,
        0.87023041832232784,
        1.0000765634332969,
    ),
    (
        -60.0,
        -0.00063095734448019298,
        0.31376443297109885,
        -0.94950085866328993,
        1.0000310577996987,
    ),
    (
        -20.0,
        -0.00063095734448019298,
        -0.91420419323965252,
        0.40525386248996575,
        1.000263632640167,
    ),
    (
        20.0,
        -0.00063095734448019298,
        0.91420419323965252,
        0.40525386248996575,
        1.000263632640167,
    ),
    (
        60.0,
        -0.00063095734448019298,
        -0.31376443297109885,
        -0.94950085866328993,
        1.0000310577996987,
    ),
    (
        100.0,
        -0.00063095734448019298,
        -0.49264492185198283,
        0.87023041832232784,
        1.0000765634332969,
    ),
    // m = -0.0039810717055349734
    (
        -100.0,
        -0.0039810717055349734,
        0.41794703691627599,
        0.9084713943393623,
        1.000347645827881,
    ),
    (
        -60.0,
        -0.0039810717055349734,
        0.3607269097954281,
        -0.93267148372266695,
        1.0002589827590855,
    ),
    (
        -20.0,
        -0.0039810717055349734,
        -0.92073813865682963,
        0.39018108619300951,
        1.0016860727060259,
    ),
    (
        20.0,
        -0.0039810717055349734,
        0.92073813865682963,
        0.39018108619300951,
        1.0016860727060259,
    ),
    (
        60.0,
        -0.0039810717055349734,
        -0.3607269097954281,
        -0.93267148372266695,
        1.0002589827590855,
    ),
    (
        100.0,
        -0.0039810717055349734,
        -0.41794703691627599,
        0.9084713943393623,
        1.000347645827881,
    ),
    // m = -0.025118864315095822
    (
        -100.0,
        -0.025118864315095822,
        -0.09144898043880885,
        0.99580976294506285,
        1.0001050279609975,
    ),
    (
        -60.0,
        -0.025118864315095822,
        0.62923541294216601,
        -0.77721476767975906,
        1.0049604434639177,
    ),
    (
        -20.0,
        -0.025118864315095822,
        -0.9560835305044626,
        0.29309432389270579,
        1.0114153678355475,
    ),
    (
        20.0,
        -0.025118864315095822,
        0.9560835305044626,
        0.29309432389270579,
        1.0114153678355475,
    ),
    (
        60.0,
        -0.025118864315095822,
        -0.62923541294216601,
        -0.77721476767975906,
        1.0049604434639177,
    ),
    (
        100.0,
        -0.025118864315095822,
        0.09144898043880885,
        0.99580976294506285,
        1.0001050279609975,
    ),
    // m = -0.15848931924611143
    (
        -100.0,
        -0.15848931924611143,
        0.10479061687186157,
        -0.99449430697999208,
        1.0008698131351963,
    ),
    (
        -60.0,
        -0.15848931924611143,
        0.5196445581827569,
        0.85438254497095589,
        1.0211742970064415,
    ),
    (
        -20.0,
        -0.15848931924611143,
        -0.94023688482685774,
        -0.34052107190464198,
        1.067760157321626,
    ),
    (
        20.0,
        -0.15848931924611143,
        0.94023688482685774,
        -0.34052107190464198,
        1.067760157321626,
    ),
    (
        60.0,
        -0.15848931924611143,
        -0.5196445581827569,
        0.85438254497095589,
        1.0211742970064415,
    ),
    (
        100.0,
        -0.15848931924611143,
        -0.10479061687186157,
        -0.99449430697999208,
        1.0008698131351963,
    ),
    // m = -1
    (
        -100.0,
        -1.0,
        -0.36119378002137797,
        0.93249077918972922,
        1.0632313702699576,
    ),
    (
        -60.0,
        -1.0,
        -0.30704980717822983,
        -0.95169344639532527,
        1.0460781921482678,
    ),
    (
        -20.0,
        -1.0,
        0.89393389569287096,
        0.44819882879294476,
        1.3413119733561736,
    ),
    (
        20.0,
        -1.0,
        -0.89393389569287096,
        0.44819882879294476,
        1.3413119733561736,
    ),
    (
        60.0,
        -1.0,
        0.30704980717822983,
        -0.95169344639532527,
        1.0460781921482678,
    ),
    (
        100.0,
        -1.0,
        0.36119378002137797,
        0.93249077918972922,
        1.0632313702699576,
    ),
    // m = -6.3095734448019432
    (
        -100.0,
        -6.3095734448019432,
        0.9053468170283242,
        0.42467298112392565,
        2.4842825752627826,
    ),
    (
        -60.0,
        -6.3095734448019432,
        0.79155169328242159,
        -0.61110221474130755,
        2.2255985722275815,
    ),
    (
        -20.0,
        -6.3095734448019432,
        0.22164715712194707,
        -0.97512693416793639,
        1.1445406638856161,
    ),
    (
        20.0,
        -6.3095734448019432,
        -0.22164715712194707,
        -0.97512693416793639,
        1.1445406638856161,
    ),
    (
        60.0,
        -6.3095734448019432,
        -0.79155169328242159,
        -0.61110221474130755,
        2.2255985722275815,
    ),
    (
        100.0,
        -6.3095734448019432,
        -0.9053468170283242,
        0.42467298112392565,
        2.4842825752627826,
    ),
    // m = -39.810717055349777
    (
        -100.0,
        -39.810717055349777,
        -0.16317219303897371,
        0.9865976056218928,
        1.4352584762655844,
    ),
    (
        -60.0,
        -39.810717055349777,
        -0.12837435905620323,
        -0.99172578061524097,
        1.2868875879915278,
    ),
    (
        -20.0,
        -39.810717055349777,
        0.72923969850110848,
        0.6842583299675733,
        4.7086051688216912,
    ),
    (
        20.0,
        -39.810717055349777,
        -0.72923969850110848,
        0.6842583299675733,
        4.7086051688216912,
    ),
    (
        60.0,
        -39.810717055349777,
        0.12837435905620323,
        -0.99172578061524097,
        1.2868875879915278,
    ),
    (
        100.0,
        -39.810717055349777,
        0.16317219303897371,
        0.9865976056218928,
        1.4352584762655844,
    ),
    // m = -251.18864315095823
    (
        -100.0,
        -251.18864315095823,
        0.075887377087293567,
        -0.99711639541169461,
        1.5641511340688534,
    ),
    (
        -60.0,
        -251.18864315095823,
        -0.4461600293341676,
        -0.89495319890178315,
        7.1415196414751527,
    ),
    (
        -20.0,
        -251.18864315095823,
        -0.19646166026649953,
        0.98051150734977632,
        3.2703477287623963,
    ),
    (
        20.0,
        -251.18864315095823,
        0.19646166026649953,
        0.98051150734977632,
        3.2703477287623963,
    ),
    (
        60.0,
        -251.18864315095823,
        0.4461600293341676,
        -0.89495319890178315,
        7.1415196414751527,
    ),
    (
        100.0,
        -251.18864315095823,
        -0.075887377087293567,
        -0.99711639541169461,
        1.5641511340688534,
    ),
    // m = -1584.893192461114
    (
        -100.0,
        -1584.893192461114,
        -0.51250020048252409,
        -0.85868710512349755,
        20.427491972080187,
    ),
    (
        -60.0,
        -1584.893192461114,
        0.76113740872068569,
        0.64859066062190562,
        30.317922402006712,
    ),
    (
        -20.0,
        -1584.893192461114,
        -0.96771332342548155,
        -0.25205341430897021,
        38.538337587835713,
    ),
    (
        20.0,
        -1584.893192461114,
        0.96771332342548155,
        -0.25205341430897021,
        38.538337587835713,
    ),
    (
        60.0,
        -1584.893192461114,
        -0.76113740872068569,
        0.64859066062190562,
        30.317922402006712,
    ),
    (
        100.0,
        -1584.893192461114,
        0.51250020048252409,
        -0.85868710512349755,
        20.427491972080187,
    ),
    // m = -10000
    (
        -100.0,
        -10000.0,
        -0.90512632236054935,
        -0.42514272964507671,
        90.518156158305231,
    ),
    (
        -60.0,
        -10000.0,
        -0.13780775321395375,
        -0.99045899620030819,
        13.817010112856536,
    ),
    (
        -20.0,
        -10000.0,
        -0.013481207944958431,
        -0.99990912438698387,
        1.678520085239406,
    ),
    (
        20.0,
        -10000.0,
        0.013481207944958431,
        -0.99990912438698387,
        1.678520085239406,
    ),
    (
        60.0,
        -10000.0,
        0.13780775321395375,
        -0.99045899620030819,
        13.817010112856536,
    ),
    (
        100.0,
        -10000.0,
        0.90512632236054935,
        -0.42514272964507671,
        90.518156158305231,
    ),
];

fn ellpj_negative_m<T: Num>() {
    let rtol = 1e-8;
    for (u, m, ref_sn, ref_cn, ref_dn) in ELLPJ_NEGATIVE_M {
        let (sn, cn, dn, ph) = specials::ellipj(T::from_f64(u), T::from_f64(m));
        let (sn, cn, dn) = (sn.to_f64(), cn.to_f64(), dn.to_f64());
        let e_sn = extended_relative_error(sn, ref_sn);
        let e_cn = extended_relative_error(cn, ref_cn);
        let e_dn = extended_relative_error(dn, ref_dn);
        // The amplitude is checked through F(ph | m) = u.
        let e_ph = extended_relative_error(specials::ellipkinc(ph, T::from_f64(m)).to_f64(), u);
        let ctx = format!("[{}] u={u:e}, m={m:e}", T::NAME);
        assert!(
            e_sn <= rtol,
            "{ctx}: sn={sn:e}, ref={ref_sn:e}, error={e_sn:e}"
        );
        assert!(
            e_cn <= rtol,
            "{ctx}: cn={cn:e}, ref={ref_cn:e}, error={e_cn:e}"
        );
        assert!(
            e_dn <= rtol,
            "{ctx}: dn={dn:e}, ref={ref_dn:e}, error={e_dn:e}"
        );
        assert!(e_ph <= rtol, "{ctx}: F(ph | m) differs from u by {e_ph:e}");
    }
}

#[test]
fn ellpj_negative_m_test() {
    for_both!(ellpj_negative_m);
}

// -------------------------------------------------------------------- agm

fn agm_check<T: Num>(a: f64, b: f64, expected: f64, rtol: f64) {
    let out = specials::agm(T::from_f64(a), T::from_f64(b)).to_f64();
    let error = extended_relative_error(out, expected);
    assert!(
        error <= rtol,
        "[{}] agm({a:e}, {b:e}): out={out:e}, expected={expected:e}, error={error:e}, rtol={rtol:e}",
        T::NAME
    );
}

fn agm_simple<T: Num>() {
    // Mirrors scipy v1.18.0rc1 test_basic.py (xsf `test_agm.cpp`, "agm simple").
    let rtol = 1e-13;
    let gauss_constant = 1.0 / specials::agm(T::from_f64(1.0), T::from_f64(2.0).sqrt()).to_f64();
    let e = extended_relative_error(gauss_constant, 0.834626841674073186);
    assert!(e <= rtol, "[{}] Gauss's constant: error={e:e}", T::NAME);
    for (a, b, expected) in [
        (1.0, 1.0, 1.0),
        (1.0, 3.0, 1.863616783244897),
        (1.0, 5.0, 2.604008190530940),
        (3.0, 1.0, 1.863616783244897),
        (3.0, 3.0, 3.0),
        (3.0, 5.0, 3.936235503649555),
    ] {
        agm_check::<T>(a, b, expected, rtol);
    }
}

#[test]
fn agm_simple_test() {
    for_both!(agm_simple);
}

fn agm_finite_cases<T: Num>() {
    // xsf `test_agm.cpp`, "agm finite cases".
    let rtol = 1e-13;
    let (min, max) = (f64::MIN_POSITIVE, f64::MAX);
    for (a, b, expected) in [
        (1.0, 2.0, 1.4567910310469068),
        (2.0, 1.0, 1.4567910310469068),
        (-1.0, -2.0, -1.4567910310469068),
        (24.0, 6.0, 13.458171481725614),
        (13.0, 123456789.5, 11111458.498599306),
        (1e30, 1.0, 2.229223055945383e+28),
        (1e-22, 1.0, 0.030182566420169886),
        (1e150, 1e180, 2.229223055945383e+178),
        (1e180, 1e-150, 2.0634722510162677e+177),
        (1e-150, 1e-170, 3.3112619670463756e-152),
        (min, max, 1.9892072050015473e+305),
        (0.75 * max, max, 1.564904312298045e+308),
        (min, 3.0 * min, 4.1466849866735005e-308),
    ] {
        agm_check::<T>(a, b, expected, rtol);
    }
}

#[test]
fn agm_finite_cases_test() {
    for_both!(agm_finite_cases);
}

fn agm_zero_nan_inf<T: Num>() {
    // xsf `test_agm.cpp`, "agm zero, nan, and inf cases".
    let agm = |a: f64, b: f64| specials::agm(T::from_f64(a), T::from_f64(b)).to_f64();
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    assert_eq!(agm(0.0, 0.0), 0.0);
    assert_eq!(agm(99.0, 0.0), 0.0);
    for (a, b) in [
        (-1.0, 10.0),
        (0.0, inf),
        (inf, 0.0),
        (0.0, -inf),
        (-inf, 0.0),
        (inf, -inf),
        (-inf, inf),
        (1.0, nan),
        (nan, -1.0),
    ] {
        assert!(
            agm(a, b).is_nan(),
            "[{}] agm({a}, {b}) = {}",
            T::NAME,
            agm(a, b)
        );
    }
    assert_eq!(agm(1.0, inf), inf);
    assert_eq!(agm(inf, 1.0), inf);
    assert_eq!(agm(-1.0, -inf), -inf);
    assert_eq!(agm(-inf, -1.0), -inf);
}

#[test]
fn agm_zero_nan_inf_test() {
    for_both!(agm_zero_nan_inf);
}

// ------------------------------------------------------- Carlson integrals

fn close<T: Num>(name: &str, args: &[f64], out: T, expected: f64, rtol: f64) {
    let out = out.to_f64();
    let error = extended_relative_error(out, expected);
    assert!(
        error <= rtol,
        "[{}] {name}{args:?}: out={out:e}, expected={expected:e}, error={error:e}, rtol={rtol:e}",
        T::NAME
    );
}

fn carlson_rc<T: Num>() {
    // Mirrors scipy v1.18.0 test_basic.py (xsf `test_ellint_carlson.cpp`, "Carlson RC"), real values.
    let f = T::from_f64;
    for (x, y, expected) in [
        (1.0, 1.0, 1.0),
        (0.0, 0.25, core::f64::consts::PI),
        (2.25, 2.0, 2.0_f64.ln()),
        (0.25, -2.0, 2.0_f64.ln() / 3.0),
    ] {
        close(
            "elliprc",
            &[x, y],
            specials::elliprc(f(x), f(y)),
            expected,
            1e-13,
        );
    }
    assert_eq!(specials::elliprc(f(1.0), f(f64::INFINITY)).to_f64(), 0.0);
    assert!(specials::elliprc(f(1.0), f(0.0)).to_f64().is_nan());
}

#[test]
fn carlson_rc_test() {
    for_both!(carlson_rc);
}

fn carlson_rd<T: Num>() {
    let f = T::from_f64;
    for (x, y, z, expected) in [
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 3.0 * 0.59907011736779610371),
        (2.0, 3.0, 4.0, 0.16510527294261),
    ] {
        close(
            "elliprd",
            &[x, y, z],
            specials::elliprd(f(x), f(y), f(z)),
            expected,
            1e-13,
        );
    }
    let rd = |x: f64, y: f64, z: f64| specials::elliprd(f(x), f(y), f(z)).to_f64();
    assert_eq!(rd(1.0, 1.0, f64::INFINITY), 0.0);
    assert!(rd(1.0, 1.0, 0.0).is_infinite());
    assert!(rd(1.0, 1.0, -1.0).is_nan());
    assert!(rd(1.0, 1.0, -f64::MIN_POSITIVE / 2.0).is_nan());
}

#[test]
fn carlson_rd_test() {
    for_both!(carlson_rd);
}

fn carlson_rf<T: Num>() {
    let f = T::from_f64;
    for (x, y, z, expected) in [
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 2.0, 1.31102877714605990523),
    ] {
        close(
            "elliprf",
            &[x, y, z],
            specials::elliprf(f(x), f(y), f(z)),
            expected,
            1e-13,
        );
    }
    let rf = |x: f64, y: f64, z: f64| specials::elliprf(f(x), f(y), f(z)).to_f64();
    assert_eq!(rf(1.0, f64::INFINITY, 1.0), 0.0);
    assert!(rf(0.0, 1.0, 0.0).is_infinite());
    assert!(rf(1.0, 1.0, -1.0).is_nan());
}

#[test]
fn carlson_rf_test() {
    for_both!(carlson_rf);
}

fn carlson_rg<T: Num>() {
    let f = T::from_f64;
    for (x, y, z, expected) in [
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 16.0, 16.0, core::f64::consts::PI),
        (2.0, 3.0, 4.0, 1.7255030280692),
        (0.0, 0.0796, 4.0, 1.0284758090288),
    ] {
        close(
            "elliprg",
            &[x, y, z],
            specials::elliprg(f(x), f(y), f(z)),
            expected,
            1e-13,
        );
    }
    let rg = |x: f64, y: f64, z: f64| specials::elliprg(f(x), f(y), f(z)).to_f64();
    assert_eq!(rg(0.0, 0.0, 0.0), 0.0);
    assert!(rg(1.0, f64::INFINITY, 1.0).is_infinite());
}

#[test]
fn carlson_rg_test() {
    for_both!(carlson_rg);
}

fn carlson_rj<T: Num>() {
    let f = T::from_f64;
    // Includes the Cauchy principal values for negative p.
    for (x, y, z, p, expected) in [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 2.0, 3.0, 0.77688623778582),
        (2.0, 3.0, 4.0, 5.0, 0.14297579667157),
        (2.0, 3.0, 4.0, -0.5, 0.24723819703052),
        (2.0, 3.0, 4.0, -5.0, -0.12711230042964),
    ] {
        close(
            "elliprj",
            &[x, y, z, p],
            specials::elliprj(f(x), f(y), f(z), f(p)),
            expected,
            1e-13,
        );
    }
    let rj = |x: f64, y: f64, z: f64, p: f64| specials::elliprj(f(x), f(y), f(z), f(p)).to_f64();
    assert_eq!(rj(1.0, 1.0, 1.0, f64::INFINITY), 0.0);
    assert_eq!(rj(1.0, 1.0, f64::INFINITY, 1.0), 0.0);
    assert!(rj(-1.0, 1.0, 1.0, 1.0).is_nan());
    assert!(rj(1.0, 0.0, 0.0, 0.0).is_nan());
}

#[test]
fn carlson_rj_test() {
    for_both!(carlson_rj);
}

fn carlson_rj_difficult_arguments<T: Num>() {
    // SciPy marks this test xfail for insufficient accuracy on 32-bit.
    let f = T::from_f64;
    for (x, y, z, p, expected) in [
        (
            6.483625725195452e-08,
            1.1649136528196886e-27,
            3.6767340167168e13,
            0.493704617023468,
            8.63426920644241857617477551054e-6,
        ),
        (
            14.375105857849121,
            9.993988969725365e-11,
            1.72844262269944e-26,
            5.898871222598245e-06,
            829774.1424801627252574054378691828,
        ),
    ] {
        let result = specials::elliprj(f(x), f(y), f(z), f(p)).to_f64();
        assert!(
            (result - expected).abs() <= 1e-20 + 5e-15 * expected.abs(),
            "[{}] elliprj({x:e}, {y:e}, {z:e}, {p:e}) = {result:e}, expected {expected:e}",
            T::NAME
        );
    }
}

#[test]
fn carlson_rj_difficult_arguments_test() {
    for_both!(carlson_rj_difficult_arguments);
}

fn legendre_carlson_k_and_e<T: Num>() {
    // Mirrors scipy v1.18.0 test_basic.py: K(m) = R_F(0, 1 - m, 1) and
    // E(m) = 2 R_G(0, 1 - m, 1), from m = lowest f64 up to m < 1.
    let f = T::from_f64;
    let check = |m: f64| {
        let k = specials::ellipk(f(m)).to_f64();
        let rf = specials::elliprf(f(0.0), f(1.0) - f(m), f(1.0)).to_f64();
        let e = specials::ellipe(f(m)).to_f64();
        let rg = 2.0 * specials::elliprg(f(0.0), f(1.0) - f(m), f(1.0)).to_f64();
        let ek = extended_relative_error(k, rf);
        let ee = extended_relative_error(e, rg);
        assert!(ek <= 1e-7, "[{}] K(m={m:e}) = {k:e}, R_F = {rf:e}", T::NAME);
        assert!(
            ee <= 1e-7,
            "[{}] E(m={m:e}) = {e:e}, 2 R_G = {rg:e}",
            T::NAME
        );
    };
    check(f64::MIN);
    for exponent in (1..=1023).rev() {
        check(-(2.0_f64.powi(exponent)));
    }
    for i in 0..200 {
        check(-1.0 + 0.01 * f64::from(i));
    }
}

#[test]
fn legendre_carlson_k_and_e_test() {
    for_both!(legendre_carlson_k_and_e);
}

fn legendre_carlson_km1<T: Num>() {
    // K(1 - m1) = R_F(0, m1, 1) for m1 = 2^-1022 ..= 2^-1.
    let f = T::from_f64;
    for exponent in -1022..0 {
        let m1 = 2.0_f64.powi(exponent);
        let result = specials::ellipkm1(f(m1)).to_f64();
        let expected = specials::elliprf(f(0.0), f(m1), f(1.0)).to_f64();
        let error = extended_relative_error(result, expected);
        assert!(
            error <= 1e-7,
            "[{}] ellipkm1({m1:e}) = {result:e}, R_F = {expected:e}, error={error:e}",
            T::NAME
        );
    }
}

#[test]
fn legendre_carlson_km1_test() {
    for_both!(legendre_carlson_km1);
}

fn carlson_rf_tiny_x_with_underflowing_yz<T: Num>() {
    // Regression test for gh-467: elliprf was NaN or -inf if x is 0 or subnormal and y * z
    // underflows. Reference values: float(mpmath.elliprf(x, y, z)) with mpmath.mp.dps = 50.
    let f = T::from_f64;
    for (x, y, z, expected) in [
        (0.0, 1e-300, 1e-30, 3.1223528191531603e+17),
        (5e-324, 1e-300, 1e-300, 1.5707963267926738e+150),
    ] {
        close(
            "elliprf",
            &[x, y, z],
            specials::elliprf(f(x), f(y), f(z)),
            expected,
            1e-15,
        );
    }
}

#[test]
fn carlson_rf_tiny_x_with_underflowing_yz_test() {
    for_both!(carlson_rf_tiny_x_with_underflowing_yz);
}

// ------------------------------------------------- non-finite agreement between strategies
//
// Both strategies must return the same non-finite value (`generic/mod.rs`). A double-double
// intermediate that overflows carries `(inf, NaN)`, so a raw `*`/`+` turned it into NaN where
// the f64 path returns +inf. Review reproducer, reported against the gamma-family PR.
fn non_finite_agreement() {
    let cases = [(1000.0f64, -300.0f64), (1000.0, -400.0), (2000.0, -350.0)];
    for (n, a) in cases {
        let f = specials::gen_harmonic(n, a);
        let d = specials::gen_harmonic(xprec::Df64::from(n), xprec::Df64::from(a)).to_f64();
        assert_eq!(f, f64::INFINITY, "gen_harmonic({n},{a}) f64");
        assert_eq!(d, f64::INFINITY, "gen_harmonic({n},{a}) Df64");
    }
    for x in [-1e-200f64, -1e-160, -1e-155] {
        let f = specials::trigamma(x);
        let d = specials::trigamma(xprec::Df64::from(x)).to_f64();
        assert_eq!(f, f64::INFINITY, "trigamma({x}) f64");
        assert_eq!(d, f64::INFINITY, "trigamma({x}) Df64");
    }
}

#[test]
fn non_finite_agreement_test() {
    non_finite_agreement();
}
