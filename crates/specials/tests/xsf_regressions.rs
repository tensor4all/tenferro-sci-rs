//! Regression tests for real arguments, ported from SciPy's xsf
//! (`tests/xsf_tests/test_gammaln.cpp`, `test_rgamma.cpp`, `test_zeta.cpp`,
//! `test_betaln.cpp`, `test_binom.cpp`, `test_trigamma.cpp`,
//! `test_digammainv.cpp`, `test_gen_harmonic.cpp`, `test_lgam1p.cpp`,
//! `test_zetac.cpp` and `test_stirling2.cpp` at commit
//! `31177fd9a458539cf05c04d215341245d0245e53`).
//!
//! The xsf sources are BSD-3-Clause licensed; see `LICENSE-BSD-3-Clause-xsf`
//! in this crate. Reference values, tolerances and regression-issue comments
//! are kept as in xsf. Every case runs for `f64` and for `xprec::Df64`; the
//! `Df64` result is rounded to its high word (`hi()`) and compared against the
//! same f64-rounded reference with the same tolerance. Complex-valued and
//! float-template cases are not ported.

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
