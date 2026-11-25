//! Regression tests for the orthogonal polynomials and Legendre functions of a
//! real argument, ported from SciPy's xsf (`tests/xsf_tests/test_orthogonal_eval.cpp`,
//! `test_assoc_legendre_p.cpp` and `test_lpmv.cpp` at commit
//! `31177fd9a458539cf05c04d215341245d0245e53`; the xsref table of
//! `test_pmv.cpp` is checked in `xsref.rs`).
//!
//! The xsf sources are BSD-3-Clause licensed; see `LICENSE-BSD-3-Clause-xsf`
//! in this crate. Reference values, tolerances and comments are kept as in xsf.
//! Every case runs for `f64` and, with the `xprec` feature (on by default), for
//! `xprec::Df64`; the `Df64` result is rounded to its high word (`hi()`) and
//! compared against the same f64 reference with the same tolerance. Without
//! `xprec` every test still runs its `f64` case. Complex-valued and
//! single-precision cases are not ported.

// Reference values keep the digits xsf's tests give them.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use specials::{LegendreType, Real};
#[cfg(feature = "xprec")]
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

#[cfg(feature = "xprec")]
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
        f64::from_bits(1)
    } else if desired.is_infinite() {
        f64::MAX
    } else if desired.is_nan() {
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

/// Runs a case for `f64` and, with `xprec`, for `Df64`.
macro_rules! for_both {
    ($f:ident) => {{
        $f::<f64>();
        #[cfg(feature = "xprec")]
        $f::<Df64>();
    }};
}

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

// ---------------------------------------------------------------- helpers

fn binom_int(a: f64, k: i64) -> f64 {
    let mut out = 1.0;
    for j in 0..k {
        out *= (a - j as f64) / (j as f64 + 1.0);
    }
    out
}

fn laguerre_coefficients(n: i64, alpha: f64) -> Vec<f64> {
    let mut out = vec![0.0; n as usize + 1];
    let mut scale = 1.0;
    for k in 0..=n {
        out[k as usize] = scale * binom_int(n as f64 + alpha, n - k);
        scale /= -(k as f64 + 1.0);
    }
    out
}

/// Coefficients in ascending powers of `x`:
/// `C_n^(alpha)(x) = sum_k (-1)^k (alpha)_{n-k} / (k! (n-2k)!) (2x)^(n-2k)`.
fn gegenbauer_coefficients(n: i64, alpha: f64) -> Vec<f64> {
    let mut out = vec![0.0; n as usize + 1];
    let mut k = 0;
    while 2 * k <= n {
        let mut c = if k % 2 == 0 { 1.0 } else { -1.0 };
        for j in 0..(n - k) {
            c *= alpha + j as f64;
        }
        for j in 1..=k {
            c /= j as f64;
        }
        for j in 1..=(n - 2 * k) {
            c *= 2.0 / j as f64;
        }
        out[(n - 2 * k) as usize] = c;
        k += 1;
    }
    out
}

/// Monomial coefficients (lowest degree first) of `P_n(x)`, or of `P_n(2x - 1)` if shifted.
fn legendre_coefficients(n: i64, shifted: bool) -> Vec<f64> {
    let mut out = vec![0.0; n as usize + 1];
    if shifted {
        for k in 0..=n {
            out[k as usize] = (-1.0_f64).powi((n + k) as i32)
                * binom_int(n as f64, k)
                * binom_int((n + k) as f64, k);
        }
    } else {
        for k in 0..=(n / 2) {
            out[(n - 2 * k) as usize] = (-1.0_f64).powi(k as i32)
                * binom_int(n as f64, k)
                * binom_int((2 * n - 2 * k) as f64, n)
                / 2.0_f64.powi(n as i32);
        }
    }
    out
}

fn linear_power(c0: f64, c1: f64, n: i64) -> Vec<f64> {
    (0..=n)
        .map(|j| binom_int(n as f64, j) * c0.powi((n - j) as i32) * c1.powi(j as i32))
        .collect()
}

/// Monomial coefficients (lowest degree first) of `scale * P_n(slope * x + offset)`, with `P_n = U_n` or `T_n`:
/// `U_n(y) = sum_k (-1)^k binom(n - k, k) (2y)^(n - 2k)`, `T_n(y)` the same with the extra factor `n / (2 (n - k))`.
/// With `T_n = eval_chebyt` and `U_n = eval_chebyu`: `S_n(x) = U_n(x / 2)`, `C_n(x) = 2 T_n(x / 2)`,
/// `T*_n(x) = T_n(2x - 1)` and `U*_n(x) = U_n(2x - 1)`.
fn chebyshev_coefficients(
    n: i64,
    second_kind: bool,
    slope: f64,
    offset: f64,
    scale: f64,
) -> Vec<f64> {
    if n == 0 {
        return vec![scale]; // avoid 0 / 0 in the T_n factor
    }
    let mut out = vec![0.0; n as usize + 1];
    for k in 0..=(n / 2) {
        let power = n - 2 * k;
        let mut c = scale
            * (-1.0_f64).powi(k as i32)
            * binom_int((n - k) as f64, k)
            * 2.0_f64.powi(power as i32);
        if !second_kind {
            c = c * n as f64 / (2.0 * (n - k) as f64);
        }
        let term = linear_power(offset, slope, power);
        for j in 0..=power {
            out[j as usize] += c * term[j as usize];
        }
    }
    out
}

fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Horner evaluation of coefficients in ascending powers (xsf's `evalpoly` of the reversed coefficients).
fn polyval(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.len() == 1 {
        return coeffs[0];
    }
    let mut it = coeffs.iter().rev();
    let mut result = *it.next().expect("coefficients");
    for &c in it {
        result = result * x + c;
    }
    result
}

fn sample(a: f64, b: f64, i: i64) -> f64 {
    let step = 0.7548776662466927;
    let u = (step * (i as f64 + 1.0)) % 1.0;
    a + (b - a) * u
}

/// `check_poly` of xsf's tests: the evaluator against the constructed
/// polynomial at deterministic parameters and arguments.
fn check_poly<T: Num>(
    coefficients: impl Fn(i64, &[f64]) -> Vec<f64>,
    eval: impl Fn(i64, &[f64], f64) -> f64,
    param_ranges: &[(f64, f64)],
    x_range: (f64, f64),
    rtol: f64,
) {
    let (nn, nparam, nx) = (10_i64, 10_i64, 10_i64);
    for n in 0..nn {
        let ncases = if param_ranges.is_empty() { 1 } else { nparam };
        for ip in 0..ncases {
            let params: Vec<f64> = param_ranges
                .iter()
                .enumerate()
                .map(|(k, r)| sample(r.0, r.1, 17 * n + nparam * k as i64 + ip))
                .collect();
            let coeffs = coefficients(n, &params);

            for ix in 0..nx {
                let mut x = sample(x_range.0, x_range.1, 31 * n + 13 * ip + ix);
                if ix == 0 {
                    x = x_range.0;
                } else if ix == 1 {
                    x = x_range.1;
                }

                let out = eval(n, &params, x);
                let expected = polyval(&coeffs, x);
                let error = extended_absolute_error(out, expected);
                let tol = 1e-12 + rtol * expected.abs();
                assert!(
                    error <= tol,
                    "[{}] n={n} params={params:?} x={x}: out={out:e}, expected={expected:e}, error={error:e}, tol={tol:e}",
                    T::NAME
                );
            }
        }
    }
}

/// `check_recurrence`: the integer-degree evaluator against the real-degree one.
fn check_recurrence<T: Num>(
    int_degree: impl Fn(i64, &[f64], f64) -> f64,
    double_degree: impl Fn(f64, &[f64], f64) -> f64,
    param_ranges: &[(f64, f64)],
    x_range: (f64, f64),
) {
    let (nn, nparam, nx) = (10_i64, 10_i64, 10_i64);
    let rtol = 1e-8;
    for n in 0..nn {
        let ncases = if param_ranges.is_empty() { 1 } else { nparam };
        for ip in 0..ncases {
            let params: Vec<f64> = param_ranges
                .iter()
                .enumerate()
                .map(|(k, r)| sample(r.0, r.1, 19 * n + nparam * k as i64 + ip))
                .collect();
            for ix in 0..nx {
                let mut x = sample(x_range.0, x_range.1, 37 * n + 11 * ip + ix);
                if ix == 0 {
                    x = x_range.0;
                } else if ix == 1 {
                    x = x_range.1;
                }

                let out = int_degree(n, &params, x);
                let expected = double_degree(n as f64, &params, x);
                let error = extended_absolute_error(out, expected);
                let tol = 1e-12 + rtol * expected.abs();
                assert!(
                    error <= tol,
                    "[{}] n={n} params={params:?} x={x}: out={out:e}, expected={expected:e}, error={error:e}, tol={tol:e}",
                    T::NAME
                );
            }
        }
    }
}

/// Wrappers: integer- and real-degree evaluations of a function of `T`; the
/// parameters are `p[0]`, `p[1]`, ... in the order of the signature.
macro_rules! wrap {
    ($name:ident, int, $f:ident, ($($p:ident = $idx:expr),*)) => {
        fn $name<T: Num>(n: i64, p: &[f64], x: f64) -> f64 {
            let _ = p;
            $( let $p = T::from_f64(p[$idx]); )*
            specials::$f(n, $($p,)* T::from_f64(x)).to_f64()
        }
    };
    ($name:ident, real, $f:ident, ($($p:ident = $idx:expr),*)) => {
        fn $name<T: Num>(n: f64, p: &[f64], x: f64) -> f64 {
            let _ = p;
            $( let $p = T::from_f64(p[$idx]); )*
            specials::$f(T::from_f64(n), $($p,)* T::from_f64(x)).to_f64()
        }
    };
}

wrap!(jacobi_i, int, eval_jacobi_int, (a = 0, b = 1));
wrap!(jacobi_r, real, eval_jacobi, (a = 0, b = 1));
wrap!(sh_jacobi_i, int, eval_sh_jacobi_int, (a = 0, b = 1));
wrap!(sh_jacobi_r, real, eval_sh_jacobi, (a = 0, b = 1));
wrap!(hermite_i, int, eval_hermite, ());
wrap!(hermitenorm_i, int, eval_hermitenorm, ());
wrap!(genlaguerre_i, int, eval_genlaguerre_int, (a = 0));
wrap!(genlaguerre_r, real, eval_genlaguerre, (a = 0));
wrap!(gegenbauer_i, int, eval_gegenbauer_int, (a = 0));
wrap!(gegenbauer_r, real, eval_gegenbauer, (a = 0));
wrap!(legendre_i, int, eval_legendre_int, ());
wrap!(legendre_r, real, eval_legendre, ());
wrap!(sh_legendre_i, int, eval_sh_legendre_int, ());
wrap!(sh_legendre_r, real, eval_sh_legendre, ());
wrap!(chebyt_i, int, eval_chebyt_int, ());
wrap!(chebyt_r, real, eval_chebyt, ());
wrap!(chebyu_i, int, eval_chebyu_int, ());
wrap!(chebyu_r, real, eval_chebyu, ());
wrap!(chebys_i, int, eval_chebys_int, ());
wrap!(chebys_r, real, eval_chebys, ());
wrap!(chebyc_i, int, eval_chebyc_int, ());
wrap!(chebyc_r, real, eval_chebyc, ());
wrap!(sh_chebyt_i, int, eval_sh_chebyt_int, ());
wrap!(sh_chebyt_r, real, eval_sh_chebyt, ());
wrap!(sh_chebyu_i, int, eval_sh_chebyu_int, ());
wrap!(sh_chebyu_r, real, eval_sh_chebyu, ());
wrap!(laguerre_i, int, eval_laguerre_int, ());
wrap!(laguerre_r, real, eval_laguerre, ());

fn jacobi_coefficients(n: i64, params: &[f64]) -> Vec<f64> {
    let (alpha, beta) = (params[0], params[1]);
    let mut out = vec![0.0; n as usize + 1];
    let scale = 2.0_f64.powi(-(n as i32));
    for m in 0..=n {
        let c = scale * binom_int(n as f64 + alpha, m) * binom_int(n as f64 + beta, n - m);
        let term = multiply(&linear_power(-1.0, 1.0, n - m), &linear_power(1.0, 1.0, m));
        for j in 0..=n as usize {
            out[j] += c * term[j];
        }
    }
    out
}

fn sh_jacobi_coefficients(n: i64, params: &[f64]) -> Vec<f64> {
    let (p, q) = (params[0], params[1]);
    let (alpha, beta) = (p - q, q - 1.0);
    let nf = n as f64;
    let scale = 1.0 / specials::binom(2.0 * nf + p - 1.0, nf);
    let mut out = vec![0.0; n as usize + 1];
    for m in 0..=n {
        let c = scale * binom_int(nf + alpha, m) * binom_int(nf + beta, n - m);
        let term = multiply(&linear_power(-1.0, 1.0, n - m), &linear_power(0.0, 1.0, m));
        for j in 0..=n as usize {
            out[j] += c * term[j];
        }
    }
    out
}

// ---------------------------------------------------------------- Jacobi

fn jacobi_constructed<T: Num>() {
    check_poly::<T>(
        jacobi_coefficients,
        |n, p, x| jacobi_r::<T>(n as f64, p, x),
        &[(-0.99, 10.0), (-0.99, 10.0)],
        (-1.0, 1.0),
        1e-5,
    );
}

#[test]
fn eval_jacobi_matches_constructed_polynomials() {
    for_both!(jacobi_constructed);
}

fn sh_jacobi_constructed<T: Num>() {
    check_poly::<T>(
        sh_jacobi_coefficients,
        |n, p, x| sh_jacobi_r::<T>(n as f64, p, x),
        &[(1.0, 10.0), (0.0, 1.0)],
        (0.0, 1.0),
        1e-5,
    );
}

#[test]
fn eval_sh_jacobi_matches_constructed_polynomials() {
    for_both!(sh_jacobi_constructed);
}

fn jacobi_recurrence<T: Num>() {
    check_recurrence::<T>(
        jacobi_i::<T>,
        jacobi_r::<T>,
        &[(-0.99, 10.0), (-0.99, 10.0)],
        (-1.0, 1.0),
    );
    check_recurrence::<T>(
        sh_jacobi_i::<T>,
        sh_jacobi_r::<T>,
        &[(1.0, 10.0), (0.0, 1.0)],
        (0.0, 1.0),
    );
}

#[test]
fn eval_jacobi_recurrence_overload() {
    for_both!(jacobi_recurrence);
}

/// gh-7001 - expected values were computed with mathematica.
fn jacobi_alpha_minus_one<T: Num>() {
    let beta_one: [(i64, [f64; 11]); 6] = [
        (0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (
            1,
            [
                -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0,
            ],
        ),
        (
            2,
            [
                3.0, 2.16, 1.44, 0.84, 0.36, 0.0, -0.24, -0.36, -0.36, -0.24, 0.0,
            ],
        ),
        (
            3,
            [
                -4.0, -1.98, -0.64, 0.14, 0.48, 0.5, 0.32, 0.06, -0.16, -0.22, 0.0,
            ],
        ),
        (
            4,
            [
                5.0, 1.332, -0.288, -0.658, -0.408, 0.0, 0.272, 0.282, 0.072, -0.148, 0.0,
            ],
        ),
        (
            5,
            [
                -6.0, -0.43308, 0.79104, 0.36876, -0.21312, -0.375, -0.14208, 0.15804, 0.19776,
                -0.04812, 0.0,
            ],
        ),
    ];
    let beta_minus_one: [(i64, [f64; 11]); 6] = [
        (0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        (1, [0.0; 11]),
        (
            2,
            [
                0.0, -0.09, -0.16, -0.21, -0.24, -0.25, -0.24, -0.21, -0.16, -0.09, 0.0,
            ],
        ),
        (
            3,
            [
                0.0, 0.144, 0.192, 0.168, 0.096, 0.0, -0.096, -0.168, -0.192, -0.144, 0.0,
            ],
        ),
        (
            4,
            [
                0.0, -0.1485, -0.096, 0.0315, 0.144, 0.1875, 0.144, 0.0315, -0.096, -0.1485, 0.0,
            ],
        ),
        (
            5,
            [
                0.0, 0.10656, -0.04608, -0.15792, -0.13056, 0.0, 0.13056, 0.15792, 0.04608,
                -0.10656, 0.0,
            ],
        ),
    ];
    for (beta, table) in [(1.0, beta_one), (-1.0, beta_minus_one)] {
        for (n, expected) in table {
            for (j, &want) in expected.iter().enumerate() {
                let x = -1.0 + 0.2 * j as f64;
                let tol = 1e-14 + 1e-10 * want.abs();
                for out in [
                    jacobi_i::<T>(n, &[-1.0, beta], x),
                    jacobi_r::<T>(n as f64, &[-1.0, beta], x),
                ] {
                    let error = extended_absolute_error(out, want);
                    assert!(
                        error <= tol,
                        "[{}] beta={beta} n={n} x={x}: out={out:e}, expected={want:e}, error={error:e}",
                        T::NAME
                    );
                }
            }
        }
    }
}

#[test]
fn eval_jacobi_alpha_minus_one() {
    for_both!(jacobi_alpha_minus_one);
}

// ---------------------------------------------------------------- Hermite

fn hermite_constructed<T: Num>() {
    check_poly::<T>(
        |n, _| {
            let mut out = vec![0.0; n as usize + 1];
            for m in 0..=(n / 2) {
                let power = n - 2 * m;
                out[power as usize] = (-1.0_f64).powi(m as i32) * specials::gamma(n as f64 + 1.0)
                    / (2.0_f64.powi(m as i32)
                        * specials::gamma(m as f64 + 1.0)
                        * specials::gamma(power as f64 + 1.0));
            }
            out
        },
        |n, p, x| hermitenorm_i::<T>(n, p, x),
        &[],
        (-100.0, 100.0),
        1e-12,
    );
    check_poly::<T>(
        |n, _| {
            let mut out = vec![0.0; n as usize + 1];
            for m in 0..=(n / 2) {
                let power = n - 2 * m;
                out[power as usize] = (-1.0_f64).powi(m as i32)
                    * specials::gamma(n as f64 + 1.0)
                    * 2.0_f64.powi(power as i32)
                    / (specials::gamma(m as f64 + 1.0) * specials::gamma(power as f64 + 1.0));
            }
            out
        },
        |n, p, x| hermite_i::<T>(n, p, x),
        &[],
        (-100.0, 100.0),
        1e-12,
    );
}

#[test]
fn eval_hermite_matches_constructed_polynomials() {
    for_both!(hermite_constructed);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L244-L255
fn hermite_domain<T: Num>() {
    assert!(hermite_i::<T>(-1, &[], 1.0).is_nan());
    assert!(hermitenorm_i::<T>(-1, &[], 1.0).is_nan());
    for n in 0..=2 {
        for x in [0.0, 1.0, NAN] {
            assert_eq!(
                hermite_i::<T>(n, &[], x).is_nan(),
                x.is_nan(),
                "n={n} x={x}"
            );
            assert_eq!(
                hermitenorm_i::<T>(n, &[], x).is_nan(),
                x.is_nan(),
                "n={n} x={x}"
            );
        }
    }
}

#[test]
fn hermite_evaluators_handle_domain_and_nan_inputs() {
    for_both!(hermite_domain);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L238-L241
fn hermite_high_order<T: Num>() {
    let expected = -1.457076485701412e60;
    let out = hermite_i::<T>(70, &[], 1.0);
    assert!(
        extended_absolute_error(out, expected) <= 1e-14 * expected.abs(),
        "{out:e}"
    );
}

#[test]
fn eval_hermite_preserves_high_order_accuracy() {
    for_both!(hermite_high_order);
}

// ---------------------------------------------------------------- Laguerre

fn genlaguerre_constructed<T: Num>() {
    check_poly::<T>(
        |n, p| laguerre_coefficients(n, p[0]),
        genlaguerre_i::<T>,
        &[(-0.99, 10.0)],
        (0.0, 100.0),
        1e-8,
    );
}

#[test]
fn eval_genlaguerre_matches_constructed_polynomials() {
    for_both!(genlaguerre_constructed);
}

fn genlaguerre_recurrence<T: Num>() {
    check_recurrence::<T>(
        genlaguerre_i::<T>,
        genlaguerre_r::<T>,
        &[(-0.99, 10.0)],
        (0.0, 100.0),
    );
}

#[test]
fn eval_genlaguerre_recurrence_overload() {
    for_both!(genlaguerre_recurrence);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L23-L26 and L258-L265
fn laguerre_domain<T: Num>() {
    for alpha in [-2.0, -1.0] {
        assert!(genlaguerre_i::<T>(0, &[alpha], 0.0).is_nan());
        assert!(genlaguerre_r::<T>(0.1, &[alpha], 0.0).is_nan());
    }
    for alpha in [1.0, NAN] {
        for x in [2.0, NAN] {
            let expected = alpha.is_nan() || x.is_nan();
            for n in [-1, 0, 1, 2] {
                assert_eq!(
                    genlaguerre_i::<T>(n, &[alpha], x).is_nan(),
                    expected,
                    "n={n} alpha={alpha} x={x}"
                );
            }
            for n in [0.0, 1.0, 2.0, 3.2] {
                assert_eq!(
                    genlaguerre_r::<T>(n, &[alpha], x).is_nan(),
                    expected,
                    "n={n} alpha={alpha} x={x}"
                );
            }
        }
    }
    for n in [-1, 0, 1, 2] {
        assert!(laguerre_i::<T>(n, &[], NAN).is_nan());
        assert!(laguerre_r::<T>(n as f64, &[], NAN).is_nan());
    }
    assert_eq!(genlaguerre_i::<T>(-1, &[0.5], 2.0), 0.0);
    assert_eq!(laguerre_i::<T>(-1, &[], 2.0), 0.0);
    assert_eq!(laguerre_i::<T>(1, &[], 1.0), 0.0);
}

#[test]
fn laguerre_evaluators_handle_domain_and_nan_inputs() {
    for_both!(laguerre_domain);
}

/// References with mpmath (`mp.laguerre`, 50 digits): the real parts of xsf's cases.
fn laguerre_noninteger<T: Num>() {
    // (n, alpha, x, eval_genlaguerre, eval_laguerre)
    let cases = [
        (
            3.0168473915183607,
            -0.43747311194333266,
            -0.899882726523523,
            3.3571359978327036,
            5.071956274504558,
        ),
        (
            3.2218471808260123,
            1.7304489192621135,
            -1.6522446694823354,
            35.43987061763628,
            12.328541041551668,
        ),
        (
            0.7025088614198185,
            0.7633882202584059,
            -1.8938561212645455,
            2.618801603517271,
            2.1831307232051684,
        ),
        (
            2.497178143317692,
            0.05110155510174175,
            0.3570627355036349,
            0.2829141374228878,
            0.22513917244388124,
        ),
        (
            3.932005885080444,
            1.2453484874705671,
            -0.6389979339280325,
            16.451903856048844,
            4.85681308204293,
        ),
        (
            1.3512699981194471,
            -0.26813539154963023,
            -1.613134492666144,
            3.0428418442147622,
            3.457758705467502,
        ),
        (
            3.9392055030090907,
            1.3243294667345447,
            0.14491236581880296,
            5.617240003306438,
            0.4880626786289751,
        ),
        (
            2.5362234720027486,
            1.5735116606324873,
            0.4740790094569842,
            3.3880086090370556,
            0.010277695053089116,
        ),
        (
            3.3751450991820793,
            -0.38543904086084446,
            -1.0884068973938126,
            5.2701572505753616,
            7.452302216322751,
        ),
        (
            0.7803498749856659,
            -0.2474964264756772,
            -0.8881055875596315,
            1.470851935394412,
            1.6628535580860702,
        ),
    ];
    for (n, alpha, x, gen_expected, expected) in cases {
        let g = genlaguerre_r::<T>(n, &[alpha], x);
        let l = laguerre_r::<T>(n, &[], x);
        assert!(
            extended_relative_error(g, gen_expected) < 1e-12,
            "[{}] genlaguerre n={n} alpha={alpha} x={x}: {g:e}",
            T::NAME
        );
        assert!(
            extended_relative_error(l, expected) < 1e-12,
            "[{}] laguerre n={n} x={x}: {l:e}",
            T::NAME
        );
    }
}

#[test]
fn laguerre_evaluators_support_noninteger_degrees() {
    for_both!(laguerre_noninteger);
}

/// Regression test for gh-457: binom(n + alpha, n) lost the digits of alpha + 1.
fn genlaguerre_alpha_near_minus_one<T: Num>() {
    for (n, alpha, x, desired) in [
        (2_i64, -0.99999999999999, 0.5, -0.375),
        (3, -0.9999999999999998, 2.0, 0.6666666666666665),
    ] {
        let out_int = genlaguerre_i::<T>(n, &[alpha], x);
        let out_double = genlaguerre_r::<T>(n as f64, &[alpha], x);
        assert!(
            extended_relative_error(out_int, desired) <= 1e-14,
            "[{}] int {out_int:e}",
            T::NAME
        );
        assert!(
            extended_relative_error(out_double, desired) <= 1e-14,
            "[{}] real {out_double:e}",
            T::NAME
        );
    }
}

#[test]
fn eval_genlaguerre_alpha_near_minus_one() {
    for_both!(genlaguerre_alpha_near_minus_one);
}

// ---------------------------------------------------------------- Gegenbauer

fn gegenbauer_constructed<T: Num>() {
    check_poly::<T>(
        |n, p| gegenbauer_coefficients(n, p[0]),
        gegenbauer_i::<T>,
        &[(-0.499, 10.0)],
        (-1.0, 1.0),
        1e-7,
    );
}

#[test]
fn eval_gegenbauer_matches_constructed_polynomials() {
    for_both!(gegenbauer_constructed);
}

/// x < 1e-5 uses a power series instead of the recurrence
fn gegenbauer_small_x<T: Num>() {
    for n in 0..10 {
        for alpha in [-0.25, 0.5, 2.0] {
            let coeffs = gegenbauer_coefficients(n, alpha);
            for x in [-1e-6, 0.0, 1e-6] {
                let expected = polyval(&coeffs, x);
                let out = gegenbauer_i::<T>(n, &[alpha], x);
                assert!(
                    extended_absolute_error(out, expected) <= 1e-14 * expected.abs(),
                    "[{}] n={n} alpha={alpha} x={x}: {out:e} vs {expected:e}",
                    T::NAME
                );
            }
        }
    }
}

#[test]
fn eval_gegenbauer_small_x_power_series() {
    for_both!(gegenbauer_small_x);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L268-L275
fn gegenbauer_nan<T: Num>() {
    for alpha in [0.0, 1.0, NAN] {
        for x in [1e-6, 2.0, NAN] {
            for n in [0.0, 1.0, 2.0, 3.2, NAN] {
                let expected = n.is_nan() || alpha.is_nan() || x.is_nan();
                assert_eq!(
                    gegenbauer_r::<T>(n, &[alpha], x).is_nan(),
                    expected,
                    "n={n} alpha={alpha} x={x}"
                );
                if !n.is_nan() && n == n.floor() {
                    assert_eq!(
                        gegenbauer_i::<T>(n as i64, &[alpha], x).is_nan(),
                        expected,
                        "n={n} alpha={alpha} x={x}"
                    );
                }
            }
        }
    }
}

#[test]
fn eval_gegenbauer_handles_nan_inputs() {
    for_both!(gegenbauer_nan);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L277-L291
fn gegenbauer_infinite_x<T: Num>() {
    for n in 0..10_i64 {
        for alpha in [-0.45, -0.25, -0.125, 0.0, 0.25, 1.0, 2.0] {
            for x in [-INF, INF] {
                let expected = if alpha == 0.0 {
                    0.0
                } else if n == 0 {
                    1.0
                } else {
                    // sign of leading coefficient: 2^n * Gamma(n+alpha) / (n! Gamma(alpha))
                    let lead_sign = if alpha < 0.0 { -1.0 } else { 1.0 };
                    let x_sign = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
                    lead_sign * x_sign * INF
                };
                assert_eq!(
                    gegenbauer_i::<T>(n, &[alpha], x),
                    expected,
                    "n={n} alpha={alpha} x={x}"
                );
                assert_eq!(
                    gegenbauer_r::<T>(n as f64, &[alpha], x),
                    expected,
                    "n={n} alpha={alpha} x={x}"
                );
            }
        }
    }
}

#[test]
fn eval_gegenbauer_handles_infinite_x() {
    for_both!(gegenbauer_infinite_x);
}

/// References with mpmath (`mp.gegenbauer`, 50 digits): the real parts of xsf's cases.
fn gegenbauer_noninteger<T: Num>() {
    let cases = [
        (
            3.0168473915183607,
            -0.3399741874655994,
            -0.4499413632617615,
            -0.15730182090720007,
        ),
        (
            3.2218471808260123,
            1.741230962491629,
            -0.8261223347411677,
            -5.743884954988635,
        ),
        (
            0.7025088614198185,
            0.8128526914480697,
            -0.9469280606322728,
            -2.1802125229493514,
        ),
        (
            2.497178143317692,
            0.129057492897672,
            0.17853136775181744,
            -0.10294922344150036,
        ),
        (
            3.932005885080444,
            1.2755345479717444,
            -0.31949896696401625,
            -0.007280823247730967,
        ),
    ];
    for (n, alpha, x, expected) in cases {
        let out = gegenbauer_r::<T>(n, &[alpha], x);
        assert!(
            extended_relative_error(out, expected) < 1e-12,
            "[{}] n={n} alpha={alpha} x={x}: {out:e}",
            T::NAME
        );
    }
}

#[test]
fn eval_gegenbauer_supports_noninteger_degrees() {
    for_both!(gegenbauer_noninteger);
}

/// Regression test: Gamma(n + 2 alpha) / Gamma(n + 1) overflowed for n > 170.
/// mpmath: `float(mp.gegenbauer(300.5, 2.5, 1.5))`.
fn gegenbauer_large_degree<T: Num>() {
    let out = gegenbauer_r::<T>(300.5, &[2.5], 1.5);
    let desired = 2.3319010819326352e+129;
    assert!(
        extended_relative_error(out, desired) <= 1e-11,
        "[{}] {out:e}",
        T::NAME
    );
}

#[test]
fn eval_gegenbauer_large_degree_prefactor() {
    for_both!(gegenbauer_large_degree);
}

// ---------------------------------------------------------------- Legendre

fn legendre_constructed<T: Num>() {
    check_poly::<T>(
        |n, _| legendre_coefficients(n, false),
        legendre_i::<T>,
        &[],
        (-1.0, 1.0),
        1e-8,
    );
    check_poly::<T>(
        |n, _| legendre_coefficients(n, true),
        sh_legendre_i::<T>,
        &[],
        (0.0, 1.0),
        1e-8,
    );
}

#[test]
fn eval_legendre_matches_constructed_polynomials() {
    for_both!(legendre_constructed);
}

fn legendre_recurrence<T: Num>() {
    check_recurrence::<T>(legendre_i::<T>, legendre_r::<T>, &[], (-1.0, 1.0));
    check_recurrence::<T>(sh_legendre_i::<T>, sh_legendre_r::<T>, &[], (0.0, 1.0));
}

#[test]
fn eval_legendre_recurrence_overload() {
    for_both!(legendre_recurrence);
}

/// Exercise both parities of the power series used near zero.
fn legendre_near_zero<T: Num>() {
    for n in 0..=11 {
        let coefficients = legendre_coefficients(n, false);
        for x in [-1e-6, 0.0, 1e-20, 1e-6] {
            let expected = polyval(&coefficients, x);
            let out = legendre_i::<T>(n, &[], x);
            assert!(
                (out - expected).abs() <= 1e-14 * expected.abs(),
                "[{}] n={n} x={x}: {out:e} vs {expected:e}",
                T::NAME
            );
        }
    }
}

#[test]
fn eval_legendre_preserves_accuracy_near_zero() {
    for_both!(legendre_near_zero);
}

/// P_{-n-1}(x) = P_n(x), n = 0, 1, 2, ...
fn legendre_negative_degrees<T: Num>() {
    for n in [-1, -2, -3, -4, -11, -12] {
        for x in [0.0, 1e-6, 0.5, 1.0] {
            assert_eq!(
                legendre_i::<T>(n, &[], x),
                legendre_i::<T>(-n - 1, &[], x),
                "n={n} x={x}"
            );
            assert_eq!(
                sh_legendre_i::<T>(n, &[], x),
                sh_legendre_i::<T>(-n - 1, &[], x),
                "n={n} x={x}"
            );
        }
    }
}

#[test]
fn legendre_evaluators_support_negative_integer_degrees() {
    for_both!(legendre_negative_degrees);
}

fn legendre_nan<T: Num>() {
    for n in [-3, -2, -1, 0, 1, 2, 3] {
        assert!(legendre_i::<T>(n, &[], NAN).is_nan(), "n={n}");
        assert!(sh_legendre_i::<T>(n, &[], NAN).is_nan(), "n={n}");
    }
    for n in [-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0] {
        assert!(legendre_r::<T>(n, &[], NAN).is_nan(), "n={n}");
        assert!(sh_legendre_r::<T>(n, &[], NAN).is_nan(), "n={n}");
    }
}

#[test]
fn legendre_evaluators_propagate_nan_inputs() {
    for_both!(legendre_nan);
}

// ---------------------------------------------------------------- Chebyshev

/// `(name, constructed coefficients, integer-degree evaluator, x range)` of the
/// six Chebyshev families.
#[allow(clippy::type_complexity)]
fn chebyshev_families<T: Num>() -> Vec<(
    &'static str,
    Box<dyn Fn(i64, &[f64]) -> Vec<f64>>,
    fn(i64, &[f64], f64) -> f64,
    fn(f64, &[f64], f64) -> f64,
    (f64, f64),
)> {
    vec![
        (
            "chebyt",
            Box::new(|n, _| chebyshev_coefficients(n, false, 1.0, 0.0, 1.0)),
            chebyt_i::<T>,
            chebyt_r::<T>,
            (-1.0, 1.0),
        ),
        (
            "chebyu",
            Box::new(|n, _| chebyshev_coefficients(n, true, 1.0, 0.0, 1.0)),
            chebyu_i::<T>,
            chebyu_r::<T>,
            (-1.0, 1.0),
        ),
        (
            "chebys",
            Box::new(|n, _| chebyshev_coefficients(n, true, 0.5, 0.0, 1.0)),
            chebys_i::<T>,
            chebys_r::<T>,
            (-2.0, 2.0),
        ),
        (
            "chebyc",
            Box::new(|n, _| chebyshev_coefficients(n, false, 0.5, 0.0, 2.0)),
            chebyc_i::<T>,
            chebyc_r::<T>,
            (-2.0, 2.0),
        ),
        (
            "sh_chebyt",
            Box::new(|n, _| chebyshev_coefficients(n, false, 2.0, -1.0, 1.0)),
            sh_chebyt_i::<T>,
            sh_chebyt_r::<T>,
            (0.0, 1.0),
        ),
        (
            "sh_chebyu",
            Box::new(|n, _| chebyshev_coefficients(n, true, 2.0, -1.0, 1.0)),
            sh_chebyu_i::<T>,
            sh_chebyu_r::<T>,
            (0.0, 1.0),
        ),
    ]
}

fn chebyshev_constructed<T: Num>() {
    for (name, coefficients, int_eval, _, x_range) in chebyshev_families::<T>() {
        let _ = name;
        check_poly::<T>(coefficients, int_eval, &[], x_range, 1e-8);
    }
}

#[test]
fn eval_chebyshev_matches_constructed_polynomials() {
    for_both!(chebyshev_constructed);
}

fn chebyshev_recurrence<T: Num>() {
    for (_, _, int_eval, real_eval, x_range) in chebyshev_families::<T>() {
        check_recurrence::<T>(int_eval, real_eval, &[], x_range);
    }
}

#[test]
fn eval_chebyshev_recurrence_overload() {
    for_both!(chebyshev_recurrence);
}

/// https://github.com/scipy/scipy/blob/v1.18.0/scipy/special/tests/test_orthogonal_eval.py#L10-L15
fn chebyt_high_order<T: Num>() {
    let x = 0.125;
    let mut n = 0;
    while n < 10000 {
        let expected = (n as f64 * f64::acos(x)).cos();
        assert!(
            (chebyt_i::<T>(n, &[], x) - expected).abs() <= 2e-11,
            "[{}] n={n}",
            T::NAME
        );
        n += 7;
    }
}

#[test]
fn eval_chebyt_preserves_high_order_accuracy() {
    for_both!(chebyt_high_order);
}

fn chebyshev_negative_degrees<T: Num>() {
    assert_eq!(chebyt_i::<T>(-3, &[], 0.5), -1.0);
    assert_eq!(chebyu_i::<T>(-1, &[], 0.5), 0.0);
    assert_eq!(chebyu_i::<T>(-2, &[], 0.5), -1.0);
    assert_eq!(chebyu_i::<T>(-3, &[], 0.5), -1.0);
    assert_eq!(chebyu_i::<T>(-4, &[], 0.5), 0.0);
}

#[test]
fn chebyshev_evaluators_support_negative_integer_degrees() {
    for_both!(chebyshev_negative_degrees);
}

fn chebyshev_nan<T: Num>() {
    for n in [-3, -2, -1, 0, 1, 2, 3] {
        for (name, _, int_eval, _, _) in chebyshev_families::<T>() {
            assert!(int_eval(n, &[], NAN).is_nan(), "{name} n={n}");
        }
    }
}

#[test]
fn chebyshev_integer_degree_evaluators_propagate_nan_inputs() {
    for_both!(chebyshev_nan);
}

/// T_nu(cos t) = cos(nu t), U_nu(cos t) = sin((nu + 1) t) / sin t.
fn chebyshev_noninteger<T: Num>() {
    for nu in [0.5, 1.3, 2.5, 4.7, 7.25] {
        for x in [-0.9, -0.3, 0.0, 0.4, 0.95] {
            let t = f64::acos(x);
            let expected_t = (nu * t).cos();
            let expected_u = ((nu + 1.0) * t).sin() / t.sin();
            assert!(
                extended_absolute_error(chebyt_r::<T>(nu, &[], x), expected_t) <= 1e-8,
                "[{}] T nu={nu} x={x}",
                T::NAME
            );
            assert!(
                extended_absolute_error(chebyu_r::<T>(nu, &[], x), expected_u) <= 1e-8,
                "[{}] U nu={nu} x={x}",
                T::NAME
            );
        }
    }
}

#[test]
fn chebyshev_evaluators_support_noninteger_degrees() {
    for_both!(chebyshev_noninteger);
}

// ---------------------------------------------------------------- Legendre functions

/// scipy/gh-23101: the normalized and unnormalized values next to z = -1.
fn assoc_legendre_gh_23101<T: Num>() {
    // (n, m, z, reference computed with mpmath, rtol)
    let below = -1.0 - f64::EPSILON; // nextafter(-1, -2)
    let above = -1.0 + f64::EPSILON / 2.0; // nextafter(-1, 0)
    let cases = [
        (1, 0, -1.0, -1.0, 1e-15),
        (1, 0, below, -1.0000000000000002, 1e-15),
        (1, 0, above, -0.9999999999999999, 1e-15),
        (2, 0, -1.0, 1.0, 1e-15),
        (2, 0, below, 1.0000000000000007, 1e-15),
        (2, 0, above, 0.9999999999999997, 1e-15),
        (3, 0, -1.0, -1.0, 1e-15),
        (3, 0, below, -1.0000000000000013, 1e-15),
        (3, 0, above, -0.9999999999999993, 1e-15),
        (4, 0, -1.0, 1.0, 1e-15),
        (4, 0, below, 1.0000000000000022, 4e-15),
        (4, 0, above, 0.9999999999999989, 1e-15),
    ];
    for (n, m, z, reference, rtol) in cases {
        let w = specials::assoc_legendre_p(n, m, T::from_f64(z), LegendreType::Ferrers).to_f64();
        let rel_error = extended_relative_error(w, reference);
        assert!(
            rel_error <= rtol,
            "[{}] n={n} m={m} z={z:e}: {w:e} vs {reference:e}, error {rel_error:e}",
            T::NAME
        );
    }
}

#[test]
fn assoc_legendre_p_scipy_gh_23101() {
    for_both!(assoc_legendre_gh_23101);
}

/// gh-78: the normalized m = 0 function is `sqrt((2n + 1) / 2)` times the unnormalized one.
fn assoc_legendre_norm_m0<T: Num>() {
    for n in 0..=10 {
        for z in linspace(-1.0, 1.0, 1000) {
            let leg_p =
                specials::assoc_legendre_p(n, 0, T::from_f64(z), LegendreType::Ferrers).to_f64();
            let leg_p_norm =
                specials::assoc_legendre_p_norm(n, 0, T::from_f64(z), LegendreType::Ferrers)
                    .to_f64();
            assert!(
                leg_p.is_finite() && leg_p_norm.is_finite(),
                "[{}] n={n} z={z}",
                T::NAME
            );
            let factor = ((2.0 * n as f64 + 1.0) / 2.0).sqrt();
            let expected = factor * leg_p;
            let rel_error = extended_relative_error(leg_p_norm, expected);
            assert!(
                rel_error <= 1e-8,
                "[{}] n={n} z={z}: {leg_p_norm:e} vs {expected:e}",
                T::NAME
            );
        }
    }
}

#[test]
fn assoc_legendre_p_norm_m0_gh_78() {
    for_both!(assoc_legendre_norm_m0);
}

/// gh-82: `pmv(0, nu, -1)` is a signed infinity for a non-integer degree.
fn pmv_endpoint<T: Num>() {
    let w = specials::pmv(T::from_f64(0.0), T::from_f64(0.5), T::from_f64(-1.0)).to_f64();
    assert!(
        w < -1e298 && w.is_sign_negative(),
        "[{}] nu = 0.5: {w}",
        T::NAME
    );
    let w = specials::pmv(T::from_f64(0.0), T::from_f64(1.5), T::from_f64(-1.0)).to_f64();
    assert!(
        w > 1e298 && w.is_sign_positive(),
        "[{}] nu = 1.5: {w}",
        T::NAME
    );
}

#[test]
fn lpmv_endpoint_x_minus_one_gives_signed_infinity_for_m_0() {
    for_both!(pmv_endpoint);
}
