//! Special functions in pure Rust, for `f64` and double-double.
//!
//! Every function at the crate root is generic over [`Real`], which is implemented for `f64`
//! and, with the `xprec` feature (on by default), for [`xprec::Df64`]. The
//! two use different algorithms ([`Strategy`]):
//!
//! * **`f64`** runs kernels ported from SciPy's
//!   [xsf](https://github.com/scipy/xsf) (Cephes), so results agree with
//!   `scipy.special` to within a few ulp.
//! * **`Df64`** runs precision-adaptive algorithms (series, asymptotic
//!   expansions and recurrences truncated at the type's epsilon), because
//!   Cephes' rational approximations stop at double precision.
//!
//! Domain errors and overflow are reported in the value, as in SciPy: NaN
//! outside the domain and at the poles where the sign is undefined, ±∞ at
//! poles with a definite sign or on overflow. There is no global error flag.
//!
//! This crate does not depend on tenferro. `tenferro-special` adds tensors,
//! autodiff rules and GPU kernels on top of it.
//!
//! # SciPy mapping
//!
//! | `scipy.special` | `specials` | `Df64` tolerance against mpmath (relative) |
//! | --- | --- | --- |
//! | `gamma(x)` | [`gamma`] | 1e-29 |
//! | `gammaln(x)` | [`lgamma`] | 1e-29 |
//! | `gammasgn(x)` | [`gammasgn`] | exact |
//! | `rgamma(x)` | [`rgamma`] | 1e-29 |
//! | `digamma(x)`, `psi(x)` | [`digamma`] | 1e-29 |
//! | `zeta(x, q)` | [`zeta`] | 1e-29 |
//! | `beta(a, b)` | [`beta`] | 1e-29 (`5e-30 · max(1, abs(ln B))` for huge arguments) |
//! | `betaln(a, b)` | [`lbeta`] | 2e-29 |
//! | `poch(a, m)` | [`poch`] | 1e-29 (grows like `ε m ln a` for `a` beyond 1e10) |
//! | `lgam1p(x)` | [`lgam1p`] | 1e-29 (relative also next to the zeros `x = 0` and `x = 1`) |
//! | `binom(n, k)` | [`binom`] | 5e-29 (grows like `ε abs(ln Γ)` for huge arguments: about 1e-22 at `k = 1e9`) |
//! | `zetac(x)` | [`zetac`] | 5e-29 (1e-29 for `x > -160`) |
//! | `polygamma(1, x)` (xsf `trigamma`) | [`trigamma`] | 1e-30 while the result is above 1e-292 (`x` below about 1e292) |
//! | `digammainv(y)` (xsf `digammainv`) | [`digammainv`] | 1e-30 for `y >= -1e290` (see [`digammainv`]) |
//! | `_gen_harmonic(n, a)` (xsf `gen_harmonic`) | [`gen_harmonic`] | 1e-30 for `a >= -3`; for `a < -3` it grows like `ε abs(a) ln n` (3.4e-30 measured at `a = -68`, `n = 70`) |
//! | `stirling2(n, k)` | [`stirling2`] | 1e-30 (exact below 2^106) |
//! | `ellipk(m)` | [`ellipk`] | 5e-30 |
//! | `ellipkm1(p)` | [`ellipkm1`] | 1e-30 |
//! | `ellipe(m)` | [`ellipe`] | 1e-30 |
//! | `ellipkinc(phi, m)` | [`ellipkinc`] | 1e-30 (the argument `phi` enters the result with the condition number of `F`, up to `1/√(1 - m)` near `phi = π/2`; for `m = 1` and `phi = 1.5707963267948966` `f64` gives `+∞` as xsf, `Df64` the finite exact value) |
//! | `ellipeinc(phi, m)` | [`ellipeinc`] | 1e-30 |
//! | `ellipj(u, m)` (`sn`, `cn`, `dn`) | [`ellipj`] | 1e-29, plus an absolute error of 24 ε `max(1, abs(u) √max(1, 1 - m))` `max(1, abs(value))`, ε = 2^-105: the phase of `u` carries a relative error ε |
//! | `ellipj(u, m)` (`ph`) | [`ellipj`] | 5e-30 |
//! | `elliprc(x, y)` | [`elliprc`] | 1e-30 |
//! | `elliprd(x, y, z)` | [`elliprd`] | 1e-30 |
//! | `elliprf(x, y, z)` | [`elliprf`] | 1e-30 |
//! | `elliprg(x, y, z)` | [`elliprg`] | 1e-30 |
//! | `elliprj(x, y, z, p)` | [`elliprj`] | 1e-30 |
//! | `agm(a, b)` | [`agm`] | 1e-30 |
//! | `cosm1(x)` | [`cosm1`] | 1e-30 for `abs(x) <= 2^50`; beyond, `f64` accuracy (absolute 1e-16) |
//! | `log1pmx(x)` (xsf `log1pmx`) | [`log1pmx`] | 1e-30 |
//! | `xlogy(x, y)` | [`xlogy`] | 1e-30 |
//! | `xlog1py(x, y)` | [`xlog1py`] | 1e-30 |
//! | `logit(x)` | [`logit`] | 1e-30 |
//! | `expit(x)` | [`expit`] | 1e-30 |
//! | `log_expit(x)` | [`log_expit`] | 1e-30 |
//! | `sinpi(x)` (xsf `sinpi`) | [`sinpi`] | 1e-30 |
//! | `cospi(x)` (xsf `cospi`) | [`cospi`] | 1e-30 |
//! | `sindg(x)` | [`sindg`] | 1e-30 |
//! | `cosdg(x)` | [`cosdg`] | 1e-30 |
//! | `tandg(x)` | [`tandg`] | 1e-30 |
//! | `cotdg(x)` | [`cotdg`] | 1e-30 |
//! | `radian(d, m, s)` | [`radian`] | 1e-30 |
//! | `exp2(x)` | [`exp2`] | 1e-30 |
//! | `exp10(x)` | [`exp10`] | 1e-30 |
//! | `cbrt(x)` | [`cbrt`] | 1e-30 |
//! | `expm1(x)` | [`expm1`] | 1e-30 |
//! | `log1p(x)` | [`log1p`] | 1e-30 |
//! | `eval_jacobi(n, α, β, x)`, integer `n` | [`eval_jacobi_int`] | 1e-28 (`α + β` not a negative integer below `-1`; NaN there) |
//! | `eval_sh_jacobi(n, p, q, x)`, integer `n` | [`eval_sh_jacobi_int`] | 1e-28 |
//! | `eval_gegenbauer(n, α, x)`, integer `n` | [`eval_gegenbauer_int`] | 1e-28 |
//! | `eval_chebyt`, `eval_chebyu`, `eval_chebyc`, `eval_chebys`, `eval_sh_chebyt`, `eval_sh_chebyu`, integer `n` | [`eval_chebyt_int`], [`eval_chebyu_int`], [`eval_chebyc_int`], [`eval_chebys_int`], [`eval_sh_chebyt_int`], [`eval_sh_chebyu_int`] | 1e-29 |
//! | `eval_legendre`, `eval_sh_legendre`, integer `n` | [`eval_legendre_int`], [`eval_sh_legendre_int`] | 2e-29 |
//! | `eval_laguerre`, `eval_genlaguerre`, integer `n` | [`eval_laguerre_int`], [`eval_genlaguerre_int`] | 1e-28 |
//! | `eval_hermite`, `eval_hermitenorm` | [`eval_hermite`], [`eval_hermitenorm`] | 1e-29 |
//! | the same twelve functions with a real `n` | [`eval_jacobi`], [`eval_sh_jacobi`], [`eval_gegenbauer`], [`eval_chebyt`], [`eval_chebyu`], [`eval_chebyc`], [`eval_chebys`], [`eval_sh_chebyt`], [`eval_sh_chebyu`], [`eval_legendre`], [`eval_sh_legendre`], [`eval_laguerre`], [`eval_genlaguerre`] | as the integer-degree row for a non-negative integer-valued `n`; **`f64` accuracy for any other `n`** (`Df64` evaluates the `f64` kernel) |
//! | `pmv(m, v, x)` (`lpmv`) | [`pmv`] | 1e-28 for an integer `v`; **`f64` accuracy (3e-13) for a non-integer `v`** |
//! | `legendre_p(n, z)` | [`legendre_p`] | 2e-29 |
//! | `assoc_legendre_p(n, m, z)` (`branch_cut` 2 and 3, `norm`) | [`assoc_legendre_p`], [`assoc_legendre_p_norm`] | 1e-29 |
//! | `sph_legendre_p(n, m, θ)` | [`sph_legendre_p`] | 1e-29 (relative accuracy of `sin θ`, an absolute 1e-32, next to multiples of `π` for `m != 0`) |
//! | `lqn(n, x)` | [`lqn`] | 2e-29 for `Q_n`; the derivative loses a factor `1/abs(1 - x^2)` |
//! | `lqmn(m, n, x)` | [`lqmn`] | 2e-29, with a factor `1/abs(1 - x^2)^m` (`m + 1` for the derivative) |
//! | `round(x)` | not provided: use [`f64::round_ties_even`] (exact; `-0` where SciPy gives `+0` for `-1/2 <= x < 0`) | - |
//!
//! The polynomials and Legendre functions of integer degree evaluate a forward recurrence in the degree
//! (work linear in `n`; Miller's backward recurrence for `lqn`, `lqmn` at `abs(x) > 1`), and
//! the `_int` suffix marks SciPy's integer-degree loop where the real-degree one has the same name.
//! The work and the limits of each function are in its documentation; the limits of the `f64` kernels
//! (xsf's, with its overflow and cancellation) are in `docs/upstream-issues.md`.
//! Not provided: the derivatives of the Legendre functions with respect to the argument (xsf's `dual`
//! numbers; `lqn` and `lqmn` return the first derivative), the complex-argument overloads, and
//! `sph_harm` (complex result).
//!
//! `exp2`, `exp10`, `cbrt`, `expm1` and `log1p` are `std`'s functions for
//! `f64` (they pass xsref's tables); the crate provides them for the `Df64`
//! path, where `xprec`'s versions have defects (`docs/upstream-issues.md`).
//!
//! A double-double result below about 1e-292 has a subnormal low word, so its
//! relative accuracy falls towards that of `f64` at the bottom of the range;
//! the figures above hold for results above that.
//!
//! # Families from `few_special_functions` (`f64` only)
//!
//! The modules [`fermi_dirac`], [`bose_einstein`], [`clausen`], [`debye`],
//! [`coulomb`], [`whittaker`], [`marcum_q`], [`dawson`], [`fresnel`],
//! [`voigt`] and [`parabolic_cylinder`] are re-exported from
//! [`few_special_functions`](https://crates.io/crates/few_special_functions)
//! 0.2.1, pinned exactly in the workspace manifest. They take `f64` (some also
//! `Complex64` from `num-complex` 0.4) and have no `Df64` form. Domain errors
//! give NaN, as above. Each function that iterates also has a `*_checked` form
//! that returns [`SpecialError`] when an iteration reaches its cap; the plain
//! form returns NaN there.
//!
//! These families are validated against mpmath in `few_special_functions`
//! itself. Maximum relative errors measured there:
//!
//! | Module | Functions | Points | Max relative error |
//! | --- | --- | --- | --- |
//! | [`fermi_dirac`] | `F_j(x)`, `F_j(x)/Γ(j+1)` | 323 | 8.1e-15 |
//! | [`bose_einstein`] | `Γ(k+1) Li_{k+1}(e^η)`, `Li_{k+1}(e^η)` | 70, 110 | 1.1e-15 |
//! | [`clausen`] | `Cl_n(θ)`, n = 1..=6 | 120 | 1.1e-14 |
//! | [`debye`] | `D_n(β, x)` | 216 | 4.8e-16 |
//! | [`coulomb`] | `F`, `G`, `H±` | 1092 | 4.7e-13 |
//! | [`coulomb`] | `phi_dot`, `psi`, `i` | 61 | 2.0e-10 (cancellation in Φ at ρ = 5; median 2e-14) |
//! | [`whittaker`] | `M`, `W` and their `z`-derivatives | 1188 | 2.9e-13 |
//! | [`marcum_q`] | `Q_μ(a, b)`, `dQ_M/db` | 392, 120 | 3.3e-14 |
//! | [`dawson`] | `D(x)` | 27 | 9.5e-16 |
//! | [`fresnel`] | `C(x)`, `S(x)` | 16 | 2.2e-12 |
//! | [`voigt`] | `K(x, y)` | 400 | 1.2e-15 |
//! | [`parabolic_cylinder`] | `U`, `V`, `D_ν` and derivatives | 176, 54 | 1.8e-13 |
//! | [`parabolic_cylinder`] | `W` and its derivative | 48 | 4.6e-11 |
//!
//! `tests/few_special_functions.rs` checks one mpmath value per family through
//! these re-exports.
//!
//! # Validation
//!
//! * **`f64`** is checked with xsf's own test suite: the reference tables of
//!   [xsref](https://github.com/scipy/xsref) (about 59,000 points for these
//!   functions) with xsf's error metric and per-point tolerances, plus xsf's
//!   hand-written regression cases. Rust does not contract `a*b + c` into FMA,
//!   so the tolerances of xsf built by gcc on x86-64 apply on every platform.
//!   They are asserted on every platform, with no relaxation. The port calls
//!   the platform's `ln`/`exp`/`powf`, so a libm that is not correctly rounded
//!   can exceed them. Linux (glibc, x86-64 and aarch64) passes every row. On
//!   macOS one `binom` row fails because Apple's `log` is off by 1 ulp, and two
//!   `pmv` rows because of another libm function; see `docs/upstream-issues.md`.
//! * **`Df64`** is checked against mpmath double-double values at the same
//!   xsref points and at points chosen for the double-double algorithms: both
//!   sides of every switch between algorithms, arguments whose low word
//!   matters (next to poles) and fixed-seed random double-double arguments.
//!   Where mpmath is undefined (poles, outside the domain), it must return
//!   SciPy's value.
//!
//! The figures are the tolerances the test suite asserts, not proven bounds.
//! Reference data is generated by `scripts/gen_specials_reference.py` and
//! `scripts/import_xsref.py`; every mpmath value is computed at two working
//! precisions that must agree.
//!
//! # Examples
//!
//! ```
//! use specials::gamma;
//!
//! // Γ(1/2) = √π
//! assert!((gamma(0.5_f64) - core::f64::consts::PI.sqrt()).abs() < 1e-15);
//! ```
//!
//! With double-double:
//!
//! ```
//! # #[cfg(feature = "xprec")] {
//! use specials::lgamma;
//! use xprec::Df64;
//!
//! // ln Γ(10) = ln 9! = ln 362880
//! let y = lgamma(Df64::from(10.0));
//! let expected = Df64::from(362880.0).ln();
//! assert!(((y - expected) / expected).abs() < Df64::from(1e-30));
//! # }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod generic;
mod real;
// Coefficients keep upstream's digits, so later xsf changes can be diffed.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod xsf;

pub use real::{Real, Strategy};

pub use few_special_functions::{
    bose_einstein, clausen, coulomb, dawson, debye, fermi_dirac, fresnel, marcum_q,
    parabolic_cylinder, voigt, whittaker, SpecialError,
};

/// Γ(x), the gamma function (DLMF 5.2.1).
///
/// Poles: `gamma(±0) = ±∞`; at negative integers and at `-∞` the result is NaN.
/// `gamma(+∞) = +∞`, and the result overflows to `+∞` above x ≈ 171.62.
///
/// # Examples
///
/// ```
/// use specials::gamma;
///
/// assert_eq!(gamma(5.0_f64), 24.0);
/// assert!(gamma(-1.0_f64).is_nan());
/// assert_eq!(gamma(-0.0_f64), f64::NEG_INFINITY);
/// ```
pub fn gamma<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::gamma::gamma(x.approx_f64())),
        Strategy::Adaptive => generic::gamma::gamma(x),
    }
}

/// ln|Γ(x)|, the logarithm of the absolute value of the gamma function.
///
/// `+∞` at the poles (zero and the negative integers). Near its zeros at
/// x = 1 and x = 2, the `f64` kernel (Cephes) is accurate in absolute rather
/// than relative terms (about 1e-16); the `Df64` path uses the Maclaurin
/// series about 1 and 2 and stays relatively accurate there.
///
/// # Examples
///
/// ```
/// use specials::lgamma;
///
/// assert_eq!(lgamma(1.0_f64), 0.0);
/// assert!((lgamma(0.5_f64) - 0.5723649429247001).abs() < 1e-15);
/// ```
pub fn lgamma<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::gamma::gammaln(x.approx_f64())),
        Strategy::Adaptive => generic::gamma::lgamma(x),
    }
}

/// The sign of Γ(x): `1` or `-1`, `±1` at `±0`, NaN at the poles and at `-∞`.
///
/// # Examples
///
/// ```
/// use specials::gammasgn;
///
/// assert_eq!(gammasgn(-0.5_f64), -1.0);
/// assert_eq!(gammasgn(-1.5_f64), 1.0);
/// ```
pub fn gammasgn<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::gamma::gammasgn(x.approx_f64())),
        Strategy::Adaptive => generic::gamma::gammasgn(x),
    }
}

/// 1/Γ(x), the reciprocal gamma function (entire; zero at the poles of Γ).
///
/// Results below `f64::MIN_POSITIVE` (x ≳ 171.6) flush to zero.
///
/// # Examples
///
/// ```
/// use specials::rgamma;
///
/// assert_eq!(rgamma(-3.0_f64), 0.0);
/// assert_eq!(rgamma(4.0_f64), 1.0 / 6.0);
/// ```
pub fn rgamma<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::gamma::rgamma(x.approx_f64())),
        Strategy::Adaptive => generic::gamma::rgamma(x),
    }
}

/// ψ(x) = Γ'(x)/Γ(x), the digamma function (DLMF 5.2.2).
///
/// `digamma(±0) = ∓∞`; NaN at the negative integers and at `-∞`. Near its
/// zeros at x ≈ 1.4616 and x ≈ -0.5041 both paths use a series about the zero
/// and are relatively accurate. Near the other negative zeros, accuracy is
/// absolute (relative to the size of the terms of the reflection formula).
///
/// # Examples
///
/// ```
/// use specials::digamma;
///
/// // ψ(1) = -γ
/// assert!((digamma(1.0_f64) + 0.5772156649015329).abs() < 1e-16);
/// ```
pub fn digamma<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::psi::digamma(x.approx_f64())),
        Strategy::Adaptive => generic::digamma::digamma(x),
    }
}

/// ζ(s, q) = Σ_{k≥0} (k + q)^-s, the Hurwitz zeta function (DLMF 25.11.1).
///
/// For `q = 1` this is the Riemann zeta function for all real `s`, with the
/// analytic continuation for `s < 1` (as SciPy/xsf). For `q != 1` it is defined
/// here for `s > 1` as in `scipy.special.zeta(x, q)`: `s = 1` gives `+∞`,
/// `s < 1` gives NaN, `q` a non-positive integer gives `+∞`, and `q < 0`
/// requires an integer `s` (NaN otherwise).
///
/// # Examples
///
/// ```
/// use specials::zeta;
///
/// // ζ(2) = π²/6
/// let pi = core::f64::consts::PI;
/// assert!((zeta(2.0_f64, 1.0) - pi * pi / 6.0).abs() < 1e-15);
/// ```
pub fn zeta<T: Real>(s: T, q: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::zeta::zeta_xsf(s.approx_f64(), q.approx_f64())),
        Strategy::Adaptive => generic::zeta::zeta(s, q),
    }
}

/// B(a, b) = Γ(a)Γ(b)/Γ(a+b), the beta function (DLMF 5.12.1).
///
/// For a negative integer argument the result follows SciPy: finite when the
/// other argument is an integer that makes B a polynomial value, `+∞`
/// otherwise. On the `Df64` path, when an argument exceeds 170 the result is
/// `exp(ln B)` and its relative error grows like `|ln B| ε`.
///
/// # Examples
///
/// ```
/// use specials::beta;
///
/// // B(2, 3) = 1!2!/4! = 1/12
/// assert!((beta(2.0_f64, 3.0) - 1.0 / 12.0).abs() < 1e-16);
/// ```
pub fn beta<T: Real>(a: T, b: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::beta::beta(a.approx_f64(), b.approx_f64())),
        Strategy::Adaptive => generic::beta::beta(a, b),
    }
}

/// ln|B(a, b)|, the logarithm of the absolute value of the beta function
/// (`scipy.special.betaln`).
///
/// Accurate in absolute terms near the zeros of ln|B| (where |B| = 1).
///
/// # Examples
///
/// ```
/// use specials::lbeta;
///
/// assert!((lbeta(2.0_f64, 3.0) + 12.0_f64.ln()).abs() < 1e-15);
/// ```
pub fn lbeta<T: Real>(a: T, b: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::beta::lbeta(a.approx_f64(), b.approx_f64())),
        Strategy::Adaptive => generic::beta::lbeta(a, b),
    }
}

/// (a)ₘ = Γ(a + m)/Γ(a), the Pochhammer symbol (rising factorial, DLMF 5.2.5).
///
/// For a non-negative integer `m` this is the product `a (a+1) ⋯ (a+m-1)`. It
/// is `0` when `a` is a non-positive integer and `a + m` is not, and `+∞` the
/// other way round (as xsf).
///
/// # Examples
///
/// ```
/// use specials::poch;
///
/// assert_eq!(poch(3.0_f64, 4.0), 3.0 * 4.0 * 5.0 * 6.0);
/// assert_eq!(poch(-2.0_f64, 3.0), 0.0);
/// ```
pub fn poch<T: Real>(a: T, m: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::poch::poch(a.approx_f64(), m.approx_f64())),
        Strategy::Adaptive => generic::poch::poch(a, m),
    }
}

/// ln Γ(1 + x), accurate for `x` near 0 and near 1 where `lgamma(1 + x)` is not.
///
/// `+∞` at `x = -1` and NaN-free elsewhere on the real line except the
/// negative integers below -1, where it is `+∞` as well (poles of Γ).
///
/// # Examples
///
/// ```
/// use specials::lgam1p;
///
/// // ln Γ(1 + x) ≈ -γx for tiny x, with full relative accuracy.
/// let y = lgam1p(1e-20_f64);
/// assert!((y + 0.5772156649015329e-20).abs() < 1e-35);
/// ```
pub fn lgam1p<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::unity::lgam1p(x.approx_f64())),
        Strategy::Adaptive => generic::lgam1p::lgam1p(x),
    }
}

/// C(n, k), the binomial coefficient, extended to real arguments as
/// Γ(n+1)/(Γ(k+1)Γ(n-k+1)).
///
/// NaN for a negative integer `n`. Integer `n` and `k` give integers exactly
/// while they fit the mantissa; `k` a negative integer gives `0` for
/// non-integer `n`.
///
/// The `f64` path is xsf's, including its large-`k` expansion, which xsf also
/// applies when `k > 1e8 |n|` but `k` is not large. There (`k < 1e6`, tiny `n`)
/// the `f64` result has a relative error up to about 3e-10, for example
/// `binom(1e-9, 0.5)`; see `docs/upstream-issues.md`. The `Df64` path is not
/// affected.
///
/// # Examples
///
/// ```
/// use specials::binom;
///
/// assert_eq!(binom(10.0_f64, 3.0), 120.0);
/// assert!((binom(0.5_f64, 2.0) + 0.125).abs() < 1e-16);
/// ```
pub fn binom<T: Real>(n: T, k: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::binom::binom(n.approx_f64(), k.approx_f64())),
        Strategy::Adaptive => generic::binom::binom(n, k),
    }
}

/// ζ(x) − 1, the Riemann zeta function minus one (accurate for large `x`
/// where ζ(x) rounds to 1).
///
/// `zetac(1) = +∞`, `zetac(+∞) = 0`, NaN at `-∞`. For `x > 1075` the result
/// underflows to zero (and is subnormal just below). For `x < 1` the result is
/// accurate in absolute terms near the isolated points where ζ(x) = 1.
///
/// # Examples
///
/// ```
/// use specials::zetac;
///
/// // ζ(2) - 1 = π²/6 - 1
/// let pi = core::f64::consts::PI;
/// assert!((zetac(2.0_f64) - (pi * pi / 6.0 - 1.0)).abs() < 1e-15);
/// assert_eq!(zetac(1.0_f64), f64::INFINITY);
/// ```
pub fn zetac<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::zetac::zetac(x.approx_f64())),
        Strategy::Adaptive => generic::zeta::zetac(x),
    }
}

/// ψ'(x), the trigamma function (DLMF 5.15.1).
///
/// `+∞` at zero and the negative integers, NaN at `-∞`. For `x > 0` it equals
/// ζ(2, x).
///
/// # Examples
///
/// ```
/// use specials::trigamma;
///
/// // ψ'(1) = π²/6
/// let pi = core::f64::consts::PI;
/// assert!((trigamma(1.0_f64) - pi * pi / 6.0).abs() < 1e-15);
/// ```
pub fn trigamma<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::trigamma::trigamma(x.approx_f64())),
        Strategy::Adaptive => generic::trigamma::trigamma(x),
    }
}

/// The `x > 0` with ψ(x) = y, the inverse of [`digamma`] on its increasing branch.
///
/// `0` for `y = -∞`; `+∞` for `y` above ln(`f64::MAX`) ≈ 709.78 and `y = +∞`.
///
/// For `y` below about -1.3e154 the root is below 7.5e-155, where ψ'
/// overflows. There the `f64` kernel (xsf's Newton iteration) returns NaN
/// unless its starting point is already exact, for example at `y = -1e300`.
/// The `Df64` path starts from the `f64` value, so it is NaN at the same
/// points. Where it is finite, it holds 1e-30 down to `y = -1e290`. Below
/// that the root is under 1e-290, where the low word of a double-double is
/// subnormal, and the accuracy falls towards that of `f64` at the bottom of the
/// normal range.
///
/// # Examples
///
/// ```
/// use specials::{digamma, digammainv};
///
/// let x = digammainv(digamma(2.5_f64));
/// assert!((x - 2.5).abs() < 1e-14);
/// ```
pub fn digammainv<T: Real>(y: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::digammainv::digammainv(y.approx_f64())),
        Strategy::Adaptive => generic::digammainv::digammainv(y),
    }
}

/// H(n, a) = Σ_{i=1}^{n} i^-a, the generalized harmonic number.
///
/// `n` must be a positive integer; a non-integer or `n < 1` gives NaN
/// (xsf leaves that check to the Python wrapper). `a` may be any real number:
/// `a = 0` gives `n`, `a = +∞` gives 1 and `a = -∞` gives `+∞`. `n = +∞`
/// gives the limit: ζ(a) for `a > 1` and `+∞` for `a <= 1`. Both strategies
/// sum at most 2^20 terms directly; beyond that they use the Euler–Maclaurin
/// formula, so the cost does not grow with `n`.
///
/// # Examples
///
/// ```
/// use specials::gen_harmonic;
///
/// assert!((gen_harmonic(4.0_f64, 1.0) - 25.0 / 12.0).abs() < 1e-15);
/// assert_eq!(gen_harmonic(4.0_f64, -1.0), 10.0);
/// ```
pub fn gen_harmonic<T: Real>(n: T, a: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::gen_harmonic::gen_harmonic(
            n.approx_f64(),
            a.approx_f64(),
        )),
        Strategy::Adaptive => generic::gen_harmonic::gen_harmonic(n, a),
    }
}

/// S(n, k), the Stirling number of the second kind: the number of ways to
/// partition `n` elements into `k` non-empty subsets.
///
/// Arguments, the same for both strategies: NaN gives NaN, and `n < 0` or
/// `k < 0` gives 0. Otherwise `n` and `k` are truncated towards zero, so
/// `stirling2(5.7, 3.2) = S(5, 3)` and `stirling2(5.1, 5.5) = S(5, 5) = 1`.
/// Then `S(n, n) = 1` (including `n = +∞`), `S(n, k) = 0` for `k > n` and for
/// `k = 0 < n`, and `S(+∞, k) = +∞` for finite `k >= 2`. Overflow gives `+∞`.
///
/// For `n > 219` the `f64` path is xsf's second-order Temme asymptotic
/// approximation (relative error up to about 1e-6 at `n = 220`, falling with
/// `n`; for `n <= 219` it is the exact recurrence), except for `k = 2` and
/// `n - k <= 64`, which are evaluated exactly, and for `(n-k) ln k > ln(f64::MAX)`,
/// which is `+∞` without evaluating Temme. The `Df64` path evaluates the
/// recurrence (with closed forms for `k = 2`, `n - 1` and `n - 2`) up to a cost
/// of `(k-1)(n-k) <= 3e7` and is exact below 2^106. Beyond that cost it uses the
/// exact sum of Graham, Knuth and Patashnik (6.43) for `n - k <= 64`; every
/// other argument beyond that cost overflows to `+∞`.
///
/// # Examples
///
/// ```
/// use specials::stirling2;
///
/// assert_eq!(stirling2(5.0_f64, 3.0), 25.0);
/// assert_eq!(stirling2(10.0_f64, 7.0), 5880.0);
/// ```
pub fn stirling2<T: Real>(n: T, k: T) -> T {
    // One argument contract for both strategies (truncation, signs, infinities).
    let (n, k) = match generic::stirling2::prepare(n, k) {
        generic::stirling2::Prepared::Value(v) => return v,
        generic::stirling2::Prepared::Kernel(n, k) => (n, k),
    };
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::stirling2::stirling2(n.approx_f64(), k.approx_f64())),
        Strategy::Adaptive => generic::stirling2::stirling2(n, k),
    }
}

/// K(m), the complete elliptic integral of the first kind with parameter
/// `m = k²` (DLMF 19.2.8): `∫₀^{π/2} dθ / √(1 − m sin²θ)`.
///
/// `ellipk(0) = π/2`. `ellipk(1) = +∞`; NaN for `m > 1`; `ellipk(-∞) = 0`.
/// For `m` close to 1 use [`ellipkm1`], which takes `1 − m` and keeps its
/// relative accuracy; `1 - m` is rounded here.
///
/// # Examples
///
/// ```
/// use specials::ellipk;
///
/// // K(0) = π/2; K(1/2) = Γ(1/4)² / (4 √π)
/// let pi = core::f64::consts::PI;
/// assert!((ellipk(0.0_f64) - pi / 2.0).abs() < 1e-15);
/// assert!((ellipk(0.5_f64) - 1.8540746773013719).abs() < 1e-15);
/// assert_eq!(ellipk(1.0_f64), f64::INFINITY);
/// ```
pub fn ellipk<T: Real>(m: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::ellip::ellipk(m.approx_f64())),
        Strategy::Adaptive => generic::ellip::ellipk(m),
    }
}

/// K(1 − p), the complete elliptic integral of the first kind as a function
/// of the complementary parameter `p = 1 − m` (`scipy.special.ellipkm1`).
///
/// Accurate for `p` near 0, where `ellipk(1 - p)` has lost the digits of `p`.
/// `ellipkm1(0) = +∞`, `ellipkm1(+∞) = 0`; NaN for `p < 0`.
///
/// # Examples
///
/// ```
/// use specials::ellipkm1;
///
/// // K(m) ≈ ln(4/√p) for small p = 1 - m
/// let p = 1e-20_f64;
/// assert!((ellipkm1(p) - (4.0 / p.sqrt()).ln()).abs() < 1e-15);
/// assert_eq!(ellipkm1(0.0_f64), f64::INFINITY);
/// ```
pub fn ellipkm1<T: Real>(p: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::ellip::ellipkm1(p.approx_f64())),
        Strategy::Adaptive => generic::ellip::ellipkm1(p),
    }
}

/// E(m), the complete elliptic integral of the second kind with parameter
/// `m = k²` (DLMF 19.2.8): `∫₀^{π/2} √(1 − m sin²θ) dθ`.
///
/// `ellipe(0) = π/2`, `ellipe(1) = 1`; NaN for `m > 1`; `ellipe(-∞) = +∞`.
///
/// # Examples
///
/// ```
/// use specials::ellipe;
///
/// let pi = core::f64::consts::PI;
/// assert!((ellipe(0.0_f64) - pi / 2.0).abs() < 1e-15);
/// assert_eq!(ellipe(1.0_f64), 1.0);
/// assert!((ellipe(0.5_f64) - 1.3506438810476755).abs() < 1e-15);
/// ```
pub fn ellipe<T: Real>(m: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::ellip::ellipe(m.approx_f64())),
        Strategy::Adaptive => generic::ellip::ellipe(m),
    }
}

/// F(φ | m), the incomplete elliptic integral of the first kind (DLMF 19.2.4):
/// `∫₀^φ dθ / √(1 − m sin²θ)`, for any real `φ`.
///
/// F is odd in `φ` and `F(φ + nπ | m) = F(φ | m) + 2n K(m)`. `ellipkinc(φ, 0) = φ`;
/// NaN for `m > 1`; `ellipkinc(±∞, m) = ±∞` and `ellipkinc(φ, -∞) = 0`. For
/// `m = 1` the integral is `asinh(tan φ)`, and `+∞` for `|φ| >= π/2`. Each type
/// compares with its own value of π/2: the `f64` value 1.5707963267948966 is
/// 6.1e-17 below π/2, so `ellipkinc(1.5707963267948966_f64, 1.0)` is `+∞` as in
/// xsf, while the same number as a double-double gives the exact
/// `atanh(sin φ)` = 38.025003373828866.
///
/// # Examples
///
/// ```
/// use specials::{ellipk, ellipkinc};
///
/// // F(π/2 | m) = K(m)
/// let m = 0.3_f64;
/// let half_pi = core::f64::consts::FRAC_PI_2;
/// assert!((ellipkinc(half_pi, m) - ellipk(m)).abs() < 1e-15);
/// assert_eq!(ellipkinc(0.7_f64, 0.0), 0.7);
/// ```
pub fn ellipkinc<T: Real>(phi: T, m: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::ellip::ellipkinc(phi.approx_f64(), m.approx_f64())),
        Strategy::Adaptive => generic::ellip::ellipkinc(phi, m),
    }
}

/// E(φ | m), the incomplete elliptic integral of the second kind (DLMF 19.2.5):
/// `∫₀^φ √(1 − m sin²θ) dθ`, for any real `φ`.
///
/// E is odd in `φ` and `E(φ + nπ | m) = E(φ | m) + 2n E(m)`. `ellipeinc(φ, 0) = φ`;
/// NaN for `m > 1`; `ellipeinc(±∞, m) = ±∞` and `ellipeinc(φ, -∞) = +∞`.
///
/// # Examples
///
/// ```
/// use specials::{ellipe, ellipeinc};
///
/// // E(π/2 | m) = E(m)
/// let m = 0.3_f64;
/// let half_pi = core::f64::consts::FRAC_PI_2;
/// assert!((ellipeinc(half_pi, m) - ellipe(m)).abs() < 1e-15);
/// assert_eq!(ellipeinc(0.7_f64, 0.0), 0.7);
/// ```
pub fn ellipeinc<T: Real>(phi: T, m: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::ellip::ellipeinc(phi.approx_f64(), m.approx_f64())),
        Strategy::Adaptive => generic::ellip::ellipeinc(phi, m),
    }
}

/// `(sn, cn, dn, ph)`: the Jacobian elliptic functions `sn(u | m)`, `cn(u | m)`,
/// `dn(u | m)` (DLMF 22.2) and the amplitude `ph = am(u | m)`, with
/// `sn = sin ph`, for real `u` and parameter `m <= 1`.
///
/// The amplitude is the continuous branch, `am(u + 2K(m)) = am(u) + π`
/// (xsf's `ph`). NaN in all four outputs for `m > 1` or NaN arguments.
///
/// # Examples
///
/// ```
/// use specials::ellipj;
///
/// // m = 0: sn = sin, cn = cos, dn = 1, am(u) = u
/// let (sn, cn, dn, ph) = ellipj(0.5_f64, 0.0);
/// assert!((sn - 0.5_f64.sin()).abs() < 1e-15);
/// assert!((cn - 0.5_f64.cos()).abs() < 1e-15);
/// assert_eq!((dn, ph), (1.0, 0.5));
/// ```
pub fn ellipj<T: Real>(u: T, m: T) -> (T, T, T, T) {
    match T::STRATEGY {
        Strategy::F64Kernels => {
            let (sn, cn, dn, ph) = xsf::ellip::ellipj(u.approx_f64(), m.approx_f64());
            (k64(sn), k64(cn), k64(dn), k64(ph))
        }
        Strategy::Adaptive => generic::ellip::ellipj(u, m),
    }
}

/// R_C(x, y), Carlson's degenerate elliptic integral (DLMF 19.2.17):
/// `R_C(x, y) = R_F(x, y, y)`.
///
/// Defined for `x >= 0` and `y != 0`. For `y < 0` it is the Cauchy principal
/// value (DLMF 19.20.14). `y = 0` and `x < 0` give NaN; an infinite argument
/// gives 0.
///
/// # Examples
///
/// ```
/// use specials::elliprc;
///
/// assert_eq!(elliprc(1.0_f64, 1.0), 1.0);
/// // R_C(0, 1/4) = π
/// assert!((elliprc(0.0_f64, 0.25) - core::f64::consts::PI).abs() < 1e-15);
/// ```
pub fn elliprc<T: Real>(x: T, y: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::carlson::elliprc(x.approx_f64(), y.approx_f64())),
        Strategy::Adaptive => generic::carlson::elliprc(x, y),
    }
}

/// R_D(x, y, z), Carlson's symmetric elliptic integral of the second kind
/// (DLMF 19.16.5), symmetric in `x` and `y`.
///
/// Defined for `x, y, z >= 0` with `z > 0` and at most one of `x`, `y` zero.
/// `+∞` if `z = 0` or if `x = y = 0`; 0 if an argument is infinite; NaN for a
/// negative argument. The `Df64` result is accurate to 1e-30 while it is above
/// about 1e-292 (a smaller double-double has a subnormal low word).
///
/// # Examples
///
/// ```
/// use specials::elliprd;
///
/// assert_eq!(elliprd(1.0_f64, 1.0, 1.0), 1.0);
/// assert!((elliprd(2.0_f64, 3.0, 4.0) - 0.16510527294261).abs() < 1e-13);
/// ```
pub fn elliprd<T: Real>(x: T, y: T, z: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::carlson::elliprd(
            x.approx_f64(),
            y.approx_f64(),
            z.approx_f64(),
        )),
        Strategy::Adaptive => generic::carlson::elliprd(x, y, z),
    }
}

/// R_F(x, y, z), Carlson's symmetric elliptic integral of the first kind
/// (DLMF 19.16.1), symmetric in its arguments.
///
/// Defined for `x, y, z >= 0` with at most one of them zero. `+∞` if two are
/// zero; 0 if an argument is infinite; NaN for a negative argument.
///
/// # Examples
///
/// ```
/// use specials::elliprf;
///
/// assert_eq!(elliprf(1.0_f64, 1.0, 1.0), 1.0);
/// // R_F(0, 1, 2) = K(1/2) / √2
/// assert!((elliprf(0.0_f64, 1.0, 2.0) - 1.3110287771460599).abs() < 1e-15);
/// ```
pub fn elliprf<T: Real>(x: T, y: T, z: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::carlson::elliprf(
            x.approx_f64(),
            y.approx_f64(),
            z.approx_f64(),
        )),
        Strategy::Adaptive => generic::carlson::elliprf(x, y, z),
    }
}

/// R_G(x, y, z), Carlson's symmetric elliptic integral of the second kind
/// (DLMF 19.16.3), symmetric in its arguments.
///
/// Defined for `x, y, z >= 0`; `+∞` if an argument is infinite, NaN for a
/// negative argument. The `f64` path scales its arguments as [`elliprj`]
/// does; arguments more than 2^600 apart are not scaled and get xsf's value,
/// as in SciPy: NaN from a largest argument of about 2^512 on, and possibly a
/// wrong finite value elsewhere (see `docs/upstream-issues.md`).
///
/// # Examples
///
/// ```
/// use specials::elliprg;
///
/// assert_eq!(elliprg(1.0_f64, 1.0, 1.0), 1.0);
/// // R_G(0, 0, 1) = 1/2
/// assert_eq!(elliprg(0.0_f64, 0.0, 1.0), 0.5);
/// ```
pub fn elliprg<T: Real>(x: T, y: T, z: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::carlson::elliprg(
            x.approx_f64(),
            y.approx_f64(),
            z.approx_f64(),
        )),
        Strategy::Adaptive => generic::carlson::elliprg(x, y, z),
    }
}

/// R_J(x, y, z, p), Carlson's symmetric elliptic integral of the third kind
/// (DLMF 19.16.2), symmetric in `x`, `y`, `z`.
///
/// Defined for `x, y, z >= 0` with at most one of them zero and `p > 0`; for
/// `p < 0` it is the Cauchy principal value (DLMF 19.20.14). `+∞` at the pole
/// (two of `x`, `y`, `z` zero, `p != 0`); 0 if an argument is infinite; NaN
/// for a negative `x`, `y` or `z` and for `p = 0`.
///
/// The `Df64` path returns NaN when the largest and the smallest nonzero finite
/// argument (in absolute value) differ by more than a factor of 2^600
/// (about 4e180), where its duplication leaves the range of the type; below
/// that it is accurate to 1e-30 (checked up to a ratio of 1e180).
///
/// The `f64` path scales its arguments by a power of four when one lies
/// beyond 2^±300, none is subnormal and they are at most 2^600 apart: they
/// are moved until the outlying one is just inside 2^±300, which leaves all
/// of them within 2^±301, so that the products xsf forms stay in range. It
/// evaluates xsf both unscaled and scaled (xsf's value depends on the scale,
/// through underflow and through absolute thresholds of its asymptotic
/// expansions) and keeps the unscaled value, which is what the port returned
/// before, unless it is NaN, infinite or 0, or the precision-adaptive
/// algorithm evaluated in `f64` shows it to be off by more than 1e-10 and the
/// scaled value is closer. The result is xsf's at one of the two scales, with
/// xsf's accuracy there, which for some argument shapes is about 1e-10 rather
/// than the 1e-14 of the reference grid (also at ordinary scales). Arguments farther apart, or with a
/// subnormal one (which xsf treats as zero), are not scaled: no power of four
/// brings them within range, and they get xsf's value, as in SciPy: NaN where
/// its products leave the range (from a largest argument of about 2^340, or
/// below about 2^-340, on; for example
/// `elliprj(2^-1000, 2^400, 2^400, 2^400)`), and possibly a wrong finite value
/// elsewhere.
///
/// For `p < 0`, both paths evaluate the principal value by a transformation
/// (DLMF 19.20.14 for `Df64`, Carlson 1995 Eq. 2.26 in xsf) that cancels
/// catastrophically when `|p|` is far below `y` and the arguments spread
/// widely. Where a path scales the arguments, such a value is NaN when the
/// cancellation of the transformation it evaluates can cost more than a
/// relative 1e-10; elsewhere it is returned as computed and can be wrong, in
/// both paths and in SciPy (see `docs/upstream-issues.md`).
///
/// # Examples
///
/// ```
/// use specials::elliprj;
///
/// assert_eq!(elliprj(1.0_f64, 1.0, 1.0, 1.0), 1.0);
/// assert!((elliprj(2.0_f64, 3.0, 4.0, 5.0) - 0.14297579667157).abs() < 1e-13);
/// ```
pub fn elliprj<T: Real>(x: T, y: T, z: T, p: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::carlson::elliprj(
            x.approx_f64(),
            y.approx_f64(),
            z.approx_f64(),
            p.approx_f64(),
        )),
        Strategy::Adaptive => generic::carlson::elliprj(x, y, z, p),
    }
}

/// agm(a, b), the arithmetic-geometric mean (DLMF 19.8.1).
///
/// `agm(a, b) = agm(b, a)`, `agm(-a, -b) = -agm(a, b)`, `agm(a, a) = a`.
/// NaN if `a` and `b` have opposite signs, if either is NaN, or if one is
/// infinite and the other zero. `agm(a, 0) = 0` for finite `a`;
/// `agm(a, ±∞) = ±∞` for `a` of the same sign.
///
/// # Examples
///
/// ```
/// use specials::agm;
///
/// // Gauss's constant: 1 / agm(1, √2) = 0.8346268…
/// let g = 1.0 / agm(1.0_f64, 2.0_f64.sqrt());
/// assert!((g - 0.8346268416740731).abs() < 1e-15);
/// assert_eq!(agm(3.0_f64, 3.0), 3.0);
/// ```
pub fn agm<T: Real>(a: T, b: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::agm::agm(a.approx_f64(), b.approx_f64())),
        Strategy::Adaptive => generic::agm::agm(a, b),
    }
}

/// cos(x) − 1, accurate where `x` is close to a multiple of 2π (`scipy.special.cosm1`).
///
/// `cosm1(±0) = +0`; NaN for infinite `x`. Not in `std`: `x.cos() - 1.0`
/// loses all digits for small `x`.
///
/// # Examples
///
/// ```
/// use specials::cosm1;
///
/// // cos(1e-10) - 1 = -5e-21, which `(1e-10_f64).cos() - 1.0` rounds to 0.
/// assert!((cosm1(1e-10_f64) + 5e-21).abs() < 1e-36);
/// ```
pub fn cosm1<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::unity::cosm1(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::cosm1(x),
    }
}

/// ln(1 + x) − x (`scipy.special.log1pmx`), accurate for small `x` where it
/// is about `-x²/2`.
///
/// `-∞` at `x = -1`; NaN for `x < -1`, for NaN and for `x = +∞` (`∞ - ∞`, as
/// SciPy).
///
/// # Examples
///
/// ```
/// use specials::log1pmx;
///
/// let x = 1e-5_f64;
/// // -x²/2 + x³/3 - x⁴/4 + ..., the next term is x⁵/5 = 2e-26.
/// let series = -x * x / 2.0 + x * x * x / 3.0 - x * x * x * x / 4.0;
/// assert!((log1pmx(x) - series).abs() < 1e-25);
/// assert_eq!(log1pmx(-1.0_f64), f64::NEG_INFINITY);
/// ```
pub fn log1pmx<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::unity::log1pmx(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::log1pmx(x),
    }
}

/// x ln y, with `xlogy(0, y) = 0` for every `y` that is not NaN
/// (`scipy.special.xlogy`), including `y = 0` and `y < 0`.
///
/// For `x != 0`: `∓∞` at `y = 0` (sign opposite to `x`), NaN for `y < 0`.
///
/// # Examples
///
/// ```
/// use specials::xlogy;
///
/// assert_eq!(xlogy(0.0_f64, 0.0), 0.0); // the convention 0 ln 0 = 0
/// assert!((xlogy(2.0_f64, core::f64::consts::E) - 2.0).abs() < 1e-15);
/// assert!(xlogy(0.0_f64, f64::NAN).is_nan());
/// ```
pub fn xlogy<T: Real>(x: T, y: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::log::xlogy(x.approx_f64(), y.approx_f64())),
        Strategy::Adaptive => generic::elementary::xlogy(x, y),
    }
}

/// x ln(1 + y), with `xlog1py(0, y) = 0` for every `y` that is not NaN
/// (`scipy.special.xlog1py`).
///
/// For `x != 0`: `∓∞` at `y = -1`, NaN for `y < -1`.
///
/// # Examples
///
/// ```
/// use specials::xlog1py;
///
/// assert_eq!(xlog1py(0.0_f64, -1.0), 0.0);
/// // ln(1 + y) ≈ y for tiny y, with full relative accuracy.
/// assert!((xlog1py(3.0_f64, 1e-20) - 3e-20).abs() < 1e-35);
/// ```
pub fn xlog1py<T: Real>(x: T, y: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::log::xlog1py(x.approx_f64(), y.approx_f64())),
        Strategy::Adaptive => generic::elementary::xlog1py(x, y),
    }
}

/// ln(x/(1 − x)), the logit function, inverse of [`expit`] (`scipy.special.logit`).
///
/// `logit(0) = -∞` (also for `-0`), `logit(1) = +∞`; NaN outside `[0, 1]`
/// and for NaN.
///
/// # Examples
///
/// ```
/// use specials::logit;
///
/// assert_eq!(logit(0.5_f64), 0.0);
/// assert_eq!(logit(0.0_f64), f64::NEG_INFINITY);
/// assert_eq!(logit(1.0_f64), f64::INFINITY);
/// assert!(logit(1.5_f64).is_nan());
/// ```
pub fn logit<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::log_exp::logit(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::logit(x),
    }
}

/// 1/(1 + e^−x), the logistic sigmoid (`scipy.special.expit`).
///
/// `expit(-∞) = 0`, `expit(+∞) = 1`. On the `f64` path the result is `0` for
/// `x < -709.78` (where `e^-x` overflows, as in SciPy), although the exact
/// value is subnormal down to `x ≈ -745`; the `Df64` path returns that
/// subnormal value.
///
/// # Examples
///
/// ```
/// use specials::expit;
///
/// assert_eq!(expit(0.0_f64), 0.5);
/// assert_eq!(expit(f64::NEG_INFINITY), 0.0);
/// assert!((expit(-50.0_f64) - 1.9287498479639178e-22).abs() < 1e-37);
/// ```
pub fn expit<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::log_exp::expit(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::expit(x),
    }
}

/// ln(expit(x)) = −ln(1 + e^−x) (`scipy.special.log_expit`), accurate for
/// large `|x|` where `expit(x).ln()` is not.
///
/// `log_expit(-∞) = -∞`, `log_expit(+∞) = -0`.
///
/// # Examples
///
/// ```
/// use specials::log_expit;
///
/// // ln expit(-1000) = -1000 - ln(1 + e^-1000) = -1000 to double precision.
/// assert_eq!(log_expit(-1000.0_f64), -1000.0);
/// assert!((log_expit(40.0_f64) + 4.248354255291589e-18).abs() < 1e-33);
/// ```
pub fn log_expit<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::log_exp::log_expit(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::log_expit(x),
    }
}

/// sin(πx), with the argument reduced exactly (`scipy.special.sinpi`).
///
/// Exact zeros at the integers, signed as in SciPy: `sinpi(±0) = ±0`;
/// for an integer `n > 0`, `+0` if `n` is even and `-0` if odd, and the
/// opposite for `n < 0`. NaN for infinite `x`. Every `f64` with `|x| >= 2^52`
/// is an integer or half-integer, so the result there is exactly `0` or `±1`.
///
/// # Examples
///
/// ```
/// use specials::sinpi;
///
/// assert_eq!(sinpi(0.5_f64), 1.0);
/// assert_eq!(sinpi(1.0_f64), 0.0); // while (π).sin() = 1.2e-16
/// assert!(sinpi(1.0_f64).is_sign_negative());
/// assert_eq!(sinpi(1e300_f64), 0.0);
/// ```
pub fn sinpi<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::sinpi(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::sinpi(x),
    }
}

/// cos(πx), with the argument reduced exactly (`scipy.special.cospi`).
///
/// Exact `+0` at every half-integer (never `-0`); NaN for infinite `x`.
///
/// # Examples
///
/// ```
/// use specials::cospi;
///
/// assert_eq!(cospi(1.0_f64), -1.0);
/// assert_eq!(cospi(-1.5_f64), 0.0);
/// assert!(cospi(-1.5_f64).is_sign_positive());
/// ```
pub fn cospi<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::cospi(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::cospi(x),
    }
}

/// sin(x°), the sine of an angle in degrees (`scipy.special.sindg`).
///
/// The argument is reduced exactly, so multiples of 90° give exact `0` and
/// `±1`, unlike `x.to_radians().sin()`. `sindg(±0) = ±0`. For
/// `|x| > 1e14` the result is `0` (no fractional degrees are left; SciPy
/// returns 0 there too, also for `±∞`). NaN for NaN.
///
/// # Examples
///
/// ```
/// use specials::sindg;
///
/// assert_eq!(sindg(180.0_f64), 0.0); // 180f64.to_radians().sin() = 1.2e-16
/// assert_eq!(sindg(-90.0_f64), -1.0);
/// assert!((sindg(30.0_f64) - 0.5).abs() < 1e-16);
/// ```
pub fn sindg<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::sindg::sindg(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::sindg(x),
    }
}

/// cos(x°), the cosine of an angle in degrees (`scipy.special.cosdg`).
///
/// Reduced exactly as [`sindg`]; `0` for `|x| > 1e14` and `±∞` (as SciPy);
/// NaN for NaN.
///
/// # Examples
///
/// ```
/// use specials::cosdg;
///
/// assert_eq!(cosdg(90.0_f64), 0.0); // 90f64.to_radians().cos() = 6.1e-17
/// assert_eq!(cosdg(180.0_f64), -1.0);
/// ```
pub fn cosdg<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::sindg::cosdg(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::cosdg(x),
    }
}

/// tan(x°), the tangent of an angle in degrees (`scipy.special.tandg`).
///
/// Exact `±0` at multiples of 180°, `±1` at odd multiples of 45° and `±∞` at
/// odd multiples of 90°, signed as in SciPy (`tandg(90) = +∞`,
/// `tandg(270) = -∞`, `tandg(540) = -0`); `0` for `|x| > 1e14` and `±∞`
/// (as SciPy); NaN for NaN.
///
/// # Examples
///
/// ```
/// use specials::tandg;
///
/// assert_eq!(tandg(45.0_f64), 1.0);
/// assert_eq!(tandg(90.0_f64), f64::INFINITY);
/// assert_eq!(tandg(-180.0_f64), 0.0);
/// ```
pub fn tandg<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::tandg::tandg(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::tancot(x, false),
    }
}

/// cot(x°), the cotangent of an angle in degrees (`scipy.special.cotdg`).
///
/// `±∞` at multiples of 180° (`cotdg(0) = +∞`, `cotdg(-0) = -∞`), exact `±0`
/// at odd multiples of 90°, signed as in SciPy; `0` for `|x| > 1e14` and `±∞`
/// (as SciPy); NaN for NaN.
///
/// # Examples
///
/// ```
/// use specials::cotdg;
///
/// assert_eq!(cotdg(45.0_f64), 1.0);
/// assert_eq!(cotdg(90.0_f64), 0.0);
/// assert_eq!(cotdg(0.0_f64), f64::INFINITY);
/// ```
pub fn cotdg<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::tandg::cotdg(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::tancot(x, true),
    }
}

/// Degrees, minutes and seconds to radians: `((60 d + m) 60 + s) π/648000`
/// (`scipy.special.radian`).
///
/// # Examples
///
/// ```
/// use specials::radian;
///
/// // 30° 30′ 36″ = 30.51°
/// assert!((radian(30.0_f64, 30.0, 36.0) - 30.51_f64.to_radians()).abs() < 1e-15);
/// ```
pub fn radian<T: Real>(d: T, m: T, s: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(xsf::sindg::radian(
            d.approx_f64(),
            m.approx_f64(),
            s.approx_f64(),
        )),
        Strategy::Adaptive => generic::elementary::radian(d, m, s),
    }
}

/// 2^x (`scipy.special.exp2`).
///
/// For `f64` this is `f64::exp2`, which passes SciPy's reference table; this
/// function exists for the double-double path, where `xprec`'s `exp2` gives
/// NaN at `±∞` and loses digits for large `|x|`. Unlike SciPy (Cephes), which
/// returns `0` below `x = -1024`, the `f64` result is subnormal down to
/// `x = -1074`.
///
/// # Examples
///
/// ```
/// use specials::exp2;
///
/// assert_eq!(exp2(10.0_f64), 1024.0);
/// assert_eq!(exp2(f64::NEG_INFINITY), 0.0);
/// ```
pub fn exp2<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(x.approx_f64().exp2()),
        Strategy::Adaptive => generic::elementary::exp2(x),
    }
}

/// 10^x (`scipy.special.exp10`).
///
/// For `f64` this is `10f64.powf(x)`, which passes SciPy's reference table;
/// it exists for the double-double path, where `10^x` as `exp(x ln 10)` loses
/// `|x ln 10|` units of the epsilon. Like [`exp2`], the `f64` result is
/// subnormal below `x = -308.25`, where SciPy returns `0`.
///
/// # Examples
///
/// ```
/// use specials::exp10;
///
/// assert_eq!(exp10(3.0_f64), 1000.0);
/// assert_eq!(exp10(-1.0_f64), 0.1);
/// ```
pub fn exp10<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(10.0_f64.powf(x.approx_f64())),
        Strategy::Adaptive => generic::elementary::exp10(x),
    }
}

/// ∛x, the real cube root (`scipy.special.cbrt`); odd, `±0`, `±∞` and NaN
/// map to themselves.
///
/// For `f64` this is `f64::cbrt`; it exists for the double-double path,
/// where `xprec` 0.2.3's `Float::cbrt` is not implemented and panics.
///
/// # Examples
///
/// ```
/// use specials::cbrt;
///
/// assert_eq!(cbrt(-27.0_f64), -3.0);
/// assert!(cbrt(-0.0_f64).is_sign_negative());
/// ```
pub fn cbrt<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(x.approx_f64().cbrt()),
        Strategy::Adaptive => generic::elementary::cbrt(x),
    }
}

/// e^x − 1 (`scipy.special.expm1`).
///
/// For `f64` this is `f64::exp_m1`; it exists for the double-double path,
/// where `xprec` 0.2.3's `exp_m1(-0)` is `+0`. `expm1(-∞) = -1`.
///
/// # Examples
///
/// ```
/// use specials::expm1;
///
/// assert!((expm1(1e-10_f64) - 1.00000000005e-10).abs() < 1e-26);
/// assert!(expm1(-0.0_f64).is_sign_negative());
/// ```
pub fn expm1<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(x.approx_f64().exp_m1()),
        Strategy::Adaptive => generic::elementary::expm1(x),
    }
}

/// ln(1 + x) (`scipy.special.log1p`).
///
/// `-∞` at `x = -1`, NaN below. For `f64` this is `f64::ln_1p`; it exists
/// for the double-double path, where `xprec` 0.2.3's `ln_1p` returns `+0`
/// for `-0` and `-∞` for arguments within `2^-53` above `-1`.
///
/// # Examples
///
/// ```
/// use specials::log1p;
///
/// assert!((log1p(1e-20_f64) - 1e-20).abs() < 1e-36);
/// assert_eq!(log1p(-1.0_f64), f64::NEG_INFINITY);
/// assert!(log1p(-0.0_f64).is_sign_negative());
/// ```
pub fn log1p<T: Real>(x: T) -> T {
    match T::STRATEGY {
        Strategy::F64Kernels => k64(x.approx_f64().ln_1p()),
        Strategy::Adaptive => generic::elementary::log1p(x),
    }
}

// ---------------------------------------------------------------------------
// Orthogonal polynomials and Legendre functions
// ---------------------------------------------------------------------------

/// Dispatch of an integer-degree kernel. `f64`: xsf's kernel. `Df64`: the
/// double-double recurrence, or xsf's value for a non-finite argument (its
/// classification of NaN and infinities is SciPy's) and where the
/// double-double result is not finite.
fn dispatch_int<T: Real, const N: usize>(
    args: [T; N],
    kernel: impl Fn([f64; N]) -> f64,
    adaptive: impl FnOnce([T; N]) -> T,
) -> T {
    let a64 = args.map(Real::approx_f64);
    match T::STRATEGY {
        Strategy::F64Kernels => k64(kernel(a64)),
        Strategy::Adaptive => {
            if a64.iter().any(|a| !a.is_finite()) {
                return k64(kernel(a64));
            }
            generic::orthogonal::finite_or(adaptive(args), || kernel(a64))
        }
    }
}

/// Dispatch of a real-degree kernel; `args[0]` is the degree. `f64`: xsf's
/// hypergeometric kernel. `Df64`: the integer-degree recurrence for a
/// non-negative integer-valued degree, otherwise the `f64` kernel.
fn dispatch_real<T: Real, const N: usize>(
    args: [T; N],
    kernel: impl Fn([f64; N]) -> f64,
    adaptive: impl FnOnce([T; N], i64) -> T,
) -> T {
    let a64 = args.map(Real::approx_f64);
    match T::STRATEGY {
        Strategy::F64Kernels => k64(kernel(a64)),
        Strategy::Adaptive => {
            let n = args[0];
            let integral = n >= T::zero() && n == n.floor() && a64[0] < 9.007_199_254_740_992e15;
            if !integral || a64.iter().any(|a| !a.is_finite()) {
                return k64(kernel(a64));
            }
            generic::orthogonal::finite_or(adaptive(args, a64[0] as i64), || kernel(a64))
        }
    }
}

/// The Jacobi polynomial `P_n^(α,β)(x)` of integer degree `n` (DLMF 18.3), from the recurrence of DLMF 18.9.2. A negative degree takes the hypergeometric form of [`eval_jacobi`] (the `f64` kernel, also for `Df64`).
///
/// `α = -1` with `abs(β) = 1` is handled as xsf does (`P_n^(-1,β) = (n + β)/(2n) (x - 1) P_(n-1)^(1,β)`); the `Df64` path also has this for every `β`, and for `β = -1`.
///
/// The recurrence is singular where `α + β` is a negative integer below `-1`: the result is NaN or infinite there (`f64`) and NaN (`Df64`).
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_jacobi_int;
/// // P_2(x) = (3 x^2 - 1) / 2 for α = β = 0
/// assert!((eval_jacobi_int(2, 0.0_f64, 0.0, 0.5) + 0.125).abs() < 1e-15);
/// ```
pub fn eval_jacobi_int<T: Real>(n: i64, alpha: T, beta: T, x: T) -> T {
    dispatch_int(
        [alpha, beta, x],
        |a| xsf::orthogonal_eval::eval_jacobi_l(n, a[0], a[1], a[2]),
        // A negative degree takes xsf's hypergeometric form.
        |a| {
            if n < 0 {
                T::nan()
            } else {
                generic::orthogonal::jacobi(n, a[0], a[1], a[2])
            }
        },
    )
}

/// The shifted Jacobi polynomial `G_n(p, q, x) = P_n^(p-q, q-1)(2x - 1) / C(2n + p - 1, n)` (SciPy's `eval_sh_jacobi`) of integer degree `n`.
///
/// The normalization `C(2n + p - 1, n)` can be zero (the result is then infinite or NaN, as in xsf).
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_sh_jacobi_int;
/// // G_2(1, 1, x) = P_2(2x - 1) / C(4, 2); at x = 1/2 that is P_2(0) / 6
/// assert!((eval_sh_jacobi_int(2, 1.0_f64, 1.0, 0.5) + 1.0 / 12.0).abs() < 1e-15);
/// ```
pub fn eval_sh_jacobi_int<T: Real>(n: i64, p: T, q: T, x: T) -> T {
    dispatch_int(
        [p, q, x],
        |a| xsf::orthogonal_eval::eval_sh_jacobi_l(n, a[0], a[1], a[2]),
        |a| {
            if n < 0 {
                T::nan()
            } else {
                generic::orthogonal::sh_jacobi(n, a[0], a[1], a[2])
            }
        },
    )
}

/// The Gegenbauer (ultraspherical) polynomial `C_n^(α)(x)` of integer degree `n` (DLMF 18.3). As in xsf the result is `0` for `α = 0` (for every `n`) and for `n < 0`; for `abs(x) < 1e-5` the `f64` path sums the explicit power series.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_gegenbauer_int;
/// // C_2^(1)(x) = U_2(x) = 4 x^2 - 1
/// assert!(eval_gegenbauer_int(2, 1.0_f64, 0.5).abs() < 1e-15);
/// ```
pub fn eval_gegenbauer_int<T: Real>(n: i64, alpha: T, x: T) -> T {
    dispatch_int(
        [alpha, x],
        |a| xsf::orthogonal_eval::eval_gegenbauer_l(n, a[0], a[1]),
        |a| generic::orthogonal::gegenbauer(n, a[0], a[1]),
    )
}

/// The Chebyshev polynomial of the first kind `T_n(x)` of integer degree `n` (DLMF 18.3); `T_(-n) = T_n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_chebyt_int;
/// assert_eq!(eval_chebyt_int(3, 0.5_f64), -1.0); // 4 x^3 - 3 x
/// ```
pub fn eval_chebyt_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_chebyt_l(n, a[0]),
        |a| generic::orthogonal::chebyt(n, a[0]),
    )
}

/// The Chebyshev polynomial of the second kind `U_n(x)` of integer degree `n`; `U_(-1) = 0` and `U_(-n-2) = -U_n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_chebyu_int;
/// assert_eq!(eval_chebyu_int(2, 0.5_f64), 0.0); // 4 x^2 - 1
/// ```
pub fn eval_chebyu_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_chebyu_l(n, a[0]),
        |a| generic::orthogonal::chebyu(n, a[0]),
    )
}

/// The Chebyshev polynomial `C_n(x) = 2 T_n(x/2)` of integer degree `n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_chebyc_int;
/// assert!((eval_chebyc_int(2, 1.0_f64) + 1.0).abs() < 1e-15); // x^2 - 2
/// ```
pub fn eval_chebyc_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_chebyc_l(n, a[0]),
        |a| generic::orthogonal::chebyc(n, a[0]),
    )
}

/// The Chebyshev polynomial `S_n(x) = U_n(x/2)` of integer degree `n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_chebys_int;
/// assert!(eval_chebys_int(2, 1.0_f64).abs() < 1e-15); // x^2 - 1
/// ```
pub fn eval_chebys_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_chebys_l(n, a[0]),
        |a| generic::orthogonal::chebys(n, a[0]),
    )
}

/// The shifted Chebyshev polynomial `T*_n(x) = T_n(2x - 1)` of integer degree `n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_sh_chebyt_int;
/// assert!((eval_sh_chebyt_int(2, 0.75_f64) + 0.5).abs() < 1e-15); // T_2(1/2)
/// ```
pub fn eval_sh_chebyt_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_sh_chebyt_l(n, a[0]),
        |a| generic::orthogonal::sh_chebyt(n, a[0]),
    )
}

/// The shifted Chebyshev polynomial `U*_n(x) = U_n(2x - 1)` of integer degree `n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_sh_chebyu_int;
/// assert!(eval_sh_chebyu_int(2, 0.75_f64).abs() < 1e-15); // U_2(1/2)
/// ```
pub fn eval_sh_chebyu_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_sh_chebyu_l(n, a[0]),
        |a| generic::orthogonal::sh_chebyu(n, a[0]),
    )
}

/// The Legendre polynomial `P_n(x)` of integer degree `n` (DLMF 18.3); `P_(-n-1) = P_n`. The `f64` path sums an explicit power series for `abs(x) < 1e-5`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_legendre_int;
/// assert!((eval_legendre_int(3, 0.5_f64) + 0.4375).abs() < 1e-15); // (5 x^3 - 3 x) / 2
/// ```
pub fn eval_legendre_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_legendre_l(n, a[0]),
        |a| generic::orthogonal::legendre(n, a[0]),
    )
}

/// The shifted Legendre polynomial `P*_n(x) = P_n(2x - 1)` of integer degree `n`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_sh_legendre_int;
/// assert!((eval_sh_legendre_int(2, 0.75_f64) + 0.125).abs() < 1e-15); // P_2(1/2)
/// ```
pub fn eval_sh_legendre_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_sh_legendre_l(n, a[0]),
        |a| generic::orthogonal::sh_legendre(n, a[0]),
    )
}

/// The Laguerre polynomial `L_n(x)` of integer degree `n` (DLMF 18.3); `0` for `n < 0`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_laguerre_int;
/// assert!((eval_laguerre_int(2, 1.0_f64) + 0.5).abs() < 1e-15); // (x^2 - 4 x + 2) / 2
/// ```
pub fn eval_laguerre_int<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_genlaguerre_l(n, 0.0, a[0]),
        |a| generic::orthogonal::genlaguerre(n, T::zero(), a[0]),
    )
}

/// The generalized Laguerre polynomial `L_n^(α)(x)` of integer degree `n` (DLMF 18.3). NaN for `α <= -1`, `0` for `n < 0`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_genlaguerre_int;
/// assert!((eval_genlaguerre_int(2, 1.0_f64, 1.0) - 0.5).abs() < 1e-15);
/// ```
pub fn eval_genlaguerre_int<T: Real>(n: i64, alpha: T, x: T) -> T {
    dispatch_int(
        [alpha, x],
        |a| xsf::orthogonal_eval::eval_genlaguerre_l(n, a[0], a[1]),
        |a| generic::orthogonal::genlaguerre(n, a[0], a[1]),
    )
}

/// The physicists' Hermite polynomial `H_n(x)` (DLMF 18.3), NaN for `n < 0`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_hermite;
/// assert!((eval_hermite(3, 0.5_f64) + 5.0).abs() < 1e-14); // 8 x^3 - 12 x
/// ```
pub fn eval_hermite<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_hermite(n, a[0]),
        |a| generic::orthogonal::hermite(n, a[0]),
    )
}

/// The probabilists' Hermite polynomial `He_n(x)`, NaN for `n < 0`.
///
/// `Df64` evaluates the forward recurrence in double-double; the running terms carry a binary exponent, so intermediate terms cannot overflow or underflow while the result is representable. The cost is linear in `n`. A non-finite argument or a result that overflows takes xsf's value (SciPy's NaN and infinities).
///
/// # Examples
///
/// ```
/// use specials::eval_hermitenorm;
/// assert!((eval_hermitenorm(3, 0.5_f64) + 1.375).abs() < 1e-14); // x^3 - 3 x
/// ```
pub fn eval_hermitenorm<T: Real>(n: i64, x: T) -> T {
    dispatch_int(
        [x],
        |a| xsf::orthogonal_eval::eval_hermitenorm(n, a[0]),
        |a| generic::orthogonal::hermitenorm(n, a[0]),
    )
}

/// The Jacobi polynomial `P_n^(α,β)(x)` of real degree `n`, as the hypergeometric function `C(n + α, n) 2F1(-n, n + α + β + 1; α + 1; (1 - x)/2)`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_jacobi_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_jacobi;
/// assert!((eval_jacobi(2.0_f64, 0.0, 0.0, 0.5) + 0.125).abs() < 1e-14);
/// ```
pub fn eval_jacobi<T: Real>(n: T, alpha: T, beta: T, x: T) -> T {
    dispatch_real(
        [n, alpha, beta, x],
        |a| xsf::orthogonal_eval::eval_jacobi(a[0], a[1], a[2], a[3]),
        |a, n| generic::orthogonal::jacobi(n, a[1], a[2], a[3]),
    )
}

/// The shifted Jacobi polynomial `G_n(p, q, x)` of real degree `n` (see [`eval_sh_jacobi_int`]).
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_sh_jacobi_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_sh_jacobi;
/// assert!((eval_sh_jacobi(2.0_f64, 1.0, 1.0, 0.5) + 1.0 / 12.0).abs() < 1e-14);
/// ```
pub fn eval_sh_jacobi<T: Real>(n: T, p: T, q: T, x: T) -> T {
    dispatch_real(
        [n, p, q, x],
        |a| xsf::orthogonal_eval::eval_sh_jacobi(a[0], a[1], a[2], a[3]),
        |a, n| generic::orthogonal::sh_jacobi(n, a[1], a[2], a[3]),
    )
}

/// The Gegenbauer polynomial `C_n^(α)(x)` of real degree `n`; xsf takes the integer-degree recurrence for an integer-valued `n` below `1e10`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_gegenbauer_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_gegenbauer;
/// assert!(eval_gegenbauer(2.0_f64, 1.0, 0.5).abs() < 1e-14);
/// ```
pub fn eval_gegenbauer<T: Real>(n: T, alpha: T, x: T) -> T {
    dispatch_real(
        [n, alpha, x],
        |a| xsf::orthogonal_eval::eval_gegenbauer(a[0], a[1], a[2]),
        |a, n| generic::orthogonal::gegenbauer(n, a[1], a[2]),
    )
}

/// The Chebyshev function `T_n(x) = 2F1(-n, n; 1/2; (1 - x)/2)` of real degree `n`; `T_n(cos t) = cos(n t)`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_chebyt_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_chebyt;
/// // T_(1/2)(cos t) = cos(t/2); t = pi/2
/// assert!((eval_chebyt(0.5_f64, 0.0) - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-14);
/// ```
pub fn eval_chebyt<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_chebyt(a[0], a[1]),
        |a, n| generic::orthogonal::chebyt(n, a[1]),
    )
}

/// The Chebyshev function `U_n(x) = (n + 1) 2F1(-n, n + 2; 3/2; (1 - x)/2)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_chebyu_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_chebyu;
/// assert!(eval_chebyu(2.0_f64, 0.5).abs() < 1e-14);
/// ```
pub fn eval_chebyu<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_chebyu(a[0], a[1]),
        |a, n| generic::orthogonal::chebyu(n, a[1]),
    )
}

/// `C_n(x) = 2 T_n(x/2)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_chebyc_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_chebyc;
/// assert!((eval_chebyc(2.0_f64, 1.0) + 1.0).abs() < 1e-14);
/// ```
pub fn eval_chebyc<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_chebyc(a[0], a[1]),
        |a, n| generic::orthogonal::chebyc(n, a[1]),
    )
}

/// `S_n(x) = U_n(x/2)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_chebys_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_chebys;
/// assert!(eval_chebys(2.0_f64, 1.0).abs() < 1e-14);
/// ```
pub fn eval_chebys<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_chebys(a[0], a[1]),
        |a, n| generic::orthogonal::chebys(n, a[1]),
    )
}

/// `T*_n(x) = T_n(2x - 1)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_sh_chebyt_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_sh_chebyt;
/// assert!((eval_sh_chebyt(2.0_f64, 0.75) + 0.5).abs() < 1e-14);
/// ```
pub fn eval_sh_chebyt<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_sh_chebyt(a[0], a[1]),
        |a, n| generic::orthogonal::sh_chebyt(n, a[1]),
    )
}

/// `U*_n(x) = U_n(2x - 1)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_sh_chebyu_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_sh_chebyu;
/// assert!(eval_sh_chebyu(2.0_f64, 0.75).abs() < 1e-14);
/// ```
pub fn eval_sh_chebyu<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_sh_chebyu(a[0], a[1]),
        |a, n| generic::orthogonal::sh_chebyu(n, a[1]),
    )
}

/// The Legendre function `P_n(x) = 2F1(-n, n + 1; 1; (1 - x)/2)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_legendre_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_legendre;
/// assert!((eval_legendre(3.0_f64, 0.5) + 0.4375).abs() < 1e-14);
/// ```
pub fn eval_legendre<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_legendre(a[0], a[1]),
        |a, n| generic::orthogonal::legendre(n, a[1]),
    )
}

/// `P*_n(x) = P_n(2x - 1)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_sh_legendre_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_sh_legendre;
/// assert!((eval_sh_legendre(2.0_f64, 0.75) + 0.125).abs() < 1e-14);
/// ```
pub fn eval_sh_legendre<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_sh_legendre(a[0], a[1]),
        |a, n| generic::orthogonal::sh_legendre(n, a[1]),
    )
}

/// The Laguerre function `L_n(x) = 1F1(-n; 1; x)` of real degree `n`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_laguerre_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_laguerre;
/// assert!((eval_laguerre(2.0_f64, 1.0) + 0.5).abs() < 1e-14);
/// ```
pub fn eval_laguerre<T: Real>(n: T, x: T) -> T {
    dispatch_real(
        [n, x],
        |a| xsf::orthogonal_eval::eval_genlaguerre(a[0], 0.0, a[1]),
        |a, n| generic::orthogonal::genlaguerre(n, T::zero(), a[1]),
    )
}

/// The generalized Laguerre function `L_n^(α)(x) = C(n + α, n) 1F1(-n; α + 1; x)` of real degree `n`, NaN for `α <= -1`.
///
/// `f64` is xsf's hypergeometric series (`hyp2f1` of Cephes, `hyp1f1` of Zhang and Jin), which loses digits for large degrees (its own header gives a peak relative error of 1.2e-11) and is limited by cancellation near the zeros.
///
/// `Df64` evaluates the integer-degree recurrence (see [`eval_genlaguerre_int`]) for a non-negative integer-valued `n`, and the `f64` kernel for any other degree, whose result has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::eval_genlaguerre;
/// assert!((eval_genlaguerre(2.0_f64, 1.0, 1.0) - 0.5).abs() < 1e-14);
/// ```
pub fn eval_genlaguerre<T: Real>(n: T, alpha: T, x: T) -> T {
    dispatch_real(
        [n, alpha, x],
        |a| xsf::orthogonal_eval::eval_genlaguerre(a[0], a[1], a[2]),
        |a, n| generic::orthogonal::genlaguerre(n, a[1], a[2]),
    )
}

/// `P_v^m(x)` of integer order `m` and real degree `v` (SciPy's `pmv`, `lpmv`).
///
/// This is the Ferrers function (DLMF 14.3.1, with the phase `(-1)^m` of 14.6.1)
/// for `-1 <= x <= 1`. NaN for a non-integer `m`, and for a negative order
/// beyond an integer degree (`-m > v`, where the relation of DLMF 14.9.3 does not
/// apply). For a non-integer degree at `x = -1` the result is `-∞` (`m = 0`,
/// `sin(πv) > 0`) or `+∞`. For `abs(x) > 1` xsf's series is not meaningful: the
/// result is NaN for `m != 0`, and `P_v(x)` for `m = 0` and an integer degree.
///
/// `f64` is xsf's: for an integer degree the explicit sum of DLMF 14.3.4 and 15.2.4
/// with the recurrence of DLMF 14.10.3 above degree 2, for a non-integer degree
/// the hypergeometric series (stopped at `1e-14`) or the digamma expansion of DLMF
/// 14.3.5 for `x < -0.35`. Next to a zero of the function (`x` near 0 for an odd
/// degree), the sum `1 + r` loses all digits: `pmv(0, 1, 1e-15)` has a relative
/// error of 10%, and `1e-30` gives `-0` (`docs/upstream-issues.md`).
///
/// `Df64` evaluates an integer degree and order by the recurrence of
/// [`assoc_legendre_p`] (`P_n(x)` for `m = 0`), which has no such loss, with
/// the sign convention and the NaN of xsf; any other degree takes the `f64` kernel
/// and has `f64` accuracy.
///
/// # Examples
///
/// ```
/// use specials::pmv;
///
/// // P_2^1(x) = -3 x sqrt(1 - x^2)
/// let v = pmv(1.0_f64, 2.0, 0.5);
/// assert!((v + 1.5 * 3.0_f64.sqrt() / 2.0 * 1.0).abs() < 1e-14);
/// ```
pub fn pmv<T: Real>(m: T, v: T, x: T) -> T {
    let kernel = || xsf::specfun::pmv(m.approx_f64(), v.approx_f64(), x.approx_f64());
    match T::STRATEGY {
        Strategy::F64Kernels => k64(kernel()),
        Strategy::Adaptive => match generic::legendre::pmv_int(m, v, x) {
            Some(r) => generic::orthogonal::finite_or(r, kernel),
            None => k64(kernel()),
        },
    }
}

/// The branch of the associated Legendre function of a real argument
/// (SciPy's `branch_cut`).
///
/// # Examples
///
/// ```
/// use specials::{assoc_legendre_p, LegendreType};
///
/// // inside (-1, 1) only the Ferrers function is defined, outside only the other
/// assert!(assoc_legendre_p(2, 1, 1.5_f64, LegendreType::Ferrers).is_nan());
/// assert!(assoc_legendre_p(2, 1, 0.5_f64, LegendreType::Outside).is_nan());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendreType {
    /// The Ferrers function on `-1 < z < 1` (cut on `abs(z) > 1`, SciPy's `branch_cut = 2`):
    /// `P_n^m(z) = (-1)^m (1 - z^2)^(m/2) d^m P_n / dz^m`. NaN for `abs(z) > 1`.
    Ferrers,
    /// The function on `abs(z) > 1` (cut on `-1 < z < 1`, SciPy's `branch_cut = 3`):
    /// `P_n^m(z) = (z^2 - 1)^(m/2) d^m P_n / dz^m`. NaN for `abs(z) < 1`, and for `m != 0` at `z < -1`.
    Outside,
}

impl LegendreType {
    fn cut(self) -> xsf::legendre::Cut {
        match self {
            LegendreType::Ferrers => xsf::legendre::Cut::Ferrers,
            LegendreType::Outside => xsf::legendre::Cut::Outside,
        }
    }
}

/// The Legendre polynomial `P_n(z)` of integer degree `n >= 0` (xsf `legendre_p`, the
/// value without derivatives). NaN for `n < 0` (xsf returns `z`).
///
/// # Examples
///
/// ```
/// use specials::legendre_p;
///
/// assert!((legendre_p(3, 0.5_f64) + 0.4375).abs() < 1e-15);
/// assert!(legendre_p(-1, 0.5_f64).is_nan());
/// ```
pub fn legendre_p<T: Real>(n: i32, z: T) -> T {
    dispatch_int(
        [z],
        |a| xsf::legendre::legendre_p(n, a[0]),
        |a| {
            if n < 0 {
                T::nan()
            } else {
                generic::orthogonal::legendre(i64::from(n), a[0])
            }
        },
    )
}

/// The associated Legendre function `P_n^m(z)` of integer degree `n >= 0` and order `m`
/// (xsf `assoc_legendre_p`, the value without derivatives), unnormalized.
///
/// `kind` selects the function for `abs(z) < 1` (Ferrers, with the Condon-Shortley
/// phase) or `abs(z) > 1`. `P_n^m = 0` for `abs(m) > n`, and `P_n^m(±1)` is `1` or
/// `(-1)^n` for `m = 0` and `0` otherwise. A negative order `m` is
/// `(-1)^m (n - m)! / (n + m)! P_n^(-m)` for the Ferrers function. NaN for `n < 0`
/// (xsf returns 0).
///
/// `f64` is xsf's recurrence; it forms `1 - z^2` as `1 - z z`, which loses digits
/// next to `abs(z) = 1` (relative error `m eps / (1 - z^2)`). `Df64` forms `(1 - z)(1 + z)`
/// and tracks a binary exponent, so a diagonal term `P_m^m` that overflows or
/// underflows in `f64` does not spoil a representable result. The work is linear in
/// `n + abs(m)`.
///
/// # Examples
///
/// ```
/// use specials::{assoc_legendre_p, LegendreType};
///
/// // P_2^1(x) = -3 x sqrt(1 - x^2)
/// let p = assoc_legendre_p(2, 1, 0.5_f64, LegendreType::Ferrers);
/// assert!((p + 1.5 * 3.0_f64.sqrt() / 2.0).abs() < 1e-15);
/// ```
pub fn assoc_legendre_p<T: Real>(n: i32, m: i32, z: T, kind: LegendreType) -> T {
    let outside = kind == LegendreType::Outside;
    dispatch_int(
        [z],
        |a| xsf::legendre::assoc_legendre_p(false, n, m, a[0], kind.cut()),
        |a| generic::legendre::assoc_legendre_p(false, n, m, a[0], outside),
    )
}

/// The normalized associated Legendre function: [`assoc_legendre_p`] times
/// `sqrt((2n + 1) (n - m)! / (2 (n + m)!))`, so that the functions of one order are orthonormal on `[-1, 1]`.
/// The normalization is built into the recurrence (no factorials are formed).
///
/// # Examples
///
/// ```
/// use specials::{assoc_legendre_p_norm, LegendreType};
///
/// // sqrt(3/2) x for n = 1, m = 0
/// let p = assoc_legendre_p_norm(1, 0, 0.5_f64, LegendreType::Ferrers);
/// assert!((p - 0.5 * 1.5_f64.sqrt()).abs() < 1e-15);
/// ```
pub fn assoc_legendre_p_norm<T: Real>(n: i32, m: i32, z: T, kind: LegendreType) -> T {
    let outside = kind == LegendreType::Outside;
    dispatch_int(
        [z],
        |a| xsf::legendre::assoc_legendre_p(true, n, m, a[0], kind.cut()),
        |a| generic::legendre::assoc_legendre_p(true, n, m, a[0], outside),
    )
}

/// The spherical Legendre function (xsf `sph_legendre_p`): the polar factor of the
/// spherical harmonic, `sqrt((2n + 1) (n - m)! / (4π (n + m)!)) P_n^m(cos θ)` with the
/// Condon-Shortley phase, for a real angle `theta` (`sin θ` enters as `abs(sin θ)`,
/// so the function is even in `θ`). `0` for `abs(m) > n`, NaN for `n < 0`. The azimuthal
/// factor and the complex value of `sph_harm` are not part of this function.
///
/// `Df64` takes `sin θ` and `cos θ` from the type: for `θ` next to a multiple of `π`, where `sin θ` is tiny, the relative
/// accuracy for `m != 0` is that of `sin θ`, an absolute `1e-32`. `f64` is xsf's; the
/// integer products under its square roots (`(2n + 1)((n - 1)^2 - m^2)`) are formed
/// in `f64`, where xsf's `int` overflows from `n = 646` (`docs/upstream-issues.md`).
///
/// # Examples
///
/// ```
/// use specials::sph_legendre_p;
///
/// // sqrt(3 / (4 pi)) cos θ
/// let y = sph_legendre_p(1, 0, 0.0_f64);
/// assert!((y - (3.0 / (4.0 * core::f64::consts::PI)).sqrt()).abs() < 1e-15);
/// ```
pub fn sph_legendre_p<T: Real>(n: i32, m: i32, theta: T) -> T {
    dispatch_int(
        [theta],
        |a| xsf::legendre::sph_legendre_p(n, m, a[0]),
        |a| generic::legendre::sph_legendre_p(n, m, a[0]),
    )
}

/// The Legendre functions of the second kind `Q_k(x)` and their derivatives `Q_k'(x)`
/// for `k = 0..=n` (xsf `lqn`, real argument), for `abs(x) < 1` the Ferrers function
/// `Q_k(x) = P_k(x) ln((1 + x)/(1 - x)) / 2 - W_(k-1)(x)` (DLMF 14.3.2) and for
/// `abs(x) > 1` the real function of DLMF 14.3.7 (odd or even in `x` as `(-1)^(k+1)`).
///
/// At `x = ±1` every entry is `1e300`, the stand-in for an infinity that xsf (and SciPy)
/// return instead of `inf`. The derivative is `Q_k' = k (Q_(k-1) - x Q_k) / (1 - x^2)`,
/// whose relative error grows like `eps / abs(1 - x^2)` next to `abs(x) = 1`.
///
/// `f64` is xsf's: the forward recurrence for `abs(x) <= 1.021`, otherwise the series of
/// DLMF 14.3.7 for `Q_(n-1)` and `Q_n` followed by the backward recurrence. `Df64`
/// takes the forward recurrence for `abs(x) < 1` and for `n acosh abs(x) <= 2` (the error
/// grows like `exp(2 n acosh abs(x))`), and Miller's backward recurrence otherwise
/// (the work is about `n + 40 / acosh abs(x)`, at most `21 n`).
///
/// # Examples
///
/// ```
/// use specials::lqn;
///
/// let (q, dq) = lqn(2, 0.5_f64);
/// // Q_0(x) = atanh(x), Q_0'(x) = 1 / (1 - x^2)
/// assert!((q[0] - 0.5_f64.atanh()).abs() < 1e-15);
/// assert!((dq[0] - 4.0 / 3.0).abs() < 1e-15);
/// ```
pub fn lqn<T: Real>(n: usize, x: T) -> (Vec<T>, Vec<T>) {
    let xf = x.approx_f64();
    let kernel = |len: usize| {
        let (mut qn, mut qd) = (vec![0.0; len], vec![0.0; len]);
        xsf::legendre::lqn(xf, &mut qn, &mut qd);
        (qn, qd)
    };
    let wrap = |(qn, qd): (Vec<f64>, Vec<f64>)| {
        (
            qn.into_iter().map(k64).collect::<Vec<T>>(),
            qd.into_iter().map(k64).collect::<Vec<T>>(),
        )
    };
    match T::STRATEGY {
        Strategy::F64Kernels => wrap(kernel(n + 1)),
        Strategy::Adaptive => {
            if !xf.is_finite() {
                return wrap(kernel(n + 1));
            }
            let (mut qn, mut qd) = (vec![T::zero(); n + 1], vec![T::zero(); n + 1]);
            generic::legendre::lqn(x, &mut qn, &mut qd);
            (qn, qd)
        }
    }
}

/// The associated Legendre functions of the second kind `Q_j^i(x)` and their derivatives for
/// `i = 0..=m`, `j = 0..=n` (xsf `lqmn`, real argument), as row-major `(m + 1) x (n + 1)`
/// tables: the entry `(i, j)` is at index `i (n + 1) + j`. For `abs(x) < 1`
/// `Q_n^m(x) = (-1)^m (1 - x^2)^(m/2) d^m Q_n / dx^m` (DLMF 14.6.2) and for `abs(x) > 1`
/// `Q_n^m(x) = (x^2 - 1)^(m/2) d^m Q_n / dx^m` (14.6.4). At `x = ±1` every entry is
/// `1e300`, as for [`lqn`].
///
/// `f64` is xsf's; for `-1.1 <= x <= -1.0001` xsf's start index uses `log(x - 1)`, a NaN
/// converted to an `int` (SciPy returns zeros), and this port uses `log(abs(x) - 1)`
/// (`docs/upstream-issues.md`). `Df64` computes rows 0 and 1 as [`lqn`] does and the
/// higher orders by the recurrence of DLMF 14.10.6, which is stable upward in the order.
///
/// # Examples
///
/// ```
/// use specials::lqmn;
///
/// let (q, _) = lqmn(1, 1, 0.5_f64);
/// // Q_0^1(x) = -1 / sqrt(1 - x^2), at index 1 * 2 + 0
/// assert!((q[2] + 1.0 / 0.75_f64.sqrt()).abs() < 1e-15);
/// ```
pub fn lqmn<T: Real>(m: usize, n: usize, x: T) -> (Vec<T>, Vec<T>) {
    let xf = x.approx_f64();
    let len = (m + 1) * (n + 1);
    let kernel = || {
        let (mut qm, mut qd) = (vec![0.0; len], vec![0.0; len]);
        xsf::legendre::lqmn(xf, m, n, &mut qm, &mut qd);
        (qm, qd)
    };
    let wrap = |(qm, qd): (Vec<f64>, Vec<f64>)| {
        (
            qm.into_iter().map(k64).collect::<Vec<T>>(),
            qd.into_iter().map(k64).collect::<Vec<T>>(),
        )
    };
    match T::STRATEGY {
        Strategy::F64Kernels => wrap(kernel()),
        Strategy::Adaptive => {
            if !xf.is_finite() {
                return wrap(kernel());
            }
            let (mut qm, mut qd) = (vec![T::zero(); len], vec![T::zero(); len]);
            generic::legendre::lqmn(x, m, n, &mut qm, &mut qd);
            (qm, qd)
        }
    }
}

#[inline]
fn k64<T: Real>(x: f64) -> T {
    T::from_dd(x, 0.0)
}

#[cfg(test)]
mod refdata;
#[cfg(test)]
mod tests;
