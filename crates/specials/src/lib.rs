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
//!   macOS one `binom` row fails because Apple's `log` is off by 1 ulp; see
//!   `docs/upstream-issues.md`.
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
/// negative argument. The `f64` path does not handle a largest argument above
/// about 2^512 with the smallest nonzero one more than about 2^1512 below it:
/// there it returns NaN, as SciPy does.
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
/// beyond 2^±300 and none is subnormal, so that the products xsf forms stay
/// in range. Arguments at most 2^600 apart are moved just into 2^±300, and
/// xsf's unscaled value is kept where it agrees with the scaled one to 1e-10;
/// arguments farther apart are scaled only when xsf's own value is NaN,
/// infinite or 0.
/// It does not handle a largest argument above about 2^340 with the smallest
/// nonzero one more than about 2^1340 below it
/// (`elliprj(2^-1000, 2^400, 2^400, 2^400)`), nor a subnormal argument (which
/// xsf treats as zero) next to arguments below about 2^-340 or above about
/// 2^340: there it returns NaN, as SciPy does.
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

#[inline]
fn k64<T: Real>(x: f64) -> T {
    T::from_dd(x, 0.0)
}

#[cfg(test)]
mod refdata;
#[cfg(test)]
mod tests;
