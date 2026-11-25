//! Elementary helpers, precision-adaptive (#19).
//!
//! Written independently from the identities in the NIST Digital Library of
//! Mathematical Functions (DLMF) chapter 4, not translated from xsf's `f64`
//! code: no rational approximation is used, only the type's own `exp`, `ln`,
//! `ln_1p`, `exp_m1`, `sin` and `cos` on exactly reduced arguments, and the
//! series noted at each function. The case analysis that decides signed
//! zeros, infinities and NaN (octants of the degree functions, the sign of
//! `sinpi` at integers, the convention `0 ln 0 = 0`) mirrors xsf so that both
//! strategies return SciPy's special values.
//!
//! Defects of xprec 0.2.3 handled here (see `docs/upstream-issues.md`):
//! `Df64::exp_m1` and `Df64::ln_1p` return `+0` for `-0`, and `Df64::ln_1p`
//! returns `-∞` for `-1 < x < -1 + 2^-53` whose high word is `-1`. `Df64::exp2`
//! is `exp(x ln 2)`, which is NaN at `±∞` and loses digits for large `|x|`,
//! and `Df64::cbrt` is not implemented (`todo!`). `Df64::sin` and `Df64::cos`
//! panic (an assertion in the kernel) for NaN and infinite arguments, so every
//! caller here returns before reaching them with one. A double-double product
//! that is zero is `+0` whatever the signs of the factors, so zero results
//! take their sign from the `f64` product of the high words.

use super::tables::TWO_PI_PIECES;
use super::{add, div, ldexp, ln, mul};
use crate::real::{k, Real};

/// `e^x - 1` with `expm1(±0) = ±0`: the type's `exp_m1`, whose only defect is
/// the sign of zero.
pub(crate) fn expm1<T: Real>(x: T) -> T {
    if x == T::zero() {
        return x;
    }
    x.exp_m1()
}

/// `ln(1 + x)`.
///
/// `ln(1 + x)` is evaluated directly where `1 + x` is exact (`x < -1/2`, by
/// Sterbenz's lemma in each word) or where `ln_1p` would overflow its internal
/// `exp_m1` (`x > 2^100`: `ln x + 1/x`, since `ln(1 + 1/x) - 1/x < 1/(2x²)`).
pub(crate) fn log1p<T: Real>(x: T) -> T {
    if x == T::zero() || x.is_nan() {
        return x;
    }
    if x < -T::one() {
        return T::nan();
    }
    if x == -T::one() {
        return T::neg_infinity();
    }
    if x.is_infinite() {
        return x;
    }
    if x < k(-0.5) {
        return (T::one() + x).ln();
    }
    if x.approx_f64() > 2.0_f64.powi(100) {
        return ln(x) + x.recip();
    }
    x.ln_1p()
}

/// `ln(1 + x) - x`.
///
/// For `|x| < 1/2` from `ln(1 + x) = 2 artanh(s)`, `s = x/(2 + x)` (DLMF 4.6.4
/// with `z = 1 + x`): `ln(1 + x) - x = -x²/(2 + x) + 2 Σ_{k≥1} s^(2k+1)/(2k+1)`,
/// whose terms shrink by `s² < 1/9`, truncated at the type's epsilon. The two
/// parts have the same sign for `x < 0` and differ by a factor of at least 18
/// for `x > 0`, so there is no cancellation. Elsewhere `log1p(x) - x`, which
/// loses at most a factor 5 (at `x = 1/2`).
pub(crate) fn log1pmx<T: Real>(x: T) -> T {
    if x.is_nan() || x < -T::one() {
        return T::nan();
    }
    if x == -T::one() {
        return T::neg_infinity();
    }
    if x.abs() < k(0.5) {
        let two = k::<T>(2.0);
        let s = x / (two + x);
        let s2 = s * s;
        let eps = T::epsilon();
        let mut sum = T::zero();
        let mut p = s * s2;
        let mut n = 3.0;
        loop {
            let term = p / k(n);
            sum = sum + term;
            if term.abs() <= eps * sum.abs() || n > 200.0 {
                break;
            }
            p = p * s2;
            n += 2.0;
        }
        return -(x * x) / (two + x) + two * sum;
    }
    // ln(1 + ∞) - ∞ is NaN, as in xsf.
    add(log1p(x), -x)
}

/// `cos x - 1 = -2 sin²(x/2)` (half-angle formula, DLMF §4.21), free of
/// cancellation; `+0` at `x = ±0` as in xsf, NaN for NaN and `±∞`.
///
/// For `|x| > π/4` the argument is first reduced to `r = x - 2πn`,
/// `|r| <= π`, with `2π` in five pieces (Cody–Waite): each `n 2π_i` is exact
/// in double-double for `|n| < 2^53`, and the pieces carry about 270 bits,
/// so for `|x| <= 2^50` `r` keeps its relative accuracy next
/// to the zeros `x = 2πn` (where `sin(x/2)` with the type's own reduction
/// would carry its absolute error of about `2^-160 |x|`). Beyond `|x| = 2^50`
/// the result is `cos(hi + lo) - 1` in `f64` (absolute accuracy about 1e-16).
pub(crate) fn cosm1<T: Real>(x: T) -> T {
    if !x.is_finite() {
        return T::nan();
    }
    if x.abs() > k(2.0_f64.powi(50)) {
        // Beyond 2^50 the quotient n = round(x / 2π) would need more than the
        // 53 bits that keep each product n 2π_i exact; the high word with the
        // platform's correctly reduced `cos` is accurate to about 1e-16
        // (absolute). The low word (up to 1/8 here) enters through
        // cos(h + l) = cos h cos l - sin h sin l.
        let h = x.approx_f64();
        let l = (x - k(h)).approx_f64();
        return T::from_dd(h.cos() * l.cos() - h.sin() * l.sin() - 1.0, 0.0);
    }
    let x = if x.abs() > k(core::f64::consts::FRAC_PI_4) {
        let n = (x.approx_f64() / TWO_PI_PIECES[0]).round();
        let mut r = x;
        for p in TWO_PI_PIECES {
            r = r - k::<T>(n) * k::<T>(p);
        }
        r
    } else {
        x
    };
    let s = (x * k(0.5)).sin();
    let r = -k::<T>(2.0) * s * s;
    if r == T::zero() {
        T::zero()
    } else {
        r
    }
}

/// `x ln y` with `0 ln y = 0` unless `y` is NaN.
pub(crate) fn xlogy<T: Real>(x: T, y: T) -> T {
    if x == T::zero() && !y.is_nan() {
        return T::zero();
    }
    signed_product(x, ln(y))
}

/// `a b`, with the sign of a zero product taken from the `f64` product of the
/// high words (a double-double product that is zero is `+0`).
fn signed_product<T: Real>(a: T, b: T) -> T {
    let r = mul(a, b);
    if r == T::zero() {
        T::from_dd(a.approx_f64() * b.approx_f64(), 0.0)
    } else {
        r
    }
}

/// `x ln(1 + y)` with `0 ln(1 + y) = 0` unless `y` is NaN.
pub(crate) fn xlog1py<T: Real>(x: T, y: T) -> T {
    if x == T::zero() && !y.is_nan() {
        return T::zero();
    }
    signed_product(x, log1p(y))
}

/// `1/(1 + e^-x)`; `e^x/(1 + e^x)` for `x < 0`, so that neither form
/// overflows before the result underflows.
pub(crate) fn expit<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return if x > T::zero() { T::one() } else { T::zero() };
    }
    if x < T::zero() {
        let e = x.exp();
        e / (T::one() + e)
    } else {
        T::one() / (T::one() + (-x).exp())
    }
}

/// `ln(x/(1 - x))`; `2 artanh(2x - 1) = ln_1p(s) - ln_1p(-s)`, `s = 2x - 1`
/// (exact), on `[0.3, 0.65]` where the quotient is close to 1. `1 - x` is
/// exact for `x >= 1/2` and has a relative error of the type's epsilon below.
pub(crate) fn logit<T: Real>(x: T) -> T {
    if x.is_nan() || x < T::zero() || x > T::one() {
        return T::nan();
    }
    if x == T::zero() {
        return T::neg_infinity();
    }
    if x == T::one() {
        return T::infinity();
    }
    if x < k(0.3) || x > k(0.65) {
        ln(x / (T::one() - x))
    } else {
        let s = k::<T>(2.0) * x - T::one();
        log1p(s) - log1p(-s)
    }
}

/// `ln expit(x) = -ln(1 + e^-x)`: `x - log1p(e^x)` for `x < 0`,
/// `-log1p(e^-x)` otherwise (`-0` at `+∞`, as in xsf).
pub(crate) fn log_expit<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x < T::zero() {
        if x.is_infinite() {
            return x;
        }
        x - log1p(x.exp())
    } else {
        let e = if x.is_infinite() {
            T::zero()
        } else {
            (-x).exp()
        };
        -log1p(e)
    }
}

/// `x mod 2` for `x >= 0`, exact: `x - 2 floor(x/2)` (NaN for `+∞`).
fn mod2<T: Real>(x: T) -> T {
    let two = k::<T>(2.0);
    x - two * (x / two).floor()
}

/// `sin(πx)`, reduced exactly modulo 2. Signed zeros as xsf: `sinpi(±0) = ±0`,
/// and at a nonzero integer `n` the zero is `+0` for even `n > 0`, `-0` for
/// odd `n > 0`, and the opposite for `n < 0` (`sinpi(1) = -0`, `sinpi(-1) = +0`).
pub(crate) fn sinpi<T: Real>(x: T) -> T {
    if !x.is_finite() {
        return T::nan();
    }
    let (a, neg) = if x.is_sign_negative() {
        (-x, true)
    } else {
        (x, false)
    };
    let r = mod2(a);
    let pi = T::PI();
    let v = if r < k(0.5) {
        (pi * r).sin()
    } else if r > k(1.5) {
        (pi * (r - k(2.0))).sin()
    } else {
        -(pi * (r - T::one())).sin()
    };
    if neg {
        -v
    } else {
        v
    }
}

/// `cos(πx)`, reduced exactly modulo 2; every zero is `+0`, as in xsf.
pub(crate) fn cospi<T: Real>(x: T) -> T {
    if !x.is_finite() {
        return T::nan();
    }
    let a = x.abs();
    let r = mod2(a);
    if r == k(0.5) || r == k(1.5) {
        return T::zero();
    }
    let pi = T::PI();
    if r < T::one() {
        -(pi * (r - k(0.5))).sin()
    } else {
        (pi * (r - k(1.5))).sin()
    }
}

/// `|x|` beyond which the degree functions return 0, as xsf (`sindg_lossth`).
const DG_LOSSTH: f64 = 1.0e14;

/// `π/180` in the type's precision.
fn pi180<T: Real>() -> T {
    T::PI() / k(180.0)
}

/// `floor(x/p)` for `0 <= x <= 1e14` and `p` in {45, 180}, corrected so that
/// `0 <= x - p q < p` holds exactly (the quotient `x/p` is rounded).
fn exact_quotient<T: Real>(x: T, p: f64) -> f64 {
    let pt = k::<T>(p);
    let mut q = (x / pt).floor().approx_f64();
    let r = x - pt * k(q);
    if r < T::zero() {
        q -= 1.0;
    } else if r >= pt {
        q += 1.0;
    }
    q
}

/// The octant of `0 <= x <= 1e14` degrees, as xsf: the multiple of 45° `y`
/// (rounded up to even) and the octant modulo 8.
fn octant<T: Real>(x: T) -> (f64, i32) {
    let mut y = exact_quotient(x, 45.0);
    // y < 2^42, so y mod 16 is exact.
    let mut j = (y % 16.0) as i32;
    if j & 1 != 0 {
        j += 1;
        y += 1.0;
    }
    (y, j & 0o7)
}

/// `sin(x°)`: exact reduction to `z ∈ [-45°, 45°]`, then `sin` or `cos` of
/// `z π/180`.
pub(crate) fn sindg<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let (a, mut neg) = if x.is_sign_negative() {
        (-x, true)
    } else {
        (x, false)
    };
    if a > k(DG_LOSSTH) {
        return T::zero();
    }
    let (y, mut j) = octant(a);
    if j > 3 {
        neg = !neg;
        j -= 4;
    }
    let z = (a - k::<T>(45.0) * k(y)) * pi180::<T>();
    let v = if j == 1 || j == 2 { z.cos() } else { z.sin() };
    if neg {
        -v
    } else {
        v
    }
}

/// `cos(x°)`, reduced as [`sindg`].
pub(crate) fn cosdg<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let a = x.abs();
    if a > k(DG_LOSSTH) {
        return T::zero();
    }
    let (y, mut j) = octant(a);
    let mut neg = false;
    if j > 3 {
        j -= 4;
        neg = !neg;
    }
    if j > 1 {
        neg = !neg;
    }
    let z = (a - k::<T>(45.0) * k(y)) * pi180::<T>();
    let v = if j == 1 || j == 2 { z.sin() } else { z.cos() };
    if neg {
        -v
    } else {
        v
    }
}

/// `tan(x°)` (`cot = false`) or `cot(x°)` (`cot = true`), with xsf's case
/// analysis: reduce modulo 180° exactly to `t`, fold into `[0°, 90°]`, return
/// exact `0`, `1` or `±∞` there, and otherwise `sin(a°)/sin(b°)` with `a` the
/// folded angle and `b = 90° - a`. Each of `a` and `b` is computed from `t`
/// with one subtraction (`t`, `90 - t`, `t - 90` or `180 - t`), which is exact
/// when the result is small (Sterbenz) and relatively accurate when it is not,
/// so the quotient is relatively accurate next to the zeros and the poles.
pub(crate) fn tancot<T: Real>(xx: T, cot: bool) -> T {
    if xx.is_nan() {
        return xx;
    }
    let (x, mut neg) = if xx.is_sign_negative() {
        (-xx, true)
    } else {
        (xx, false)
    };
    if x > k(DG_LOSSTH) {
        return T::zero();
    }
    let q = exact_quotient(x, 180.0);
    let t = x - k::<T>(180.0) * k(q);
    let ninety = k::<T>(90.0);
    let half_turn = k::<T>(180.0);
    // (a, b): the folded angle and its complement 90 - a.
    let (a, b) = if cot {
        if t <= ninety {
            (ninety - t, t)
        } else {
            neg = !neg;
            (t - ninety, half_turn - t)
        }
    } else if t > ninety {
        neg = !neg;
        (half_turn - t, t - ninety)
    } else {
        (t, ninety - t)
    };
    let q_odd = q % 2.0 != 0.0;
    let signed = |v: T| if neg { -v } else { v };
    if a == T::zero() {
        if q_odd {
            neg = !neg;
        }
        return if neg { -T::zero() } else { T::zero() };
    } else if a == k(45.0) {
        return signed(T::one());
    } else if a == ninety {
        if q_odd {
            neg = !neg;
        }
        return if neg {
            T::neg_infinity()
        } else {
            T::infinity()
        };
    }
    let r = pi180::<T>();
    let num = (a * r).sin();
    // Below 2^-400 degrees sin(b r) = b r to double-double precision; divide
    // by r and b in turn so that no intermediate has a subnormal low word
    // (cotdg(1e-300) = 5.7e301).
    let v = if b < k(2.0_f64.powi(-400)) {
        div(num / r, b)
    } else {
        num / (b * r).sin()
    };
    signed(v)
}

/// `((60 d + m) 60 + s) π/648000`: degrees, minutes and seconds to radians.
pub(crate) fn radian<T: Real>(d: T, m: T, s: T) -> T {
    let sixty = k::<T>(60.0);
    let sec = add(mul(add(mul(d, sixty), m), sixty), s);
    if sec == T::zero() {
        // The signed zero of the f64 formula (radian(-0, -0, -0) = -0).
        let (d, m, s) = (d.approx_f64(), m.approx_f64(), s.approx_f64());
        return T::from_dd((d * 60.0 + m) * 60.0 + s, 0.0);
    }
    signed_product(sec, T::PI() / k(648000.0))
}

/// `2^x = 2^n e^(f ln 2)` with `n = round(x)` and `f = x - n` exact, so that
/// the argument of `exp` stays in `[-ln 2 / 2, ln 2 / 2]` (DLMF §4.2).
pub(crate) fn exp2<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x > k(1024.0) {
        return T::infinity();
    }
    if x < k(-1080.0) {
        return T::zero();
    }
    let n = x.round();
    let f = x - n;
    ldexp((f * T::LN_2()).exp(), n.approx_f64() as i32)
}

/// `10^n` for an integer `0 <= n <= 400` by binary powering (at most 9
/// squarings and 9 products, each with a relative error of the epsilon).
fn pow10<T: Real>(mut n: u32) -> T {
    let mut base = k::<T>(10.0);
    let mut acc = T::one();
    while n > 0 {
        if n & 1 == 1 {
            acc = mul(acc, base);
        }
        n >>= 1;
        if n > 0 {
            base = mul(base, base);
        }
    }
    acc
}

/// `10^x = 10^n e^(f ln 10)` with `n = round(x)` and `f = x - n` exact
/// (DLMF §4.2), `10^n` by [`pow10`]. Rounding `x ln 10` as a whole would
/// cost `|x ln 10|` units of the epsilon; this split costs a few.
pub(crate) fn exp10<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x > k(309.0) {
        return T::infinity();
    }
    if x < k(-345.0) {
        return T::zero();
    }
    let n = x.round();
    let f = x - n;
    let e = (f * T::LN_10()).exp();
    let n = n.approx_f64();
    if n >= 0.0 {
        mul(e, pow10::<T>(n as u32))
    } else {
        // 10^-n overflows below n = -308: divide in two steps.
        let m = (-n) as u32;
        let first = m.min(300);
        let r = div(e, pow10::<T>(first));
        if m > first {
            div(r, pow10::<T>(m - first))
        } else {
            r
        }
    }
}

/// `∛x`: one Newton step `y + (x/y² - y)/3` from the `f64` cube root, which
/// doubles its 53 correct bits. Arguments outside `[2^-900, 2^900]` are first
/// scaled by `2^∓999` (a power of 8) so that `y²` and the low words stay normal.
pub(crate) fn cbrt<T: Real>(x: T) -> T {
    if x == T::zero() || !x.is_finite() {
        return x;
    }
    let (a, neg) = if x < T::zero() {
        (-x, true)
    } else {
        (x, false)
    };
    let h = a.approx_f64();
    let (a, e) = if h < 2.0_f64.powi(-900) {
        (ldexp(a, 999), -333)
    } else if h > 2.0_f64.powi(900) {
        (ldexp(a, -999), 333)
    } else {
        (a, 0)
    };
    let y = k::<T>(a.approx_f64().cbrt());
    let y = y + (a / (y * y) - y) / k(3.0);
    let r = ldexp(y, e);
    if neg {
        -r
    } else {
        r
    }
}
