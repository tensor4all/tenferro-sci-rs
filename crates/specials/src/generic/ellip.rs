//! Complete and incomplete elliptic integrals and the Jacobian elliptic
//! functions, precision-adaptive.
//!
//! Written from DLMF; no code was translated.
//!
//! * `K(m) = π / (2 agm(1, √(1 - m)))` (DLMF 19.8(i)).
//! * `E(m) = K(m) (1 - Σ_{n≥0} 2^(n-1) c_n²)` with `c_0² = m` and
//!   `c_{n+1} = c_n² / (2(a_n + g_n))` (DLMF 19.8(i)) for `m <= 1/2`. For
//!   `m < 0` the transformation `E(m) = √(1 - m) E(m / (m - 1))`
//!   (DLMF 19.7(ii)) gives a parameter in `(0, 1)`. For `1/2 < m < 1` the sum
//!   above cancels like `ε K(m)`, so Legendre's relation (DLMF 19.7(i)) is used
//!   instead: `E(m) = (π/2 + K(m) (p/3) R_D(0, m, 1)) / K(p)` with `p = 1 - m`
//!   and `K(p) - E(p) = (p/3) R_D(0, 1 - p, 1)` (DLMF 19.25(i)), in which every
//!   term is positive.
//! * `F(φ | m) = sin φ R_F(cos² φ, 1 - m sin² φ, 1)` and
//!   `E(φ | m) = sin φ R_F(...) - (m/3) sin³ φ R_D(...)` for `|φ| <= π/2`
//!   (DLMF 19.25(i)), extended to all `φ` by the quasi-periodicity
//!   `F(φ + nπ | m) = F(φ | m) + 2n K(m)` and `E(φ + nπ | m) = E(φ | m) +
//!   2n E(m)` (DLMF 19.2(ii)).
//! * `sn`, `cn`, `dn` and the amplitude are evaluated by the descending
//!   Landen (AGM) transformation (DLMF 22.20(ii)): with `a_0 = 1`,
//!   `g_0 = √(1 - m)`, `c_0 = √m`, the amplitudes satisfy
//!   `φ_N = 2^N a_N u` and `sin(2φ_{n-1} - φ_n) = (c_n/a_n) sin φ_n`. The
//!   latter is solved for `2φ_{n-1} - φ_n ∈ (-π/2, π/2)` with `atan2`, from
//!   `r sin φ_n` and `cos = √((1 - r sin φ_n)(1 + r sin φ_n))` (`r = c_n/a_n`),
//!   whose two factors are computed as `(1 - r) + r (1 ∓ sin φ_n)` with
//!   `1 ∓ sin φ_n = 2 sin²(φ_n/2 ∓ π/4)`, so that they keep their relative
//!   accuracy where `r sin φ_n` is close to ±1 (an `asin` would lose half the
//!   digits there). `dn² = (1 - m) + m cn²` is a sum of positive terms. For
//!   `m < 0` the transformation to a positive parameter of DLMF 22.17(i)
//!   (imaginary modulus) is applied, and `m = 1` gives `tanh`, `sech` and the
//!   Gudermannian (DLMF 22.10(ii) and 4.23).
//!
//! The conventions for non-finite and zero results are those of
//! `scipy.special`.

use super::agm::agm_positive;
use super::carlson::{elliprd, elliprf, rd_core};
use super::{add, div, mul, sqrt};
use crate::real::{k, Real};

/// K(m) for `p = 1 - m` finite and positive.
fn k_of_p<T: Real>(p: T) -> T {
    T::FRAC_PI_2() / agm_positive(T::one(), sqrt(p))
}

pub(crate) fn ellipkm1<T: Real>(p: T) -> T {
    if p.is_nan() {
        return p;
    }
    if p < T::zero() {
        return T::nan();
    }
    if p == T::zero() {
        return T::infinity();
    }
    if p.is_infinite() {
        return T::zero();
    }
    k_of_p(p)
}

pub(crate) fn ellipk<T: Real>(m: T) -> T {
    if m.is_nan() || m > T::one() {
        return T::nan();
    }
    ellipkm1(add(T::one(), -m))
}

/// E(m) for `0 <= m < 1` with `p = 1 - m` supplied by the caller (so that
/// `p` keeps its accuracy when `m` is close to 1 or was obtained by a
/// transformation).
pub(crate) fn ellipe_core<T: Real>(m: T, p: T) -> T {
    if m > k::<T>(0.5) {
        let k_m = k_of_p(p);
        let k_p = k_of_p(m);
        let kd = rd_core(T::zero(), m, T::one());
        return (T::FRAC_PI_2() + k_m * (p / k::<T>(3.0)) * kd) / k_p;
    }
    let eps = T::epsilon();
    let (mut a, mut g) = (T::one(), sqrt(p));
    let mut c2 = m;
    let mut s = m * k::<T>(0.5);
    let mut w = T::one();
    for _ in 0..200 {
        let c = c2 / (k::<T>(2.0) * (a + g));
        let an = (a + g) * k::<T>(0.5);
        g = (a * g).sqrt();
        a = an;
        c2 = c * c;
        s = s + w * c2;
        w = w * k::<T>(2.0);
        if c.abs() <= eps * a {
            return T::FRAC_PI_2() / a * (T::one() - s);
        }
    }
    T::nan()
}

pub(crate) fn ellipe<T: Real>(m: T) -> T {
    if m.is_nan() || m > T::one() {
        return T::nan();
    }
    if m == T::one() {
        return T::one();
    }
    if m == T::neg_infinity() {
        return T::infinity();
    }
    if m < T::zero() {
        // E(m) = √p E(1 - 1/p), p = 1 - m (DLMF 19.7.1).
        let p = T::one() - m;
        return mul(sqrt(p), ellipe_core(div(-m, p), T::one() / p));
    }
    ellipe_core(m, T::one() - m)
}

/// `φ = nπ + ψ` with `|ψ| <= π/2`.
fn reduce<T: Real>(phi: T) -> (T, T) {
    let n = (phi / T::PI()).round();
    (n, phi - n * T::PI())
}

/// Largest `|φ|` reduced modulo π: beyond it the reduced amplitude has no
/// correct digit, and `F(φ | m) = 2 φ K(m) / π` to within `O(K(m))`, a relative
/// error below `π/(2|φ|) < ε`.
fn reduction_limit<T: Real>() -> T {
    k::<T>(2.0_f64.powi(106))
}

/// `(F(ψ | m), E(ψ | m))`, or only the first when `want_e` is false, for
/// `|ψ| <= π/2`, `m < 1`, with `p = 1 - m`.
fn incomplete_core<T: Real>(psi: T, m: T, p: T, want_e: bool) -> (T, T) {
    let (s, c) = psi.sin_cos();
    let c2 = c * c;
    let y = if m >= T::zero() {
        p + m * c2
    } else {
        T::one() + (-m) * s * s
    };
    let rf = elliprf(c2, y, T::one());
    let f = mul(s, rf);
    if !want_e {
        return (f, T::nan());
    }
    let rd = elliprd(c2, y, T::one());
    let e = add(f, -mul(mul(mul(m / k::<T>(3.0), s) * s, s), rd));
    (f, e)
}

pub(crate) fn ellipkinc<T: Real>(phi: T, m: T) -> T {
    if phi.is_nan() || m.is_nan() || m > T::one() {
        return T::nan();
    }
    if phi.is_infinite() || m.is_infinite() {
        return if m.is_infinite() && phi.is_finite() {
            T::zero()
        } else if phi.is_infinite() && m.is_finite() {
            phi
        } else {
            T::nan()
        };
    }
    if m == T::zero() {
        return phi;
    }
    let p = T::one() - m;
    if p == T::zero() && phi.abs() >= T::FRAC_PI_2() {
        return T::infinity();
    }
    if phi.abs() > reduction_limit::<T>() {
        return mul(div(phi, T::FRAC_PI_2()), ellipk(m));
    }
    let (n, psi) = reduce(phi);
    let f0 = incomplete_core(psi, m, p, false).0;
    if n == T::zero() {
        return f0;
    }
    add(mul(k::<T>(2.0) * n, ellipk(m)), f0)
}

pub(crate) fn ellipeinc<T: Real>(phi: T, m: T) -> T {
    if phi.is_nan() || m.is_nan() || m > T::one() {
        return T::nan();
    }
    if phi.is_infinite() {
        return phi;
    }
    if m.is_infinite() {
        return -m;
    }
    if m == T::zero() {
        return phi;
    }
    if phi.abs() > reduction_limit::<T>() {
        return mul(div(phi, T::FRAC_PI_2()), ellipe(m));
    }
    let p = T::one() - m;
    let (n, psi) = reduce(phi);
    let e0 = if p == T::zero() {
        psi.sin()
    } else {
        incomplete_core(psi, m, p, true).1
    };
    if n == T::zero() {
        return e0;
    }
    add(mul(k::<T>(2.0) * n, ellipe(m)), e0)
}

/// Most AGM levels `ellipj` supports: `c_n/a_n` falls quadratically, so a
/// parameter `1 - m` down to the smallest normal number needs about a dozen.
const LEVELS: usize = 40;

/// `(sn, cn, dn, ph)` for `0 <= m < 1`, `p = 1 - m`, finite nonzero `u`.
fn ellipj_core<T: Real>(u: T, m: T, p: T) -> (T, T, T, T) {
    let eps = T::epsilon();
    let two = k::<T>(2.0);
    let nan = (T::nan(), T::nan(), T::nan(), T::nan());
    // a[n], c[n] and g_prev[n] = g_{n-1} of the AGM sequence (DLMF 22.20.1).
    let mut a = [T::zero(); LEVELS];
    let mut c = [T::zero(); LEVELS];
    let mut g_prev = [T::zero(); LEVELS];
    a[0] = T::one();
    c[0] = sqrt(m);
    let mut g = sqrt(p);
    let mut n = 0;
    while c[n] > eps * a[n] {
        if n + 1 >= LEVELS {
            return nan;
        }
        g_prev[n + 1] = g;
        c[n + 1] = c[n] * c[n] / (two * (a[n] + g));
        a[n + 1] = (a[n] + g) * k::<T>(0.5);
        g = (a[n] * g).sqrt();
        n += 1;
    }
    let levels = n;
    // φ_N = 2^N a_N u carries the phase of u; with ε|φ_N| >= 1 no digit of
    // sn or cn is correct.
    let mut phi = mul(k::<T>(2.0_f64.powi(levels as i32)) * a[levels], u);
    if !(phi.abs() < k::<T>(2.0_f64.powi(104))) {
        return nan;
    }
    let quarter_pi = T::FRAC_PI_4();
    for n in (1..=levels).rev() {
        let r = c[n] / a[n];
        let s = phi.sin();
        let half = phi * k::<T>(0.5);
        let sm = (half - quarter_pi).sin();
        let sp = (half + quarter_pi).sin();
        let one_minus_r = g_prev[n] / a[n];
        // 1 - r sin φ and 1 + r sin φ, each as a sum of positive terms.
        let q_minus = one_minus_r + r * two * sm * sm;
        let q_plus = one_minus_r + r * two * sp * sp;
        let theta = (r * s).atan2((q_minus * q_plus).sqrt());
        phi = (phi + theta) * k::<T>(0.5);
    }
    let (sn, cn) = phi.sin_cos();
    let dn = (p + m * cn * cn).sqrt();
    (sn, cn, dn, phi)
}

/// `(sn, cn, dn, ph)` for `m = 1`.
fn ellipj_m1<T: Real>(u: T) -> (T, T, T, T) {
    // Beyond |u| = 400, sech u < 1e-173 and tanh u = ±1 to double-double precision.
    let au = u.abs();
    if au > k::<T>(400.0) {
        let sech = mul(k::<T>(2.0), (-au).exp());
        let ph = T::FRAC_PI_2() - sech;
        return (T::one().copysign(u), sech, sech, ph.copysign(u));
    }
    let sech = T::one() / u.cosh();
    // Gudermannian: gd(u) = 2 atan(tanh(u/2)).
    let ph = k::<T>(2.0) * (u * k::<T>(0.5)).tanh().atan();
    (u.tanh(), sech, sech, ph)
}

pub(crate) fn ellipj<T: Real>(u: T, m: T) -> (T, T, T, T) {
    let nan = (T::nan(), T::nan(), T::nan(), T::nan());
    if u.is_nan() || m.is_nan() || m > T::one() || u.is_infinite() || m.is_infinite() {
        return nan;
    }
    if u == T::zero() {
        return (u, T::one(), T::one(), u);
    }
    if m < T::zero() {
        // DLMF 22.17(i): sn(u | m) = sn(√p u | m1) / (√p dn(√p u | m1)),
        // cn = cn1 / dn1, dn = 1 / dn1, with p = 1 - m and m1 = -m/p.
        let p = T::one() - m;
        let rp = sqrt(p);
        let m1 = div(-m, p);
        let p1 = T::one() / p;
        let (sn1, cn1, dn1, ph1) = ellipj_core(mul(rp, u), m1, p1);
        let sn = sn1 / (rp * dn1);
        let cn = cn1 / dn1;
        let dn = T::one() / dn1;
        let ph0 = sn1.atan2(rp * cn1);
        let ph = ph0 + T::PI() * ((ph1 - ph0) / T::PI()).round();
        return (sn, cn, dn, ph);
    }
    let p = T::one() - m;
    if p == T::zero() {
        return ellipj_m1(u);
    }
    ellipj_core(u, m, p)
}
