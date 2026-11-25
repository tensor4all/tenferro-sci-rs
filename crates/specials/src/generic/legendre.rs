//! Legendre functions of integer degree and order, precision-adaptive: the
//! associated Legendre function `P_n^m` (unnormalized and normalized), the
//! spherical Legendre function, `pmv` of an integer degree and the functions
//! of the second kind `Q_n` and `Q_n^m` with their derivatives.
//!
//! Written from the NIST Digital Library of Mathematical Functions (DLMF),
//! chapter 14; no code was translated.
//!
//! * `P_n^m` (DLMF 14.10.3): the diagonal `P_m^m = (-1)^m (2m - 1)!! (1 -
//!   x^2)^(m/2)` (DLMF 14.7.15; no phase for `x > 1`, 14.7.16) by its two-term
//!   recurrence in `m`, then the three-term recurrence in `n`. A negative order
//!   has its own diagonal recurrence (a factor `1 / (2j (2j - 2))` per step), so
//!   no factorial ratio is formed. The normalized function carries the factor
//!   `sqrt((2n + 1)(n - m)! / (2 (n + m)!))` and the spherical one in addition
//!   `1 / sqrt(2 pi)`, built into the recurrences.
//! * `Q_n` (DLMF 14.10.3): forward recurrence from `Q_0 = ln|(1 + x)/(1 - x)| / 2`
//!   where it is stable (`|x| < 1`, or `|x| > 1` with `n acosh|x|` small, the
//!   growth being `exp(2 n acosh|x|)`), otherwise Miller's backward recurrence
//!   from a start index `n + 40 / acosh|x|` normalized by `Q_0`. `Q_n' = n (Q_(n-1)
//!   - x Q_n) / (1 - x^2)` (14.10.5).
//! * `Q_n^m` from `Q_n^0` and `Q_n^1 = -n (Q_(n-1) - x Q_n) / sqrt(|1 - x^2|)` by
//!   the recurrence in the order (14.10.6), which is stable upward because
//!   `P_n^m` vanishes for `m > n`.
//!
//! Terms are kept as a mantissa and a binary exponent, so a diagonal element
//! that underflows while the result does not is still correct.

use super::orthogonal::{Pair, Sc};
use super::{div, ldexp, mul, sqrt};
use crate::real::{k, Real};

fn two<T: Real>() -> T {
    k::<T>(2.0)
}

/// Exact small integer as `T`.
fn int<T: Real>(v: i64) -> T {
    k::<T>(v as f64)
}

/// `(1 - z)(1 + z)`, without the cancellation of `1 - z^2` next to `|z| = 1`.
fn one_minus_sq<T: Real>(z: T) -> T {
    (T::one() - z) * (T::one() + z)
}

/// `P_n^m(z)` at `z = +-1`.
fn pm1<T: Real>(norm: bool, n: i64, m: i32, z: T) -> T {
    if m != 0 {
        return T::zero();
    }
    let one = T::one();
    let mut res = if z >= T::zero() || n % 2 == 0 {
        one
    } else {
        -one
    };
    if norm {
        res = res * sqrt(int::<T>(2 * n + 1) / two::<T>());
    }
    res
}

/// `P_n^m(z)` for real `z`. `outside` selects the function for `|z| > 1`
/// (xsf branch cut 3: NaN for `z < 1`), otherwise the Ferrers function (NaN
/// for `|z| > 1`). `norm` selects the normalized function. NaN for `n < 0`.
pub(crate) fn assoc_legendre_p<T: Real>(norm: bool, n: i32, m: i32, z: T, outside: bool) -> T {
    let n = i64::from(n);
    let m_abs = i64::from(m.unsigned_abs());
    if n < 0 {
        return T::nan();
    }
    if m_abs > n {
        return T::zero();
    }
    if z.abs() == T::one() {
        return pm1(norm, n, m, z);
    }
    let one = T::one();
    let s2 = one_minus_sq(z);
    let m_neg = m < 0;

    // The first two terms of the diagonal.
    let w = if outside {
        sqrt(z - one) * sqrt(z + one)
    } else {
        let w = -sqrt(s2);
        if m_neg {
            -w
        } else {
            w
        }
    };
    let (r0, r1) = if norm {
        (one / sqrt(two::<T>()), sqrt(k::<T>(3.0)) * w / two::<T>())
    } else if m_neg {
        (one, w / two::<T>())
    } else {
        (one, w)
    };
    let bcs = if outside { -one } else { one };
    let mut d = Pair::new(r0, r1);
    for kk in 2..=m_abs {
        let fac = if norm {
            bcs * sqrt(int::<T>((2 * kk + 1) * (2 * kk - 1)) / int::<T>(4 * kk * (kk - 1)))
        } else if m_neg {
            bcs / int::<T>((2 * kk) * (2 * kk - 2))
        } else {
            bcs * int::<T>((2 * kk - 1) * (2 * kk - 3))
        };
        // The second coefficient is zero but propagates a NaN of the seed.
        let next = mul(fac * s2, d.a) + T::zero() * d.b;
        d.push(next);
    }
    let diag = if m_abs == 0 { d.a } else { d.b };
    if n == m_abs {
        return Sc { m: diag, e: d.e }.value();
    }

    // The recurrence in the degree.
    let mf = i64::from(m);
    let fac = if norm {
        sqrt(int::<T>(2 * m_abs + 3))
    } else {
        // (2 |m| + 1) / (|m| + 1 - m)
        int::<T>(2 * m_abs + 1) / int::<T>(m_abs + 1 - mf)
    };
    let mut p = Pair::new(diag, fac * z * diag);
    p.e += d.e;
    for j in (m_abs + 2)..=n {
        let (c0, c1) = if norm {
            // sqrt((2j+1) ((j-1)^2 - m^2) / ((2j-3) (j^2 - m^2))) and
            // sqrt((2j+1) (4 (j-1)^2 - 1) / ((2j-3) (j^2 - m^2)))
            let den = int::<T>(2 * j - 3) * int::<T>(j - mf) * int::<T>(j + mf);
            let a = int::<T>(2 * j + 1) * int::<T>(j - 1 - mf) * int::<T>(j - 1 + mf);
            let b = int::<T>(2 * j + 1) * int::<T>(2 * j - 3) * int::<T>(2 * j - 1);
            (-sqrt(a / den), sqrt(b / den))
        } else {
            (
                -int::<T>(j + mf - 1) / int::<T>(j - mf),
                int::<T>(2 * j - 1) / int::<T>(j - mf),
            )
        };
        let next = mul(c0, p.a) + mul(c1 * z, p.b);
        p.push(next);
    }
    Sc { m: p.b, e: p.e }.value()
}

/// The normalized spherical Legendre function of integer degree and order.
pub(crate) fn sph_legendre_p<T: Real>(n: i32, m: i32, theta: T) -> T {
    let n = i64::from(n);
    let m_abs = i64::from(m.unsigned_abs());
    if n < 0 {
        return T::nan();
    }
    if m_abs > n {
        return T::zero();
    }
    let one = T::one();
    let s = theta.sin();
    let c = theta.cos();
    let pi = T::PI();

    let r0 = one / (two::<T>() * sqrt(pi));
    let mut r1 = -sqrt(k::<T>(3.0)) / (two::<T>() * sqrt(two::<T>() * pi));
    if m < 0 {
        r1 = -r1;
    }
    let r1 = r1 * s.abs();
    let mut d = Pair::new(r0, r1);
    for kk in 2..=m_abs {
        let fac = sqrt(int::<T>((2 * kk + 1) * (2 * kk - 1)) / int::<T>(4 * kk * (kk - 1)));
        let next = mul(fac * s * s, d.a) + T::zero() * d.b;
        d.push(next);
    }
    let diag = if m_abs == 0 { d.a } else { d.b };
    if n == m_abs {
        return Sc { m: diag, e: d.e }.value();
    }
    let mf = i64::from(m);
    let mut p = Pair::new(diag, sqrt(int::<T>(2 * m_abs + 3)) * c * diag);
    p.e += d.e;
    for j in (m_abs + 2)..=n {
        let den = int::<T>(2 * j - 3) * int::<T>(j - mf) * int::<T>(j + mf);
        let a = int::<T>(2 * j + 1) * int::<T>(j - 1 - mf) * int::<T>(j - 1 + mf);
        let b = int::<T>(2 * j + 1) * int::<T>(2 * j - 3) * int::<T>(2 * j - 1);
        let next = mul(-sqrt(a / den), p.a) + mul(sqrt(b / den) * c, p.b);
        p.push(next);
    }
    Sc { m: p.b, e: p.e }.value()
}

/// `Q_0(x) = ln|(1 + x)/(1 - x)| / 2` without cancellation, for real `x != +-1`.
fn lq0<T: Real>(x: T) -> T {
    let ax = x.abs();
    let t = if ax < T::one() {
        two::<T>() * ax / (T::one() - ax)
    } else {
        two::<T>() / (ax - T::one())
    };
    let q = super::elementary::log1p(t) / two::<T>();
    if x < T::zero() {
        -q
    } else {
        q
    }
}

/// `acosh|x|` in `f64`, for `|x| > 1`.
fn acosh_f64(x: f64) -> f64 {
    let ax = x.abs();
    (ax + (ax * ax - 1.0).sqrt()).ln()
}

/// `Q_k(x)` for `k = 0..=n`, `x` finite with `|x| != 1`.
fn q_values<T: Real>(x: T, n: usize) -> Vec<T> {
    let one = T::one();
    let mut q = vec![T::zero(); n + 1];
    let q0 = lq0(x);
    let ach = if x.abs() > one {
        acosh_f64(x.approx_f64())
    } else {
        0.0
    };
    if x.abs() < one || n as f64 * ach <= 2.0 {
        // Forward recurrence: the growth of the error is exp(2 n acosh|x|) <= e^4.
        q[0] = q0;
        if n >= 1 {
            q[1] = x * q0 - one;
        }
        for j in 2..=n {
            let jf = k::<T>(j as f64);
            q[j] = (mul((two::<T>() * jf - one) * x, q[j - 1]) - mul(jf - one, q[j - 2])) / jf;
        }
        return q;
    }

    // Miller's algorithm: backward from a start index beyond which the
    // dominant solution has grown by exp(80), normalized by Q_0.
    let start = n + (40.0 / ach).ceil() as usize + 5;
    let mut mant = vec![T::zero(); n + 1];
    let mut expo = vec![0_i64; n + 1];
    let mut p = Pair::new(T::zero(), one);
    for j in (1..=start).rev() {
        let jf = k::<T>(j as f64);
        // q_(j-1) = ((2j + 1) x q_j - (j + 1) q_(j+1)) / j
        let next = (mul((two::<T>() * jf + one) * x, p.b) - mul(jf + one, p.a)) / jf;
        p.push(next);
        if j - 1 <= n {
            mant[j - 1] = p.b;
            expo[j - 1] = p.e;
        }
    }
    // `p` holds q_0 in its second term; the first n + 1 terms are in mant.
    let (m0, e0) = (mant[0], expo[0]);
    for j in 0..=n {
        let r = div(mant[j], m0);
        q[j] = ldexp(mul(q0, r), (expo[j] - e0).clamp(-3000, 3000) as i32);
    }
    q
}

/// `Q_n(x)` and `Q_n'(x)` for `n = 0..qn.len()`, `x` finite. At `x = +-1` every
/// entry is `1e300`, the stand-in for an infinity of xsf.
pub(crate) fn lqn<T: Real>(x: T, qn: &mut [T], qd: &mut [T]) {
    let n = qn.len() - 1;
    if x.abs() == T::one() {
        for j in 0..=n {
            qn[j] = T::from_dd(1.0e300, 0.0);
            qd[j] = T::from_dd(1.0e300, 0.0);
        }
        return;
    }
    let q = q_values(x, n);
    let s2 = one_minus_sq(x);
    qn[..=n].copy_from_slice(&q);
    qd[0] = T::one() / s2;
    for j in 1..=n {
        let jf = k::<T>(j as f64);
        qd[j] = mul(jf, q[j - 1] - x * q[j]) / s2;
    }
}

/// `Q_n^m(x)` and its derivative for `m = 0..=mmax`, `n = 0..=nmax`, row-major with
/// `nmax + 1` columns, `x` finite. At `x = +-1` every entry is `1e300`.
pub(crate) fn lqmn<T: Real>(x: T, mmax: usize, nmax: usize, qm: &mut [T], qd: &mut [T]) {
    let w = nmax + 1;
    let one = T::one();
    if x.abs() == one {
        for v in qm.iter_mut().chain(qd.iter_mut()) {
            *v = T::from_dd(1.0e300, 0.0);
        }
        return;
    }
    let s2 = one_minus_sq(x);
    let outside = x.abs() > one;
    let ls = if outside { -one } else { one };
    let xq = sqrt(ls * s2);
    let q = q_values(x, nmax);

    // Rows 0 and 1.
    qm[..=nmax].copy_from_slice(&q);
    if mmax >= 1 {
        qm[w] = -one / xq;
        for j in 1..=nmax {
            let jf = k::<T>(j as f64);
            qm[w + j] = -mul(jf, q[j - 1] - x * q[j]) / xq;
        }
    }
    // Q_n^(i+2) = -2 (i + 1) x / xq Q_n^(i+1) - ls (n - i)(n + i + 1) Q_n^i
    for i in 0..mmax.saturating_sub(1) {
        let fi = k::<T>(i as f64);
        for j in 0..=nmax {
            let jf = k::<T>(j as f64);
            let a = mul(-two::<T>() * (fi + one) * x / xq, qm[(i + 1) * w + j]);
            let b = mul(ls * (jf - fi) * (jf + fi + one), qm[i * w + j]);
            qm[(i + 2) * w + j] = a - b;
        }
    }

    // Derivatives.
    qd[0] = one / s2;
    for j in 1..=nmax {
        let jf = k::<T>(j as f64);
        qd[j] = mul(jf, qm[j - 1] - x * qm[j]) / s2;
    }
    for i in 1..=mmax {
        let fi = k::<T>(i as f64);
        for j in 0..=nmax {
            let jf = k::<T>(j as f64);
            qd[i * w + j] = mul(fi * x / s2, qm[i * w + j])
                + mul((fi + jf) * (jf - fi + one) / xq, qm[(i - 1) * w + j]);
        }
    }
}

/// `P_v^m(x)` of integer order and degree, as xsf's `pmv` defines it: `None` where
/// the arguments are not finite integers in range (the caller takes the `f64`
/// kernel). `m = 0` is the Legendre polynomial `P_v(x)` for every `x`; `m != 0`
/// is the Ferrers function, NaN for `|x| > 1` (xsf takes `sqrt(1 - x^2)`) and for a
/// negative order beyond the degree (where xsf does not apply DLMF 14.9.3).
pub(crate) fn pmv_int<T: Real>(m: T, v: T, x: T) -> Option<T> {
    let lim = 2.0_f64.powi(31);
    let (mf, vf) = (m.approx_f64(), v.approx_f64());
    if !(m.is_finite() && v.is_finite() && x.is_finite())
        || m != m.floor()
        || v != v.floor()
        || mf.abs() >= lim
        || vf.abs() >= lim
    {
        return None;
    }
    let vx = if vf < 0.0 { -vf - 1.0 } else { vf } as i64;
    let mi = mf as i64;
    if mi < 0 && vx + mi < 0 {
        return Some(T::nan());
    }
    if mi == 0 {
        return Some(super::orthogonal::legendre(vx, x));
    }
    if x.abs() > T::one() {
        return Some(T::nan());
    }
    Some(assoc_legendre_p(false, vx as i32, mi as i32, x, false))
}
