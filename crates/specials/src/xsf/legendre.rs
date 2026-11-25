//! Legendre functions of integer degree and order for `f64`: the Legendre
//! polynomial `P_n`, the associated Legendre function `P_n^m` (unnormalized
//! and normalized), the spherical Legendre function and the functions of the
//! second kind `Q_n` and `Q_n^m` with their derivatives.
//!
//! Ported from xsf (scipy/xsf at `31177fd9a458539cf05c04d215341245d0245e53`)
//! `include/xsf/cpu/legendre.h` (the real overloads, without
//! the derivative orders of the `dual` number types: the value only), with
//! the recurrences of `include/xsf/cpu/recur.h` unrolled. The `lqn` and
//! `lqmn` routines are SciPy's C++ translation of Zhang and Jin's Fortran
//! (`LQNA`, `LQMNA` in "Computation of Special Functions", Wiley, 1996), see
//! `LICENSE-NOTICE-zhang-jin-specfun`.
//!
//! Deviations from upstream, each where the C code has undefined behavior or
//! overflows `int`:
//!
//! * the integer products under the square roots of the normalized recurrences
//!   (`(2n + 1) (4 (n - 1)^2 - 1)`, `(2n + 1) ((n - 1)^2 - m^2)`, `(2n - 3) (n^2 - m^2)`)
//!   are formed in `f64`; the `int` products overflow from `n = 646`;
//! * `lqmn` for `-1.1 <= x <= -1.0001` evaluates `log(|x| - 1)` where xsf
//!   evaluates `log(x - 1)`, a NaN converted to an `int`;
//! * `legendre_p` returns NaN for a negative degree (xsf returns `z`);
//! * `lqn` and `lqmn` return the infinities of the functions at `x = ±1`, where xsf stores `1e300`.

use core::f64::consts::PI;

/// The branch of the associated Legendre function for a real argument
/// (xsf `branch_cut` 2 and 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cut {
    /// Cut on `|z| > 1`: the Ferrers function for `-1 < z < 1`.
    Ferrers,
    /// Cut on `-1 < z < 1`: the function for `|z| > 1`.
    Outside,
}

/// `P_n(z)` (`legendre_p`).
pub(crate) fn legendre_p(n: i32, z: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    // The recurrence P_k = -(k-1)/k P_(k-2) + (2k-1)/k z P_(k-1) from (P_0, P_1).
    let (mut a, mut b) = (1.0, z);
    if n == 0 {
        return a;
    }
    for k in 2..=i64::from(n) {
        let kf = k as f64;
        let fac0 = -(kf - 1.0) / kf;
        let fac1 = (2.0 * kf - 1.0) / kf;
        let tmp = fac0 * a + fac1 * z * b;
        a = b;
        b = tmp;
    }
    b
}

/// `P_|m|^m(z)` (`assoc_legendre_p_for_each_m_abs_m` for the last `m`).
fn p_m_abs_m(norm: bool, m: i32, z: f64, cut: Cut) -> f64 {
    let m_signbit = m < 0;
    let w = if cut == Cut::Outside {
        // form for this branch cut
        (z - 1.0).sqrt() * (z + 1.0).sqrt()
    } else {
        // form for this branch cut
        let w = -(1.0 - z * z).sqrt();
        if m_signbit {
            -w
        } else {
            w
        }
    };
    let (mut r0, mut r1) = if norm {
        (1.0 / 2.0_f64.sqrt(), 3.0_f64.sqrt() * w / 2.0)
    } else {
        let r1 = if m_signbit { w / 2.0 } else { w };
        (1.0, r1)
    };

    let branch_cut_sign = if cut == Cut::Outside { -1.0 } else { 1.0 };
    let m_abs = i64::from(m.unsigned_abs());
    if m_abs == 0 {
        return r0;
    }
    // k = 2..=|m|: P_k^k = fac (1 - z^2) P_(k-2)^(k-2), the second coefficient is 0.
    for k in 2..=m_abs {
        let fac = if norm {
            branch_cut_sign * (((2 * k + 1) * (2 * k - 1)) as f64 / (4 * k * (k - 1)) as f64).sqrt()
        } else if m_signbit {
            branch_cut_sign / ((2 * k) * (2 * k - 2)) as f64
        } else {
            branch_cut_sign * ((2 * k - 1) * (2 * k - 3)) as f64
        };
        let coef0 = fac * (1.0 - z * z);
        let tmp = coef0 * r0 + 0.0 * r1;
        r0 = r1;
        r1 = tmp;
    }
    r1
}

/// `P_n^m(z)` at `z = +-1` (`assoc_legendre_p_pm1`).
fn p_pm1(norm: bool, n: i64, m: i32, z: f64) -> f64 {
    if m == 0 {
        let mut res = if z >= 0.0 || n % 2 == 0 { 1.0 } else { -1.0 };
        // Apply normalization for m=0
        if norm {
            res *= ((2 * n + 1) as f64 / 2.0).sqrt();
        }
        res
    } else {
        0.0
    }
}

/// `P_n^m(z)` for a real `z` (`assoc_legendre_p`). `norm` selects the
/// normalized function, `P_n^m` times `sqrt((2n + 1) (n - m)! / (2 (n + m)!))`.
pub(crate) fn assoc_legendre_p(norm: bool, n: i32, m: i32, z: f64, cut: Cut) -> f64 {
    let n = i64::from(n);
    let m_abs = i64::from(m.unsigned_abs());
    if n < 0 {
        return f64::NAN;
    }
    let rmm = p_m_abs_m(norm, m, z, cut);

    if m_abs > n {
        return 0.0;
    }
    if z.abs() == 1.0 {
        return p_pm1(norm, n, m, z);
    }
    // init_n: (P_|m|^m, fac z P_|m|^m)
    let fac = if norm {
        ((2 * m_abs + 3) as f64).sqrt()
    } else {
        (2 * (m_abs + 1) - 1) as f64 / (m_abs + 1 - i64::from(m)) as f64
    };
    let (mut a, mut b) = (rmm, fac * z * rmm);
    if n == m_abs {
        return rmm;
    }
    let mf = i64::from(m);
    for j in (m_abs + 2)..=n {
        let (fac0, fac1) = if norm {
            (
                -(((2 * j + 1) * ((j - 1) * (j - 1) - mf * mf)) as f64
                    / ((2 * j - 3) * (j * j - mf * mf)) as f64)
                    .sqrt(),
                (((2 * j + 1) * (4 * (j - 1) * (j - 1) - 1)) as f64
                    / ((2 * j - 3) * (j * j - mf * mf)) as f64)
                    .sqrt(),
            )
        } else {
            (
                -((j + mf - 1) as f64) / (j - mf) as f64,
                (2 * j - 1) as f64 / (j - mf) as f64,
            )
        };
        let tmp = fac0 * a + fac1 * z * b;
        a = b;
        b = tmp;
    }
    b
}

/// The normalized spherical Legendre function `P_n^m(cos theta)` (`sph_legendre_p`).
pub(crate) fn sph_legendre_p(n: i32, m: i32, theta: f64) -> f64 {
    let n = i64::from(n);
    let m_abs = i64::from(m.unsigned_abs());
    let m_signbit = m < 0;
    let theta_sin = theta.sin();

    let fac0 = 1.0 / (2.0 * PI.sqrt());
    let mut fac1 = -3.0_f64.sqrt() / (2.0 * (2.0 * PI).sqrt());
    if m_signbit {
        fac1 = -fac1;
    }
    let (mut r0, mut r1) = (fac0, fac1 * theta_sin.abs());
    let rmm = if m_abs == 0 {
        r0
    } else {
        for k in 2..=m_abs {
            let fac = (((2 * k + 1) * (2 * k - 1)) as f64 / (4 * k * (k - 1)) as f64).sqrt();
            let tmp = fac * theta_sin * theta_sin * r0 + 0.0 * r1;
            r0 = r1;
            r1 = tmp;
        }
        r1
    };

    if n < 0 {
        return f64::NAN;
    }
    if m_abs > n {
        return 0.0;
    }
    let theta_cos = theta.cos();
    let fac = ((2 * m_abs + 3) as f64).sqrt();
    let (mut a, mut b) = (rmm, fac * theta_cos * rmm);
    if n == m_abs {
        return rmm;
    }
    let mf = i64::from(m);
    for j in (m_abs + 2)..=n {
        let fac0 = -(((2 * j + 1) * ((j - 1) * (j - 1) - mf * mf)) as f64
            / ((2 * j - 3) * (j * j - mf * mf)) as f64)
            .sqrt();
        let fac1 = (((2 * j + 1) * (4 * (j - 1) * (j - 1) - 1)) as f64
            / ((2 * j - 3) * (j * j - mf * mf)) as f64)
            .sqrt();
        let tmp = fac0 * a + fac1 * theta_cos * b;
        a = b;
        b = tmp;
    }
    b
}

/// `Q_0(x) = log(|(1 + x) / (1 - x)|) / 2` without cancellation, for real
/// `x != +-1` (`lq0`).
pub(crate) fn lq0(x: f64) -> f64 {
    let ax = x.abs();
    let t = if ax < 1.0 {
        2.0 * ax / (1.0 - ax)
    } else {
        2.0 / (ax - 1.0)
    };
    (t.ln_1p() / 2.0).copysign(x)
}

/// `Q_n(x)` and `Q_n'(x)` for `n = 0..qn.len()` (`lqn`, real argument). At
/// `x = +-1` the values are infinite and the derivatives NaN.
pub(crate) fn lqn(x: f64, qn: &mut [f64], qd: &mut [f64]) {
    let n = qn.len() - 1;
    let eps = 1.0e-16;

    if x.abs() == 1.0 {
        // Deviation from xsf, which stores 1e300: Q_k(1) = +inf and Q_k(-1) = (-1)^(k+1) inf
        // (Q_k(-x) = (-1)^(k+1) Q_k(x)), and the derivative has opposite infinities on the two sides.
        for k in 0..=n {
            qn[k] = if x > 0.0 || k % 2 == 1 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            qd[k] = f64::NAN;
        }
        return;
    }

    if x.abs() <= 1.021 {
        let mut q0 = lq0(x);
        let mut q1 = x * q0 - 1.0;
        qn[0] = q0;
        if n >= 1 {
            qn[1] = q1;
        }
        qd[0] = 1.0 / (1.0 - x * x);
        if n >= 1 {
            qd[1] = qn[0] + x * qd[0];
        }

        for k in 2..=n {
            let kf = k as f64;
            let qf = ((2.0 * kf - 1.0) * x * q1 - (kf - 1.0) * q0) / kf;
            qn[k] = qf;
            qd[k] = (qn[k - 1] - x * qf) * kf / (1.0 - x * x);
            q0 = q1;
            q1 = qf;
        }
    } else {
        let mut qc1 = if n == 0 { 0.0 } else { 1.0 / x };
        let mut qc2 = 1.0 / x;

        for j in 1..=n {
            let jf = j as f64;
            qc2 *= jf / ((2.0 * jf + 1.0) * x);
            if j == n - 1 {
                qc1 = qc2;
            }
        }

        for l in 0..=1 {
            let nl = (n + l) as f64;
            let mut qf = 1.0;
            let mut qr = 1.0;

            // For |x| just above 1.021, the terms decay like 1/x^2 ~ 0.96, so the tail can be tens of
            // times the last term, and in double up to ~3800 terms are needed (at n ~ 1000). Skip the
            // series if its prefactor underflowed, since qf can then overflow.
            let mut k = 1;
            while k <= 5000 && (if l == 0 { qc1 } else { qc2 }) != 0.0 {
                let kf = f64::from(k);
                qr = qr * (0.5 * nl + kf - 1.0) * (0.5 * (nl - 1.0) + kf)
                    / ((nl + kf - 0.5) * kf * x * x);
                qf += qr;
                if (qr / qf).abs() < eps {
                    break;
                }
                k += 1;
            }

            if l == 0 {
                if n >= 1 {
                    qn[n - 1] = qf * qc1;
                }
            } else {
                qn[n] = qf * qc2;
            }
        }

        let mut qf2 = qn[n];
        let mut qf1 = if n >= 1 { qn[n - 1] } else { 0.0 };

        for k in (2..=n).rev() {
            let kf = k as f64;
            let qf0 = ((2.0 * kf - 1.0) * x * qf1 - kf * qf2) / (kf - 1.0);
            qn[k - 2] = qf0;
            qf2 = qf1;
            qf1 = qf0;
        }

        qd[0] = 1.0 / (1.0 - x * x);

        for k in 1..=n {
            qd[k] = k as f64 * (qn[k - 1] - x * qn[k]) / (1.0 - x * x);
        }
    }
}

/// `Q_n^m(x)` and its derivative for `m = 0..=mmax` and `n = 0..=nmax`
/// (`lqmn`, real argument), row-major with `nmax + 1` columns. At `x = +-1`
/// the values are infinite and the derivatives NaN.
pub(crate) fn lqmn(x: f64, mmax: usize, nmax: usize, qm: &mut [f64], qd: &mut [f64]) {
    let m = mmax;
    let n = nmax;
    let w = n + 1;
    let at = |i: usize, j: usize| i * w + j;

    if x.abs() == 1.0 {
        // Deviation from xsf, which stores 1e300: Q_j^i(x) ~ (-1)^i (1 - x)^(-i/2) at x = 1 and
        // Q_j^i(-x) = (-1)^(j+i+1) Q_j^i(x), so the entry is (-1)^i inf at 1 and (-1)^(j+1) inf
        // at -1; the derivative has opposite infinities on the two sides.
        for i in 0..(m + 1) {
            for j in 0..(n + 1) {
                let negative = if x > 0.0 { i % 2 == 1 } else { j % 2 == 0 };
                qm[at(i, j)] = if negative {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                qd[at(i, j)] = f64::NAN;
            }
        }
        return;
    }
    let ls: i32 = if x.abs() > 1.0 { -1 } else { 1 };
    let lsf = f64::from(ls);
    let xs = lsf * (1.0 - x * x);
    let xq = xs.sqrt();
    let q0 = lq0(x);
    if x.abs() < 1.0001 {
        qm[at(0, 0)] = q0;
        if n >= 1 {
            qm[at(0, 1)] = x * q0 - 1.0;
        }
        if m >= 1 {
            qm[at(1, 0)] = -1.0 / xq;
            if n >= 1 {
                qm[at(1, 1)] = -lsf * xq * (q0 + x / (1. - x * x));
            }
        }
        for i in 0..=m.min(1) {
            for j in 2..=n {
                let jf = j as f64;
                qm[at(i, j)] = ((2.0 * jf - 1.) * x * qm[at(i, j - 1)]
                    - (jf + i as f64 - 1.0) * qm[at(i, j - 2)])
                    / (jf - i as f64);
            }
        }
        for i in 2..=m {
            let if_ = i as f64;
            for j in 0..=n {
                let jf = j as f64;
                qm[at(i, j)] = -2.0 * (if_ - 1.0) * x / xq * qm[at(i - 1, j)]
                    - lsf * (jf + if_ - 1.0) * (jf - if_ + 2.0) * qm[at(i - 2, j)];
            }
        }
    } else {
        let km: i64 = if x.abs() > 1.1 {
            (40 + m + n) as i64
        } else {
            // |x| in the logarithm: xsf has log(x - 1), NaN for negative x
            (40 + m + n) as i64 * ((-1. - 1.8 * (x.abs() - 1.).ln()) as i64)
        };
        // rescale by a power of 2, since the recurrences grow like (2 x)^k and overflow for large |x|
        let big = 2.0_f64.powi(512);
        let small = 2.0_f64.powi(-512);
        let mut qf2 = 0.0;
        let mut qf1 = 1.0;
        let mut qf0 = 0.0;
        let mut k = km;
        while k >= 0 {
            let kf = k as f64;
            qf0 = ((2.0 * kf + 3.0) * x * qf1 - (kf + 2.0) * qf2) / (kf + 1.0);
            if k as usize <= n {
                qm[at(0, k as usize)] = qf0;
            }
            if qf0.abs() > big {
                qf0 *= small;
                qf1 *= small;
                for l in (k as usize)..=n {
                    qm[at(0, l)] *= small;
                }
            }
            qf2 = qf1;
            qf1 = qf0;
            k -= 1;
        }

        for k in 0..=n {
            qm[at(0, k)] *= q0 / qf0;
        }

        if m >= 1 {
            qf2 = 0.0;
            qf1 = 1.0;
            let mut k = km;
            while k >= 0 {
                let kf = k as f64;
                qf0 = ((2.0 * kf + 3.0) * x * qf1 - (kf + 1.0) * qf2) / (kf + 2.0);
                if k as usize <= n {
                    qm[at(1, k as usize)] = qf0;
                }
                if qf0.abs() > big {
                    qf0 *= small;
                    qf1 *= small;
                    for l in (k as usize)..=n {
                        qm[at(1, l)] *= small;
                    }
                }
                qf2 = qf1;
                qf1 = qf0;
                k -= 1;
            }

            let q10 = -1.0 / xq;
            for k in 0..=n {
                qm[at(1, k)] *= q10 / qf0;
            }
        }

        for j in 0..=n {
            if m < 2 {
                break;
            }
            let mut q0 = qm[at(0, j)];
            let mut q1 = qm[at(1, j)];
            for i in 0..=(m - 2) {
                let if_ = i as f64;
                let jf = j as f64;
                let qf = -2. * (if_ + 1.) * x / xq * q1 + (jf - if_) * (jf + if_ + 1.) * q0;
                qm[at(i + 2, j)] = qf;
                q0 = q1;
                q1 = qf;
            }
        }
    }

    qd[at(0, 0)] = lsf / xs;
    for j in 1..=n {
        let jf = j as f64;
        qd[at(0, j)] = lsf * jf * (qm[at(0, j - 1)] - x * qm[at(0, j)]) / xs;
    }

    for i in 1..=m {
        let if_ = i as f64;
        for j in 0..=n {
            let jf = j as f64;
            qd[at(i, j)] = lsf * if_ * x / xs * qm[at(i, j)]
                + (if_ + jf) * (jf - if_ + 1.0) / xq * qm[at(i - 1, j)];
        }
    }
}
