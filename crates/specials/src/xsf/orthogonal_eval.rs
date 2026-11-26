//! Classical orthogonal polynomials for `f64`: Jacobi, Gegenbauer, Chebyshev
//! (`T`, `U`, `C`, `S` and their shifted forms), Legendre, Laguerre,
//! generalized Laguerre and Hermite.
//!
//! Ported from xsf (scipy/xsf at `31177fd9a458539cf05c04d215341245d0245e53`)
//! `include/xsf/orthogonal_eval.h` and
//! `include/xsf/cpu/orthogonal_eval.h`, the real overloads: the `Int`
//! templates (`*_l`, integer degree, here `i64`) and the `double` degree
//! overloads, which evaluate a terminating hypergeometric series with
//! [`hyp2f1`](super::hyp2f1) (Cephes, real argument) and
//! [`hyp1f1`](super::specfun) (Zhang and Jin). The integer-degree code is by
//! Pauli Virtanen and the SciPy developers (Copyright 2024 SciPy developers).
//!
//! The complex and `float` overloads are not ported. xsf reports a domain
//! error (`sf_error`) with a NaN result; this port returns the same NaN.

use super::beta::beta;
use super::binom::binom;
use super::gamma::gamma;
use super::hyp2f1::hyp2f1;
use super::specfun::hyp1f1;

/// `binom(n + alpha, n) = 1 / (n B(alpha + 1, n))` (DLMF 5.12.1), keeping the
/// digits of `alpha + 1` for `alpha` near -1.
fn binom_alpha(n: f64, alpha: f64) -> f64 {
    if alpha < -0.5 && n >= 1.0 {
        return 1.0 / n / beta(alpha + 1.0, n);
    }
    binom(n + alpha, n)
}

/// `P_n^(alpha, beta)(x)` for a real degree `n`.
pub(crate) fn eval_jacobi(n: f64, alpha: f64, beta_: f64, x: f64) -> f64 {
    if alpha == -1.0 && beta_.abs() == 1.0 {
        if n == 0.0 {
            return 1.0;
        } else if n == 1.0 {
            return 0.5 * (1.0 + beta_) * (x - 1.0);
        } else if n > 1.0 {
            return ((n + beta_) / (2.0 * n)) * (x - 1.0) * eval_jacobi(n - 1.0, 1.0, beta_, x);
        }
    }

    let d = binom(n + alpha, n);
    let a = -n;
    let b = n + alpha + beta_ + 1.0;
    let c = alpha + 1.0;
    let g = 0.5 * (1.0 - x);
    d * hyp2f1(a, b, c, g)
}

/// `P_n^(alpha, beta)(x)` for an integer degree `n`.
pub(crate) fn eval_jacobi_l(n: i64, alpha: f64, beta_: f64, x: f64) -> f64 {
    let nf = n as f64;
    if n < 0 {
        eval_jacobi(nf, alpha, beta_, x)
    } else if n == 0 {
        1.0
    } else if n == 1 {
        0.5 * (2.0 * (alpha + 1.0) + (alpha + beta_ + 2.0) * (x - 1.0))
    } else if alpha == -1.0 && beta_.abs() == 1.0 {
        ((nf + beta_) / (2.0 * nf)) * (x - 1.0) * eval_jacobi(nf - 1.0, 1.0, beta_, x)
    } else {
        let mut d = (alpha + beta_ + 2.0) * (x - 1.0) / (2.0 * (alpha + 1.0));
        let mut p = d + 1.0;
        for kk in 0..n - 1 {
            let k = kk as f64 + 1.0;
            let t = 2.0 * k + alpha + beta_;
            d = ((t * (t + 1.0) * (t + 2.0)) * (x - 1.0) * p
                + 2.0 * k * (k + beta_) * (t + 2.0) * d)
                / (2.0 * (k + alpha + 1.0) * (k + alpha + beta_ + 1.0) * t);
            p += d;
        }
        binom(nf + alpha, nf) * p
    }
}

/// Shifted Jacobi polynomial `G_n(p, q, x)`, real degree.
pub(crate) fn eval_sh_jacobi(n: f64, p: f64, q: f64, x: f64) -> f64 {
    eval_jacobi(n, p - q, q - 1.0, 2.0 * x - 1.0) / binom(2.0 * n + p - 1.0, n)
}

/// Shifted Jacobi polynomial `G_n(p, q, x)`, integer degree.
pub(crate) fn eval_sh_jacobi_l(n: i64, p: f64, q: f64, x: f64) -> f64 {
    eval_jacobi_l(n, p - q, q - 1.0, 2.0 * x - 1.0) / binom(2.0 * n as f64 + p - 1.0, n as f64)
}

/// Probabilists' Hermite polynomial `He_n(x)`.
pub(crate) fn eval_hermitenorm(n: i64, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }

    if n < 0 {
        return f64::NAN;
    } else if n == 0 {
        return 1.0;
    } else if n == 1 {
        return x;
    }

    let mut y3 = 0.0;
    let mut y2 = 1.0;
    let mut k = n;
    while k > 1 {
        let y1 = x * y2 - k as f64 * y3;
        y3 = y2;
        y2 = y1;
        k -= 1;
    }
    x * y2 - y3
}

/// Physicists' Hermite polynomial `H_n(x)`.
pub(crate) fn eval_hermite(n: i64, x: f64) -> f64 {
    if n < 0 {
        return f64::NAN;
    }
    eval_hermitenorm(n, 2.0_f64.sqrt() * x) * 2.0_f64.powf(n as f64 / 2.0)
}

/// `L_n^(alpha)(x)` for an integer degree: forward recurrence on the
/// normalized values `p_k = L_k^(alpha)(x) / binom(k + alpha, k)` and their
/// differences `d_k = p_k - p_(k-1)`:
/// `d_(k+1) = (k d_k - x p_k) / (k + alpha + 1)`, `p_(k+1) = p_k + d_(k+1)`,
/// starting from `p_0 = 1`, `d_0 = 0`.
pub(crate) fn eval_genlaguerre_l(n: i64, alpha: f64, x: f64) -> f64 {
    if alpha <= -1.0 {
        return f64::NAN;
    }

    if alpha.is_nan() || x.is_nan() {
        return f64::NAN;
    }

    if n < 0 {
        return 0.0;
    } else if n == 0 {
        return 1.0;
    } else if n == 1 {
        return -x + alpha + 1.0;
    }

    let mut d = -x / (alpha + 1.0);
    let mut p = d + 1.0;
    for kk in 0..n - 1 {
        let k = kk as f64 + 1.0;
        d = -x / (k + alpha + 1.0) * p + (k / (k + alpha + 1.0)) * d;
        p += d;
    }
    binom_alpha(n as f64, alpha) * p
}

/// `L_n^(alpha)(x)` for a real degree.
pub(crate) fn eval_genlaguerre(n: f64, alpha: f64, x: f64) -> f64 {
    if alpha <= -1.0 {
        return f64::NAN;
    }

    let d = binom_alpha(n, alpha);
    d * hyp1f1(-n, alpha + 1.0, x)
}

/// `C_n^(alpha)(x)` for an integer degree. For `|x| >= 1e-5`, forward
/// recurrence on the normalized values `p_k = C_k^(alpha)(x) / binom(k + 2
/// alpha - 1, k)` and their differences `d_k = p_k - p_(k-1)`:
/// `d_(k+1) = 2 (k + alpha) / (k + 2 alpha) (x - 1) p_k + k / (k + 2 alpha)
/// d_k`, `p_(k+1) = p_k + d_(k+1)`, starting from `p_1 = x`, `d_1 = x - 1`.
/// For `|x| < 1e-5`, the explicit power series in `x` is used instead.
pub(crate) fn eval_gegenbauer_l(n: i64, alpha: f64, mut x: f64) -> f64 {
    if alpha.is_nan() || x.is_nan() {
        return f64::NAN;
    }

    if alpha == 0.0 {
        return 0.0;
    }
    if n < 0 {
        return 0.0;
    } else if n == 0 {
        return 1.0;
    } else if n == 1 {
        return 2.0 * alpha * x;
    }

    // Symmetry property: C_n^alpha(-x) = (-1)^n C_n^alpha(x) for integer n
    let sign = if x < 0.0 && n % 2 != 0 { -1.0 } else { 1.0 };
    x = x.abs();

    let nf = n as f64;
    if x < 1e-5 {
        // Power series rather than recurrence due to loss of precision
        // http://functions.wolfram.com/Polynomials/GegenbauerC3/02/
        let a = n / 2;
        let af = a as f64;

        let mut d = if a % 2 == 0 { 1.0 } else { -1.0 };
        d /= beta(alpha, 1.0 + af);
        if n == 2 * a {
            d /= af + alpha;
        } else {
            d *= 2.0 * x;
        }

        let mut p = 0.0;
        for kk in 0..=a {
            let kkf = kk as f64;
            p += d;
            d *= -4.0 * x * x * (af - kkf) * (alpha - af + kkf + nf)
                / ((nf + 1.0 - 2.0 * af + 2.0 * kkf) * (nf + 2.0 - 2.0 * af + 2.0 * kkf));
            if d.abs() < 1e-20 * p.abs() {
                // converged
                break;
            }
        }
        return sign * p;
    }

    let mut d = x - 1.0;
    let mut p = x;
    for kk in 0..n - 1 {
        let k = kk as f64 + 1.0;
        d = (2.0 * (k + alpha) / (k + 2.0 * alpha)) * (x - 1.0) * p + (k / (k + 2.0 * alpha)) * d;
        p += d;
    }

    if (alpha / nf).abs() < 1e-8 {
        // avoid loss of precision
        return sign * 2.0 * alpha / nf * p;
    }
    sign * binom(nf + 2.0 * alpha - 1.0, nf) * p
}

/// `C_n^(alpha)(x)` for a real degree (the integer-valued ones below 1e10 take
/// the recurrence).
pub(crate) fn eval_gegenbauer(n: f64, alpha: f64, x: f64) -> f64 {
    // If n is an integer, use more stable `eval_gegenbauer_l`
    if (0.0..1e10).contains(&n) && n == (n as i64) as f64 {
        return eval_gegenbauer_l(n as i64, alpha, x);
    }

    // Gamma(n + 2 alpha) / Gamma(n + 1) / Gamma(2 alpha) = 1 / (n B(2 alpha, n)) (DLMF 5.12.1), avoiding overflow
    // of the Gamma factors; at the poles of Gamma(2 alpha) (2 alpha = 0, -1, ...) the Gamma form is kept.
    let alpha2 = 2.0 * alpha;
    let d = if n >= 1.0 && (alpha2 > 0.0 || alpha2 != alpha2.floor()) {
        1.0 / n / beta(alpha2, n)
    } else {
        gamma(n + alpha2) / gamma(1.0 + n) / gamma(alpha2)
    };
    let a = -n;
    let b = n + 2.0 * alpha;
    let c = alpha + 0.5;
    let g = (1.0 - x) / 2.0;
    d * hyp2f1(a, b, c, g)
}

/// `P_n(x)` for a real degree.
pub(crate) fn eval_legendre(n: f64, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    hyp2f1(-n, n + 1.0, 1.0, 0.5 * (1.0 - x))
}

/// `P_n(x)` for an integer degree: forward recurrence on the values
/// `p_k = P_k(x)` and their differences `d_k = p_k - p_(k-1)`:
/// `d_(k+1) = ((2k + 1) (x - 1) p_k + k d_k) / (k + 1)`,
/// `p_(k+1) = p_k + d_(k+1)`, starting from `p_1 = x`, `d_1 = x - 1`. Near
/// `x = 0` an explicit power series is used instead.
pub(crate) fn eval_legendre_l(mut n: i64, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }

    if n < 0 {
        // P_{-n-1}(x) = P_n(x); add before negating to avoid signed overflow.
        n = -(n + 1);
    }

    if n == 0 {
        return 1.0;
    } else if n == 1 {
        return x;
    } else if x.abs() < 1e-5 {
        // Power series rather than recurrence due to loss of precision
        // http://functions.wolfram.com/Polynomials/LegendreP/02/
        let a = n / 2;
        let af = a as f64;
        let nf = n as f64;
        let mut d = if a % 2 == 0 { 1.0 } else { -1.0 };
        if n % 2 == 0 {
            d *= -2.0 / beta(af + 1.0, -0.5);
        } else {
            d *= 2.0 * x / beta(af + 1.0, 0.5);
        }

        let mut p = 0.0;
        for kk in 0..=a {
            let kkf = kk as f64;
            p += d;
            d *= -2.0 * x * x * (af - kkf) * (2.0 * nf + 1.0 - 2.0 * af + 2.0 * kkf)
                / ((nf + 1.0 - 2.0 * af + 2.0 * kkf) * (nf + 2.0 - 2.0 * af + 2.0 * kkf));
            if d.abs() < 1e-20 * p.abs() {
                // converged
                break;
            }
        }
        return p;
    }

    let mut d = x - 1.0;
    let mut p = x;
    for kk in 0..n - 1 {
        let k = kk as f64 + 1.0;
        d = ((2.0 * k + 1.0) / (k + 1.0)) * (x - 1.0) * p + (k / (k + 1.0)) * d;
        p += d;
    }
    p
}

/// `T_n(x)` for a real degree.
pub(crate) fn eval_chebyt(n: f64, x: f64) -> f64 {
    hyp2f1(-n, n, 0.5, 0.5 * (1.0 - x))
}

/// `T_n(x)` for an integer degree: the Chebyshev `T` recurrence directly, see
/// Mason & Handscomb, Chebyshev Polynomials, CRC Press (2003).
pub(crate) fn eval_chebyt_l(mut k: i64, mut x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }

    if k < 0 {
        // symmetry: T_{-k}(x) = T_k(x)
        k = -k;
    }

    let mut b2 = 0.0;
    let mut b1 = -1.0;
    let mut b0 = 0.0;
    x *= 2.0;
    for _ in 0..k + 1 {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2;
    }
    (b0 - b2) / 2.0
}

/// `U_n(x)` for a real degree.
pub(crate) fn eval_chebyu(n: f64, x: f64) -> f64 {
    (n + 1.0) * hyp2f1(-n, n + 2.0, 1.5, 0.5 * (1.0 - x))
}

/// `U_n(x)` for an integer degree.
pub(crate) fn eval_chebyu_l(mut k: i64, mut x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }

    let mut sign = 1.0;
    if k < 0 {
        if k == -1 {
            return 0.0;
        }
        // symmetry: U_{-k-2}(x) = -U_k(x)
        k = -k - 2;
        sign = -1.0;
    }

    let mut b1 = -1.0;
    let mut b0 = 0.0;
    x *= 2.0;
    for _ in 0..k + 1 {
        let b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2;
    }
    b0 * sign
}

/// `S_n(x) = U_n(x/2)`, real degree.
pub(crate) fn eval_chebys(n: f64, x: f64) -> f64 {
    eval_chebyu(n, 0.5 * x)
}

/// `S_n(x) = U_n(x/2)`, integer degree.
pub(crate) fn eval_chebys_l(n: i64, x: f64) -> f64 {
    eval_chebyu_l(n, 0.5 * x)
}

/// `C_n(x) = 2 T_n(x/2)`, real degree.
pub(crate) fn eval_chebyc(n: f64, x: f64) -> f64 {
    2.0 * eval_chebyt(n, 0.5 * x)
}

/// `C_n(x) = 2 T_n(x/2)`, integer degree.
pub(crate) fn eval_chebyc_l(n: i64, x: f64) -> f64 {
    2.0 * eval_chebyt_l(n, 0.5 * x)
}

/// `T*_n(x) = T_n(2x - 1)`, real degree.
pub(crate) fn eval_sh_chebyt(n: f64, x: f64) -> f64 {
    eval_chebyt(n, 2.0 * x - 1.0)
}

/// `T*_n(x) = T_n(2x - 1)`, integer degree.
pub(crate) fn eval_sh_chebyt_l(n: i64, x: f64) -> f64 {
    eval_chebyt_l(n, 2.0 * x - 1.0)
}

/// `U*_n(x) = U_n(2x - 1)`, real degree.
pub(crate) fn eval_sh_chebyu(n: f64, x: f64) -> f64 {
    eval_chebyu(n, 2.0 * x - 1.0)
}

/// `U*_n(x) = U_n(2x - 1)`, integer degree.
pub(crate) fn eval_sh_chebyu_l(n: i64, x: f64) -> f64 {
    eval_chebyu_l(n, 2.0 * x - 1.0)
}

/// `P*_n(x) = P_n(2x - 1)`, real degree.
pub(crate) fn eval_sh_legendre(n: f64, x: f64) -> f64 {
    eval_legendre(n, 2.0 * x - 1.0)
}

/// `P*_n(x) = P_n(2x - 1)`, integer degree.
pub(crate) fn eval_sh_legendre_l(n: i64, x: f64) -> f64 {
    eval_legendre_l(n, 2.0 * x - 1.0)
}
