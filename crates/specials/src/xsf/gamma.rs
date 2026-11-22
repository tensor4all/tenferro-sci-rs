//! Γ, ln|Γ|, sign of Γ and 1/Γ for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/gamma.h` (`Gamma`, `lgam_sgn`,
//! `gammasgn`), `include/xsf/cephes/rgamma.h` (`rgamma`),
//! `include/xsf/loggamma.h` (real `loggamma` and `loggamma_taylor`) and
//! `include/xsf/gamma.h` (`gammaln`).
//! Cephes Math Library Release 2.2, Copyright 1984, 1987, 1989, 1992 by
//! Stephen L. Moshier; C++ translation by SciPy developers (2024).

use super::consts::{LOGPI, MAXGAM, SCIPY_EULER, SQRT2PI};
use super::polevl::{chbevl, p1evl, polevl};
use super::sinpi;

const GAMMA_P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];

const GAMMA_Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];

const GAMMA_STIR: [f64; 5] = [
    7.87311395793093628397E-4,
    -2.29549961613378126380E-4,
    -2.68132617805781232825E-3,
    3.47222221605458667310E-3,
    8.33333333333482257126E-2,
];

const MAXSTIR: f64 = 143.01608;

/// Γ by Stirling's formula, valid for 33 <= x <= 172.
fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return f64::INFINITY;
    }
    let mut w = 1.0 / x;
    w = 1.0 + w * polevl(w, &GAMMA_STIR);
    let mut y = x.exp();
    if x > MAXSTIR {
        // Avoid overflow in pow().
        let v = x.powf(0.5 * x - 0.25);
        y = v * (v / y);
    } else {
        y = x.powf(x - 0.5) / y;
    }
    SQRT2PI * y * w
}

pub(crate) fn gamma(mut x: f64) -> f64 {
    if !x.is_finite() {
        // gamma(+inf) = +inf; gamma(NaN) and gamma(-inf) are NaN.
        return if x > 0.0 { x } else { f64::NAN };
    }
    if x == 0.0 {
        // The pole at zero takes the sign of the zero.
        return f64::INFINITY.copysign(x);
    }

    let q = x.abs();
    if q > 33.0 {
        let mut sgngam = 1.0;
        let z = if x < 0.0 {
            let mut p = q.floor();
            if p == q {
                // Negative integer: a pole.
                return f64::NAN;
            }
            if p % 2.0 == 0.0 {
                sgngam = -1.0;
            }
            let mut z = q - p;
            if z > 0.5 {
                p += 1.0;
                z = q - p;
            }
            z = q * sinpi(z);
            if z == 0.0 {
                return sgngam * f64::INFINITY;
            }
            core::f64::consts::PI / (z.abs() * stirf(q))
        } else {
            stirf(x)
        };
        return sgngam * z;
    }

    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1.0e-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0e-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    z * polevl(x, &GAMMA_P) / polevl(x, &GAMMA_Q)
}

fn small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        // x started as a negative integer.
        f64::NAN
    } else {
        z / ((1.0 + SCIPY_EULER * x) * x)
    }
}

const GAMMA_A: [f64; 5] = [
    8.11614167470508450300E-4,
    -5.95061904284301438324E-4,
    7.93650340457716943945E-4,
    -2.77777777730099687205E-3,
    8.33333333333331927722E-2,
];

const GAMMA_B: [f64; 6] = [
    -1.37825152569120859100E3,
    -3.88016315134637840924E4,
    -3.31612992738871184744E5,
    -1.16237097492762307383E6,
    -1.72173700820839662146E6,
    -8.53555664245765465627E5,
];

const GAMMA_C: [f64; 6] = [
    -3.51815701436523470549E2,
    -1.70642106651881159223E4,
    -2.20528590553854454839E5,
    -1.13933444367982507207E6,
    -2.53252307177582951285E6,
    -2.01889141433532773231E6,
];

/// `ln √(2π)`.
const LS2PI: f64 = 0.91893853320467274178;
const MAXLGM: f64 = 2.556348e305;

fn lgam_large_x(x: f64) -> f64 {
    let q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1.0e8 {
        return q;
    }
    let p = 1.0 / (x * x);
    let p = ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
        + 0.0833333333333333333333)
        / x;
    q + p
}

/// `(ln|Γ(x)|, sign of Γ(x))`.
pub(crate) fn lgam_sgn(mut x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (x, 1.0);
    }

    if x < -34.0 {
        let q = -x;
        let (w, _) = lgam_sgn(q);
        let mut p = q.floor();
        if p == q {
            return (f64::INFINITY, 1.0);
        }
        let sign = if p % 2.0 == 0.0 { -1.0 } else { 1.0 };
        let mut z = q - p;
        if z > 0.5 {
            p += 1.0;
            z = p - q;
        }
        z = q * sinpi(z);
        if z == 0.0 {
            return (f64::INFINITY, sign);
        }
        return (LOGPI - z.ln() - w, sign);
    }

    if x != 0.0 && x.abs() < f64::MIN_POSITIVE {
        // DLMF 5.7.3: ln|Γ(x)| = -ln|x| - γx + ..., and γx < eps.
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        return (-x.abs().ln(), sign);
    }

    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return (f64::INFINITY, 1.0);
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        let sign = if z < 0.0 {
            z = -z;
            -1.0
        } else {
            1.0
        };
        if u == 2.0 {
            return (z.ln(), sign);
        }
        p -= 2.0;
        x += p;
        let p = x * polevl(x, &GAMMA_B) / p1evl(x, &GAMMA_C);
        return (z.ln() + p, sign);
    }

    if x > MAXLGM {
        return (f64::INFINITY, 1.0);
    }
    if x >= 1000.0 {
        return (lgam_large_x(x), 1.0);
    }
    let q = (x - 0.5) * x.ln() - x + LS2PI;
    let p = 1.0 / (x * x);
    (q + polevl(p, &GAMMA_A) / x, 1.0)
}

pub(crate) fn gammasgn(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0_f64.copysign(x);
    }
    if x.is_infinite() {
        return f64::NAN;
    }
    let fx = x.floor();
    if x - fx == 0.0 {
        return f64::NAN;
    }
    // The sign of Γ on (-n, -n+1) is (-1)^n.
    if fx % 2.0 != 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Chebyshev coefficients of `1/(x Γ(x)) - 1` on [0, 1].
const RGAMMA_R: [f64; 16] = [
    3.13173458231230000000E-17,
    -6.70718606477908000000E-16,
    2.20039078172259550000E-15,
    2.47691630348254132600E-13,
    -6.60074100411295197440E-12,
    5.13850186324226978840E-11,
    1.08965386454418662084E-9,
    -3.33964630686836942556E-8,
    2.68975996440595483619E-7,
    2.96001177518801696639E-6,
    -8.04814124978471142852E-5,
    4.16609138709688864714E-4,
    5.06579864028608725080E-3,
    -6.41925436109158228810E-2,
    -4.98558728684003594785E-3,
    1.27546015610523951063E-1,
];

pub(crate) fn rgamma(x: f64) -> f64 {
    if x.abs() < f64::MIN_POSITIVE {
        // 1/Γ(x) = x + γx² + ... rounds to x (DLMF 5.7.1), keeping the sign of zero.
        return x;
    }
    if x < 0.0 && x == x.floor() {
        // Poles of Γ.
        return 0.0;
    }
    if -178.0 < x && x < -170.0 {
        // Γ(x) underflows here, so use 1/Γ(x) = x/Γ(x + 1).
        return x * rgamma(x + 1.0);
    }
    if x.abs() > 4.0 {
        return 1.0 / gamma(x);
    }

    let mut z = 1.0;
    let mut w = x;
    while w > 1.0 {
        w -= 1.0;
        z *= w;
    }
    while w < 0.0 {
        z /= w;
        w += 1.0;
    }
    if w == 0.0 {
        return 0.0;
    }
    if w == 1.0 {
        return 1.0 / z;
    }
    w * (1.0 + chbevl(4.0 * w - 2.0, &RGAMMA_R)) / z
}

const LOGGAMMA_TAYLOR_RADIUS: f64 = 0.2;

/// Taylor series for log-Gamma around z = 1:
/// `loggamma(z + 1) = -gamma*z + zeta(2)*z**2/2 - zeta(3)*z**3/3 ...`
fn loggamma_taylor(mut z: f64) -> f64 {
    const COEFFS: [f64; 23] = [
        -4.3478266053040259361E-2,
        4.5454556293204669442E-2,
        -4.7619070330142227991E-2,
        5.000004769810169364E-2,
        -5.2631679379616660734E-2,
        5.5555767627403611102E-2,
        -5.8823978658684582339E-2,
        6.2500955141213040742E-2,
        -6.6668705882420468033E-2,
        7.1432946295361336059E-2,
        -7.6932516411352191473E-2,
        8.3353840546109004025E-2,
        -9.0954017145829042233E-2,
        1.0009945751278180853E-1,
        -1.1133426586956469049E-1,
        1.2550966952474304242E-1,
        -1.4404989676884611812E-1,
        1.6955717699740818995E-1,
        -2.0738555102867398527E-1,
        2.7058080842778454788E-1,
        -4.0068563438653142847E-1,
        8.2246703342411321824E-1,
        -5.7721566490153286061E-1,
    ];
    z -= 1.0;
    z * polevl(z, &COEFFS)
}

/// Real `loggamma` of `xsf/loggamma.h`: NaN for `x < 0`.
fn loggamma(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    // lgam loses relative accuracy near its zeros at 1 and 2
    if (x - 1.0).abs() < LOGGAMMA_TAYLOR_RADIUS {
        // `+ 0.0` keeps +0 at x = 1
        return loggamma_taylor(x) + 0.0;
    }
    if 2.0 - LOGGAMMA_TAYLOR_RADIUS < x && x < 2.0 {
        return (x - 1.0).ln() + loggamma_taylor(x - 1.0);
    }
    lgam_sgn(x).0
}

/// `gammaln` of `xsf/gamma.h`.
pub(crate) fn gammaln(x: f64) -> f64 {
    if x < 0.0 {
        lgam_sgn(x).0
    } else {
        loggamma(x)
    }
}
