//! Complete and incomplete elliptic integrals and Jacobian elliptic functions
//! for `f64`.
//!
//! Ported from xsf `include/xsf/ellip.h` (the public wrappers `ellipk`,
//! `ellipkm1`, `ellipe`, `ellipkinc`, `ellipeinc`, `ellipj`) and from
//! `include/xsf/cephes/ellpk.h`, `ellpe.h`, `ellik.h`, `ellie.h` and
//! `ellpj.h`.
//! Cephes Math Library, Copyright 1984, 1987, 1989, 1993 by Stephen L.
//! Moshier; the negative-`m` branches of `ellik` and `ellie` are Copyright
//! 2014, Eric W. Moore; C++ translation by SciPy developers (2024).
//!
//! Deviation from upstream: `1 << 2*n` in the Carlson-style iterations of
//! `ellik_neg_m` and `ellie_neg_m` is `4^n` computed in `f64`, which is
//! identical where the C shift is defined.

use core::f64::consts::{FRAC_PI_2, PI};

use super::consts::MACHEP;
use super::polevl::polevl;
use super::unity::cosm1;

const ELLPK_P: [f64; 11] = [
    1.37982864606273237150E-4,
    2.28025724005875567385E-3,
    7.97404013220415179367E-3,
    9.85821379021226008714E-3,
    6.87489687449949877925E-3,
    6.18901033637687613229E-3,
    8.79078273952743772254E-3,
    1.49380448916805252718E-2,
    3.08851465246711995998E-2,
    9.65735902811690126535E-2,
    1.38629436111989062502E0,
];

const ELLPK_Q: [f64; 11] = [
    2.94078955048598507511E-5,
    9.14184723865917226571E-4,
    5.94058303753167793257E-3,
    1.54850516649762399335E-2,
    2.39089602715924892727E-2,
    3.01204715227604046988E-2,
    3.73774314173823228969E-2,
    4.88280347570998239232E-2,
    7.03124996963957469739E-2,
    1.24999999999870820058E-1,
    4.99999999999999999821E-1,
];

/// log(4)
const ELLPK_C1: f64 = 1.3862943611198906188E0;

const ELLPE_P: [f64; 11] = [
    1.53552577301013293365E-4,
    2.50888492163602060990E-3,
    8.68786816565889628429E-3,
    1.07350949056076193403E-2,
    7.77395492516787092951E-3,
    7.58395289413514708519E-3,
    1.15688436810574127319E-2,
    2.18317996015557253103E-2,
    5.68051945617860553470E-2,
    4.43147180560990850618E-1,
    1.00000000000000000299E0,
];

const ELLPE_Q: [f64; 10] = [
    3.27954898576485872656E-5,
    1.00962792679356715133E-3,
    6.50609489976927491433E-3,
    1.68862163993311317300E-2,
    2.61769742454493659583E-2,
    3.34833904888224918614E-2,
    4.27180926518931511717E-2,
    5.85936634471101055642E-2,
    9.37499997197644278445E-2,
    2.49999999999888314361E-1,
];

/// K(1 - x), the complete elliptic integral of the first kind with parameter
/// `m = 1 - x` (`cephes/ellpk.h`).
pub(crate) fn ellpk(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x > 1.0 {
        if x.is_infinite() {
            return 0.0;
        }
        return ellpk(1.0 / x) / x.sqrt();
    }
    if x > MACHEP {
        polevl(x, &ELLPK_P) - x.ln() * polevl(x, &ELLPK_Q)
    } else if x == 0.0 {
        f64::INFINITY
    } else {
        ELLPK_C1 - 0.5 * x.ln()
    }
}

/// E(1 - x), the complete elliptic integral of the second kind with
/// parameter `m = 1 - x` (`cephes/ellpe.h`, called with `m` in xsf's wrapper).
pub(crate) fn ellpe(x: f64) -> f64 {
    let x = 1.0 - x;
    if x <= 0.0 {
        if x == 0.0 {
            return 1.0;
        }
        return f64::NAN;
    }
    if x > 1.0 {
        return ellpe(1.0 - 1.0 / x) * x.sqrt();
    }
    polevl(x, &ELLPE_P) - x.ln() * (x * polevl(x, &ELLPE_Q))
}

/// `4^n` as an exact `f64` (C: `1 << 2*n`).
fn pow4(n: i32) -> f64 {
    4.0_f64.powi(n)
}

/// F(phi, m) for `m < 0` and `0 <= phi <= pi/2` (`cephes/ellik.h`,
/// `ellik_neg_m`): a power series for small `m phi^2`, an asymptotic series
/// for large `m phi^2`, and Carlson's R_F otherwise (Carlson 1994,
/// <https://arxiv.org/abs/math/9409227>).
fn ellik_neg_m(phi: f64, m: f64) -> f64 {
    let mut n = 0;
    let mpp = (m * phi) * phi;

    if -mpp < 1e-6 && phi < -m {
        return phi + (-mpp * phi * phi / 30.0 + 3.0 * mpp * mpp / 40.0 + mpp / 6.0) * phi;
    }

    if -mpp > 4e7 {
        let sm = (-m).sqrt();
        let sp = phi.sin();
        let cp = phi.cos();

        let a = (4.0 * sp * sm / (1.0 + cp)).ln();
        let b = -(1.0 + cp / sp / sp - a) / 4.0 / m;
        return (a + b) / sm;
    }

    let (scale, x, y, z);
    if phi > 1e-153 && m > -1e305 {
        let s = phi.sin();
        let csc2 = 1.0 / (s * s);
        scale = 1.0;
        x = 1.0 / (phi.tan() * phi.tan());
        y = csc2 - m;
        z = csc2;
    } else {
        scale = phi;
        x = 1.0;
        y = 1.0 - m * scale * scale;
        z = 1.0;
    }

    if x == y && x == z {
        return scale / x.sqrt();
    }

    let a0 = (x + y + z) / 3.0;
    let mut a = a0;
    let (mut x1, mut y1, mut z1) = (x, y, z);
    // Carlson gives 1/pow(3*r, 1.0/6.0) for this constant. if r == eps, it is ~338.38.
    let mut q = 400.0 * (a0 - x).abs().max((a0 - y).abs().max((a0 - z).abs()));

    while q > a.abs() && n <= 100 {
        let sx = x1.sqrt();
        let sy = y1.sqrt();
        let sz = z1.sqrt();
        let lam = sx * sy + sx * sz + sy * sz;
        x1 = (x1 + lam) / 4.0;
        y1 = (y1 + lam) / 4.0;
        z1 = (z1 + lam) / 4.0;
        a = (x1 + y1 + z1) / 3.0;
        n += 1;
        q /= 4.0;
    }
    let xx = (a0 - x) / a / pow4(n);
    let yy = (a0 - y) / a / pow4(n);
    let zz = -(xx + yy);

    let e2 = xx * yy - zz * zz;
    let e3 = xx * yy * zz;

    scale * (1.0 - e2 / 10.0 + e3 / 14.0 + e2 * e2 / 24.0 - 3.0 * e2 * e3 / 44.0) / a.sqrt()
}

/// F(phi | m), the incomplete elliptic integral of the first kind
/// (`cephes/ellik.h`, `ellik`).
pub(crate) fn ellik(mut phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        return f64::NAN;
    }
    if phi.is_infinite() || m.is_infinite() {
        if m.is_infinite() && phi.is_finite() {
            return 0.0;
        } else if phi.is_infinite() && m.is_finite() {
            return phi;
        } else {
            return f64::NAN;
        }
    }
    if m == 0.0 {
        return phi;
    }
    let mut a = 1.0 - m;
    if a == 0.0 {
        if phi.abs() >= FRAC_PI_2 {
            return f64::INFINITY;
        }
        // DLMF 19.6.8, and 4.23.42
        return phi.tan().asinh();
    }
    // From about 2^53 on, the reduction below loses the residual.
    // F(phi, m) = 2 phi K(m) / pi + O(K(m)), DLMF 19.2.10
    if phi.abs() > 9e15 {
        return phi / FRAC_PI_2 * ellpk(a);
    }
    let mut npio2 = (phi / FRAC_PI_2).floor();
    if (npio2.abs() % 2.0) == 1.0 {
        npio2 += 1.0;
    }
    let mut k;
    if npio2 != 0.0 {
        k = ellpk(a);
        phi -= npio2 * FRAC_PI_2;
    } else {
        k = 0.0;
    }
    let sign;
    if phi < 0.0 {
        phi = -phi;
        sign = -1;
    } else {
        sign = 0;
    }
    let mut temp;
    'done: {
        if a > 1.0 {
            temp = ellik_neg_m(phi, m);
            break 'done;
        }
        let mut b = a.sqrt();
        let mut t = phi.tan();
        if t.abs() > 10.0 {
            // Transform the amplitude
            let e = 1.0 / (b * t);
            // ... but avoid multiple recursions.
            if e.abs() < 10.0 {
                let e = e.atan();
                if npio2 == 0.0 {
                    k = ellpk(a);
                }
                temp = k - ellik(e, m);
                break 'done;
            }
        }
        a = 1.0;
        let mut c = m.sqrt();
        let mut d = 1_i32;
        let mut modulus = 0_i32;

        while (c / a).abs() > MACHEP {
            temp = b / a;
            phi = phi + (t * temp).atan() + f64::from(modulus) * PI;
            let denom = 1.0 - temp * t * t;
            if denom.abs() > 10.0 * MACHEP {
                t = t * (1.0 + temp) / denom;
                modulus = ((phi + FRAC_PI_2) / PI) as i32;
            } else {
                t = phi.tan();
                modulus = ((phi - t.atan()) / PI).floor() as i32;
            }
            c = (a - b) / 2.0;
            temp = (a * b).sqrt();
            a = (a + b) / 2.0;
            b = temp;
            d += d;
        }

        temp = (t.atan() + f64::from(modulus) * PI) / (f64::from(d) * a);
    }
    if sign < 0 {
        temp = -temp;
    }
    temp += npio2 * k;
    temp
}

/// E(phi, m) for `m < 0` and `0 <= phi <= pi/2` (`cephes/ellie.h`,
/// `ellie_neg_m`): a power series for small `m phi^2`, an asymptotic series
/// for large `m phi^2`, and Carlson's R_F and R_D otherwise (Carlson 1994,
/// <https://arxiv.org/abs/math/9409227>).
fn ellie_neg_m(phi: f64, m: f64) -> f64 {
    let mut n = 0;
    let mpp = (m * phi) * phi;

    if -mpp < 1e-6 && phi < -m {
        return phi + (mpp * phi * phi / 30.0 - mpp * mpp / 40.0 - mpp / 6.0) * phi;
    }

    if -mpp > 1e6 {
        let sm = (-m).sqrt();
        let sp = phi.sin();
        let cp = phi.cos();

        let a = -cosm1(phi);
        let b1 = (4.0 * sp * sm / (1.0 + cp)).ln();
        let b = -(0.5 + b1) / 2.0 / m;
        let c = (0.75 + cp / sp / sp - b1) / 16.0 / m / m;
        return (a + b + c) * sm;
    }

    let (scalef, scaled, x, y, z);
    if phi > 1e-153 && m > -1e200 {
        let s = phi.sin();
        let csc2 = 1.0 / s / s;
        scalef = 1.0;
        scaled = m / 3.0;
        x = 1.0 / phi.tan() / phi.tan();
        y = csc2 - m;
        z = csc2;
    } else {
        scalef = phi;
        scaled = mpp * phi / 3.0;
        x = 1.0;
        y = 1.0 - mpp;
        z = 1.0;
    }

    if x == y && x == z {
        return (scalef + scaled / x) / x.sqrt();
    }

    let a0f = (x + y + z) / 3.0;
    let mut af = a0f;
    let a0d = (x + y + 3.0 * z) / 5.0;
    let mut ad = a0d;
    let (mut x1, mut y1, mut z1) = (x, y, z);
    let mut seriesd = 0.0;
    let mut seriesn = 1.0;
    // Carlson gives 1/pow(3*r, 1.0/6.0) for this constant. if r == eps, it is ~338.38.
    let mut q = 400.0 * (a0f - x).abs().max((a0f - y).abs().max((a0f - z).abs()));

    while q > af.abs() && q > ad.abs() && n <= 100 {
        let sx = x1.sqrt();
        let sy = y1.sqrt();
        let sz = z1.sqrt();
        let lam = sx * sy + sx * sz + sy * sz;
        seriesd += seriesn / (sz * (z1 + lam));
        x1 = (x1 + lam) / 4.0;
        y1 = (y1 + lam) / 4.0;
        z1 = (z1 + lam) / 4.0;
        af = (x1 + y1 + z1) / 3.0;
        ad = (ad + lam) / 4.0;
        n += 1;
        q /= 4.0;
        seriesn /= 4.0;
    }

    let xf = (a0f - x) / af / pow4(n);
    let yf = (a0f - y) / af / pow4(n);
    let zf = -(xf + yf);

    let e2f = xf * yf - zf * zf;
    let e3f = xf * yf * zf;

    let mut ret = scalef
        * (1.0 - e2f / 10.0 + e3f / 14.0 + e2f * e2f / 24.0 - 3.0 * e2f * e3f / 44.0)
        / af.sqrt();

    let xd = (a0d - x) / ad / pow4(n);
    let yd = (a0d - y) / ad / pow4(n);
    let zd = -(xd + yd) / 3.0;

    let e2d = xd * yd - 6.0 * zd * zd;
    let e3d = (3.0 * xd * yd - 8.0 * zd * zd) * zd;
    let e4d = 3.0 * (xd * yd - zd * zd) * zd * zd;
    let e5d = xd * yd * zd * zd * zd;

    ret -= scaled
        * (1.0 - 3.0 * e2d / 14.0 + e3d / 6.0 + 9.0 * e2d * e2d / 88.0
            - 3.0 * e4d / 22.0
            - 9.0 * e2d * e3d / 52.0
            + 3.0 * e5d / 26.0)
        / pow4(n)
        / ad
        / ad.sqrt();
    ret -= 3.0 * scaled * seriesd;
    ret
}

/// E(phi | m), the incomplete elliptic integral of the second kind
/// (`cephes/ellie.h`, `ellie`).
pub(crate) fn ellie(phi: f64, m: f64) -> f64 {
    if phi.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m > 1.0 {
        return f64::NAN;
    }
    if phi.is_infinite() {
        return phi;
    }
    if m.is_infinite() {
        return -m;
    }
    if m == 0.0 {
        return phi;
    }
    // From about 2^53 on, the reduction below loses the residual.
    // E(phi, m) = 2 phi E(m) / pi + O(E(m)), DLMF 19.2.10
    if phi.abs() > 9e15 {
        return phi / FRAC_PI_2 * ellpe(m);
    }
    let mut lphi = phi;
    let mut npio2 = (lphi / FRAC_PI_2).floor();
    if (npio2.abs() % 2.0) == 1.0 {
        npio2 += 1.0;
    }
    lphi -= npio2 * FRAC_PI_2;
    let sign;
    if lphi < 0.0 {
        lphi = -lphi;
        sign = -1;
    } else {
        sign = 1;
    }
    let a = 1.0 - m;
    let ee = ellpe(m);
    let mut temp;
    'done: {
        if a == 0.0 {
            temp = lphi.sin();
            break 'done;
        }
        if a > 1.0 {
            temp = ellie_neg_m(lphi, m);
            break 'done;
        }

        if lphi < 0.135 {
            let m11 = (((((-7.0 / 2816.0) * m + (5.0 / 1056.0)) * m - (7.0 / 2640.0)) * m
                + (17.0 / 41580.0))
                * m
                - (1.0 / 155925.0))
                * m;
            let m9 = ((((-5.0 / 1152.0) * m + (1.0 / 144.0)) * m - (1.0 / 360.0)) * m
                + (1.0 / 5670.0))
                * m;
            let m7 = ((-m / 112.0 + (1.0 / 84.0)) * m - (1.0 / 315.0)) * m;
            let m5 = (-m / 40.0 + (1.0 / 30.0)) * m;
            let m3 = -m / 6.0;
            let p2 = lphi * lphi;

            temp = ((((m11 * p2 + m9) * p2 + m7) * p2 + m5) * p2 + m3) * p2 * lphi + lphi;
            break 'done;
        }
        let mut t = lphi.tan();
        let mut b = a.sqrt();
        // Thanks to Brian Fitzgerald <fitzgb@mml0.meche.rpi.edu>
        // for pointing out an instability near odd multiples of pi/2.
        if t.abs() > 10.0 {
            // Transform the amplitude
            let e = 1.0 / (b * t);
            // ... but avoid multiple recursions.
            if e.abs() < 10.0 {
                let e = e.atan();
                temp = ee + m * lphi.sin() * e.sin() - ellie(e, m);
                break 'done;
            }
        }
        let mut c = m.sqrt();
        let mut a = 1.0;
        let mut d = 1_i32;
        let mut e = 0.0;
        let mut modulus = 0_i32;

        while (c / a).abs() > MACHEP {
            temp = b / a;
            lphi = lphi + (t * temp).atan() + f64::from(modulus) * PI;
            let denom = 1.0 - temp * t * t;
            if denom.abs() > 10.0 * MACHEP {
                t = t * (1.0 + temp) / denom;
                modulus = ((lphi + FRAC_PI_2) / PI) as i32;
            } else {
                t = lphi.tan();
                modulus = ((lphi - t.atan()) / PI).floor() as i32;
            }
            c = (a - b) / 2.0;
            temp = (a * b).sqrt();
            a = (a + b) / 2.0;
            b = temp;
            d += d;
            e += c * lphi.sin();
        }

        temp = ee / ellpk(1.0 - m);
        temp *= (t.atan() + f64::from(modulus) * PI) / (f64::from(d) * a);
        temp += e;
    }
    if sign < 0 {
        temp = -temp;
    }
    temp += npio2 * ee;
    temp
}

/// `(sn, cn, dn, ph)` of `u` with parameter `m` (`cephes/ellpj.h`).
pub(crate) fn ellpj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    // Check for special cases
    if m > 1.0 || m.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    // Handle negative m by transformation (DLMF 22.17)
    if m < 0.0 {
        let m1 = -m / (1.0 - m);
        let sqrt_1_minus_m = (1.0 - m).sqrt();
        let u1 = sqrt_1_minus_m * u;

        let (sn_p, cn_p, dn_p, ph_p) = ellpj(u1, m1);

        let sn = sn_p / (sqrt_1_minus_m * dn_p);
        let cn = cn_p / dn_p;
        let dn = 1.0 / dn_p;
        let ph0 = sn_p.atan2(sqrt_1_minus_m * cn_p);
        let ph = ph0 + PI * ((ph_p - ph0) / PI).round();
        return (sn, cn, dn, ph);
    }
    if m < 1.0e-9 {
        let t = u.sin();
        let b = u.cos();
        let ai = 0.25 * m * (u - t * b);
        let sn = t - ai * b;
        let cn = b + ai * t;
        let ph = u - ai;
        let dn = 1.0 - 0.5 * m * t * t;
        return (sn, cn, dn, ph);
    }
    if m >= 0.9999999999 {
        let mut ai = 0.25 * (1.0 - m);
        let b = u.cosh();
        let t = u.tanh();
        let phi = 1.0 / b;
        let twon = b * u.sinh();
        let sn = t + ai * (twon - u) / (b * b);
        let ph = 2.0 * u.exp().atan() - FRAC_PI_2 + ai * (twon - u) / b;
        ai *= t * phi;
        let cn = phi - ai * (twon - u);
        let dn = phi + ai * (twon + u);
        return (sn, cn, dn, ph);
    }

    // A. G. M. scale. See DLMF 22.20(ii)
    let mut a = [0.0_f64; 9];
    let mut c = [0.0_f64; 9];
    a[0] = 1.0;
    let mut b = (1.0 - m).sqrt();
    c[0] = m.sqrt();
    let mut twon = 1.0;
    let mut i = 0_usize;

    while (c[i] / a[i]).abs() > MACHEP {
        if i > 7 {
            // xsf reports an overflow here and continues with the table full.
            break;
        }
        let ai = a[i];
        i += 1;
        c[i] = (ai - b) / 2.0;
        let t = (ai * b).sqrt();
        a[i] = (ai + b) / 2.0;
        b = t;
        twon *= 2.0;
    }

    // backward recurrence
    let mut phi = twon * a[i] * u;
    loop {
        let t = c[i] * phi.sin() / a[i];
        b = phi;
        phi = (t.asin() + phi) / 2.0;
        i -= 1;
        if i == 0 {
            break;
        }
    }

    let sn = phi.sin();
    let t = phi.cos();
    let cn = t;
    let dnfac = (phi - b).cos();
    // See discussion after DLMF 22.20.5
    let dn = if dnfac.abs() < 0.1 {
        (1.0 - m * sn * sn).sqrt()
    } else {
        t / dnfac
    };
    (sn, cn, dn, phi)
}

/// K(m) (`ellip.h`: `ellpk(1 - m)`).
pub(crate) fn ellipk(m: f64) -> f64 {
    ellpk(1.0 - m)
}

/// K(1 - p) (`ellip.h`: `ellpk(p)`).
pub(crate) fn ellipkm1(p: f64) -> f64 {
    ellpk(p)
}

/// E(m).
pub(crate) fn ellipe(m: f64) -> f64 {
    ellpe(m)
}

/// F(phi | m).
pub(crate) fn ellipkinc(phi: f64, m: f64) -> f64 {
    ellik(phi, m)
}

/// E(phi | m).
pub(crate) fn ellipeinc(phi: f64, m: f64) -> f64 {
    ellie(phi, m)
}

/// `(sn(u|m), cn(u|m), dn(u|m), am(u|m))`.
pub(crate) fn ellipj(u: f64, m: f64) -> (f64, f64, f64, f64) {
    ellpj(u, m)
}
