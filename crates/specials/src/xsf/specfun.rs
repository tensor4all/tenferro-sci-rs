//! Kummer's function `M(a, b, x)` and the Legendre function of integer order
//! and arbitrary degree for `f64`.
//!
//! Ported from xsf (scipy/xsf at `31177fd9a458539cf05c04d215341245d0245e53`)
//! `include/xsf/cpu/specfun/specfun.h` (`chgm`, `cgama`,
//! `gamma2`, `psi_spec`, `lpmv`, `lpmv0`) and from the real wrappers `hyp1f1`
//! and `pmv` of `include/xsf/cpu/specfun.h`.
//!
//! `specfun.h` is SciPy's C++ translation (Copyright (C) 2024 SciPy
//! developers, BSD-3-Clause) of the Fortran code of Shanjie Zhang and Jianming
//! Jin, "Computation of Special Functions" (Wiley, 1996), "Copyrighted but
//! permission granted to use code in programs"; see
//! `LICENSE-NOTICE-zhang-jin-specfun`.
//!
//! Deviations from upstream: the integer arguments of the C code are `i32`
//! (the C signed overflow of `j * j` for orders above 46340 is computed in
//! `f64`); the complex logarithm of the gamma function is evaluated for the
//! real arguments `chgm` passes it, as `(re, im)` pairs; the `1e300` overflow
//! sentinel of `lpmv` is converted by [`pmv`], as in xsf.

use core::f64::consts::PI;

/// `static_cast<int>(x) == x`.
fn is_i32(x: f64) -> bool {
    f64::from(x as i32) == x
}

/// Coefficients of the asymptotic series of `ln Γ` (`cgama`).
const CGAMA_A: [f64; 10] = [
    8.333333333333333e-02,
    -2.777777777777778e-03,
    7.936507936507937e-04,
    -5.952380952380952e-04,
    8.417508417508418e-04,
    -1.917526917526918e-03,
    6.410256410256410e-03,
    -2.955065359477124e-02,
    1.796443723688307e-01,
    -1.392432216905900e+00,
];

/// `ln Γ(xx + i yy)` as `(re, im)` (`cgama` with `kf = 0`). `1e300` for a
/// pole on the real axis.
fn cgama_ln(mut xx: f64, mut yy: f64) -> (f64, f64) {
    let z1_real;
    if yy == 0.0 && xx <= 0.0 && is_i32(xx) {
        return (1e300, 0.0);
    } else if xx < 0.0 {
        z1_real = xx;
        xx = -xx;
        yy = -yy;
    } else {
        z1_real = xx;
    }
    let mut x0 = xx;
    let mut na = 0;
    if xx <= 7.0 {
        na = (7.0 - xx) as i32;
        x0 = xx + f64::from(na);
    }
    let mut az0 = x0.hypot(yy);
    let th = yy.atan2(x0);
    let mut gr = (x0 - 0.5) * az0.ln() - th * yy - x0 + 0.5 * (2.0 * PI).ln();
    let mut gi = th * (x0 - 0.5) + yy * az0.ln() - yy;
    for k in 1..11 {
        let t = az0.powf(f64::from(1 - 2 * k));
        gr += CGAMA_A[k as usize - 1] * t * ((2.0 * f64::from(k) - 1.0) * th).cos();
        gi += -CGAMA_A[k as usize - 1] * t * ((2.0 * f64::from(k) - 1.0) * th).sin();
    }
    if xx <= 7.0 {
        let mut gr1 = 0.0;
        let mut gi1 = 0.0;
        for j in 0..na {
            gr1 += 0.5 * ((xx + f64::from(j)).powf(2.0) + yy * yy).ln();
            gi1 += yy.atan2(xx + f64::from(j));
        }
        gr -= gr1;
        gi -= gi1;
    }
    if z1_real < 0.0 {
        az0 = xx.hypot(yy);
        let th1 = yy.atan2(xx);
        let sr = -(PI * xx).sin() * (PI * yy).cosh();
        let si = -(PI * xx).cos() * (PI * yy).sinh();
        let az1 = sr.hypot(si);
        let th2 = si.atan2(sr);
        gr = (PI / (az0 * az1)).ln() - gr;
        gi = -th1 - th2 - gi;
    }
    (gr, gi)
}

/// Real part of `exp(re + i im)`.
fn exp_re(re: f64, im: f64) -> f64 {
    re.exp() * im.cos()
}

/// Kummer's function `M(a, b, x)` (`chgm`); NaN for a NaN argument.
pub(crate) fn hyp1f1(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    chgm(x, a, b)
}

fn chgm(mut x: f64, mut a: f64, b: f64) -> f64 {
    let mut a0 = a;
    let x0 = x;
    let mut hg = 0.0;

    // DLMF 13.2.39
    if x < 0.0 {
        a = b - a;
        a0 = a;
        x = x.abs();
    }
    let mut nl = 0;
    let mut la = 0;
    if a >= 2.0 {
        // preparing terms for DLMF 13.3.1
        nl = 1;
        la = a as i32;
        a -= f64::from(la + 1);
    }
    let mut y0 = 0.0;
    let mut y1 = 0.0;
    for n in 0..(nl + 1) {
        if a0 >= 2.0 {
            a += 1.0;
        }
        if x <= 30.0 + b.abs() || a < 0.0 {
            hg = 1.0;
            let mut rg = 1.0;
            for j in 1..501 {
                let jf = f64::from(j);
                rg = rg * (a + (jf - 1.0)) / (jf * (b + (jf - 1.0))) * x;
                hg += rg;
                if hg != 0.0 && (rg / hg).abs() < 1e-15 {
                    // DLMF 13.2.39 (cf. above)
                    if x0 < 0.0 {
                        hg *= x0.exp();
                    }
                    break;
                }
            }
        } else {
            // DLMF 13.7.2 & 13.2.4, SUM2 corresponds to first sum
            let (cta_re, cta_im) = cgama_ln(a, 0.0);
            let (ctb_re, ctb_im) = cgama_ln(b, 0.0);
            let xg = b - a;
            let (ctba_re, ctba_im) = cgama_ln(xg, 0.0);
            let mut sum1 = 1.0;
            let mut sum2 = 1.0;
            let mut r1 = 1.0;
            let mut r2 = 1.0;
            for i in 1..9 {
                let i = f64::from(i);
                r1 = -r1 * (a + i - 1.0) * (a - b + i) / (x * i);
                r2 = -r2 * (b - a + i - 1.0) * (a - i) / (x * i);
                sum1 += r1;
                sum2 += r2;
            }
            let (hg1, hg2);
            if x0 >= 0.0 {
                hg1 =
                    exp_re(ctb_re - ctba_re, ctb_im - ctba_im) * x.powf(-a) * (PI * a).cos() * sum1;
                hg2 = exp_re(ctb_re - cta_re + x, ctb_im - cta_im) * x.powf(a - b) * sum2;
            } else {
                // DLMF 13.2.39 (cf. above)
                hg1 = exp_re(ctb_re - ctba_re + x0, ctb_im - ctba_im)
                    * x.powf(-a)
                    * (PI * a).cos()
                    * sum1;
                hg2 = exp_re(ctb_re - cta_re, ctb_im - cta_im) * x.powf(a - b) * sum2;
            }
            hg = hg1 + hg2;
        }
        if n == 0 {
            y0 = hg;
        }
        if n == 1 {
            y1 = hg;
        }
    }
    if a0 >= 2.0 {
        // DLMF 13.3.1
        for _ in 1..la {
            hg = ((2.0 * a - b + x) * y1 + (b - a) * y0) / a;
            y0 = y1;
            y1 = hg;
            a += 1.0;
        }
    }
    hg
}

/// Coefficients of the series of `1/Γ` around 0 (`gamma2`).
const GAMMA2_G: [f64; 26] = [
    1.0000000000000000e+00,
    0.5772156649015329e+00,
    -0.6558780715202538e+00,
    -0.4200263503409520e-01,
    0.1665386113822915e+00,
    -0.4219773455554430e-01,
    -0.9621971527877000e-02,
    0.7218943246663000e-02,
    -0.1165167591859100e-02,
    -0.2152416741149000e-03,
    0.1280502823882000e-03,
    -0.2013485478070000e-04,
    -0.1250493482100000e-05,
    0.1133027232000000e-05,
    -0.2056338417000000e-06,
    0.6116095000000000e-08,
    0.5002007500000000e-08,
    -0.1181274600000000e-08,
    0.1043427000000000e-09,
    0.7782300000000000e-11,
    -0.3696800000000000e-11,
    0.5100000000000000e-12,
    -0.2060000000000000e-13,
    -0.5400000000000000e-14,
    0.1400000000000000e-14,
    0.1000000000000000e-15,
];

/// `Γ(x)` for `x` not a non-positive integer (`gamma2`); `1e300` there.
fn gamma2(x: f64) -> f64 {
    if x == f64::from(x as i32) {
        if x > 0.0 {
            let mut ga = 1.0;
            let m = (x - 1.0) as i32;
            for k in 2..(m + 1) {
                ga *= f64::from(k);
            }
            ga
        } else {
            1e300
        }
    } else {
        let mut r = 1.0;
        let z = if x.abs() > 1.0 {
            let za = x.abs();
            let m = za as i32;
            for k in 1..(m + 1) {
                r *= za - f64::from(k);
            }
            za - f64::from(m)
        } else {
            x
        };
        let mut gr = GAMMA2_G[25];
        for k in (1..26).rev() {
            gr *= z;
            gr += GAMMA2_G[k - 1];
        }
        let mut ga = 1.0 / (gr * z);
        if x.abs() > 1.0 {
            ga *= r;
            if x < 0.0 {
                ga = -PI / (x * ga * (PI * x).sin());
            }
        }
        ga
    }
}

/// `ψ(x)` (`psi_spec`); `1e300` at the poles.
fn psi_spec(x: f64) -> f64 {
    const A: [f64; 8] = [
        -0.8333333333333e-01,
        0.83333333333333333e-02,
        -0.39682539682539683e-02,
        0.41666666666666667e-02,
        -0.75757575757575758e-02,
        0.21092796092796093e-01,
        -0.83333333333333333e-01,
        0.4432598039215686,
    ];
    const EL: f64 = 0.5772156649015329;
    let mut s = 0.0;
    let mut xa = x.abs();
    let mut ps;

    if x == f64::from(x as i32) && x <= 0.0 {
        return 1e300;
    } else if xa == f64::from(xa as i32) {
        let n = xa as i32;
        for k in 1..n {
            s += 1.0 / f64::from(k);
        }
        ps = -EL + s;
    } else if (xa + 0.5) == f64::from((xa + 0.5) as i32) {
        let n = (xa - 0.5) as i32;
        for k in 1..(n + 1) {
            s += 1.0 / (2.0 * f64::from(k) - 1.0);
        }
        ps = -EL + 2.0 * s - 1.386294361119891; // 2*log(2)
    } else {
        if xa < 10.0 {
            let n = (10.0 - f64::from(xa as i32)) as i32;
            for k in 0..n {
                s += 1.0 / (xa + f64::from(k));
            }
            xa += f64::from(n);
        }
        let x2 = 1.0 / (xa * xa);
        ps = xa.ln() - 0.5 / xa;
        ps += x2
            * (((((((A[7] * x2 + A[6]) * x2 + A[5]) * x2 + A[4]) * x2 + A[3]) * x2 + A[2]) * x2
                + A[1])
                * x2
                + A[0]);
        ps -= s;
    }
    if x < 0.0 {
        ps -= PI * (PI * x).cos() / (PI * x).sin() + 1.0 / x;
    }
    ps
}

/// `P_v^m(x)` for `-1 <= x <= 1`, order `m`, degree `v` (`lpmv`). `±1e300`
/// stands for an infinity.
fn lpmv(x: f64, m: i32, v: f64) -> f64 {
    if x == -1.0 && v != f64::from(v as i32) {
        return if m == 0 {
            let vs = (v * PI).sin() / (v * PI);
            if vs > 0.0 {
                -1e300
            } else {
                1e300
            }
        } else {
            1e300
        };
    }
    let mut vx = v;
    let mut mx = m;
    // DLMF 14.9.5
    if v < 0.0 {
        vx = -vx - 1.0;
    }
    let mut neg_m = false;
    if m < 0 {
        if (vx + f64::from(m) + 1.0) > 0.0 || vx != f64::from(vx as i32) {
            neg_m = true;
            mx = -m;
        } else {
            // We don't handle cases where DLMF 14.9.3 doesn't help
            return f64::NAN;
        }
    }
    let nv = vx as i32;
    let v0 = vx - f64::from(nv);
    let mut pmv;
    if nv > 2 && nv > mx {
        // Up-recursion on degree, AMS 8.5.3 / DLMF 14.10.3
        let mut p0 = lpmv0(v0 + f64::from(mx), mx, x);
        let mut p1 = lpmv0(v0 + f64::from(mx) + 1.0, mx, x);
        pmv = p1;
        for j in (mx + 2)..(nv + 1) {
            let jf = f64::from(j);
            pmv = ((2.0 * (v0 + jf) - 1.0) * x * p1 - (v0 + jf - 1.0 + f64::from(mx)) * p0)
                / (v0 + jf - f64::from(mx));
            p0 = p1;
            p1 = pmv;
        }
    } else {
        pmv = lpmv0(vx, mx, x);
    }
    if neg_m && pmv.abs() < 1.0e300 {
        // DLMF 14.9.3
        let g1 = gamma2(vx - f64::from(mx) + 1.0);
        let g2 = gamma2(vx + f64::from(mx) + 1.0);
        pmv = pmv * g1 / g2 * f64::powi(-1.0, mx);
    }
    pmv
}

/// `P_v^m(x)` for `-1 <= x <= 1`, order `m` and degree `v >= 0` (`lpmv0`).
fn lpmv0(v: f64, m: i32, x: f64) -> f64 {
    const PI_: f64 = 3.141592653589793;
    const EL: f64 = 0.5772156649015329;
    const EPS: f64 = 1e-14;

    let nv = v as i32;
    let v0 = v - f64::from(nv);

    if x == -1.0 && v != f64::from(nv) {
        if m == 0 {
            return -1.0e+300;
        }
        return 1.0e+300;
    }

    let mf = f64::from(m);
    let mut c0 = 1.0;
    if m != 0 {
        let mut rg = v * (v + mf);
        for j in 1..m {
            let jf = f64::from(j);
            rg *= v * v - jf * jf;
        }
        let xq = (1.0 - x * x).sqrt();
        let mut r0 = 1.0;
        for j in 1..(m + 1) {
            r0 = 0.5 * r0 * xq / f64::from(j);
        }
        c0 = r0 * rg;
    }

    if v0 == 0.0 {
        // DLMF 14.3.4, 14.7.17, 15.2.4
        let mut pmv = 1.0;
        let mut r = 1.0;
        for k in 1..(nv - m + 1) {
            r = 0.5 * r * (f64::from(-nv + m + k) - 1.0) * f64::from(nv + m + k)
                / f64::from(k * (k + m))
                * (1.0 + x);
            pmv += r;
        }
        f64::powi(-1.0, nv) * c0 * pmv
    } else if x >= -0.35 {
        // DLMF 14.3.4, 15.2.1
        let mut pmv = 1.0;
        let mut r = 1.0;
        for k in 1..101 {
            let kf = f64::from(k);
            r = 0.5 * r * (-v + mf + kf - 1.0) * (v + mf + kf) / (kf * (mf + kf)) * (1.0 - x);
            pmv += r;
            if k > 12 && (r / pmv).abs() < EPS {
                break;
            }
        }
        f64::powi(-1.0, m) * c0 * pmv
    } else {
        // DLMF 14.3.5, 15.8.10
        let vs = (v * PI_).sin() / PI_;
        let mut pv0 = 0.0;
        if m != 0 {
            let qr = ((1.0 - x) / (1.0 + x)).sqrt();
            let mut r2 = 1.0;
            for j in 1..(m + 1) {
                r2 *= qr * f64::from(j);
            }
            let mut s0 = 1.0;
            let mut r1 = 1.0;
            for k in 1..m {
                let kf = f64::from(k);
                r1 = 0.5 * r1 * (-v + kf - 1.0) * (v + kf) / (kf * (kf - mf)) * (1.0 + x);
                s0 += r1;
            }
            pv0 = -vs * r2 / mf * s0;
        }

        let pa = 2.0 * (psi_spec(v) + EL) + PI_ / (PI_ * v).tan() + 1.0 / v;
        let mut s1 = 0.0;
        for j in 1..(m + 1) {
            let jf = f64::from(j);
            s1 += (jf * jf + v * v) / (jf * (jf * jf - v * v));
        }
        let mut pmv = pa + s1 - 1.0 / (mf - v) + (0.5 * (1.0 + x)).ln();
        let mut r = 1.0;
        for k in 1..101 {
            let kf = f64::from(k);
            r = 0.5 * r * (-v + mf + kf - 1.0) * (v + mf + kf) / (kf * (kf + mf)) * (1.0 + x);
            let mut s = 0.0;
            for j in 1..(m + 1) {
                let kj = f64::from(k + j);
                s += (kj * kj + v * v) / (kj * (kj * kj - v * v));
            }

            let mut s2 = 0.0;
            for j in 1..(k + 1) {
                let jf = f64::from(j);
                s2 += 1.0 / (jf * (jf * jf - v * v));
            }

            let pss = pa + s + 2.0 * v * v * s2 - 1.0 / (mf + kf - v) + (0.5 * (1.0 + x)).ln();
            let r2 = pss * r;
            pmv += r2;
            if (r2 / pmv).abs() < EPS {
                break;
            }
        }
        pv0 + pmv * vs * c0
    }
}

/// `P_v^m(x)` of integer order `m` (xsf `pmv`): NaN for a non-integer `m`,
/// and the `±1e300` sentinel of `lpmv` converted to an infinity.
pub(crate) fn pmv(m: f64, v: f64, x: f64) -> f64 {
    if m != m.floor() {
        return f64::NAN;
    }
    let int_m = m as i32;
    let out = lpmv(x, int_m, v);
    if out == 1.0e300 {
        f64::INFINITY
    } else if out == -1.0e300 {
        f64::NEG_INFINITY
    } else {
        out
    }
}
