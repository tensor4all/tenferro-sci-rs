//! Hurwitz ζ(x, q) for `f64`.
//!
//! Ported from xsf `include/xsf/cephes/zeta.h` and the real `zeta(double,
//! double)` wrapper of `include/xsf/zeta.h`.
//! Cephes Math Library, Copyright 1984, 1987 by Stephen L. Moshier.

use super::consts::MACHEP;

/// `(2k)! / B_{2k}` for the Euler–Maclaurin tail.
const ZETA_A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9,
    7.47242496e10,
    -2.950130727918164224e12,
    1.1646782814350067249e14,
    -4.5979787224074726105e15,
    1.8152105401943546773e17,
    -7.1661652561756670113e18,
];

pub(crate) fn zeta(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            // q^-x is not defined.
            return f64::NAN;
        }
    }

    // Asymptotic expansion, DLMF 25.11.43.
    if q > 1e8 {
        let xq = if x.is_infinite() { 0.0 } else { x / q };
        return (1.0 / (x - 1.0) + (6.0 + xq) / q / 12.0) * q.powf(1.0 - x);
    }

    // Euler–Maclaurin summation.
    let mut s = q.powf(-x);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = a.powf(-x);
        s += b;
        if (b / s).abs() < MACHEP && (b * a / ((x - 1.0) * s)).abs() < MACHEP {
            return s;
        }
    }

    let w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    let mut a = 1.0;
    let mut k = 0.0;
    for za in ZETA_A {
        a *= x + k;
        b /= w;
        let t = a * b / za;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    s
}

/// `zeta(x, q)` of `xsf/zeta.h`: Riemann zeta for `q == 1`, else Hurwitz zeta.
pub(crate) fn zeta_xsf(x: f64, q: f64) -> f64 {
    if q == 1.0 {
        return super::zetac::riemann_zeta(x);
    }
    zeta(x, q)
}
