//! ψ'(x), the trigamma function, for real `f64` arguments.
//!
//! Ported from xsf `include/xsf/trigamma.h` (`trigamma`, real branch).
//! Copyright (c) 2017 Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and
//! others (MIT License, JuliaMath/SpecialFunctions.jl), as reproduced in xsf;
//! see `LICENSE-MIT-julia-specialfunctions`.
//! Based on the differentiation of equation 12 of K.S. Kölbig, Programs for
//! computing the logarithm of the gamma function, and the digamma function,
//! for complex argument, Computer Physics Communications, 1972.

use super::polevl::polevl;
use super::sinpi;

pub(crate) fn trigamma(mut z: f64) -> f64 {
    let x = z;
    if x <= 0.0 {
        // Reflection formula, DLMF 5.15.6.
        let s = sinpi(z);
        let c = core::f64::consts::PI / s;
        return c * c - trigamma(1.0 - z);
    }

    let mut psi = 0.0;
    let n_const = 10;
    if x < f64::from(n_const) {
        // Recursion based on DLMF 5.15.1.
        let n = n_const - x.floor() as i32;
        psi += 1.0 / (z * z);
        for v in 1..n {
            let w = z + f64::from(v);
            psi += 1.0 / (w * w);
        }
        z += f64::from(n);
    }

    // DLMF 5.15.8.
    let t = 1.0 / z;
    let w = t * t;
    psi += t + 0.5 * w;
    // The order of the Bernoulli numbers is reversed (xsf's `evalpoly` convention).
    const COEFFS: [f64; 8] = [
        -7.092156862745098,
        1.1666666666666667,
        -0.2531135531135531,
        0.07575757575757576,
        -0.03333333333333333,
        0.023809523809523808,
        -0.03333333333333333,
        0.16666666666666666,
    ];
    psi += t * w * polevl(w, &COEFFS);
    psi
}
