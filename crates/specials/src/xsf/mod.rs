//! `f64` kernels ported from SciPy's xsf (Cephes).
//!
//! Ported from <https://github.com/scipy/xsf> at commit
//! `31177fd9a458539cf05c04d215341245d0245e53` (2026-11-21). Each file names its
//! upstream header. These are close translations and remain under xsf's and
//! Cephes' BSD-3-Clause licenses (`LICENSE-BSD-3-Clause-xsf`,
//! `LICENSE-BSD-3-Clause-cephes`); `trigamma` is the MIT-licensed code of
//! JuliaMath/SpecialFunctions.jl that xsf took (`LICENSE-MIT-julia-specialfunctions`).
//!
//! Deviation from upstream: xsf reports domain and overflow errors through the
//! process-global `sf_error`; this port returns the same values (NaN, ±inf)
//! without a global flag.

pub(crate) mod beta;
pub(crate) mod binom;
pub(crate) mod digammainv;
pub(crate) mod gamma;
pub(crate) mod gen_harmonic;
pub(crate) mod lambertw;
pub(crate) mod lanczos;
pub(crate) mod poch;
pub(crate) mod polevl;
pub(crate) mod psi;
pub(crate) mod stirling2;
pub(crate) mod trigamma;
pub(crate) mod unity;
pub(crate) mod zeta;
pub(crate) mod zetac;

/// Constants from `xsf/cephes/const.h`.
pub(crate) mod consts {
    pub(crate) const MACHEP: f64 = 1.1102230246251565404236316680908203125E-16;
    pub(crate) const MAXLOG: f64 = 7.097827128933839730962063185871E2;
    pub(crate) const SQRT2PI: f64 = 2.506628274631000502415765284811045253007;
    pub(crate) const SQRT2OPI: f64 = 7.978845608028653558798921198687637369517E-1;
    pub(crate) const LOGPI: f64 = 1.144729885849400174143427351353058711647;
    pub(crate) const MAXGAM: f64 = 171.624376956302725;
    pub(crate) const SCIPY_EULER: f64 = 0.577215664901532860606512090082402431;
}

/// `sin(πx)`, from `xsf/cephes/trig.h`.
pub(crate) fn sinpi(mut x: f64) -> f64 {
    let mut s = 1.0;
    if x.is_sign_negative() {
        x = -x;
        s = -1.0;
    }
    let r = x % 2.0;
    if r < 0.5 {
        s * (core::f64::consts::PI * r).sin()
    } else if r > 1.5 {
        s * (core::f64::consts::PI * (r - 2.0)).sin()
    } else {
        -s * (core::f64::consts::PI * (r - 1.0)).sin()
    }
}
