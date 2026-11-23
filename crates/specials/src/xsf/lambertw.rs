//! Principal branch `W₀` of the Lambert W function on `(-1/e, 0]`, for `f64`.
//!
//! This is the part of xsf's `lambertw` that `stirling2` needs, taken from the
//! complex-argument path of `include/xsf/lambertw.h` (`lambertw_branchpt`,
//! `lambertw_pade0` as the starting guess, then Halley's method; see 5.9 in
//! Corless et al., "On the Lambert W function", 1996), specialised to real
//! arguments. xsf itself routes a real argument above -1/e through Fukushima's
//! piecewise minimax approximation (`lambertw_real`), which is not ported here;
//! Halley iteration converges cubically to the same value (within a few ulp).
//! The general lambertw belongs to tenferro-sci-rs#10.
//!
//! Original Lambert W code: Copyright Yosef Meller, 2009, distributed under
//! the same license as SciPy (BSD-3-Clause).

/// exp(-1)
const EXPN1: f64 = 0.36787944117144232159553;

/// `W₀(z)` for `-1/e < z <= 0` (NaN for NaN).
pub(crate) fn lambertw0_neg(z: f64, tol: f64) -> f64 {
    if z.is_nan() {
        return z;
    }
    if z == 0.0 {
        return z;
    }
    if z <= -EXPN1 {
        // The branch point.
        return -1.0;
    }
    // Initial guess for Halley's method.
    let mut w = if (z + EXPN1).abs() < 0.3 {
        // Series around the branch point; see 4.22 in Corless et al.
        let p = (2.0 * (core::f64::consts::E * z + 1.0)).sqrt();
        (-1.0 / 3.0 * p + 1.0) * p - 1.0
    } else {
        // (3, 2) Padé approximation around 0.
        let num = (1.0 * z + 12.34042553191489361902) * z + 12.85106382978723404255;
        let den = (1.0 * z + 14.34042553191489361702) * z + 32.53191489361702127660;
        z * num / den
    };
    // W₀ < 0 here, so the form without the exp(-w) rearrangement applies.
    for _ in 0..100 {
        let ew = w.exp();
        let wew = w * ew;
        let wewz = wew - z;
        let wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
        if (wn - w).abs() <= tol * wn.abs() {
            return wn;
        }
        w = wn;
    }
    f64::NAN
}
