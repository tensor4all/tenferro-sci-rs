//! Carlson's symmetric elliptic integrals R_C, R_D, R_F, R_G, R_J for real
//! `f64` arguments.
//!
//! Ported from xsf `include/xsf/cpu/ellint_carlson.h` (the public wrappers,
//! with the relative error bound `5e-16`) and `include/xsf/ellint_carlson/`
//! (`_rc.hh`, `_rd.hh`, `_rf.hh`, `_rg.hh`, `_rj.hh`, `ellint_arithmetic.hh`,
//! `ellint_arith_aux.hh`, `ellint_argcheck.hh`, `ellint_common.hh`), real
//! branches only. The directory carries no license header of its own; it is
//! part of xsf and covered by xsf's BSD-3-Clause license
//! (`LICENSE-BSD-3-Clause-xsf`). Algorithms: B. C. Carlson, "Numerical
//! computation of real or complex elliptic integrals", Numer. Algorithms 10
//! (1995) 13-26, <https://arxiv.org/abs/math/9409227>, and DLMF 19.36.
//!
//! Deviations from upstream: the functions return the value that xsf stores
//! in `res`, with no global error flag; the complex branches, the unused
//! asymptotic case `tinyx` (never selected upstream) and the compensated
//! complex arithmetic are not ported; `std::sort` on at most three entries is
//! an insertion sort. The wrappers [`elliprg`] and [`elliprj`] scale their
//! arguments by an exact power of four when one lies beyond 2^±300, none is
//! subnormal and they are at most 2^600 apart (see [`Scaling`]): xsf forms
//! products of up to three arguments before any scaling and returns NaN, 0
//! or a wrong value from about 2^±341 (R_J) and 2^±512 (R_G) on, although
//! the integrals are finite there (tensor4all/tenferro-sci-rs#25). All other
//! arguments take xsf's path unchanged.

use core::f64::consts::PI;

use crate::generic::carlson::pv_ill_conditioned;
use crate::generic::ldexp;

/// Exit status of the Carlson routines (`ellint_common.hh`). The variants no
/// routine here produces are kept so that `is_horrible` mirrors upstream.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ExitStatus {
    Success,
    Singular,
    NIter,
    NoResult,
    BadArgs,
    BadRerr,
    Other,
}

use ExitStatus::{BadArgs, BadRerr, NIter, Singular, Success};

const MAX_ITER: u32 = 1000;
const ASYM_ZERO_UL: f64 = 5.0e-14;
const ASYM_CLOSE_UL: f64 = 1.0e-9;

const RC_C: [f64; 8] = [
    80080.0, 0.0, 24024.0, 11440.0, 30030.0, 32760.0, 61215.0, 90090.0,
];
const RF_C1: [f64; 4] = [0.0, -24024.0, 10010.0, -5775.0];
const RF_C2: [f64; 3] = [17160.0, -16380.0, 15015.0];
const RF_C33: f64 = 6930.0;
const RF_DENOM: f64 = 240240.0;
const RDJ_C1: [f64; 4] = [0.0, -875160.0, 417690.0, -255255.0];
const RDJ_C2: [f64; 3] = [0.0, 680680.0, 306306.0];
const RDJ_C3: [f64; 3] = [0.0, -706860.0, 675675.0];
const RDJ_C4: [f64; 2] = [-556920.0, 612612.0];
const RDJ_C5: [f64; 2] = [471240.0, -540540.0];
const RDJ_DENOM: f64 = 4084080.0;

fn is_horrible(s: ExitStatus) -> bool {
    matches!(
        s,
        ExitStatus::NoResult | BadArgs | BadRerr | ExitStatus::Other
    )
}

fn is_troublesome(s: ExitStatus) -> bool {
    !(s == Success || is_horrible(s))
}

// ---- argcheck

fn invalid_rerr(r: f64, upper: f64) -> bool {
    r <= 0.0 || r > upper
}

fn too_small(x: f64) -> bool {
    x == 0.0 || x.is_subnormal()
}

/// The argument is not negative and not NaN.
fn ph_good(x: f64) -> bool {
    x >= 0.0
}

// ---- arithmetic (compensated sums, dot products and Horner scheme)

/// Knuth's TwoSum: `(s, corr)` with `s + corr = x + y` exactly.
fn eft_sum(x: f64, y: f64) -> (f64, f64) {
    let s = x + y;
    let z = s - x;
    (s, (x - (s - z)) + (y - z))
}

/// `(prod, corr)` with `prod + corr = x y` exactly.
fn eft_prod(x: f64, y: f64) -> (f64, f64) {
    let prod = x * y;
    (prod, x.mul_add(y, -prod))
}

fn sum2_acc(summand: f64, acc: &mut f64, corr: &mut f64) {
    let (s, e) = eft_sum(summand, *acc);
    *acc = s;
    *corr += e;
}

/// Compensated sum of the first `n` entries.
fn nsum2(x: &[f64], n: usize) -> f64 {
    let (mut p, mut s) = (0.0, 0.0);
    for &v in x.iter().take(n) {
        sum2_acc(v, &mut p, &mut s);
    }
    p + s
}

fn fdot2_acc(x: f64, y: f64, acc: &mut f64, corr: &mut f64) {
    let (h, r) = eft_prod(x, y);
    let (t, q) = eft_sum(*acc, h);
    *acc = t;
    *corr += q + r;
}

/// Compensated dot product of the first `n` entries.
fn ndot2(x: &[f64], y: &[f64], n: usize) -> f64 {
    let (mut p, mut s) = (0.0, 0.0);
    for (&a, &b) in x.iter().zip(y.iter()).take(n) {
        fdot2_acc(a, b, &mut p, &mut s);
    }
    p + s
}

fn dot2(x: &[f64], y: &[f64]) -> f64 {
    ndot2(x, y, x.len().min(y.len()))
}

/// Compensated Horner scheme; `poly` in ascending order.
fn comp_horner(x: f64, poly: &[f64]) -> f64 {
    let degree = poly.len() - 1;
    let mut s = poly[degree];
    let mut r = 0.0;
    for i in (0..degree).rev() {
        let (ptmp, pp) = eft_prod(s, x);
        let (s2, ps) = eft_sum(ptmp, poly[i]);
        s = s2;
        r = r * x + (pp + ps);
    }
    s + r
}

/// Eighth root.
fn ocrt(x: f64) -> f64 {
    x.sqrt().sqrt().sqrt()
}

/// `(sx + sy) * t`, given `sx = sqrt(x)` and `sy = sqrt(y)`, with
/// `sx + sy = (x - y) / (sx - sy)` if the sum cancels.
fn sqrt_sum_mul(x: f64, y: f64, sx: f64, sy: f64, t: f64) -> f64 {
    let s = sx + sy;
    if s.abs() >= sx.abs() {
        return s * t;
    }
    let r = t / (sx - sy);
    let d = x - y;
    if d.abs().is_infinite() {
        return (x * 0.5 - y * 0.5) * r * 2.0;
    }
    d * r
}

/// `x + lam` in the duplication step (DLMF 19.26.18), where
/// `lam = sx sy + sy sz + sz sx` (DLMF 19.26.19).
fn add_lam(x: f64, y: f64, z: f64, sx: f64, sy: f64, sz: f64, lam: f64) -> f64 {
    let mut xl = x + lam;
    if xl.abs() < x.abs() {
        if (sx + sz).abs() < sx.abs() {
            xl = sqrt_sum_mul(x, z, sx, sz, sqrt_sum_mul(x, y, sx, sy, 1.0));
        } else {
            xl = sqrt_sum_mul(x, y, sx, sy, sqrt_sum_mul(x, z, sx, sz, 1.0));
        }
    }
    xl
}

/// The entry of `a`, `b`, `c` with the largest absolute value.
fn max_abs3(a: f64, b: f64, c: f64) -> f64 {
    let mut m = a;
    if m.abs() < b.abs() {
        m = b;
    }
    if m.abs() < c.abs() {
        m = c;
    }
    m
}

/// Sort three values ascending by `less`.
fn sort3(v: &mut [f64; 3], less: impl Fn(f64, f64) -> bool) {
    for i in 1..3 {
        let mut j = i;
        while j > 0 && less(v[j], v[j - 1]) {
            v.swap(j, j - 1);
            j -= 1;
        }
    }
}

// ---- R_F

fn agm_update(x: &mut f64, y: &mut f64) {
    let xnext = (*x + *y) * 0.5;
    let ynext = (*x * *y).sqrt();
    *x = xnext;
    *y = ynext;
}

fn rf0(x: f64, y: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    let rsq = 2.0 * rerr.sqrt();
    let mut xm = x.sqrt();
    let mut ym = y.sqrt();
    let mut m = 0_u32;
    while (xm - ym).abs() >= rsq * xm.abs().min(ym.abs()) {
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        agm_update(&mut xm, &mut ym);
        m += 1;
    }
    *res = PI / (xm + ym);
    status
}

pub(crate) fn rf(x: f64, y: f64, z: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    if invalid_rerr(rerr, 3.0e-4) {
        *res = f64::NAN;
        return BadRerr;
    }
    if ph_good(x) && ph_good(y) && ph_good(z) {
        if x.is_infinite() || y.is_infinite() || z.is_infinite() {
            *res = 0.0;
            return Success;
        }
    } else {
        *res = f64::NAN;
        return BadArgs;
    }
    let mut cct1 = [x, y, z];
    sort3(&mut cct1, |a, b| a.abs() < b.abs());
    let (mut xm, mut ym, mut zm) = (cct1[0], cct1[1], cct1[2]);
    if too_small(xm) {
        if too_small(ym) {
            *res = f64::INFINITY;
            return Singular;
        } else {
            let mut tmpres = 0.0;
            status = rf0(ym, zm, rerr * 0.5, &mut tmpres);
            // Correction for non-zero x, see Eq. 19.27.3 in https://dlmf.nist.gov/19.27.E3
            *res = tmpres - xm.sqrt() / (ym.sqrt() * zm.sqrt());
            return status;
        }
    }
    let mut am = nsum2(&cct1, 3) / 3.0;
    let mut xxm = am - xm;
    let mut yym = am - ym;
    let mut fterm = max_abs3(xxm, yym, am - zm).abs() / ocrt(3.0 * rerr);
    let mut m = 0_u32;
    loop {
        let aam = am.abs();
        if !(aam <= fterm || aam <= max_abs3(xxm, yym, am - zm).abs()) {
            break;
        }
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        let (sx, sy, sz) = (xm.sqrt(), ym.sqrt(), zm.sqrt());
        let lam = dot2(&[sx, sy, sz], &[sy, sz, sx]);
        let xl = add_lam(xm, ym, zm, sx, sy, sz, lam);
        let yl = add_lam(ym, zm, xm, sy, sz, sx, lam);
        let zl = add_lam(zm, xm, ym, sz, sx, sy, lam);
        am = (am + lam) * 0.25;
        xm = xl * 0.25;
        ym = yl * 0.25;
        zm = zl * 0.25;
        xxm *= 0.25;
        yym *= 0.25;
        fterm *= 0.25;
        m += 1;
    }
    // Burn some extra cycles re-balancing Am as the "true" centroid.
    am = nsum2(&[xm, ym, zm], 3) / 3.0;
    xxm /= am;
    yym /= am;
    // Prepare the E_2 and E_3 terms used in the expansion
    let zzm = -(xxm + yym);
    let e2 = xxm * yym - zzm * zzm;
    let e3 = xxm * (yym * zzm);
    // Evaluate the 7th-degree expansion using the E_2 and E_3 terms, following
    // Eq. 19.36.1 of https://dlmf.nist.gov/19.36.E1
    let mut s = comp_horner(e2, &RF_C1);
    s += e3 * (comp_horner(e2, &RF_C2) + e3 * RF_C33);
    s /= RF_DENOM;
    s += 1.0;
    *res = s / am.sqrt();
    status
}

// ---- R_C

pub(crate) fn rc(x: f64, y: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    if invalid_rerr(rerr, 2.0e-4) {
        *res = f64::NAN;
        return BadRerr;
    }
    // Cauchy principal value with real negative y
    if y < 0.0 {
        // Carlson (1995) Eq. 2.14
        let mut tmpres = 0.0;
        status = rc(x - y, -y, rerr, &mut tmpres);
        if is_horrible(status) {
            *res = f64::NAN;
        } else {
            *res = tmpres * (x / (x - y)).sqrt();
        }
        return status;
    } else if too_small(y) || !ph_good(x) {
        *res = f64::NAN;
        return BadArgs;
    } else if x.is_infinite() || y.is_infinite() {
        *res = 0.0;
        return Success;
    }
    let mut am = (x + 2.0 * y) / 3.0;
    let mut fterm = (am - x).abs() / ocrt(3.0 * rerr);
    let mut xm = x;
    let mut ym = y;
    let mut sm = y - am;
    let mut m = 0_u32;
    while (xm - ym).abs().max(fterm) >= am.abs() {
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        let lam = 2.0 * xm.sqrt() * ym.sqrt() + ym;
        am = (am + lam) * 0.25;
        xm = (xm + lam) * 0.25;
        ym = (ym + lam) * 0.25;
        sm *= 0.25;
        fterm *= 0.25;
        m += 1;
    }
    am = (xm + ym + ym) / 3.0;
    sm /= am;
    // Eq. (20) of Carlson (1995)
    *res = comp_horner(sm, &RC_C) / (am.sqrt() * RC_C[0]);
    status
}

// ---- R_D

pub(crate) fn rd(x: f64, y: f64, z: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    if invalid_rerr(rerr, 1.0e-4) {
        *res = f64::NAN;
        return BadRerr;
    }
    if !(ph_good(x) && ph_good(y) && ph_good(z)) {
        *res = f64::NAN;
        return BadArgs;
    }
    if too_small(z) {
        *res = f64::INFINITY;
        return Singular;
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        *res = 0.0;
        return status;
    }
    if too_small(x) && too_small(y) {
        *res = f64::INFINITY;
        return Singular;
    }
    let mut am = nsum2(&[x, y, z, z, z], 5) / 5.0;
    let (mut xm, mut ym, mut zm) = (x, y, z);
    let mut xxm = am - xm;
    let mut yym = am - ym;
    let mut fterm = max_abs3(xxm, yym, am - z).abs() / ocrt(rerr / 5.0);
    let mut d4m = 1.0;
    let (mut adt, mut ade) = (0.0, 0.0);
    let mut m = 0_u32;
    loop {
        let aam = am.abs();
        if !(aam <= fterm || aam <= max_abs3(xxm, yym, am - zm).abs()) {
            break;
        }
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        let (sx, sy, sz) = (xm.sqrt(), ym.sqrt(), zm.sqrt());
        let lam = ndot2(&[sx, sy, sz], &[sy, sz, sx], 3);
        let xl = add_lam(xm, ym, zm, sx, sy, sz, lam);
        let yl = add_lam(ym, zm, xm, sy, sz, sx, lam);
        let zl = add_lam(zm, xm, ym, sz, sx, sy, lam);
        // The cumulative sum term of Carlson (1995) Eq. (41).
        let tmp = d4m / (sz * zl);
        sum2_acc(tmp, &mut adt, &mut ade);
        am = (am + lam) * 0.25;
        xm = xl * 0.25;
        ym = yl * 0.25;
        zm = zl * 0.25;
        xxm *= 0.25;
        yym *= 0.25;
        fterm *= 0.25;
        d4m *= 0.25;
        m += 1;
    }
    // Burn some extra cycles re-balancing Am as the "true" centroid
    am = nsum2(&[xm, ym, zm, zm, zm], 5) / 5.0;
    xxm /= am;
    yym /= am;
    let zzm = (xxm + yym) / -3.0;
    // The elementary terms E_n of Carlson (1995) Eqs. (39-40)
    let xy = xxm * yym;
    let zz2 = zzm * zzm;
    let e2 = xy - zz2 * 6.0;
    let e3 = (xy * 3.0 - zz2 * 8.0) * zzm;
    let e4 = (xy - zz2) * zz2 * 3.0;
    let e5 = xy * zz2 * zzm;
    let mut t = am.sqrt();
    let tmp = d4m / (t * t * t);
    // The 7th-degree expansion of Eq. 19.36.2 of https://dlmf.nist.gov/19.36#E2
    let c1 = [
        comp_horner(e2, &RDJ_C1),
        comp_horner(e3, &RDJ_C2),
        comp_horner(e2, &RDJ_C3),
        comp_horner(e2, &RDJ_C4),
        comp_horner(e2, &RDJ_C5),
        e3 * RDJ_C5[1],
    ];
    let c2 = [1.0, 1.0, e3, e4, e5, e4];
    t = dot2(&c1, &c2) / RDJ_DENOM + 1.0;
    // Combine into the final result, Eq. (41) of Carlson (1995).
    *res = ndot2(&[tmp, 3.0, 3.0], &[t, adt, ade], 3);
    status
}

// ---- R_G

fn rg0(x: f64, y: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    let rsq = 2.0 * rerr.sqrt();
    let mut fac = 0.25;
    let mut xm = x.sqrt();
    let mut ym = y.sqrt();
    let mut dm = (xm + ym) * 0.5;
    let mut sum = -dm * dm;
    let mut cor = 0.0;
    dm = xm - ym;
    let mut m = 0_u32;
    while dm.abs() >= rsq * xm.abs().min(ym.abs()) {
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        agm_update(&mut xm, &mut ym);
        // Carlson (1995) Eq. 2.39
        fac *= 2.0;
        dm = xm - ym;
        fdot2_acc(fac, dm * dm, &mut sum, &mut cor);
        m += 1;
    }
    *res = PI / (xm + ym);
    *res *= -0.5 * (sum + cor);
    status
}

pub(crate) fn rg(x: f64, y: f64, z: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let mut status = Success;
    if invalid_rerr(rerr, 1.0e-4) {
        *res = f64::NAN;
        return BadRerr;
    }
    let mut cct = [x, y, z];
    sort3(&mut cct, |a, b| a.abs() < b.abs());
    if (cct[0].is_infinite() || cct[1].is_infinite() || cct[2].is_infinite())
        && ph_good(cct[0])
        && ph_good(cct[1])
        && ph_good(cct[2])
    {
        *res = f64::INFINITY;
        return Singular;
    }
    if too_small(cct[0]) {
        if too_small(cct[1]) {
            // Special case -- also covers the case of z ~ zero.
            *res = cct[2].sqrt() * 0.5;
            return status;
        } else {
            // Special case -- use the AGM algorithm.
            return rg0(cct[1], cct[2], rerr, res);
        }
    }
    // DLMF 19.21.11 (second identity): 6 R_G is the sum of x (y + z) R_D over
    // cyclic permutations of (x, y, z), which is manifestly symmetric.
    let mut cct2 = [0.0; 3];
    let mut status_tmp;
    macro_rules! check_status_or_fail {
        () => {
            if status_tmp != Success {
                status = status_tmp;
            }
            if is_horrible(status) {
                *res = f64::NAN;
                return status;
            }
        };
    }
    status_tmp = rd(y, z, x, rerr, &mut cct2[0]);
    check_status_or_fail!();
    status_tmp = rd(z, x, y, rerr, &mut cct2[1]);
    check_status_or_fail!();
    status_tmp = rd(x, y, z, rerr, &mut cct2[2]);
    check_status_or_fail!();
    let cct1 = [
        dot2(&[x, x], &[y, z]),
        dot2(&[y, y], &[x, z]),
        dot2(&[z, z], &[x, y]),
    ];
    *res = dot2(&cct1, &cct2) / 6.0;
    status
}

// ---- R_J

#[derive(Clone, Copy, PartialEq, Eq)]
enum AsymFlag {
    Nothing,
    /// x, y, z << p
    HugeP,
    /// p << geometric mean of x, y, z
    TinyP,
    /// max(x, p) << min(y, z) == y
    HugeY,
    /// max(x, y) == y << min(z, p)
    TinyY,
    /// max(x, y, p) == max(y, p) << z
    HugeZ,
}

#[derive(Default)]
struct ArgCases {
    maybe_asymp: bool,
    retry_caupv: bool,
    good_infinity: bool,
    hit_pole: bool,
}

fn asymp_zero(r: f64) -> bool {
    0.0 < r && r <= ASYM_ZERO_UL
}

fn abs_close_zero(r: f64) -> bool {
    0.0 < r && r <= ASYM_CLOSE_UL
}

/// Whether the arguments are in the domain of the algorithm; `x <= y <= z`.
fn good_args(x: f64, y: f64, z: f64, p: f64, classify: &mut ArgCases) -> bool {
    classify.hit_pole = too_small(x) && too_small(y) && ph_good(z) && !too_small(p);
    if classify.hit_pole {
        return false;
    }
    classify.good_infinity =
        (x.is_infinite() || y.is_infinite() || z.is_infinite() || p.is_infinite())
            && (ph_good(x) && ph_good(y) && ph_good(z));
    if classify.good_infinity {
        return false;
    }
    // Carlson (1995), text above Eq. 2.26: "If x, y, z are real and
    // nonnegative, at most one of them is 0, and the fourth variable of RJ is
    // negative, the Cauchy principal value ..."
    let xyzreal_nonneg_atmost1z = x >= 0.0 && y > 0.0;
    if xyzreal_nonneg_atmost1z {
        classify.retry_caupv = p < 0.0;
        if classify.retry_caupv {
            return false;
        }
        // DLMF 19.27(iv): x, y, z real and nonnegative, at most one of them 0, p > 0.
        classify.maybe_asymp = p > 0.0;
        if classify.maybe_asymp {
            return true;
        }
    }
    // Carlson (1995), text above Eq. 2.17: x, y, z with nonnegative real part
    // and at most one of them 0, while Re p > 0.
    let (x0, y0, z0) = (too_small(x), too_small(y), too_small(z));
    if p > 0.0 && x >= 0.0 && !((y0 && (x0 || z0)) || (x0 && z0)) {
        return true;
    }
    // "Alternatively, if p != 0 and |ph p| < pi ..."
    if !too_small(p) && ph_good(p) {
        let r1conj2 = |a: f64, b: f64, c: f64| {
            too_small(a - b)
                && !(too_small(a) || too_small(b))
                && c >= 0.0
                && ph_good(a)
                && ph_good(b)
        };
        return xyzreal_nonneg_atmost1z || r1conj2(x, y, z) || r1conj2(y, z, x) || r1conj2(z, x, y);
    }
    false
}

/// Cauchy principal value for negative `p`, Carlson (1995) Eq. 2.26.
fn rj_cpv_dispatch(x: f64, y: f64, z: f64, p: f64, rerr: f64, res: &mut f64) -> ExitStatus {
    let xct1 = [x, y, -p, z];
    let mut xct2 = [0.0; 4];
    let xy = x * y;
    xct2[3] = 1.0 - p / z;
    let pn = (nsum2(&xct1, 3) - xy / xct1[3]) / xct2[3];
    let mut status = Success;
    let mut status_tmp = rj(x, y, z, pn, rerr, &mut xct2[0], false);
    if is_horrible(status_tmp) {
        return status_tmp;
    } else if is_troublesome(status_tmp) {
        status = status_tmp;
    }
    status_tmp = rf(x, y, z, rerr, &mut xct2[1]);
    if is_horrible(status_tmp) {
        return status_tmp;
    } else if is_troublesome(status_tmp) {
        status = status_tmp;
    }
    let pq = pn * xct1[2];
    let xypq = xy + pq;
    status_tmp = rc(xypq, pq, rerr, &mut xct2[2]);
    if is_horrible(status_tmp) {
        return status_tmp;
    } else if is_troublesome(status_tmp) {
        status = status_tmp;
    }
    let xct1 = [pn - z, -3.0, 3.0 * (xy * z / xypq).sqrt()];
    let mut tmpres = ndot2(&xct1, &xct2, 3);
    tmpres /= z - p;
    *res = tmpres;
    status
}

/// DLMF 19.27(i) notation.
#[derive(Default)]
struct AsymConfig {
    a: f64,
    b: f64,
    f: f64,
    g: f64,
    h: f64,
}

fn rj_asym_conf(x: f64, y: f64, z: f64, p: f64, conf: &mut AsymConfig) -> AsymFlag {
    let t = z / p;
    if asymp_zero(t) {
        return AsymFlag::HugeP;
    }
    // This bound is sharp. RJ behaves with a logarithmic singularity as p -> +0.
    if abs_close_zero(p) || (x != 0.0 && asymp_zero(p / x)) {
        conf.f = (x * y * z).sqrt();
        return AsymFlag::TinyP;
    }
    let t = y / z.min(p);
    if (0.0 < y && y <= 1e-26) || asymp_zero(t) {
        // bound fairly sharp even if p is large
        conf.a = 0.5 * (x + y);
        conf.g = (x * y).sqrt();
        if (conf.a / z + conf.a / p) * (p / conf.a).ln().abs() <= 1.0 {
            return AsymFlag::TinyY;
        }
    }
    if x != 0.0 && asymp_zero(z.max(p) / x) {
        // The bound might not be sharp if x + 2p is much larger than (yz)^2,
        // but this is unlikely to be true anyway.
        return AsymFlag::HugeY;
    }
    if z != 0.0 && asymp_zero(y.max(p) / z) {
        conf.b = 0.5 * (x + y);
        conf.h = (x * y).sqrt();
        // when bounds are sharp
        if (z / (conf.b + conf.h)).ln().abs() <= z.sqrt() {
            return AsymFlag::HugeZ;
        }
    }
    AsymFlag::Nothing
}

/// atan(sqrt(x)) / sqrt(x), continued to negative `x` with atanh and to
/// `x = 0` with its limit 1.
fn safe_atan_sqrt_div(x: f64) -> f64 {
    if too_small(x) {
        return 1.0;
    }
    if x < 0.0 {
        let s = (-x).sqrt();
        return s.atanh() / s;
    }
    let s = x.sqrt();
    s.atan() / s
}

/// State of the duplication of `rj` (Carlson & FitzSimmons, Theorem A.1).
struct RjState {
    xm: f64,
    ym: f64,
    zm: f64,
    pm: f64,
    am: f64,
    xxm: f64,
    yym: f64,
    zzm: f64,
    d4m: f64,
    fterm: f64,
}

impl RjState {
    /// `(sqrt(x), sqrt(y), sqrt(z), lam, dm)` of the current iterate.
    fn calc(&self) -> (f64, f64, f64, f64, f64) {
        let prm = self.pm.sqrt();
        let (sx, sy, sz) = (self.xm.sqrt(), self.ym.sqrt(), self.zm.sqrt());
        let lam = ndot2(&[sx, sy, sz], &[sz, sx, sy], 3);
        let dm = (prm + sx) * (prm + sy) * (prm + sz);
        (sx, sy, sz, lam, dm)
    }

    fn update(&mut self, sx: f64, sy: f64, sz: f64, lam: f64) {
        let xl = add_lam(self.xm, self.ym, self.zm, sx, sy, sz, lam);
        let yl = add_lam(self.ym, self.zm, self.xm, sy, sz, sx, lam);
        let zl = add_lam(self.zm, self.xm, self.ym, sz, sx, sy, lam);
        self.am = (self.am + lam) * 0.25;
        self.xm = xl * 0.25;
        self.ym = yl * 0.25;
        self.zm = zl * 0.25;
        self.pm = (self.pm + lam) * 0.25;
        self.xxm *= 0.25;
        self.yym *= 0.25;
        self.zzm *= 0.25;
        self.d4m *= 0.25;
        self.fterm *= 0.25;
    }
}

pub(crate) fn rj(
    x: f64,
    y: f64,
    z: f64,
    p: f64,
    rerr: f64,
    res: &mut f64,
    noasymp: bool,
) -> ExitStatus {
    let mut status = Success;
    if invalid_rerr(rerr, 1.0e-4) {
        *res = f64::NAN;
        return BadRerr;
    }
    // Put the symmetric arguments in order.
    let mut cct1 = [x, y, z];
    sort3(&mut cct1, |a, b| a < b);
    let (xm, ym, zm) = (cct1[0], cct1[1], cct1[2]);
    let mut classify = ArgCases::default();
    if !good_args(xm, ym, zm, p, &mut classify) {
        if classify.good_infinity {
            *res = 0.0;
            return Success;
        }
        if classify.retry_caupv {
            // Retry with principal value evaluation, valid for reals.
            let mut tmpres = 0.0;
            status = rj_cpv_dispatch(xm, ym, zm, p, rerr, &mut tmpres);
            if is_horrible(status) {
                *res = f64::NAN;
            } else {
                *res = tmpres;
            }
            return status;
        } else if classify.hit_pole {
            *res = f64::INFINITY;
            return Singular;
        } else {
            *res = f64::NAN;
            return BadArgs;
        }
    }
    if classify.maybe_asymp && !noasymp {
        // Might be dealt with by an asymptotic expansion of the real-argument RJ.
        let mut tmpres = 0.0;
        let mut config = AsymConfig::default();
        let (xr, yr, zr, pr) = (xm, ym, zm, p);
        let cres = rj_asym_conf(xr, yr, zr, pr, &mut config);
        match cres {
            AsymFlag::Nothing => {}
            AsymFlag::HugeP => {
                // DLMF 19.27.11
                status = rf(xr, yr, zr, rerr * 0.5, &mut tmpres);
                tmpres = 3.0 * (tmpres - 0.5 * PI / pr.sqrt()) / pr;
            }
            AsymFlag::TinyP => {
                // Gustafson (1982), Table 2 and Theorem 11 (Eq. 4.12)
                let mut xct2 = [0.0; 3];
                let r = rerr * 0.5;
                let (sx, sy, sz) = (xr.sqrt(), yr.sqrt(), zr.sqrt());
                let xct1 = [sz, sx, sy];
                let xct2_in = [sx, sy, sz];
                let lamt = dot2(&xct1, &xct2_in);
                let xct3 = [pr, pr, pr];
                let mut alpha = dot2(&xct1, &xct3) + config.f;
                alpha *= alpha;
                let mut beta = pr + lamt;
                beta = beta * beta * pr;
                let status_tmp = rc(alpha, beta, r, &mut xct2[0]);
                status = rj(
                    xr + lamt,
                    yr + lamt,
                    zr + lamt,
                    pr + lamt,
                    r,
                    &mut xct2[1],
                    false,
                );
                if status_tmp != Success {
                    status = status_tmp;
                }
                tmpres = ndot2(&[3.0, 2.0], &xct2, 2);
            }
            AsymFlag::HugeY => {
                // DLMF 19.27.14
                let (mut t1, mut t2) = (0.0, 0.0);
                let r = rerr / 3.0;
                tmpres = 1.0 / (yr * zr).sqrt();
                let status_tmp = rc(xr, pr, r, &mut t1);
                status = rg(0.0, yr, zr, r, &mut t2);
                if status_tmp != Success {
                    status = status_tmp;
                }
                tmpres *= 3.0 * t1 - 2.0 * t2 * tmpres;
            }
            AsymFlag::TinyY => {
                // Carlson & Gustafson (1994), Proposition J3
                let mut tx = 0.0;
                status = rc(1.0, pr / zr, rerr, &mut tx);
                tmpres = (8.0 * zr / (config.a + config.g)).ln() - 2.0 * tx;
                tx = (2.0 * pr / (config.a + config.g)).ln() / pr;
                let rt_h = tx * config.a * (1.0 + 0.5 * pr / zr) / (1.0 - config.a / pr);
                tx = 1.5 / (zr.sqrt() * pr);
                tmpres *= tx;
                tmpres += rt_h * tx;
            }
            AsymFlag::HugeZ => {
                // Switch between Carlson & Gustafson (1994), Proposition J6,
                // and direct computation without asymptotics.
                let mut tt = config.h + pr;
                tt *= tt;
                let tm = config.b + config.h;
                status = rc(tt, 2.0 * tm * pr, rerr * 0.5, &mut tmpres);
                tt = (2.0 * zr / tm).ln();
                let rt_h = 0.75 * (2.0 * pr * tmpres / zr - tt / (zr - config.h));
                tt = 3.0 / zr.sqrt();
                tmpres *= tt;
                // Also try if direct computation gives reasonable approximation.
                let mut tmp_direct = 0.0;
                let status_direct = rj(xr, yr, zr, pr, rerr * 0.5, &mut tmp_direct, true);
                if tmp_direct > tmpres || tmp_direct < tmpres + rt_h * tt {
                    tmpres += tt * 0.3 * rt_h;
                } else {
                    tmpres = tmp_direct;
                    status = status_direct;
                }
            }
        }
        if cres != AsymFlag::Nothing {
            *res = tmpres;
            return status;
        }
    }
    let am = nsum2(&[xm, ym, zm, p, p], 5) / 5.0;
    let delta = (p - xm) * (p - ym) * (p - zm);
    let mut st = RjState {
        xm,
        ym,
        zm,
        pm: p,
        am,
        xxm: am - xm,
        yym: am - ym,
        zzm: am - zm,
        d4m: 1.0,
        fterm: 0.0,
    };
    st.fterm = max_abs3(st.xxm, st.yym, st.zzm)
        .abs()
        .max((st.am - p).abs())
        / ocrt(rerr / 5.0);
    // The algorithm is a direct implementation of Carlson & FitzSimmons (2000),
    // Theorem A.1, where a purely arithmetic recurrence relation is used
    // instead of evaluating special functions in the inner loop.
    let (sx, sy, sz, lam, dm) = st.calc();
    let mut sm = dm * 0.5;
    st.update(sx, sy, sz, lam);
    let mut m = 1_u32;
    loop {
        let aam = st.am.abs();
        let tail = max_abs3(st.xxm, st.yym, st.zzm)
            .abs()
            .max((st.am - st.pm).abs());
        if !(aam <= st.fterm || aam <= tail) {
            break;
        }
        if m > MAX_ITER {
            status = NIter;
            break;
        }
        let rm = sm * (((delta * st.d4m) / (sm * sm) + 1.0).sqrt() + 1.0);
        let (sx, sy, sz, lam, dm) = st.calc();
        sm = (rm * dm - delta * (st.d4m * st.d4m)) * 0.5 / (dm + rm * st.d4m);
        st.update(sx, sy, sz, lam);
        m += 1;
    }
    // Burn some extra cycles re-balancing Am as the "true" centroid
    let am = nsum2(&[st.xm, st.ym, st.zm, st.pm, st.pm], 5) / 5.0;
    let (xxm, yym, zzm) = (st.xxm / am, st.yym / am, st.zzm / am);
    let pp = nsum2(&[xxm, yym, zzm], 3) * -0.5;
    let pp2 = pp * pp;
    let xyz = yym * zzm * xxm;
    // The elementary terms E_n of Carlson (1995) Eqs. (31-32):
    // e2 = xxm yym + zzm xxm + yym zzm - 3 pp^2
    let e2 = ndot2(&[xxm, yym, zzm, pp * -3.0], &[yym, zzm, xxm, pp], 4);
    let e3 = xyz + pp * 2.0 * (e2 + pp2 * 2.0);
    let e4 = (2.0 * xyz + (e2 + 3.0 * pp2) * pp) * pp;
    let e5 = xyz * pp2;
    let mut t = am.sqrt();
    let mut tmp = st.d4m / (t * t * t);
    // The 7th-degree expansion of Eq. 19.36.2 of https://dlmf.nist.gov/19.36#E2
    let c1 = [
        comp_horner(e2, &RDJ_C1),
        comp_horner(e3, &RDJ_C2),
        comp_horner(e2, &RDJ_C3),
        comp_horner(e2, &RDJ_C4),
        comp_horner(e2, &RDJ_C5),
        e3 * RDJ_C5[1],
    ];
    let c2 = [1.0, 1.0, e3, e4, e5, e4];
    t = dot2(&c1, &c2) / RDJ_DENOM + 1.0;
    tmp *= t;
    t = delta * st.d4m / (sm * sm);
    // The library atan, following Johansson (2019), Sec. 1.6.1.
    tmp += safe_atan_sqrt_div(t) * 3.0 / sm;
    *res = tmp;
    status
}

// ---- public wrappers (`cpu/ellint_carlson.h`)

const RERR: f64 = 5e-16;

pub(crate) fn elliprc(x: f64, y: f64) -> f64 {
    let mut r = 0.0;
    rc(x, y, RERR, &mut r);
    r
}

pub(crate) fn elliprd(x: f64, y: f64, z: f64) -> f64 {
    let mut r = 0.0;
    rd(x, y, z, RERR, &mut r);
    r
}

pub(crate) fn elliprf(x: f64, y: f64, z: f64) -> f64 {
    let mut r = 0.0;
    rf(x, y, z, RERR, &mut r);
    r
}

/// Largest exponent, in bits, of a nonzero argument that [`elliprg`] and
/// [`elliprj`] pass to xsf unscaled. R_J forms `(p - x)(p - y)(p - z)` and
/// R_G `z (x + y)`, which leave the `f64` range from 2^±341 and 2^±512 on.
const UNSCALED_BITS: i32 = 300;

/// How [`elliprg`] and [`elliprj`] scale their arguments by `4^-j` before
/// xsf forms its products (not in xsf; tensor4all/tenferro-sci-rs#25). R_G
/// and R_J are homogeneous of degree 1/2 and -3/2 (DLMF 19.16.3, 19.16.2),
/// and scaling a normal argument by a power of four is exact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scaling {
    /// Every nonzero finite argument lies within 2^±[`UNSCALED_BITS`], one is
    /// subnormal (xsf treats a subnormal argument as zero, and scaling it up
    /// would undo that), or the largest and the smallest nonzero argument are
    /// more than 2^(2 [`UNSCALED_BITS`]) apart: xsf's path, unchanged. No
    /// power of four brings such a spread within range, and evaluating xsf
    /// on it with only the largest moved into range gave values off by up to
    /// 13 orders of magnitude where xsf returns NaN (review of 470803a).
    None,
    /// The largest and the smallest nonzero argument are at most
    /// 2^(2 [`UNSCALED_BITS`]) apart: they are moved by the power of four that
    /// brings the argument beyond 2^±[`UNSCALED_BITS`] just inside it (to
    /// 2^±299 or 2^±300), which leaves every argument within
    /// 2^±([`UNSCALED_BITS`] + 1), so every product of three stays in range. xsf picks some asymptotic
    /// expansions by absolute thresholds (`p <= 1e-9`, `y <= 1e-26`), so its
    /// value can depend on the scale by about 1e-11; xsf's unscaled value is
    /// therefore kept when it agrees with the scaled one to [`AGREE`], and the
    /// scaled value is used when it does not (the unscaled one is then
    /// damaged by underflow: off by more than 1e-10, up to a wrong sign).
    Window(i32),
}

fn scaling(args: &[f64]) -> Scaling {
    let (mut max, mut min) = (i32::MIN, i32::MAX);
    for v in args.iter().map(|a| a.abs()) {
        if v.is_subnormal() {
            return Scaling::None;
        }
        if v != 0.0 && v.is_finite() {
            let e = v.log2().floor() as i32;
            max = max.max(e);
            min = min.min(e);
        }
    }
    if max == i32::MIN
        || (max <= UNSCALED_BITS && min >= -UNSCALED_BITS)
        || max - min > 2 * UNSCALED_BITS
    {
        Scaling::None
    } else if max > UNSCALED_BITS {
        Scaling::Window((max - UNSCALED_BITS + 1).div_euclid(2))
    } else {
        Scaling::Window((min + UNSCALED_BITS).div_euclid(2))
    }
}

/// True for a value of R_G or R_J that can be kept: finite and nonzero. A
/// product leaving the range gives NaN, an infinity or 0.
fn usable(r: f64) -> bool {
    r.is_finite() && r != 0.0
}

/// Relative agreement between xsf's unscaled and scaled value within which
/// [`Scaling::Window`] keeps the unscaled one without further evaluation.
/// xsf's value depends on the scale both through underflow (the unscaled
/// value is then wrong) and through absolute thresholds of its asymptotic
/// expansions (`p <= 1e-9`, `y <= 1e-26`; either value can then be the wrong
/// one, by up to 2e-6), so a disagreement is settled by [`arbiter`].
const AGREE: f64 = 1e-10;

/// The value [`Scaling::Window`] returns when xsf's unscaled value `r` and
/// scaled value `rs` disagree beyond [`AGREE`] (or one is not [`usable`]).
///
/// Invariant: a usable `r` (xsf's value at the caller's arguments, which is
/// what the port returned before scaling existed) is returned unchanged
/// unless an independent evaluation, `reference` (the precision-adaptive
/// algorithm of [`crate::generic::carlson`] evaluated in `f64`, which scales
/// its own arguments and has no absolute thresholds), shows it to be off by
/// more than [`AGREE`] and `rs` is closer to `reference`. Only an unusable
/// `r` (NaN, infinite or 0) is replaced without that evidence, by a usable
/// `rs`. A scaled evaluation therefore never replaces a value the reference
/// agrees with, and when the reference is unusable `r` is kept.
fn arbiter(r: f64, rs: f64, reference: impl FnOnce() -> f64) -> f64 {
    if !usable(r) {
        return if usable(rs) { rs } else { r };
    }
    if !usable(rs) {
        return r;
    }
    let reference = reference();
    let r_off = (r - reference).abs();
    if usable(reference) && r_off > AGREE * reference.abs() && (rs - reference).abs() < r_off {
        rs
    } else {
        r
    }
}

/// Evaluates `f` (R_G or R_J at arguments scaled by `4^-j`, rescaled) as
/// [`Scaling`] says; `reference` (see [`arbiter`]) only when the unscaled and
/// the scaled value disagree.
fn scaled(scaling: Scaling, f: impl Fn(i32) -> f64, reference: impl FnOnce() -> f64) -> f64 {
    match scaling {
        Scaling::None => f(0),
        Scaling::Window(j) => {
            let r = f(0);
            let rs = f(j);
            if usable(r) && usable(rs) && (r - rs).abs() <= AGREE * rs.abs() {
                r
            } else {
                arbiter(r, rs, reference)
            }
        }
    }
}

/// Condition number `(|t1| + |t2| + |t3|) / |t1 + t2 + t3|` of the
/// combination [`rj_cpv_dispatch`] evaluates for the principal value of R_J
/// with `p < 0` (Carlson 1995 Eq. 2.26, for sorted `x <= y <= z`): the terms
/// `(q - z) R_J(x, y, z, q)`, `-3 R_F(x, y, z)` and
/// `3 sqrt(x y z / (x y + p q)) R_C(x y + p q, p q)` with
/// `q = (x + y - p - x y / z) / (1 - p / z)`, formed as it forms them. The
/// condition number of DLMF 19.20.14, which the `Df64` path evaluates, is a
/// different one and can be much larger (review of 3a7bc8c).
fn pv_condition(x: f64, y: f64, z: f64, p: f64) -> f64 {
    let mut v = [x, y, z];
    sort3(&mut v, |a, b| a < b);
    let [x, y, z] = v;
    let xy = x * y;
    let q = (nsum2(&[x, y, -p], 3) - xy / z) / (1.0 - p / z);
    let pq = q * -p;
    let xypq = xy + pq;
    let (mut a, mut b, mut c) = (0.0, 0.0, 0.0);
    rj(x, y, z, q, RERR, &mut a, false);
    rf(x, y, z, RERR, &mut b);
    rc(xypq, pq, RERR, &mut c);
    let t = [(q - z) * a, -3.0 * b, 3.0 * (xy * z / xypq).sqrt() * c];
    (t[0].abs() + t[1].abs() + t[2].abs()) / (t[0] + t[1] + t[2]).abs()
}

pub(crate) fn elliprg(x: f64, y: f64, z: f64) -> f64 {
    scaled(
        scaling(&[x, y, z]),
        |j| {
            let s = |v: f64| ldexp(v, -2 * j);
            let mut r = 0.0;
            rg(s(x), s(y), s(z), RERR, &mut r);
            ldexp(r, j)
        },
        || crate::generic::carlson::elliprg(x, y, z),
    )
}

pub(crate) fn elliprj(x: f64, y: f64, z: f64, p: f64) -> f64 {
    let scaling = scaling(&[x, y, z, p]);
    // Where the arguments are scaled, a principal value whose transformation
    // cancels has lost its digits whichever of the unscaled and the scaled
    // value is returned (the condition number does not depend on the scale,
    // so the two agree); return NaN rather than that.
    // Only where xsf takes that path (Carlson 1995, text above Eq. 2.26:
    // `x, y, z` real, nonnegative, at most one zero, `p < 0`, all finite); the
    // pole of two zero arguments, an infinite argument and the domain errors
    // keep xsf's value.
    let mut v = [x, y, z];
    sort3(&mut v, |a, b| a < b);
    let principal = p < 0.0 && p.is_finite() && v[0] >= 0.0 && v[1] > 0.0 && v[2].is_finite();
    if let (Scaling::Window(j), true) = (scaling, principal) {
        let s = |v: f64| ldexp(v, -2 * j);
        if pv_ill_conditioned(pv_condition(s(x), s(y), s(z), s(p)), f64::EPSILON) {
            return f64::NAN;
        }
    }
    scaled(
        scaling,
        |j| {
            let s = |v: f64| ldexp(v, -2 * j);
            let mut r = 0.0;
            rj(s(x), s(y), s(z), s(p), RERR, &mut r, false);
            ldexp(r, -3 * j)
        },
        || crate::generic::carlson::elliprj(x, y, z, p),
    )
}
