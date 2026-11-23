//! Carlson's symmetric elliptic integrals R_C, R_D, R_F, R_G, R_J,
//! precision-adaptive.
//!
//! Written from the duplication theorems of DLMF 19.26(ii) and the series of
//! DLMF 19.36(i) (Eqs. 19.36.1 for R_F, 19.36.2 for R_D, 19.36.3 for R_J and
//! the R_C series that follows them), with R_G from the cyclic form of
//! DLMF 19.21.11, R_G(0, y, z) from E(m) = 2 R_G(0, 1 - m, 1) (DLMF 19.25.1),
//! the principal value of R_J for `p < 0` from DLMF 19.20.14 and that of R_C
//! for `y < 0` from DLMF 19.2.20. No code was translated.
//!
//! Each duplication step replaces the arguments by `(x + λ)/4`, `λ = √x√y +
//! √y√z + √z√x`, which shrinks their relative spread by a factor of 4. The
//! iteration stops when the spread `ε` satisfies `ε⁸ <= 10⁻³` times the
//! epsilon of the scalar type, and the series, whose error is O(ε⁸), is
//! evaluated at the remaining arguments. Every term of the sums is positive
//! for the real arguments of the integrals, so there is no cancellation,
//! except in the principal value of R_J, whose result can be small against
//! its terms.
//!
//! The conventions for zeros, infinities and domain errors (including xsf's
//! treatment of a subnormal argument as zero when deciding between a finite
//! and an infinite result) are those of `scipy.special.elliprc` and friends.

use super::ellip::ellipe_core;
use super::{div, ldexp, mul, sqrt};
use crate::real::{k, Real};

/// Longest duplication loop; each step divides the spread by 4, so 800 steps
/// reach any double-double precision from any representable ratio.
const MAX_STEPS: usize = 800;

/// `ε` below which the `O(ε⁸)` series is accurate to `10⁻³` epsilon.
fn spread_limit<T: Real>() -> T {
    (T::epsilon() * k::<T>(1e-3)).sqrt().sqrt().sqrt()
}

/// True for zero and for a subnormal `f64` high word, which xsf treats as zero.
fn is_small<T: Real>(x: T) -> bool {
    x.abs().approx_f64() < f64::MIN_POSITIVE
}

/// `j` such that the arguments scaled by `4^-j` have a largest entry of about
/// 1, so that the sums and products of the duplication neither overflow nor
/// push the double-double low word into the subnormal range. 0 (no scaling)
/// for moderate arguments, and when scaling would push a nonzero argument
/// below 2^-960.
fn scale_exponent<T: Real>(args: &[T]) -> i32 {
    let max = args
        .iter()
        .map(|a| a.abs().approx_f64())
        .fold(0.0, f64::max);
    if !max.is_finite() || max == 0.0 {
        return 0;
    }
    let e = max.log2().floor() as i32;
    if e.abs() < 200 {
        return 0;
    }
    let j = e / 2;
    for a in args {
        let v = a.abs().approx_f64();
        if v != 0.0 && v.log2() - f64::from(2 * j) < -960.0 {
            return 0;
        }
    }
    j
}

/// Sort three values ascending (NaN-safe: only `<` is used).
fn sort3<T: Real>(v: &mut [T; 3]) {
    for i in 1..3 {
        let mut j = i;
        while j > 0 && v[j] < v[j - 1] {
            v.swap(j, j - 1);
            j -= 1;
        }
    }
}

fn max_of<T: Real>(vals: &[T]) -> T {
    vals.iter()
        .fold(T::zero(), |m, &v| if v > m { v } else { m })
}

// ---- cores: finite arguments in the domain, no scaling

/// R_C(x, y) for `x >= 0`, `y > 0` (DLMF 19.36.4).
fn rc_core<T: Real>(x: T, y: T) -> T {
    let three = k::<T>(3.0);
    let four = k::<T>(4.0);
    let limit = spread_limit::<T>();
    let (mut x, mut y) = (x, y);
    for _ in 0..MAX_STEPS {
        let a = (x + y + y) / three;
        let spread = max_of(&[(a - x).abs(), (a - y).abs()]) / a;
        if spread < limit {
            let s = (y - a) / a;
            // DLMF 19.36.4: 1 + 3s²/10 + s³/7 + 3s⁴/8 + 9s⁵/22 + 159s⁶/208 + 9s⁷/8.
            let poly = T::one()
                + s * s
                    * (k::<T>(3.0) / k::<T>(10.0)
                        + s * (T::one() / k::<T>(7.0)
                            + s * (k::<T>(3.0 / 8.0)
                                + s * (k::<T>(9.0) / k::<T>(22.0)
                                    + s * (k::<T>(159.0) / k::<T>(208.0)
                                        + s * k::<T>(9.0 / 8.0))))));
            return poly / sqrt(a);
        }
        let lam = k::<T>(2.0) * sqrt(x) * sqrt(y) + y;
        x = (x + lam) / four;
        y = (y + lam) / four;
    }
    T::nan()
}

/// The polynomial of DLMF 19.36.2 and 19.36.3 in the elementary symmetric
/// functions `E2..E5` (the same polynomial serves R_D and R_J).
fn rdj_poly<T: Real>(e2: T, e3: T, e4: T, e5: T) -> T {
    let c = |num: f64, den: f64| k::<T>(num) / k::<T>(den);
    T::one() - c(3.0, 14.0) * e2 + e3 / k::<T>(6.0) + c(9.0, 88.0) * e2 * e2
        - c(3.0, 22.0) * e4
        - c(9.0, 52.0) * e2 * e3
        + c(3.0, 26.0) * e5
        - e2 * e2 * e2 / k::<T>(16.0)
        + c(3.0, 40.0) * e3 * e3
        + c(3.0, 20.0) * e2 * e4
        + c(45.0, 272.0) * e2 * e2 * e3
        - c(9.0, 68.0) * (e3 * e4 + e2 * e5)
}

/// R_F(x, y, z) for `x, y, z >= 0` with at most one zero (DLMF 19.36.1).
fn rf_core<T: Real>(x: T, y: T, z: T) -> T {
    let three = k::<T>(3.0);
    let four = k::<T>(4.0);
    let limit = spread_limit::<T>();
    let (mut x, mut y, mut z) = (x, y, z);
    for _ in 0..MAX_STEPS {
        let a = (x + y + z) / three;
        let spread = max_of(&[(a - x).abs(), (a - y).abs(), (a - z).abs()]) / a;
        if spread < limit {
            let xx = (a - x) / a;
            let yy = (a - y) / a;
            let zz = -(xx + yy);
            let e2 = xx * yy - zz * zz;
            let e3 = xx * yy * zz;
            let c = |num: f64, den: f64| k::<T>(num) / k::<T>(den);
            // DLMF 19.36.1.
            let poly = T::one() - e2 / k::<T>(10.0) + e3 / k::<T>(14.0) + e2 * e2 / k::<T>(24.0)
                - c(3.0, 44.0) * e2 * e3
                - c(5.0, 208.0) * e2 * e2 * e2
                + c(3.0, 104.0) * e3 * e3
                + e2 * e2 * e3 / k::<T>(16.0);
            return poly / sqrt(a);
        }
        let (sx, sy, sz) = (sqrt(x), sqrt(y), sqrt(z));
        let lam = sx * sy + sy * sz + sz * sx;
        x = (x + lam) / four;
        y = (y + lam) / four;
        z = (z + lam) / four;
    }
    T::nan()
}

/// R_D(x, y, z) for `x, y >= 0` (not both zero) and `z > 0` (DLMF 19.36.2).
pub(super) fn rd_core<T: Real>(x: T, y: T, z: T) -> T {
    let three = k::<T>(3.0);
    let four = k::<T>(4.0);
    let five = k::<T>(5.0);
    let limit = spread_limit::<T>();
    let (mut x, mut y, mut z) = (x, y, z);
    let mut sum = T::zero();
    let mut fac = T::one();
    for _ in 0..MAX_STEPS {
        let a = (x + y + three * z) / five;
        let spread = max_of(&[(a - x).abs(), (a - y).abs(), (a - z).abs()]) / a;
        if spread < limit {
            let xx = (a - x) / a;
            let yy = (a - y) / a;
            let zz = -(xx + yy) / three;
            let e2 = xx * yy - k::<T>(6.0) * zz * zz;
            let e3 = (three * xx * yy - k::<T>(8.0) * zz * zz) * zz;
            let e4 = three * (xx * yy - zz * zz) * zz * zz;
            let e5 = xx * yy * zz * zz * zz;
            return three * sum + fac * rdj_poly(e2, e3, e4, e5) / a / sqrt(a);
        }
        let (sx, sy, sz) = (sqrt(x), sqrt(y), sqrt(z));
        let lam = sx * sy + sy * sz + sz * sx;
        // DLMF 19.26.19: R_D(x, y, z) = 3 Σ 4^-n / (√z_n (z_n + λ_n)) + 4^-N R_D(x_N, ...).
        sum = sum + fac / sz / (z + lam);
        fac = fac / four;
        x = (x + lam) / four;
        y = (y + lam) / four;
        z = (z + lam) / four;
    }
    T::nan()
}

/// R_J(x, y, z, p) for `x, y, z >= 0` (at most one zero) and `p > 0`
/// (DLMF 19.36.3, with R_C in the duplication sum).
fn rj_core<T: Real>(x: T, y: T, z: T, p: T) -> T {
    let three = k::<T>(3.0);
    let four = k::<T>(4.0);
    let five = k::<T>(5.0);
    let two = k::<T>(2.0);
    let limit = spread_limit::<T>();
    let (mut x, mut y, mut z, mut p) = (x, y, z, p);
    let mut sum = T::zero();
    let mut fac = T::one();
    for _ in 0..MAX_STEPS {
        let a = (x + y + z + two * p) / five;
        let spread = max_of(&[(a - x).abs(), (a - y).abs(), (a - z).abs(), (a - p).abs()]) / a;
        if spread < limit {
            let xx = (a - x) / a;
            let yy = (a - y) / a;
            let zz = (a - z) / a;
            let pp = -(xx + yy + zz) / two;
            let e2 = xx * yy + xx * zz + yy * zz - three * pp * pp;
            let xyz = xx * yy * zz;
            let e3 = xyz + two * e2 * pp + k::<T>(4.0) * pp * pp * pp;
            let e4 = (two * xyz + e2 * pp + three * pp * pp * pp) * pp;
            let e5 = xyz * pp * pp;
            return three * sum + fac * rdj_poly(e2, e3, e4, e5) / a / sqrt(a);
        }
        let (sx, sy, sz) = (sqrt(x), sqrt(y), sqrt(z));
        let lam = sx * sy + sy * sz + sz * sx;
        // DLMF 19.26(ii): the duplication of R_J carries R_C(α, β).
        let s = p * (sx + sy + sz) + sx * sy * sz;
        let alpha = s * s;
        let t = p + lam;
        let beta = p * t * t;
        sum = sum + fac * rc_core(alpha, beta);
        fac = fac / four;
        x = (x + lam) / four;
        y = (y + lam) / four;
        z = (z + lam) / four;
        p = (p + lam) / four;
    }
    T::nan()
}

// ---- public functions: domain, special values and scaling

/// R_C(x, y); for `y < 0`, the Cauchy principal value
/// R_C(x, y) = √(x/(x - y)) R_C(x - y, -y) (DLMF 19.2.20).
pub(crate) fn elliprc<T: Real>(x: T, y: T) -> T {
    if x.is_nan() || y.is_nan() {
        return T::nan();
    }
    if y < T::zero() {
        let t = elliprc(x - y, -y);
        return mul(t, sqrt(div(x, x - y)));
    }
    if is_small(y) || !(x >= T::zero()) {
        return T::nan();
    }
    if x.is_infinite() || y.is_infinite() {
        return T::zero();
    }
    let j = scale_exponent(&[x, y]);
    let r = rc_core(ldexp(x, -2 * j), ldexp(y, -2 * j));
    ldexp(r, -j)
}

pub(crate) fn elliprf<T: Real>(x: T, y: T, z: T) -> T {
    if x.is_nan()
        || y.is_nan()
        || z.is_nan()
        || !(x >= T::zero() && y >= T::zero() && z >= T::zero())
    {
        return T::nan();
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return T::zero();
    }
    if [x, y, z].iter().filter(|&&v| is_small(v)).count() >= 2 {
        return T::infinity();
    }
    let j = scale_exponent(&[x, y, z]);
    let r = rf_core(ldexp(x, -2 * j), ldexp(y, -2 * j), ldexp(z, -2 * j));
    ldexp(r, -j)
}

pub(crate) fn elliprd<T: Real>(x: T, y: T, z: T) -> T {
    if x.is_nan()
        || y.is_nan()
        || z.is_nan()
        || !(x >= T::zero() && y >= T::zero() && z >= T::zero())
    {
        return T::nan();
    }
    if is_small(z) {
        return T::infinity();
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return T::zero();
    }
    if is_small(x) && is_small(y) {
        return T::infinity();
    }
    let j = scale_exponent(&[x, y, z]);
    let r = rd_core(ldexp(x, -2 * j), ldexp(y, -2 * j), ldexp(z, -2 * j));
    ldexp(r, -3 * j)
}

/// R_G(x, y, z) for `x, y, z >= 0` (finite).
fn rg_finite<T: Real>(x: T, y: T, z: T) -> T {
    let mut v = [x, y, z];
    sort3(&mut v);
    let (x, y, z) = (v[0], v[1], v[2]);
    if is_small(x) {
        if is_small(y) {
            return sqrt(z) * k::<T>(0.5);
        }
        // R_G(0, y, z) = (√z / 2) E(1 - y/z): homogeneity of degree 1/2 and
        // E(m) = 2 R_G(0, 1 - m, 1).
        let p = y / z;
        let e = ellipe_core(T::one() - p, p);
        return sqrt(z) * e * k::<T>(0.5);
    }
    let j = scale_exponent(&[x, y, z]);
    let (xs, ys, zs) = (ldexp(x, -2 * j), ldexp(y, -2 * j), ldexp(z, -2 * j));
    // DLMF 19.21.11: 6 R_G = Σ x (y + z) R_D(y, z, x) over cyclic permutations.
    let six = rd_core(ys, zs, xs) * xs * (ys + zs)
        + rd_core(zs, xs, ys) * ys * (zs + xs)
        + rd_core(xs, ys, zs) * zs * (xs + ys);
    ldexp(six / k::<T>(6.0), j)
}

pub(crate) fn elliprg<T: Real>(x: T, y: T, z: T) -> T {
    if x.is_nan()
        || y.is_nan()
        || z.is_nan()
        || !(x >= T::zero() && y >= T::zero() && z >= T::zero())
    {
        return T::nan();
    }
    if x.is_infinite() || y.is_infinite() || z.is_infinite() {
        return T::infinity();
    }
    rg_finite(x, y, z)
}

/// R_J for `x <= y <= z`, `y > 0`, `p < 0`: the Cauchy principal value
/// (DLMF 19.20.14).
fn rj_principal<T: Real>(x: T, y: T, z: T, p: T) -> T {
    let g = y + (z - y) * (y - x) / (y - p);
    let rj = rj_scaled(x, y, z, g);
    let rf = elliprf(x, y, z);
    let rc = elliprc(x * z / y, p * g / y);
    ((g - y) * rj - k::<T>(3.0) * rf + k::<T>(3.0) * rc) / (y - p)
}

fn rj_scaled<T: Real>(x: T, y: T, z: T, p: T) -> T {
    let j = scale_exponent(&[x, y, z, p]);
    let r = rj_core(
        ldexp(x, -2 * j),
        ldexp(y, -2 * j),
        ldexp(z, -2 * j),
        ldexp(p, -2 * j),
    );
    ldexp(r, -3 * j)
}

pub(crate) fn elliprj<T: Real>(x: T, y: T, z: T, p: T) -> T {
    if x.is_nan() || y.is_nan() || z.is_nan() || p.is_nan() {
        return T::nan();
    }
    let mut v = [x, y, z];
    sort3(&mut v);
    let zero = T::zero();
    if is_small(v[0]) && is_small(v[1]) && v[2] >= zero && !is_small(p) {
        return T::infinity();
    }
    let nonneg = v[0] >= zero && v[1] >= zero && v[2] >= zero;
    if nonneg && [x, y, z, p].iter().any(|a| a.is_infinite()) {
        return zero;
    }
    if v[0] >= zero && v[1] > zero {
        if p > zero {
            return rj_scaled(v[0], v[1], v[2], p);
        }
        if p < zero {
            return rj_principal(v[0], v[1], v[2], p);
        }
    }
    T::nan()
}
