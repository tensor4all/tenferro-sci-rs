//! Generalized harmonic number `H_{n,a} = Σ_{i=1}^{n} i^-a`, precision-adaptive.
//!
//! Written from the Euler–Maclaurin formula (DLMF 2.10.1). The first
//! `N - 1` terms are summed directly (smallest first for `a >= 0`), with
//! `N = 40 + ⌈|a|⌉`, and the tail `Σ_{i=N}^{n} i^-a` is
//!
//! `∫_N^n x^-a dx + (N^-a + n^-a)/2
//!  - Σ_{j>=1} B_{2j}/(2j)! (a)_{2j-1} (n^{-a-2j+1} - N^{-a-2j+1})`,
//!
//! with `B_{2j}/(2j)! = (-1)^{j+1} 2 ζ(2j) / (2π)^{2j}` (DLMF 25.6.2) and the
//! integral written as `N^{1-a} expm1((1-a) ln(n/N)) / (1-a)` (or `ln(n/N)`
//! for `a = 1`), which is accurate for `a` near 1. With `N >= 40 + |a|` the
//! terms of the series decrease like `((a+2j)/(2πN))²` until they pass the
//! double-double epsilon. The special values follow xsf's `gen_harmonic`.

use super::tables::ZETA_M1;
use super::{add, div, mul};
use crate::real::{c, k, Real};

/// `i^-a` for a positive `i`: an integer power when `a` is an integer, else
/// `exp(-a ln i)`; `+∞` on overflow and 0 on underflow.
fn pow_neg<T: Real>(i: T, a: T) -> T {
    if i == T::one() {
        return T::one();
    }
    if a == a.floor() && a.abs() < k(1e6) {
        let e = a.approx_f64() as i32;
        let p = i.powi(e.abs());
        return if !p.is_finite() {
            if e > 0 {
                T::zero()
            } else {
                T::infinity()
            }
        } else if e >= 0 {
            p.recip()
        } else {
            p
        };
    }
    let l = -a * i.ln();
    if l.approx_f64() > 709.7 {
        T::infinity()
    } else if l.approx_f64() < -745.0 {
        T::zero()
    } else {
        l.exp()
    }
}

/// `Σ_{i=lo}^{hi} i^-a` over integers, smallest terms first.
fn sum_powers<T: Real>(lo: u64, hi: u64, a: T) -> T {
    let mut sum = T::zero();
    // `add`, not `+`: once a term overflows the double-double sum carries (inf, NaN), and a
    // raw addition turns the whole sum into NaN instead of the infinity the f64 path gives.
    if a >= T::zero() {
        for i in (lo..=hi).rev() {
            sum = add(sum, pow_neg(k::<T>(i as f64), a));
        }
    } else {
        for i in lo..=hi {
            sum = add(sum, pow_neg(k::<T>(i as f64), a));
        }
    }
    sum
}

/// `ζ(2j)` for `j >= 1`, from the table of `ζ(k) - 1` (beyond it, `1 + 2^-2j`).
fn zeta_even<T: Real>(j: usize) -> T {
    match ZETA_M1.get(2 * j - 2) {
        Some(&p) => T::one() + c::<T>(p),
        None => T::one() + k::<T>(2.0).powi(-(2 * j as i32)),
    }
}

pub(crate) fn gen_harmonic<T: Real>(n: T, a: T) -> T {
    if n.is_nan() || n < T::one() || n != n.floor() {
        return T::nan();
    }
    if n == T::one() {
        return T::one();
    }
    if a.is_nan() {
        return T::nan();
    }
    if a.is_infinite() {
        return if a > T::zero() {
            T::one()
        } else {
            T::infinity()
        };
    }
    if a == T::zero() {
        return n;
    }
    let big_n = k::<T>(40.0) + a.abs().ceil();
    if n <= big_n {
        return sum_powers(1, n.approx_f64() as u64, a);
    }
    let nn = big_n.approx_f64() as u64;
    let head = sum_powers(1, nn - 1, a);
    // Judge the value, not the type predicate: a double-double that overflowed carries
    // (hi = inf, lo = NaN), and `is_infinite()` on that pair is false, so the guard would
    // let the NaN through and the result would differ from the f64 path.
    let head_f64 = head.approx_f64();
    if !head_f64.is_finite() {
        return if head_f64.is_nan() {
            T::nan()
        } else {
            T::infinity()
        };
    }

    let eps = T::epsilon();
    let one = T::one();
    let pn = pow_neg(n, a); // n^-a
                            // `a < 0` and large `|a|` overflow here, and then the sum diverges: return +inf rather
                            // than carry an infinity into the correction loop, where `inf - inf` is NaN in both
                            // strategies and only the double-double one used to survive to the result.
    if !pn.approx_f64().is_finite() {
        return if pn.approx_f64().is_nan() {
            T::nan()
        } else {
            T::infinity()
        };
    }
    let p_big = pow_neg(big_n, a); // N^-a
                                   // Integral.
    let oma = one - a;
    let integral = if oma == T::zero() {
        (n / big_n).ln()
    } else {
        let u = oma * ((n - big_n) / big_n).ln_1p();
        if u.approx_f64() > 709.0 {
            return T::infinity();
        }
        big_n * p_big * u.exp_m1() / oma
    };
    // `add`/`mul`, not raw operators: `p_big` can be infinite (1000^-(-300)), and a raw
    // multiply would make the Df64 error term NaN where the f64 path returns +inf.
    let mut tail = add(integral, div(add(p_big, pn), k(2.0)));
    // Correction terms; `xn = n^(-a-2j+1)` and `xb = N^(-a-2j+1)`.
    let mut xn = pn / n;
    let mut xb = p_big / big_n;
    let in2 = (n * n).recip();
    let ib2 = (big_n * big_n).recip();
    let two_pi = k::<T>(2.0) * T::PI();
    let inv_two_pi2 = (two_pi * two_pi).recip();
    let mut bern = k::<T>(2.0) * inv_two_pi2 * zeta_even::<T>(1); // B_2/2!
    let mut poch_odd = a; // (a)_{2j-1}
    for j in 1..60_usize {
        let term = -mul(mul(bern, poch_odd), add(xn, -xb));
        tail = add(tail, term);
        if term.abs() <= eps * (head + tail).abs() {
            break;
        }
        // Advance to j + 1.
        poch_odd = mul(
            mul(poch_odd, add(a, k(2.0 * j as f64 - 1.0))),
            add(a, k(2.0 * j as f64)),
        );
        xn = xn * in2;
        xb = xb * ib2;
        let sign = if (j + 1) % 2 == 1 {
            T::one()
        } else {
            -T::one()
        };
        bern = sign * k::<T>(2.0) * zeta_even::<T>(j + 1) * inv_two_pi2.powi(j as i32 + 1);
    }
    head + tail
}
