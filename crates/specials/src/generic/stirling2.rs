//! Stirling numbers of the second kind `S(n, k)`, precision-adaptive.
//!
//! `S(n, k)` counts the partitions of `n` elements into `k` non-empty subsets.
//! Written from the recurrence `S(n, k) = k S(n-1, k) + S(n-1, k-1)`
//! (DLMF 26.8.22), evaluated by dynamic programming in the working type for
//! all `n` whose cost `(k-1)(n-k)` stays below [`WORK_MAX`]. The values are
//! integers: they are exact in double-double up to 2^106, and beyond that each
//! addition rounds by `ε`. Overflow gives `+∞`. The closed forms
//! `S(n, 2) = 2^(n-1) - 1`, `S(n, n-1) = n(n-1)/2` and
//! `S(n, n-2) = n(n-1)(n-2)(3n-5)/24` (elementary identities, checked against the
//! recurrence by the tests for `n <= 2000`) cover the large-`n` cases that matter most.
//!
//! Beyond the work limit (`n` in the millions or more with `k` neither 2 nor
//! within 2 of `n`, for results that do not overflow) there is no
//! double-double algorithm here: the result is xsf's second-order Temme
//! asymptotic approximation evaluated in `f64`, converted to the type, so it
//! has only the accuracy of that approximation (about 1e-3 there).
//!
//! The argument conventions follow xsf's `stirling2`: `n < 0` or `k < 0` or
//! `k > n` or `0 < k < 1` give 0 (`S(0, 0) = 1`), and `n`, `k` are truncated
//! to integers.

use super::{add, div, mul};
use crate::real::{k, Real};
use crate::xsf;

/// Largest `(k-1)(n-k)` handled by the dynamic programme.
const WORK_MAX: f64 = 3e7;

pub(crate) fn stirling2<T: Real>(n: T, kk: T) -> T {
    if n.is_nan() || kk.is_nan() {
        return T::nan();
    }
    let (nf, kf) = (n.approx_f64(), kk.approx_f64());
    if (nf == 0.0 && kf == 0.0) || (nf == 1.0 && kf == 1.0) {
        return T::one();
    }
    if kf < 1.0 || kf > nf || nf < 0.0 {
        return T::zero();
    }
    if nf == kf || kf == 1.0 {
        return T::one();
    }
    if nf.is_infinite() {
        return T::from_dd(xsf::stirling2::stirling2(nf, kf), 0.0);
    }
    let nt = k::<T>(nf.trunc());
    if kf.trunc() == 2.0 {
        // 2^(n-1) - 1 overflows beyond n = 1025.
        return if nf >= 1025.0 {
            T::infinity()
        } else {
            k::<T>(2.0_f64.powi(nf as i32 - 1)) - T::one() // an exact power of two
        };
    }
    if kf.trunc() == nf.trunc() - 1.0 {
        return div(mul(nt, nt - T::one()), k(2.0));
    }
    if kf.trunc() == nf.trunc() - 2.0 {
        let p = mul(mul(nt, nt - T::one()), nt - k(2.0));
        return div(mul(p, k::<T>(3.0) * nt - k(5.0)), k(24.0));
    }
    let ni = nf as i64;
    let ki = kf as i64;
    let rows = ni - ki; // rows of the triangle to advance
    let width = ki - 1; // inner loop length
    if (rows as f64) * (width as f64) > WORK_MAX {
        return T::from_dd(xsf::stirling2::stirling2_temme(nf, kf), 0.0);
    }
    // curr[j] holds S(j + i + 1, j + 1) as the rows advance (the diagonal
    // formulation of xsf, with the shorter side as the array).
    let size = ki.min(ni - ki + 1) as usize;
    let mut curr = vec![T::one(); size];
    if ki <= ni - ki + 1 {
        for _ in 1..ni - ki + 1 {
            for j in 1..ki as usize {
                curr[j] = add(mul(k::<T>((j + 1) as f64), curr[j]), curr[j - 1]);
                if curr[j].is_infinite() {
                    return T::infinity();
                }
            }
        }
    } else {
        for i in 1..ki {
            for j in 1..(ni - ki + 1) as usize {
                curr[j] = add(mul(k::<T>((i + 1) as f64), curr[j - 1]), curr[j]);
                if curr[j].is_infinite() {
                    return T::infinity();
                }
            }
        }
    }
    curr[size - 1]
}
