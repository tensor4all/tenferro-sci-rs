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
//! The argument contract is shared with the `f64` path and applied by
//! [`prepare`] before either kernel runs: NaN gives NaN; `n < 0` or `k < 0`
//! gives 0; otherwise `n` and `k` are truncated towards zero, and then
//! `S(n, n) = 1` (also for `n = +∞`), `S(n, k) = 0` for `k > n` or `k = 0 < n`,
//! `S(n, 1) = 1` and `S(+∞, k) = +∞` for finite `k >= 2`. The kernels see only
//! finite integers with `2 <= k < n`. The choice between the closed forms and
//! the recurrence is made on the double-double values, so arguments above 2^53
//! keep their low word; only the Temme fallback rounds them to `f64`.

use super::{add, div, mul};
use crate::real::{k, Real};
use crate::xsf;

/// Largest `(k-1)(n-k)` handled by the dynamic programme.
const WORK_MAX: f64 = 3e7;

/// The arguments after the contract above: either the value, or integers
/// `2 <= k < n` with `n` finite for a kernel.
pub(crate) enum Prepared<T> {
    Value(T),
    Kernel(T, T),
}

/// `trunc` that leaves infinities alone (double-double `trunc(∞)` is NaN).
fn trunc<T: Real>(x: T) -> T {
    if x.is_finite() {
        x.trunc()
    } else {
        x
    }
}

pub(crate) fn prepare<T: Real>(n: T, kk: T) -> Prepared<T> {
    if n.is_nan() || kk.is_nan() {
        return Prepared::Value(T::nan());
    }
    if n < T::zero() || kk < T::zero() {
        return Prepared::Value(T::zero());
    }
    let (n, kk) = (trunc(n), trunc(kk));
    if n == kk {
        return Prepared::Value(T::one());
    }
    if kk > n || kk == T::zero() {
        return Prepared::Value(T::zero());
    }
    if kk == T::one() {
        return Prepared::Value(T::one());
    }
    if n.is_infinite() {
        return Prepared::Value(T::infinity());
    }
    Prepared::Kernel(n, kk)
}

/// S(n, k) for integers `2 <= k < n`, `n` finite (see [`prepare`]).
pub(crate) fn stirling2<T: Real>(n: T, kk: T) -> T {
    if kk == k(2.0) {
        // 2^(n-1) - 1 overflows beyond n = 1025.
        return if n >= k(1025.0) {
            T::infinity()
        } else {
            k::<T>(2.0_f64.powi(n.approx_f64() as i32 - 1)) - T::one() // an exact power of two
        };
    }
    if kk == n - T::one() {
        return div(mul(n, n - T::one()), k(2.0));
    }
    if kk == n - k(2.0) {
        let p = mul(mul(n, n - T::one()), n - k(2.0));
        return div(mul(p, k::<T>(3.0) * n - k(5.0)), k(24.0));
    }
    let rows = (n - kk).approx_f64(); // rows of the triangle to advance
    let width = (kk - T::one()).approx_f64(); // inner loop length
    if rows * width > WORK_MAX {
        return T::from_dd(
            xsf::stirling2::stirling2_temme(n.approx_f64(), kk.approx_f64()),
            0.0,
        );
    }
    // Both are below 3e7 + 2 here, so exact in f64 and i64.
    let ni = n.approx_f64() as i64;
    let ki = kk.approx_f64() as i64;
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
