//! Binomial coefficient `C(n, k) = Γ(n+1) / (Γ(k+1) Γ(n-k+1))`, precision-adaptive.
//!
//! Written from the definition and:
//!
//! * for a non-negative integer `k` below [`PRODUCT_MAX`], the product
//!   `Π_{j<k} (n-j)/(j+1)` for every `n` (accumulated as a numerator and a
//!   denominator, so integer results stay exact as long as they fit the
//!   mantissa);
//! * for `k > 1e8 |n|`, the reflection formula (DLMF 5.5.3) applied to
//!   Γ(n-k+1), which gives
//!   `C(n, k) = Γ(n+1) sin(π(k-n)) / (π (k-n)_{n+1})` with the Pochhammer
//!   symbol of [`poch`](super::poch) (accurate for large `k-n`);
//! * for `-1 < k < 100`, `(n-k+1)_k / Γ(k+1)` with the Pochhammer symbol of
//!   [`poch`](super::poch) (accurate for every `n`), when that is finite;
//! * for `n >= 1e10 k`, or when B(n-k+1, k+1) is outside the range of the type
//!   (below 1e-300 or above 1e300), `±exp(-ln B(n-k+1, k+1) - ln|n+1|)`;
//! * otherwise `1 / ((n+1) B(n-k+1, k+1))` (DLMF 5.12.1).
//!
//! The domain conventions (undefined at negative integer `n`; `n` and `k`
//! integer-valued handled exactly) follow xsf's `binom`.

use super::beta::{beta, lbeta};
use super::gamma::{gamma, gammasgn};
use super::poch::poch;
use super::{div, mul, sinpi};
use crate::real::{k, Real};
use crate::xsf;

/// Largest integer `k` for which the product formula is used.
const PRODUCT_MAX: f64 = 100.0;

pub(crate) fn binom<T: Real>(n: T, kk: T) -> T {
    if n.is_nan() || kk.is_nan() {
        return T::nan();
    }
    if !n.is_finite() || !kk.is_finite() {
        // Special values (0, ±∞, NaN): take xsf's.
        return T::from_dd(xsf::binom::binom(n.approx_f64(), kk.approx_f64()), 0.0);
    }
    let zero = T::zero();
    let one = T::one();
    if n < zero && n == n.floor() {
        return T::nan();
    }

    let mut kx = kk.floor();
    if kk == kx {
        let nx = n.floor();
        if nx == n && kx > nx / k(2.0) && nx > zero {
            kx = nx - kx;
        }
        if kx >= zero && kx < k(PRODUCT_MAX) {
            let mut num = one;
            let mut den = one;
            let big = k::<T>(1e50);
            let kn = kx.approx_f64() as i32;
            for i in 0..kn {
                // The factors n - j without the cancellation of xsf's
                // `i + n - k`, which is why xsf must exclude small `n`.
                num = mul(num, n - k::<T>(f64::from(i)));
                den = den * k::<T>(f64::from(i + 1));
                if num.abs() > big {
                    num = div(num, den);
                    den = one;
                }
            }
            return div(num, den);
        }
    }

    if n >= k::<T>(1e10) * kk && kk > zero {
        let l = -lbeta(one + n - kk, one + kk) - (n + one).ln();
        return if l.approx_f64() > 709.7 {
            T::infinity()
        } else {
            l.exp()
        };
    }
    if kk > k::<T>(1e8) * n.abs() {
        // C(n, k) = Γ(n+1) sin(π(k-n)) / (π (k-n)_{n+1}). The Pochhammer
        // symbol alone may overflow while the quotient does not (it is about
        // k^-n), so for a moderate `n + 1` the factors are divided out one at
        // a time.
        let a = kk - n;
        let mut m = n + one;
        let mut g = div(mul(gamma(n + one), sinpi(kk - n)), T::PI());
        if m.abs() <= k(1000.0) {
            while m >= one {
                m = m - one;
                g = div(g, a + m);
            }
            while m <= -one {
                g = mul(g, a + m);
                m = m + one;
            }
            return if m == zero { g } else { div(g, poch(a, m)) };
        }
        return div(g, poch(a, m));
    }
    if kk > -one && kk < k(PRODUCT_MAX) {
        let p = poch(one + n - kk, kk);
        if p.is_finite() {
            return div(p, gamma(kk + one));
        }
    }
    let (a, b) = (one + n - kk, one + kk);
    let bt = beta(a, b);
    if bt.is_infinite() || (bt.abs() > k(1e-300) && bt.abs() < k(1e300)) {
        // (B is infinite at the poles: the result is zero.)
        return div(div(one, n + one), bt);
    }
    // B(a, b) is out of range (or its sign is needed separately).
    let l = -lbeta(a, b) - (n + one).abs().ln();
    let sign = mul(gammasgn(a), mul(gammasgn(b), gammasgn(a + b)));
    let sign = mul(sign, if n + one < zero { -one } else { one });
    if l.approx_f64() > 709.7 {
        mul(sign, T::infinity())
    } else {
        mul(sign, l.exp())
    }
}
