//! ln Γ(1 + x), precision-adaptive.
//!
//! Written from the Maclaurin series DLMF 5.7.3,
//! `ln Γ(1+x) = -γx + Σ_{k>=2} ζ(k) (-x)^k / k` (`|x| < 1`), applied for
//! `|x| <= 1/2` with ζ(k) = 1 + (ζ(k) - 1) so that the factor 1 sums as the
//! logarithmic series; ln Γ(2+z) from [`lgamma2p`](super::gamma::lgamma2p) for
//! `|x - 1| < 1/2`; and ln Γ(x+1) otherwise. The series keep the relative
//! accuracy next to the two zeros x = 0 and x = 1, where `lgamma(1 + x)` would
//! lose the digits of `x`.

use super::add;
use super::gamma::{lgamma, lgamma2p};
use super::tables::{EULER_GAMMA, ZETA_M1};
use crate::real::{c, k, Real};

/// `-γx + Σ_{k>=2} (1 + ζ(k) - 1)(-x)^k / k` for `|x| <= 1/2`.
fn taylor<T: Real>(x: T) -> T {
    if x == T::zero() {
        return x;
    }
    let eps = T::epsilon();
    let mx = -x;
    let mut sum = -c::<T>(EULER_GAMMA) * x;
    let mut xk = mx;
    for kk in 2..400_usize {
        xk = xk * mx;
        // ζ(k) - 1 beyond the table is below 2^-80 and enters at (x/2)^k.
        let zm1 = ZETA_M1.get(kk - 2).map_or(T::zero(), |&p| c::<T>(p));
        let term = (T::one() + zm1) * xk / k::<T>(kk as f64);
        sum = sum + term;
        if term.abs() <= eps * sum.abs() {
            break;
        }
    }
    sum
}

pub(crate) fn lgam1p<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let half = k::<T>(0.5);
    if x.abs() <= half {
        taylor(x)
    } else if (x - T::one()).abs() < half {
        lgamma2p(x - T::one())
    } else {
        lgamma(add(x, T::one()))
    }
}
