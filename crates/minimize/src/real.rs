//! The scalar type of the line search, as a private trait.
//!
//! Only what the algorithm uses: arithmetic, comparisons, `abs`, `sqrt`, a finiteness test and
//! constants through [`Real::from_f64`]. It is implemented for `f64` only; the public types of
//! `minimize::linesearch` are concrete `f64` and never name this trait. NaN propagates through
//! every helper here (see [`min`], [`max`], [`clip`]); `f64::clamp` is not used anywhere because
//! it panics on NaN bounds or `min > max`.

#![allow(dead_code)] // stub commit; removed when the algorithm lands

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A real scalar: the operations `dcstep` and `DCSRCH` use, nothing more.
pub(crate) trait Real:
    Copy
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The constant `x` (every literal of the algorithm goes through here).
    fn from_f64(x: f64) -> Self;
    /// The value as `f64`, for the public types.
    fn to_f64(self) -> f64;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Square root; NaN for a negative argument.
    fn sqrt(self) -> Self;
    /// Neither NaN nor infinite.
    fn is_finite(self) -> bool;
    /// NaN.
    fn is_nan(self) -> bool;
}

impl Real for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn abs(self) -> Self {
        f64::abs(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

/// The smaller of `a` and `b`; NaN if either is NaN (unlike `f64::min`, which drops it).
pub(crate) fn min<T: Real>(a: T, b: T) -> T {
    if a.is_nan() {
        a
    } else if b.is_nan() || b < a {
        b
    } else {
        a
    }
}

/// The larger of `a` and `b`; NaN if either is NaN (unlike `f64::max`, which drops it).
pub(crate) fn max<T: Real>(a: T, b: T) -> T {
    if a.is_nan() {
        a
    } else if b.is_nan() || b > a {
        b
    } else {
        a
    }
}

/// `x` limited to `[lo, hi]`; a NaN `x` passes through unchanged. Never panics (not `f64::clamp`).
pub(crate) fn clip<T: Real>(x: T, lo: T, hi: T) -> T {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_propagates_through_min_max_clip() {
        let nan = f64::NAN;
        assert!(min(nan, 1.0).is_nan() && min(1.0, nan).is_nan());
        assert!(max(nan, 1.0).is_nan() && max(1.0, nan).is_nan());
        assert!(clip(nan, 0.0, 1.0).is_nan());
        assert_eq!((min(1.0, 2.0), max(1.0, 2.0)), (1.0, 2.0));
        assert_eq!(
            (
                clip(-1.0, 0.0, 1.0),
                clip(2.0, 0.0, 1.0),
                clip(0.5, 0.0, 1.0)
            ),
            (0.0, 1.0, 0.5)
        );
    }

    #[test]
    fn clip_does_not_panic_on_crossed_bounds() {
        // f64::clamp would panic here; clip just applies the lower test first.
        assert_eq!(clip(0.5, 1.0, 0.0), 1.0);
        assert!(clip(0.5, f64::NAN, 1.0) == 0.5);
    }
}
