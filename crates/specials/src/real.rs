//! The scalar trait every function in this crate is generic over.

use core::fmt::Debug;

use num_traits::{Float, FloatConst};

/// How a scalar type evaluates special functions.
///
/// Rational approximations such as Cephes' are fitted to one precision: their
/// coefficients are good to about `f64::EPSILON` and no further. A wider type
/// therefore cannot reuse them; it needs algorithms whose truncation follows
/// the type's own epsilon (series, asymptotic expansions, recurrences).
///
/// # Examples
///
/// ```
/// use specials::{Real, Strategy};
///
/// assert_eq!(<f64 as Real>::STRATEGY, Strategy::F64Kernels);
/// # #[cfg(feature = "xprec")]
/// assert_eq!(<xprec::Df64 as Real>::STRATEGY, Strategy::Adaptive);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Strategy {
    /// Evaluate in `f64` with the kernels ported from SciPy's xsf, and round
    /// the result to the type. Results match SciPy to within a few ulp.
    F64Kernels,
    /// Evaluate with the precision-adaptive algorithms, which truncate their
    /// series at `Float::epsilon()` of the type.
    Adaptive,
}

/// A real floating-point scalar.
///
/// Implemented for `f64` and, with the `xprec` feature (on by default), for
/// [`xprec::Df64`] (double-double, about 32 significant digits).
///
/// # Examples
///
/// ```
/// use specials::Real;
///
/// let x = <f64 as Real>::from_dd(0.5, 0.0);
/// assert_eq!(x.approx_f64(), 0.5);
/// ```
pub trait Real: Float + FloatConst + Debug + Send + Sync + 'static {
    /// The evaluation strategy used by this type.
    const STRATEGY: Strategy;

    /// The value `hi + lo`, rounded to `Self`.
    ///
    /// The crate stores its constants as double-double pairs, so that one
    /// table serves `f64` (which keeps `hi`) and double-double (which keeps
    /// both). The pair must be normalized: `hi + lo == hi` in `f64`.
    fn from_dd(hi: f64, lo: f64) -> Self;

    /// The nearest `f64`.
    fn approx_f64(self) -> f64;
}

impl Real for f64 {
    const STRATEGY: Strategy = Strategy::F64Kernels;

    #[inline]
    fn from_dd(hi: f64, _lo: f64) -> Self {
        hi
    }

    #[inline]
    fn approx_f64(self) -> f64 {
        self
    }
}

#[cfg(feature = "xprec")]
impl Real for xprec::Df64 {
    const STRATEGY: Strategy = Strategy::Adaptive;

    #[inline]
    fn from_dd(hi: f64, lo: f64) -> Self {
        if hi.is_finite() {
            xprec::Df64::new_full(hi, lo)
        } else {
            xprec::Df64::new(hi)
        }
    }

    #[inline]
    fn approx_f64(self) -> f64 {
        self.hi()
    }
}

/// `Self::from_dd` of a table pair.
#[inline]
pub(crate) fn c<T: Real>(pair: (f64, f64)) -> T {
    T::from_dd(pair.0, pair.1)
}

/// A small exact constant.
#[inline]
pub(crate) fn k<T: Real>(x: f64) -> T {
    T::from_dd(x, 0.0)
}
