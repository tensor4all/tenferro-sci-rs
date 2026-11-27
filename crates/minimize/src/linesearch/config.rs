//! Validated parameters of the Moré–Thuente search.
//!
//! Ported from SciPy v1.16.3 (b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! scipy/optimize/_dcsrch.py, the `task == b"START"` argument checks of `DCSRCH._iterate`, a 2023
//! translation of MINPACK-2 dcsrch (Argonne National Laboratory / University of Minnesota; see
//! LICENSE-NOTICE-minpack2). Deviations:
//!
//! - The checks are split from the start checks (alpha0, φ'(0)) and made once, at construction.
//!   Each violation is its own [`ConfigError`] variant; MINPACK's "the last failing check
//!   overwrites the earlier ones" has no counterpart.
//! - NaN and ±inf are rejected ([`ConfigError::NonFinite`]); `stpmax = +inf` is allowed. DCSRCH's
//!   comparisons let NaN through, because `NaN < 0` is false.

/// Why a [`MoreThuenteConfig`] was refused.
///
/// A configuration with several violations reports the first of them in the order of the variants
/// below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// A parameter is NaN or infinite (`stpmax = +inf` excepted).
    #[error("ftol, gtol, xtol and stpmin must be finite, and stpmax not NaN or -inf")]
    NonFinite,
    /// `ftol < 0`.
    #[error("ftol must be nonnegative")]
    NegativeFtol,
    /// `gtol < 0`.
    #[error("gtol must be nonnegative")]
    NegativeGtol,
    /// `xtol < 0`.
    #[error("xtol must be nonnegative")]
    NegativeXtol,
    /// `stpmin < 0`.
    #[error("stpmin must be nonnegative")]
    NegativeStpmin,
    /// `stpmax < stpmin`.
    #[error("stpmax must not be below stpmin")]
    StpmaxBelowStpmin,
}

/// Parameters of the Moré–Thuente search, validated once.
///
/// - `ftol` (μ in the paper) is the tolerance of the sufficient decrease condition (1.1),
/// - `gtol` (η) the tolerance of the curvature condition (1.2),
/// - `xtol` the relative width below which the bracket ends the search with a warning,
/// - `stpmin` and `stpmax` bound the step.
///
/// `ftol == gtol` is allowed (the paper's tests IV–VI use μ = η); the stricter `0 < c1 < c2 < 1`
/// of Nocedal & Wright belongs to [`scalar_search_wolfe1`](crate::linesearch::scalar_search_wolfe1).
///
/// # Examples
///
/// ```
/// use minimize::linesearch::MoreThuenteConfig;
///
/// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
/// assert_eq!((cfg.ftol(), cfg.gtol()), (1e-4, 0.9));
/// # Ok::<(), minimize::linesearch::ConfigError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoreThuenteConfig {
    ftol: f64,
    gtol: f64,
    xtol: f64,
    stpmin: f64,
    stpmax: f64,
}

impl MoreThuenteConfig {
    /// Builds a configuration from `ftol`, `gtol`, `xtol`, `stpmin` and `stpmax`.
    ///
    /// # Errors
    ///
    /// - [`ConfigError::NonFinite`] if `ftol`, `gtol`, `xtol` or `stpmin` is NaN or infinite, or
    ///   `stpmax` is NaN or `-inf` (`stpmax = +inf` means "no upper bound" and is allowed),
    /// - [`ConfigError::NegativeFtol`], [`NegativeGtol`](ConfigError::NegativeGtol),
    ///   [`NegativeXtol`](ConfigError::NegativeXtol) or
    ///   [`NegativeStpmin`](ConfigError::NegativeStpmin) if that parameter is negative,
    /// - [`ConfigError::StpmaxBelowStpmin`] if `stpmax < stpmin`.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::linesearch::{ConfigError, MoreThuenteConfig};
    ///
    /// assert!(MoreThuenteConfig::new(0.1, 0.1, 0.0, 0.0, f64::INFINITY).is_ok());
    /// assert_eq!(
    ///     MoreThuenteConfig::new(-1.0, 0.9, 1e-14, 0.0, 1.0),
    ///     Err(ConfigError::NegativeFtol)
    /// );
    /// assert_eq!(
    ///     MoreThuenteConfig::new(1e-4, f64::NAN, 1e-14, 0.0, 1.0),
    ///     Err(ConfigError::NonFinite)
    /// );
    /// ```
    pub fn new(
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
    ) -> Result<Self, ConfigError> {
        if !(ftol.is_finite() && gtol.is_finite() && xtol.is_finite() && stpmin.is_finite())
            || stpmax.is_nan()
            || stpmax == f64::NEG_INFINITY
        {
            return Err(ConfigError::NonFinite);
        }
        if ftol < 0.0 {
            return Err(ConfigError::NegativeFtol);
        }
        if gtol < 0.0 {
            return Err(ConfigError::NegativeGtol);
        }
        if xtol < 0.0 {
            return Err(ConfigError::NegativeXtol);
        }
        if stpmin < 0.0 {
            return Err(ConfigError::NegativeStpmin);
        }
        if stpmax < stpmin {
            return Err(ConfigError::StpmaxBelowStpmin);
        }
        Ok(Self {
            ftol,
            gtol,
            xtol,
            stpmin,
            stpmax,
        })
    }

    /// Tolerance μ of the sufficient decrease condition (1.1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::MoreThuenteConfig;
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// assert_eq!(cfg.ftol(), 1e-4);
    /// # Ok::<(), minimize::linesearch::ConfigError>(())
    /// ```
    pub fn ftol(&self) -> f64 {
        self.ftol
    }

    /// Tolerance η of the curvature condition (1.2).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::MoreThuenteConfig;
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// assert_eq!(cfg.gtol(), 0.9);
    /// # Ok::<(), minimize::linesearch::ConfigError>(())
    /// ```
    pub fn gtol(&self) -> f64 {
        self.gtol
    }

    /// Relative bracket width below which the search stops with
    /// [`Reason::XtolSatisfied`](crate::linesearch::Reason::XtolSatisfied).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::MoreThuenteConfig;
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// assert_eq!(cfg.xtol(), 1e-14);
    /// # Ok::<(), minimize::linesearch::ConfigError>(())
    /// ```
    pub fn xtol(&self) -> f64 {
        self.xtol
    }

    /// Lower bound of the step.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::MoreThuenteConfig;
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// assert_eq!(cfg.stpmin(), 1e-8);
    /// # Ok::<(), minimize::linesearch::ConfigError>(())
    /// ```
    pub fn stpmin(&self) -> f64 {
        self.stpmin
    }

    /// Upper bound of the step (possibly `+inf`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::linesearch::MoreThuenteConfig;
    /// let cfg = MoreThuenteConfig::new(1e-4, 0.9, 1e-14, 1e-8, 50.0)?;
    /// assert_eq!(cfg.stpmax(), 50.0);
    /// # Ok::<(), minimize::linesearch::ConfigError>(())
    /// ```
    pub fn stpmax(&self) -> f64 {
        self.stpmax
    }
}
