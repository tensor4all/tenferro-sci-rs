//! Validated parameters of the Moré–Thuente search.
//!
//! Stub: the validation lands with the algorithm.

/// Why a [`MoreThuenteConfig`] was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// `ftol` is negative.
    #[error("ftol must be nonnegative")]
    NegativeFtol,
    /// `gtol` is negative.
    #[error("gtol must be nonnegative")]
    NegativeGtol,
    /// `xtol` is negative.
    #[error("xtol must be nonnegative")]
    NegativeXtol,
    /// `stpmin` is negative.
    #[error("stpmin must be nonnegative")]
    NegativeStpmin,
    /// `stpmax < stpmin`.
    #[error("stpmax must not be below stpmin")]
    StpmaxBelowStpmin,
    /// A parameter is NaN or infinite (`stpmax = +inf` excepted).
    #[error("parameters must be finite (stpmax may be +inf)")]
    NonFinite,
}

/// Parameters of the search: `ftol`, `gtol`, `xtol`, `stpmin`, `stpmax`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoreThuenteConfig {
    ftol: f64,
    gtol: f64,
    xtol: f64,
    stpmin: f64,
    stpmax: f64,
}

impl MoreThuenteConfig {
    /// Builds a configuration (stub: no validation yet).
    ///
    /// # Errors
    ///
    /// See [`ConfigError`].
    pub fn new(
        ftol: f64,
        gtol: f64,
        xtol: f64,
        stpmin: f64,
        stpmax: f64,
    ) -> Result<Self, ConfigError> {
        Ok(Self {
            ftol,
            gtol,
            xtol,
            stpmin,
            stpmax,
        })
    }

    /// Sufficient-decrease tolerance μ of (1.1).
    pub fn ftol(&self) -> f64 {
        self.ftol
    }
    /// Curvature tolerance η of (1.2).
    pub fn gtol(&self) -> f64 {
        self.gtol
    }
    /// Relative width tolerance of the bracket.
    pub fn xtol(&self) -> f64 {
        self.xtol
    }
    /// Lower bound of the step.
    pub fn stpmin(&self) -> f64 {
        self.stpmin
    }
    /// Upper bound of the step.
    pub fn stpmax(&self) -> f64 {
        self.stpmax
    }
}
