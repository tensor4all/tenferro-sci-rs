//! Validated parameters of the BFGS iteration.
//!
//! Mirrors the options of SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34)
//! `scipy/optimize/_optimize.py::_minimize_bfgs`: `gtol`, `xrtol`, `c1`, `c2`, `maxiter`. The
//! `norm` option is fixed to ∞ (SciPy's default) and `hess_inv0` to the identity. Deviations:
//!
//! - Invalid options are an error of the configuration, raised before anything is evaluated;
//!   SciPy validates `c1` and `c2` only (`_check_c1_c2`, in the line search, after the first
//!   evaluation) and lets NaN tolerances through.
//! - `maxiter = 0` is refused ([`ConfigError::ZeroMaxIter`]); SciPy accepts it and reports
//!   "maximum number of iterations" after one evaluation, even at a stationary point.

use crate::linesearch::MoreThuenteConfig;

/// SciPy's defaults, validated when the crate is compiled: a wrong constant is a build error, so
/// [`BfgsConfig::scipy_default`] cannot fail at run time (as `LBFGSB_LINE_SEARCH` in `lbfgs`).
const SCIPY_DEFAULTS: BfgsConfig = match BfgsConfig::new(1e-5, 0.0, 1e-4, 0.9, 200) {
    Ok(c) => c,
    Err(_) => panic!("SciPy's BFGS defaults are valid"),
};

/// Why a [`BfgsConfig`] was refused.
///
/// A configuration with several violations reports the first of them in the order of the variants.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::{BfgsConfig, ConfigError};
///
/// assert_eq!(BfgsConfig::new(-1.0, 0.0, 1e-4, 0.9, 10), Err(ConfigError::NegativeGtol));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    /// `gtol`, `xrtol`, `c1` or `c2` is NaN or infinite.
    #[error("gtol, xrtol, c1 and c2 must be finite")]
    NonFinite,
    /// `gtol < 0`.
    #[error("gtol must be nonnegative")]
    NegativeGtol,
    /// `xrtol < 0`.
    #[error("xrtol must be nonnegative")]
    NegativeXrtol,
    /// `0 < c1 < c2 < 1` does not hold.
    #[error("c1 and c2 must satisfy 0 < c1 < c2 < 1")]
    InvalidWolfeParameters,
    /// `maxiter == 0`.
    #[error("maxiter must be at least 1")]
    ZeroMaxIter,
}

/// Parameters of [`BfgsState`](crate::bfgs::BfgsState), validated once.
///
/// - `gtol`: stop when `‖g‖∞ <= gtol` (SciPy's `gtol` with `norm = inf`),
/// - `xrtol`: stop when `α‖p‖₂ <= xrtol (xrtol + ‖x‖₂)` for the last step,
/// - `c1`, `c2`: the strong Wolfe parameters of the line search,
/// - `maxiter`: the iteration limit.
///
/// The line search is the Moré–Thuente search of [`linesearch`](crate::linesearch) with SciPy's
/// `_line_search_wolfe12` numbers (`amin = 1e-100`, `amax = 1e100`, `xtol = 1e-14`, at most 100
/// trials); they are not options, as in SciPy.
///
/// # Examples
///
/// ```
/// use minimize::bfgs::BfgsConfig;
///
/// let cfg = BfgsConfig::scipy_default(5);
/// assert_eq!((cfg.gtol(), cfg.xrtol(), cfg.c1(), cfg.c2(), cfg.maxiter()), (1e-5, 0.0, 1e-4, 0.9, 1000));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BfgsConfig {
    gtol: f64,
    xrtol: f64,
    c1: f64,
    c2: f64,
    maxiter: usize,
    ls: MoreThuenteConfig,
}

impl BfgsConfig {
    /// Builds a configuration from `gtol`, `xrtol`, `c1`, `c2` and `maxiter`.
    ///
    /// # Errors
    ///
    /// - [`ConfigError::NonFinite`] if `gtol`, `xrtol`, `c1` or `c2` is NaN or infinite,
    /// - [`ConfigError::NegativeGtol`] or [`ConfigError::NegativeXrtol`] if that tolerance is negative,
    /// - [`ConfigError::InvalidWolfeParameters`] unless `0 < c1 < c2 < 1`,
    /// - [`ConfigError::ZeroMaxIter`] if `maxiter == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::bfgs::{BfgsConfig, ConfigError};
    ///
    /// assert!(BfgsConfig::new(0.0, 1e-8, 1e-4, 0.9, 400).is_ok());
    /// assert_eq!(BfgsConfig::new(1e-5, 0.0, 0.9, 0.9, 400), Err(ConfigError::InvalidWolfeParameters));
    /// ```
    pub const fn new(
        gtol: f64,
        xrtol: f64,
        c1: f64,
        c2: f64,
        maxiter: usize,
    ) -> Result<Self, ConfigError> {
        if !(gtol.is_finite() && xrtol.is_finite() && c1.is_finite() && c2.is_finite()) {
            return Err(ConfigError::NonFinite);
        }
        if gtol < 0.0 {
            return Err(ConfigError::NegativeGtol);
        }
        if xrtol < 0.0 {
            return Err(ConfigError::NegativeXrtol);
        }
        if !(0.0 < c1 && c1 < c2 && c2 < 1.0) {
            return Err(ConfigError::InvalidWolfeParameters);
        }
        if maxiter == 0 {
            return Err(ConfigError::ZeroMaxIter);
        }
        // `_line_search_wolfe12(..., amin=1e-100, amax=1e100, c1, c2)`: `wolfe1` with `xtol = 1e-14`. With
        // 0 < c1 < c2 < 1 checked above this cannot fail; the arm exists because a `const fn` cannot `unwrap`.
        let ls = match MoreThuenteConfig::new(c1, c2, 1e-14, 1e-100, 1e100) {
            Ok(ls) => ls,
            Err(_) => return Err(ConfigError::InvalidWolfeParameters),
        };
        Ok(Self {
            gtol,
            xrtol,
            c1,
            c2,
            maxiter,
            ls,
        })
    }

    /// SciPy's defaults for a problem of dimension `n`: `gtol = 1e-5`, `xrtol = 0`, `c1 = 1e-4`,
    /// `c2 = 0.9`, `maxiter = 200 n` (at least 200).
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::bfgs::BfgsConfig;
    ///
    /// assert_eq!(BfgsConfig::scipy_default(2).maxiter(), 400);
    /// assert_eq!(BfgsConfig::scipy_default(0).maxiter(), 200);
    /// ```
    pub const fn scipy_default(n: usize) -> Self {
        let mut c = SCIPY_DEFAULTS;
        c.maxiter = 200 * if n == 0 { 1 } else { n };
        c
    }

    /// The gradient tolerance, in the ∞-norm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// assert_eq!(BfgsConfig::scipy_default(1).gtol(), 1e-5);
    /// ```
    pub fn gtol(&self) -> f64 {
        self.gtol
    }

    /// The relative step tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// assert_eq!(BfgsConfig::scipy_default(1).xrtol(), 0.0);
    /// ```
    pub fn xrtol(&self) -> f64 {
        self.xrtol
    }

    /// The sufficient decrease parameter of the line search.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// assert_eq!(BfgsConfig::scipy_default(1).c1(), 1e-4);
    /// ```
    pub fn c1(&self) -> f64 {
        self.c1
    }

    /// The curvature parameter of the line search.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// assert_eq!(BfgsConfig::scipy_default(1).c2(), 0.9);
    /// ```
    pub fn c2(&self) -> f64 {
        self.c2
    }

    /// The iteration limit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// assert_eq!(BfgsConfig::scipy_default(1).maxiter(), 200);
    /// ```
    pub fn maxiter(&self) -> usize {
        self.maxiter
    }

    /// The Moré–Thuente configuration of the `wolfe1` line search these options define
    /// (`ftol = c1`, `gtol = c2`, `xtol = 1e-14`, `stpmin = 1e-100`, `stpmax = 1e100`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::bfgs::BfgsConfig;
    /// let ls = *BfgsConfig::scipy_default(1).line_search();
    /// assert_eq!((ls.ftol(), ls.gtol(), ls.xtol()), (1e-4, 0.9, 1e-14));
    /// ```
    pub fn line_search(&self) -> &MoreThuenteConfig {
        &self.ls
    }
}
