//! Validated configuration of the L-BFGS iteration.
//!
//! Written for tensor4all/tenferro-sci-rs#40. The defaults are those of SciPy v1.16.3's (commit
//! `b9105ccc2237f57acb1060202cd77f6dd264fb34`) `minimize(method="L-BFGS-B")` (`_lbfgsb_py.py`, `_minimize_lbfgsb`) and the line-search numbers are the ones
//! `lnsrlb` hard-codes (`__lbfgsb.c`: `ftol = 1e-3, gtol = 0.9, xtol = 0.1`, `stpmin = 0`, `stpmax = 1e10`).

use crate::linesearch::MoreThuenteConfig;

/// The line search of SciPy's L-BFGS-B, `lnsrlb` in `__lbfgsb.c`: `ftol = 1e-3`, `gtol = 0.9`, `xtol = 0.1`,
/// `stpmin = 0`, `stpmax = 1e10` (unconstrained). These numbers belong to L-BFGS, not to `linesearch`. The validation
/// of [`MoreThuenteConfig::new`] runs at compile time (it is a `const fn`), so a wrong constant is a build error and
/// the configuration of the optimizer cannot fail at run time.
const LBFGSB_LINE_SEARCH: MoreThuenteConfig =
    match MoreThuenteConfig::new(1e-3, 0.9, 0.1, 0.0, 1e10) {
        Ok(c) => c,
        Err(_) => panic!("the L-BFGS-B line search constants are valid"),
    };

/// The initial inverse-Hessian approximation `H₀` of the two-loop recursion.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::H0;
///
/// assert_eq!(H0::default(), H0::ScaledGamma);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum H0 {
    /// `H₀ = γ I` with `γ = sᵀy / yᵀy` of the newest stored pair (the usual scaling of the initial matrix); `γ = 1` while no pair is
    /// stored. This is what SciPy's L-BFGS-B does (it stores the reciprocal `θ = yᵀy / sᵀy`).
    #[default]
    ScaledGamma,
    /// `H₀ = I`, whatever the pairs are. A test hook: with it the method is the textbook limited-memory BFGS
    /// whose `m ≥ k` iterates coincide with dense BFGS from `H₀ = I` (decision 4 of tensor4all/tenferro-sci-rs#40).
    FixedIdentity,
}

/// Why a [`LbfgsConfig`] was refused.
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{ConfigError, LbfgsConfig, H0};
///
/// let e = LbfgsConfig::new(0, 1e-5, 1e-9, 100, 100, 20, H0::ScaledGamma).unwrap_err();
/// assert_eq!(e, ConfigError::ZeroMemory);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum ConfigError {
    /// `m = 0`: the memory must hold at least one pair (SciPy: `ERROR: M <= 0`).
    #[error("the memory size m must be at least 1")]
    ZeroMemory,
    /// `maxiter = 0`. SciPy accepts it and behaves like 1 (the test comes after the first iteration).
    #[error("maxiter must be at least 1")]
    ZeroMaxiter,
    /// `maxfun = 0`. SciPy accepts it and behaves like 1.
    #[error("maxfun must be at least 1")]
    ZeroMaxfun,
    /// `maxls = 0` (SciPy: `ValueError: maxls must be positive`).
    #[error("maxls must be at least 1")]
    ZeroMaxls,
    /// `gtol` or `ftol` is NaN or infinite. SciPy accepts both (a NaN tolerance never triggers its test).
    #[error("gtol and ftol must be finite")]
    NonFinite,
    /// `gtol < 0`: the gradient test could never hold (SciPy accepts it silently).
    #[error("gtol must not be negative")]
    NegativeGtol,
    /// `ftol < 0` (SciPy: `ERROR: FACTR < 0`).
    #[error("ftol must not be negative")]
    NegativeFtol,
}

/// The configuration of [`LbfgsState`](super::LbfgsState) and [`minimize`](super::minimize), validated at
/// construction and immutable afterwards.
///
/// | field | meaning | SciPy option |
/// |---|---|---|
/// | `m` | number of stored pairs | `maxcor` |
/// | `gtol` | stop when `‖g‖∞ ≤ gtol` | `gtol` |
/// | `ftol` | stop when `(f_old − f) ≤ ftol · max(\|f_old\|, \|f\|, 1)` (one-sided) | `ftol` |
/// | `maxiter` | stop after this many accepted iterations | `maxiter` |
/// | `maxfun` | stop when the evaluations exceed this, tested after an accepted iteration | `maxfun` |
/// | `maxls` | at most this many trial evaluations per line search | `maxls` |
/// | `h0` | the initial inverse Hessian | (always `ScaledGamma`) |
///
/// # Examples
///
/// ```
/// use minimize::lbfgs::{LbfgsConfig, H0};
///
/// let cfg = LbfgsConfig::scipy_default();
/// assert_eq!((cfg.m(), cfg.gtol(), cfg.maxls(), cfg.h0()), (10, 1e-5, 20, H0::ScaledGamma));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsConfig {
    m: usize,
    gtol: f64,
    ftol: f64,
    maxiter: usize,
    maxfun: usize,
    maxls: usize,
    h0: H0,
    ls: MoreThuenteConfig,
}

impl LbfgsConfig {
    /// Builds a configuration. The line search is Moré–Thuente with the numbers L-BFGS-B hard-codes
    /// (`ftol = 1e-3`, `gtol = 0.9`, `xtol = 0.1`, `stpmin = 0`, `stpmax = 1e10`).
    ///
    /// # Errors
    ///
    /// - [`ConfigError::ZeroMemory`], [`ConfigError::ZeroMaxiter`], [`ConfigError::ZeroMaxfun`] or
    ///   [`ConfigError::ZeroMaxls`] if that count is zero,
    /// - [`ConfigError::NonFinite`] if `gtol` or `ftol` is NaN or infinite (checked before the signs),
    /// - [`ConfigError::NegativeGtol`] or [`ConfigError::NegativeFtol`] if that tolerance is negative.
    ///
    /// When several violations hold, the first of this list is reported.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::{ConfigError, LbfgsConfig, H0};
    ///
    /// assert!(LbfgsConfig::new(5, 1e-8, 0.0, 50, 100, 10, H0::FixedIdentity).is_ok());
    /// assert_eq!(
    ///     LbfgsConfig::new(5, f64::NAN, 0.0, 50, 100, 10, H0::ScaledGamma),
    ///     Err(ConfigError::NonFinite)
    /// );
    /// ```
    pub fn new(
        m: usize,
        gtol: f64,
        ftol: f64,
        maxiter: usize,
        maxfun: usize,
        maxls: usize,
        h0: H0,
    ) -> Result<Self, ConfigError> {
        if m == 0 {
            return Err(ConfigError::ZeroMemory);
        }
        if maxiter == 0 {
            return Err(ConfigError::ZeroMaxiter);
        }
        if maxfun == 0 {
            return Err(ConfigError::ZeroMaxfun);
        }
        if maxls == 0 {
            return Err(ConfigError::ZeroMaxls);
        }
        if !(gtol.is_finite() && ftol.is_finite()) {
            return Err(ConfigError::NonFinite);
        }
        if gtol < 0.0 {
            return Err(ConfigError::NegativeGtol);
        }
        if ftol < 0.0 {
            return Err(ConfigError::NegativeFtol);
        }
        Ok(Self {
            m,
            gtol,
            ftol,
            maxiter,
            maxfun,
            maxls,
            h0,
            ls: LBFGSB_LINE_SEARCH,
        })
    }

    /// SciPy 1.16.3's defaults for `minimize(method="L-BFGS-B")`: `maxcor = 10`, `gtol = 1e-5`,
    /// `ftol = 2.220446049250313e-9` (`1e7 · eps`), `maxiter = maxfun = 15000`, `maxls = 20`, and the line search
    /// L-BFGS-B hard-codes. It cannot fail: every value is a valid constant.
    ///
    /// # Examples
    ///
    /// ```
    /// use minimize::lbfgs::LbfgsConfig;
    ///
    /// let cfg = LbfgsConfig::scipy_default();
    /// assert_eq!((cfg.maxiter(), cfg.maxfun()), (15000, 15000));
    /// assert_eq!(cfg.ftol(), 2.220446049250313e-9);
    /// ```
    pub fn scipy_default() -> Self {
        Self {
            m: 10,
            gtol: 1e-5,
            ftol: 2.220446049250313e-9,
            maxiter: 15000,
            maxfun: 15000,
            maxls: 20,
            h0: H0::ScaledGamma,
            ls: LBFGSB_LINE_SEARCH,
        }
    }

    /// The number of stored pairs `m`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-5, 0.0, 10, 10, 20, H0::ScaledGamma)?;
    /// assert_eq!(cfg.m(), 3);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn m(&self) -> usize {
        self.m
    }

    /// The gradient tolerance, in the ∞-norm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 0.0, 10, 10, 20, H0::ScaledGamma)?;
    /// assert_eq!(cfg.gtol(), 1e-7);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn gtol(&self) -> f64 {
        self.gtol
    }

    /// The relative one-sided decrease tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 1e-3, 10, 10, 20, H0::ScaledGamma)?;
    /// assert_eq!(cfg.ftol(), 1e-3);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn ftol(&self) -> f64 {
        self.ftol
    }

    /// The maximum number of accepted iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 1e-3, 11, 12, 20, H0::ScaledGamma)?;
    /// assert_eq!(cfg.maxiter(), 11);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn maxiter(&self) -> usize {
        self.maxiter
    }

    /// The evaluation budget, tested after an accepted iteration with a strict `>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 1e-3, 11, 12, 20, H0::ScaledGamma)?;
    /// assert_eq!(cfg.maxfun(), 12);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn maxfun(&self) -> usize {
        self.maxfun
    }

    /// The maximum number of trial evaluations of one line search.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 1e-3, 11, 12, 13, H0::ScaledGamma)?;
    /// assert_eq!(cfg.maxls(), 13);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn maxls(&self) -> usize {
        self.maxls
    }

    /// The initial inverse-Hessian rule.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::{LbfgsConfig, H0};
    /// let cfg = LbfgsConfig::new(3, 1e-7, 1e-3, 11, 12, 13, H0::FixedIdentity)?;
    /// assert_eq!(cfg.h0(), H0::FixedIdentity);
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn h0(&self) -> H0 {
        self.h0
    }

    /// The Moré–Thuente configuration of the line search.
    ///
    /// # Examples
    ///
    /// ```
    /// # use minimize::lbfgs::LbfgsConfig;
    /// let cfg = LbfgsConfig::scipy_default();
    /// let ls = cfg.line_search();
    /// assert_eq!((ls.ftol(), ls.gtol(), ls.xtol(), ls.stpmin(), ls.stpmax()), (1e-3, 0.9, 0.1, 0.0, 1e10));
    /// # Ok::<(), minimize::lbfgs::ConfigError>(())
    /// ```
    pub fn line_search(&self) -> &MoreThuenteConfig {
        &self.ls
    }
}
