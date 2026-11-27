//! The BFGS update of the inverse Hessian approximation.
//!
//! Written independently from the formula `H' = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ`,
//! `ρ = 1/(yᵀs)` (Nocedal & Wright, *Numerical Optimization*, the BFGS inverse update; SciPy's
//! `_minimize_bfgs` evaluates the same expression). Dense products go through `colmat`.
//!
//! **Red stage:** [`inverse_update`] checks its arguments and leaves `H` unchanged.

use colmat::Mat;

/// Why [`inverse_update`] refused a pair.
///
/// # Examples
///
/// ```
/// use colmat::Mat;
/// use minimize::bfgs::{inverse_update, UpdateError};
///
/// let mut h = Mat::<f64>::identity(2, 2);
/// assert_eq!(inverse_update(&mut h, &[1.0], &[1.0, 2.0]), Err(UpdateError::DimensionMismatch));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UpdateError {
    /// `h` is not square of the dimension of `s` and `y`, or `s` and `y` differ in length.
    #[error("h must be n x n and s, y of length n")]
    DimensionMismatch,
    /// `yᵀs <= 0`: the update needs `ρ = 1/(yᵀs) > 0` to keep `H` positive definite. What a
    /// BFGS iteration does with such a pair is an open question of #40 (SciPy: `ρ = 1000` at
    /// exactly 0, the formula as it stands for a negative `yᵀs`).
    #[error("the curvature yᵀs must be positive")]
    NonPositiveCurvature,
}

/// Applies the BFGS update to the inverse Hessian approximation `h` in place.
///
/// For `yᵀs > 0` and a symmetric positive definite `h` the result is symmetric positive definite
/// and satisfies the secant equation `h' y = s`.
///
/// # Errors
///
/// - [`UpdateError::DimensionMismatch`] if `h` is not `n × n` for `s.len() == y.len() == n`,
/// - [`UpdateError::NonPositiveCurvature`] if `yᵀs <= 0`.
///
/// # Examples
///
/// ```
/// use colmat::Mat;
/// use minimize::bfgs::inverse_update;
///
/// let mut h = Mat::<f64>::identity(2, 2);
/// let (s, y) = ([1.0, 0.0], [2.0, 0.0]);
/// inverse_update(&mut h, &s, &y)?;
/// // secant equation: h y = s
/// assert!((h[(0, 0)] * y[0] + h[(0, 1)] * y[1] - s[0]).abs() < 1e-15);
/// assert!((h[(1, 0)] * y[0] + h[(1, 1)] * y[1] - s[1]).abs() < 1e-15);
/// # Ok::<(), minimize::bfgs::UpdateError>(())
/// ```
pub fn inverse_update(h: &mut Mat<f64>, s: &[f64], y: &[f64]) -> Result<(), UpdateError> {
    let n = s.len();
    if y.len() != n || h.nrows() != n || h.ncols() != n {
        return Err(UpdateError::DimensionMismatch);
    }
    Ok(())
}
