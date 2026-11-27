//! The vector operations the quasi-Newton iterations need (tensor4all/tenferro-sci-rs#40, decision 1).
//!
//! The set follows the design's "operations krylov uses" plus the ∞-norm SciPy's `gtol` is measured in. The trait
//! belongs to `minimize`, not to krylov, so the two cores release independently. `tenferro-optimize` implements it
//! for tenferro tensors. Workspace is allocated by cloning `x0` when a state is created; [`Clone::clone_from`] reuses it
//! afterwards, so an iteration allocates nothing.
//!
//! The scalar is `f64` for now (the scalar trait is still private, as decided on #38).

/// A vector space element as the L-BFGS and BFGS iterations use it.
///
/// Every operation takes the caller's execution context, so an implementation can run on the caller's engine
/// without global state.
///
/// # Examples
///
/// ```
/// use minimize::Vector;
///
/// let mut y = vec![1.0, 2.0];
/// let x = vec![3.0, -4.0];
/// y.axpy(&mut (), 2.0, &x); // y += 2 x
/// assert_eq!(y, vec![7.0, -6.0]);
/// assert_eq!(x.norm_inf(&mut ()), 4.0);
/// ```
pub trait Vector: Clone {
    /// The caller-held execution context passed to every operation (`()` for plain `Vec<f64>`).
    type Ctx;
    /// `self += a * x`.
    fn axpy(&mut self, ctx: &mut Self::Ctx, a: f64, x: &Self);
    /// `self *= a`.
    fn scale(&mut self, ctx: &mut Self::Ctx, a: f64);
    /// The inner product `selfᵀ x`.
    fn dot(&self, ctx: &mut Self::Ctx, x: &Self) -> f64;
    /// The Euclidean norm.
    fn norm(&self, ctx: &mut Self::Ctx) -> f64;
    /// The maximum absolute entry (SciPy's `gtol` norm). NaN if any entry is NaN.
    fn norm_inf(&self, ctx: &mut Self::Ctx) -> f64;
}

// The `Vec<f64>` implementation is the BLAS-1 reference for plain vectors. The kernels are handwritten
// loops by design: the tenferro implementation (`tenferro-optimize`) goes through tenferro. Each loop
// carries the repository's `contraction-kernel-ok` marker (REPOSITORY_RULES.md, "Kernels").
//
// Length mismatches are a caller bug in every BLAS-1 routine; `zip` stops at the shorter operand and
// `debug_assert_eq!` reports it in debug and test builds. Nothing here can panic in release builds.
impl Vector for Vec<f64> {
    type Ctx = ();

    fn axpy(&mut self, _ctx: &mut (), a: f64, x: &Self) {
        debug_assert_eq!(self.len(), x.len());
        // contraction-kernel-ok: BLAS-1 reference implementation for plain Vec<f64>; the tenferro implementation goes through tenferro
        for (yi, xi) in self.iter_mut().zip(x) {
            *yi += a * xi;
        }
    }

    fn scale(&mut self, _ctx: &mut (), a: f64) {
        // contraction-kernel-ok: BLAS-1 reference implementation for plain Vec<f64>; the tenferro implementation goes through tenferro
        for yi in self.iter_mut() {
            *yi *= a;
        }
    }

    fn dot(&self, _ctx: &mut (), x: &Self) -> f64 {
        debug_assert_eq!(self.len(), x.len());
        let mut sum = 0.0;
        // contraction-kernel-ok: BLAS-1 reference implementation for plain Vec<f64>; the tenferro implementation goes through tenferro
        for (a, b) in self.iter().zip(x) {
            sum += a * b;
        }
        sum
    }

    fn norm(&self, ctx: &mut ()) -> f64 {
        // Scaled by the largest entry so that the squares neither overflow nor underflow:
        // |x|₂ = m sqrt(Σ (xᵢ/m)²), m = |x|∞. NaN and ∞ pass through `norm_inf`.
        let m = self.norm_inf(ctx);
        if m == 0.0 || !m.is_finite() {
            return m;
        }
        let mut sum = 0.0;
        // contraction-kernel-ok: BLAS-1 reference implementation for plain Vec<f64>; the tenferro implementation goes through tenferro
        for xi in self.iter() {
            let t = xi / m;
            sum += t * t;
        }
        m * sum.sqrt()
    }

    fn norm_inf(&self, _ctx: &mut ()) -> f64 {
        let mut m = 0.0_f64;
        // contraction-kernel-ok: BLAS-1 reference implementation for plain Vec<f64>; the tenferro implementation goes through tenferro
        for xi in self.iter() {
            let a = xi.abs();
            // `f64::max` would drop a NaN (1.0_f64.max(NAN) == 1.0) and let a NaN gradient pass `‖g‖∞ ≤ gtol`.
            if a.is_nan() {
                return f64::NAN;
            }
            if a > m {
                m = a;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_scale_dot_on_small_vectors() {
        let mut y = vec![1.0, 2.0, 3.0];
        y.axpy(&mut (), -2.0, &vec![0.5, 1.0, 1.5]);
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
        let mut z = vec![1.0, -2.0, 4.0];
        z.scale(&mut (), 0.5);
        assert_eq!(z, vec![0.5, -1.0, 2.0]);
        // exact in binary floating point: 0.5·4 + (−1)·(−2) + 2·0.25
        assert_eq!(z.dot(&mut (), &vec![4.0, -2.0, 0.25]), 4.5);
    }

    #[test]
    fn norm_matches_the_pythagorean_triples_exactly() {
        assert_eq!(vec![3.0, 4.0].norm(&mut ()), 5.0);
        assert_eq!(vec![-3.0, 0.0, 4.0, 0.0].norm(&mut ()), 5.0);
        assert_eq!(Vec::<f64>::new().norm(&mut ()), 0.0);
        assert_eq!(vec![0.0, -0.0].norm(&mut ()), 0.0);
    }

    #[test]
    fn norm_does_not_overflow_or_underflow() {
        let big = vec![3.0e200, 4.0e200];
        assert!((big.norm(&mut ()) / 5.0e200 - 1.0).abs() < 4.0 * f64::EPSILON);
        let tiny = vec![3.0e-200, 4.0e-200];
        assert!((tiny.norm(&mut ()) / 5.0e-200 - 1.0).abs() < 4.0 * f64::EPSILON);
    }

    #[test]
    fn norm_inf_is_the_largest_absolute_entry() {
        assert_eq!(vec![1.0, -7.5, 3.0].norm_inf(&mut ()), 7.5);
        assert_eq!(Vec::<f64>::new().norm_inf(&mut ()), 0.0);
    }

    #[test]
    fn norm_inf_propagates_a_nan_among_finite_entries() {
        // f64::max would return 2.0 here and a NaN gradient would pass `‖g‖∞ ≤ gtol`.
        assert!(vec![1.0, f64::NAN, 2.0].norm_inf(&mut ()).is_nan());
        assert!(vec![f64::NAN, 1.0].norm_inf(&mut ()).is_nan());
        assert!(vec![1.0, 2.0, f64::NAN].norm_inf(&mut ()).is_nan());
        // a NaN next to an infinity is still NaN, whichever comes first
        assert!(vec![f64::INFINITY, f64::NAN].norm_inf(&mut ()).is_nan());
        assert!(vec![f64::NAN, f64::NEG_INFINITY].norm_inf(&mut ()).is_nan());
    }

    #[test]
    fn norm_inf_and_norm_of_infinite_entries() {
        assert_eq!(
            vec![1.0, f64::NEG_INFINITY].norm_inf(&mut ()),
            f64::INFINITY
        );
        assert_eq!(vec![1.0, f64::INFINITY].norm_inf(&mut ()), f64::INFINITY);
        assert_eq!(vec![1.0, f64::NEG_INFINITY].norm(&mut ()), f64::INFINITY);
        assert!(vec![1.0, f64::NAN].norm(&mut ()).is_nan());
    }

    #[test]
    fn clone_from_reuses_the_allocation() {
        // The workspace contract of the module docs: `clone_from` into a vector of the same length
        // does not allocate, so its buffer address does not change.
        let mut dst = vec![0.0; 8];
        let before = dst.as_ptr();
        dst.clone_from(&vec![1.0; 8]);
        assert_eq!(dst.as_ptr(), before);
        assert_eq!(dst, vec![1.0; 8]);
    }
}
