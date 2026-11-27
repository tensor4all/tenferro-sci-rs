//! `dcstep`: the safeguarded step and interval update of Moré–Thuente §4.
//!
//! Stub: the step computation lands next.

#![allow(dead_code)] // stub commit; removed when dcstep lands

use crate::real::Real;

/// A point `(α, φ(α), φ'(α))` in the units of the caller (also the ψ form of §3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pt<T> {
    pub a: T,
    pub f: T,
    pub g: T,
}

/// Which branch of §4 (p. 299–300) produced the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Case {
    /// Case 1: a higher function value.
    HigherValue,
    /// Case 2: a lower value and derivatives of opposite sign.
    OppositeSlopes,
    /// Case 3: a lower value, same sign, the magnitude of the derivative decreases.
    DecreasingSlope,
    /// Case 4: a lower value, same sign, the magnitude does not decrease.
    SteepSlope,
}

/// Result of [`dcstep`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct DcStep<T> {
    pub x: Pt<T>,
    pub y: Pt<T>,
    pub stp: T,
    pub brackt: bool,
    pub case: Case,
}

/// One safeguarded step (stub: returns its inputs).
pub(crate) fn dcstep<T: Real>(
    x: Pt<T>,
    y: Pt<T>,
    t: Pt<T>,
    brackt: bool,
    _stpmin: T,
    _stpmax: T,
) -> DcStep<T> {
    DcStep {
        x,
        y,
        stp: t.a,
        brackt,
        case: Case::SteepSlope,
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `dcstep`, the cases of Moré–Thuente (1994) §4 (p. 299–300). The expected
    //! values are computed by hand or by an independent cubic-interpolant formula (Nocedal &
    //! Wright (3.59)), not by the code under test.

    use super::*;

    fn pt(a: f64, f: f64, g: f64) -> Pt<f64> {
        Pt { a, f, g }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        a == b || (a - b).abs() <= tol * a.abs().max(b.abs())
    }

    /// Minimizer of the cubic through `(a1, f1, g1)` and `(a2, f2, g2)` (N&W (3.59)).
    fn cubic_min(p1: Pt<f64>, p2: Pt<f64>) -> f64 {
        let d1 = p1.g + p2.g - 3.0 * (p1.f - p2.f) / (p1.a - p2.a);
        let d2 = (p2.a - p1.a).signum() * (d1 * d1 - p1.g * p2.g).sqrt();
        p2.a - (p2.a - p1.a) * (p2.g + d2 - d1) / (p2.g - p1.g + 2.0 * d2)
    }

    /// The secant step of two derivative values: the zero of the line through `(a1, g1)`, `(a2, g2)`.
    fn secant(a1: f64, g1: f64, a2: f64, g2: f64) -> f64 {
        a1 + g1 / (g1 - g2) * (a2 - a1)
    }

    /// The minimizer of the quadratic through `(a1, f1, g1)` and `(a2, f2)`.
    fn quad_min(p1: Pt<f64>, a2: f64, f2: f64) -> f64 {
        let h = a2 - p1.a;
        p1.a - 0.5 * p1.g * h * h / (f2 - p1.f - p1.g * h)
    }

    fn run(x: Pt<f64>, y: Pt<f64>, t: Pt<f64>, brackt: bool) -> DcStep<f64> {
        dcstep(x, y, t, brackt, -1e3, 1e3)
    }

    #[test]
    fn every_case_is_selected() {
        let x = pt(0.0, 0.0, -1.0);
        let y = pt(10.0, 5.0, 1.0);
        for brackt in [false, true] {
            // Case 1: a higher function value
            let d = run(x, y, pt(1.0, 0.5, -0.2), brackt);
            assert_eq!(d.case, Case::HigherValue);
            assert!(d.brackt);
            // Case 2: lower value, derivatives of opposite sign
            let d = run(x, y, pt(2.0, -1.2, 0.2), brackt);
            assert_eq!(d.case, Case::OppositeSlopes);
            assert!(d.brackt);
            // Case 3: lower value, same sign, |dp| < |dx|
            let d = run(x, y, pt(0.5, -0.4, -0.6), brackt);
            assert_eq!(d.case, Case::DecreasingSlope);
            assert_eq!(d.brackt, brackt);
            // Case 4: lower value, same sign, |dp| > |dx|, and the tie |dp| == |dx|
            for g in [-1.5, -1.0] {
                let d = run(x, y, pt(1.0, -1.5, g), brackt);
                assert_eq!(d.case, Case::SteepSlope, "dp = {g}");
                assert_eq!(d.brackt, brackt);
            }
        }
    }

    #[test]
    fn interval_update_follows_the_case() {
        let x = pt(0.0, 0.0, -1.0);
        let y = pt(10.0, 5.0, 1.0);
        // Case 1: stp becomes the other endpoint, x is unchanged
        let t = pt(1.0, 0.5, -0.2);
        let d = run(x, y, t, false);
        assert_eq!((d.x, d.y), (x, t));
        // Case 2: the old best becomes the other endpoint, stp the new best
        let t = pt(2.0, -1.2, 0.2);
        let d = run(x, y, t, true);
        assert_eq!((d.x, d.y), (t, x));
        // Cases 3 and 4: stp is the new best, y is unchanged
        for t in [pt(0.5, -0.4, -0.6), pt(1.0, -1.5, -1.5)] {
            let d = run(x, y, t, true);
            assert_eq!((d.x, d.y), (t, y));
        }
    }

    #[test]
    fn case_one_bounds_and_combination() {
        // fp > fx, dx < 0: the cubic and quadratic minimizers lie in the interval, with
        // |a_c - a_l| <= 2/3 |a_u - a_l| and |a_q - a_l| <= 1/2 |a_u - a_l| (p. 299), and the
        // step is a_c if it is closer to a_l than a_q, else their average: |a+ - a_l| <= 7/12 |..|.
        let (stx, stp) = (1.0, 3.5);
        let w = stp - stx;
        let mut n = 0;
        for rise in [1e-3, 0.1, 1.0, 10.0, 1e3] {
            for dp in [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0] {
                for dx in [-0.01, -1.0, -50.0] {
                    let x = pt(stx, 2.0, dx);
                    let t = pt(stp, 2.0 + rise, dp);
                    let ac = cubic_min(x, t);
                    let aq = quad_min(x, stp, t.f);
                    assert!(
                        (ac - stx).abs() <= 2.0 / 3.0 * w * (1.0 + 1e-12),
                        "cubic bound {ac}"
                    );
                    assert!(
                        (aq - stx).abs() <= 0.5 * w * (1.0 + 1e-12),
                        "quadratic bound {aq}"
                    );
                    let want = if (ac - stx).abs() < (aq - stx).abs() {
                        ac
                    } else {
                        0.5 * (ac + aq)
                    };
                    let d = run(x, pt(0.0, 0.0, 0.0), t, false);
                    assert_eq!(d.case, Case::HigherValue);
                    assert!(close(d.stp, want, 1e-10), "step {} want {want}", d.stp);
                    assert!((d.stp - stx).abs() <= 7.0 / 12.0 * w * (1.0 + 1e-12));
                    n += 1;
                }
            }
        }
        assert_eq!(n, 5 * 7 * 3);
    }

    #[test]
    fn case_one_takes_the_cubic_when_it_is_closer() {
        // phi = a^3 - 3a^2 ... chosen so that |a_c - a_l| < |a_q - a_l|: a steep cubic rise.
        // x = (0, 0, -1), t = (2, 8, 4): a_c < a_q.
        let x = pt(0.0, 0.0, -1.0);
        let t = pt(2.0, 8.0, 4.0);
        let ac = cubic_min(x, t);
        let aq = quad_min(x, 2.0, 8.0);
        assert!(ac.abs() < aq.abs(), "test setup: {ac} vs {aq}");
        assert!(close(run(x, x, t, false).stp, ac, 1e-12));
        // and the average when it is not: x = (0, 0, -1), t = (2, 1, 5): the cubic is farther
        let t = pt(2.0, 1.0, -0.5);
        let ac = cubic_min(x, t);
        let aq = quad_min(x, 2.0, 1.0);
        assert!(ac.abs() >= aq.abs(), "test setup: {ac} vs {aq}");
        assert!(close(run(x, x, t, false).stp, 0.5 * (ac + aq), 1e-12));
    }

    #[test]
    fn case_two_takes_the_farther_of_cubic_and_secant() {
        // A cubic phi(a) = -a + 0.1 a^3 through x = 0 and stp = 2 (opposite slopes).
        let f = |a: f64| -a + 0.1 * a * a * a;
        let g = |a: f64| -1.0 + 0.3 * a * a;
        let (x, t) = (pt(0.0, f(0.0), g(0.0)), pt(2.0, f(2.0), g(2.0)));
        let ac = cubic_min(x, t);
        let aq = secant(t.a, t.g, x.a, x.g);
        let want = if (ac - t.a).abs() > (aq - t.a).abs() {
            ac
        } else {
            aq
        };
        let d = run(x, x, t, false);
        assert_eq!(d.case, Case::OppositeSlopes);
        assert!(close(d.stp, want, 1e-12), "{} vs {want}", d.stp);
        // phi has its minimizer at sqrt(10/3) between them
        assert!(d.stp > 0.0 && d.stp < 2.0);
    }

    #[test]
    fn quadratic_data_is_hit_exactly_by_every_interpolant() {
        // phi(a) = (a - m)^2 - m^2: cubic, quadratic and secant steps all give the minimizer m.
        let m = 1.25;
        let f = |a: f64| (a - m) * (a - m) - m * m;
        let g = |a: f64| 2.0 * (a - m);
        let x = pt(0.0, f(0.0), g(0.0));
        let far = pt(10.0 * m, f(10.0 * m), g(10.0 * m));
        // Case 1 (stp = 4 > 2m): a_c = a_q = m, so the step is m whichever rule applies
        let d = run(x, x, pt(4.0, f(4.0), g(4.0)), false);
        assert_eq!(d.case, Case::HigherValue);
        assert!(close(d.stp, m, 1e-12), "{}", d.stp);
        // Case 2 (m < stp < 2m)
        let d = run(x, x, pt(2.0, f(2.0), g(2.0)), false);
        assert_eq!(d.case, Case::OppositeSlopes);
        assert!(close(d.stp, m, 1e-12), "{}", d.stp);
        // Case 3 (stp < m), unbracketed and bracketed with the 0.66 cap beyond m
        for (brackt, y) in [(false, x), (true, far)] {
            let d = run(x, y, pt(0.5, f(0.5), g(0.5)), brackt);
            assert_eq!(d.case, Case::DecreasingSlope);
            assert!(close(d.stp, m, 1e-12), "brackt = {brackt}: {}", d.stp);
        }
    }

    #[test]
    fn case_three_is_capped_and_clipped() {
        // phi = -a + 0.4 a^2: x = 0, stp = 0.5; the cubic, quadratic and secant steps are all 1.25.
        let x = pt(0.0, 0.0, -1.0);
        let t = pt(0.5, -0.4, -0.6);
        // bracketed with sty = 1: capped at stp + 0.66 (sty - stp) = 0.83
        let d = dcstep(x, pt(1.0, 5.0, 1.0), t, true, 0.0, 10.0);
        assert!(close(d.stp, 0.5 + 0.66 * 0.5, 1e-14), "{}", d.stp);
        // unbracketed: clipped into [stpmin, stpmax], and not clipped when inside
        assert!(close(dcstep(x, x, t, false, 0.0, 1.0).stp, 1.0, 1e-14));
        assert!(close(dcstep(x, x, t, false, 0.0, 10.0).stp, 1.25, 1e-12));
        assert!(close(dcstep(x, x, t, false, 2.0, 10.0).stp, 2.0, 1e-14));
    }

    #[test]
    fn case_four_unbracketed_goes_to_the_bound() {
        let x = pt(0.0, 0.0, -1.0);
        let t = pt(1.0, -1.5, -1.5);
        let d = dcstep(x, x, t, false, 0.25, 7.0);
        assert_eq!((d.case, d.stp), (Case::SteepSlope, 7.0));
        // stepping to the left (stp < stx) goes to stpmin
        let x = pt(0.0, 0.0, 1.0);
        let t = pt(-1.0, -1.5, 1.5);
        let d = dcstep(x, x, t, false, -7.0, -0.25);
        assert_eq!((d.case, d.stp), (Case::SteepSlope, -7.0));
    }

    #[test]
    fn case_four_bracketed_is_the_cubic_through_stp_and_y() {
        // Case 4 needs a lower value with a slope that does not shrink: phi(a) = -(a^3 - 3a) on
        // x = 2, stp = 2.5 (phi' = -9, -15.75), with y = 4 beyond. The step is the stationary
        // point of the cubic through (stp, fp, dp) and (sty, fy, dy), here phi itself: a = sqrt(1).
        let f = |a: f64| -(a * a * a - 3.0 * a);
        let g = |a: f64| -(3.0 * a * a - 3.0);
        let (x, t, y) = (
            pt(2.0, f(2.0), g(2.0)),
            pt(2.5, f(2.5), g(2.5)),
            pt(4.0, f(4.0), g(4.0)),
        );
        let d = dcstep(x, y, t, true, 0.0, 10.0);
        assert_eq!(d.case, Case::SteepSlope);
        assert!(d.brackt);
        let want = cubic_min(t, y);
        assert!(close(d.stp, want, 1e-12), "{} vs {want}", d.stp);
        assert!(close(d.stp, 1.0, 1e-12), "phi is the cubic: {}", d.stp);
    }

    #[test]
    fn zero_slope_at_stp_is_not_opposite() {
        // dp = 0 is not "opposite sign" to dx < 0 (sgnd = 0): the lower value updates x and leaves y.
        let x = pt(0.0, 0.0, -1.0);
        let y = pt(9.0, 4.0, 2.0);
        let t = pt(1.0, -0.5, 0.0);
        let d = dcstep(x, y, t, false, 0.0, 5.0);
        assert_eq!(d.case, Case::DecreasingSlope);
        assert!(!d.brackt);
        assert_eq!((d.x, d.y), (t, y));
        // phi = -a + a^2/2 is a quadratic whose cubic is degenerate (r = 0): the unbracketed step
        // is the bound, the secant step stp itself; the farther one (the bound) is taken
        assert_eq!(d.stp, 5.0);
        // bracketed, the cubic/secant choice picks the secant step = stp (capped by the 0.66 rule)
        let d = dcstep(x, y, t, true, 0.0, 5.0);
        assert!(!d.stp.is_nan() && d.stp >= 1.0, "{}", d.stp);
        // dx = 0 as well: sign tests are false in both directions
        let d = dcstep(pt(0.0, 0.0, 0.0), y, pt(1.0, -0.5, 0.0), false, 0.0, 5.0);
        assert_ne!(d.case, Case::OppositeSlopes);
    }

    #[test]
    fn degenerate_cubic_gives_a_non_finite_step() {
        // Two collinear points (Moré–Thuente's example (3.1) in the limit): phi = a, so
        // theta = -s, the discriminant is 0, gamma = 0 and q = 0: r = -2s/0 = -inf.
        let x = pt(0.0, 0.0, 1.0);
        let t = pt(1.0, 1.0, 1.0);
        let d = run(x, x, t, false);
        assert_eq!(d.case, Case::HigherValue);
        assert!(!d.stp.is_finite(), "{}", d.stp);
    }

    #[test]
    fn nan_input_propagates_to_the_step() {
        let x = pt(0.0, 0.0, -1.0);
        // Case 1 with a NaN slope at x
        let d = run(pt(0.0, 0.0, f64::NAN), x, pt(1.0, 1.0, 0.5), false);
        assert_eq!(d.case, Case::HigherValue);
        assert!(d.stp.is_nan());
        // Case 4 bracketed with a NaN slope at stp
        let d = run(x, pt(3.0, 1.0, 1.0), pt(1.0, -1.5, f64::NAN), true);
        assert!(d.stp.is_nan(), "{}", d.stp);
        // overflow of the cubic (theta = -inf) also ends in NaN, not a finite step
        let d = run(x, x, pt(1.0, 1e308, 1.0), false);
        assert_eq!(d.case, Case::HigherValue);
        assert!(d.stp.is_nan(), "{}", d.stp);
    }

    #[test]
    fn mirror_invariance() {
        // phi(-a) mirrors the search direction: every step is negated, every case is the same.
        let mirror = |p: Pt<f64>| pt(-p.a, p.f, -p.g);
        let x = pt(0.0, 0.0, -1.0);
        let y = pt(10.0, 5.0, 1.0);
        for t in [
            pt(1.0, 0.5, -0.2),
            pt(2.0, -1.2, 0.2),
            pt(0.5, -0.4, -0.6),
            pt(1.0, -1.5, -1.5),
        ] {
            for brackt in [false, true] {
                let a = dcstep(x, y, t, brackt, 0.25, 7.0);
                let b = dcstep(mirror(x), mirror(y), mirror(t), brackt, -7.0, -0.25);
                assert_eq!(a.case, b.case);
                assert_eq!(a.brackt, b.brackt);
                assert!(close(a.stp, -b.stp, 1e-12), "{} vs {}", a.stp, b.stp);
                assert_eq!((mirror(a.x), mirror(a.y)), (b.x, b.y));
            }
        }
    }
}
