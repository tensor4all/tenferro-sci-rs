//! Gauss hypergeometric function `2F1(a, b; c; x)` for real `x`, `f64`.
//!
//! Ported from xsf (scipy/xsf at `31177fd9a458539cf05c04d215341245d0245e53`)
//! `include/xsf/cephes/hyp2f1.h`, the kernel behind the real
//! overload `hyp2f1(double, double, double, double)` of `include/xsf/hyp2f1.h`.
//! Cephes Math Library Release 2.8, Copyright 1984, 1987, 1992, 2000 by
//! Stephen L. Moshier; the `x < -1` transformation (AMS 15.3.7) is by Travis
//! Oliphant; C++ translation by SciPy developers (2024).
//!
//! The C code reaches its exits with `goto`; here the labels `hypdon`, `hypf`
//! and `hypdiv` are the variants of [`Exit`]. The loss-of-precision flags
//! (`SF_ERROR_LOSS`, `SF_ERROR_SLOW`) are not reported; the value is returned
//! as in xsf, and NaN where xsf returns NaN.

use super::consts::MACHEP;
use super::gamma::{gamma, lgam_sgn, rgamma};
use super::psi::psi;

const EPS: f64 = 1.0e-13;
const ETHRESH: f64 = 1.0e-12;
const MAXITER: u64 = 10000;

/// Power series of the Gauss hypergeometric function. Returns the value and
/// the estimated loss of significance.
fn hys2f1(mut a: f64, mut b: f64, c: f64, x: f64) -> (f64, f64) {
    let mut intflag = false;

    if b.abs() > a.abs() {
        // Ensure that |a| > |b| ...
        core::mem::swap(&mut a, &mut b);
    }

    let ib = b.round();

    if (b - ib).abs() < EPS && ib <= 0.0 && (b.abs() < a.abs() || (b == -a && a <= MAXITER as f64))
    {
        // ... except when `b` is a negative integer with |b| <= |a|
        core::mem::swap(&mut a, &mut b);
        intflag = a.abs() != b.abs(); // not for the tie `b == -a`
    }

    if (a.abs() > c.abs() + 1.0 || intflag) && (c - a).abs() > 2.0 && a.abs() > 2.0 {
        // |a| >> |c| implies that large cancellation error is to be expected.
        //
        // We try to reduce it with the recurrence relations
        return hyp2f1ra(a, b, c, x);
    }

    let mut i: u64 = 0;
    let mut umax = 0.0;
    let f = a;
    let g = b;
    let h = c;
    let mut s = 1.0;
    let mut u = 1.0;
    let mut k = 0.0;
    loop {
        if h.abs() < EPS {
            return (f64::INFINITY, 1.0);
        }
        let m = k + 1.0;
        u *= (f + k) * (g + k) * x / ((h + k) * m);
        s += u;
        k = u.abs(); // remember largest term summed
        if k > umax {
            umax = k;
        }
        k = m;
        i += 1;
        if i > MAXITER {
            // should never happen
            return (s, 1.0);
        }
        if !(s == 0.0 || (u / s).abs() > MACHEP) {
            break;
        }
    }

    // return estimated relative error
    (s, (MACHEP * umax) / s.abs() + (MACHEP * i as f64))
}

/// Transformations for `|x|` near 1, then the power series.
fn hyt2f1(a: f64, b: f64, c: f64, x: f64) -> (f64, f64) {
    let ia = a.round();
    let ib = b.round();

    let neg_int_a = a <= 0.0 && (a - ia).abs() < EPS; // a is a negative integer
    let neg_int_b = b <= 0.0 && (b - ib).abs() < EPS; // b is a negative integer

    let s = 1.0 - x;
    if x < -0.5 && !(neg_int_a || neg_int_b) {
        let (v, err) = if b > a {
            let (v, err) = hys2f1(a, c - b, c, -x / s);
            (s.powf(-a) * v, err)
        } else {
            let (v, err) = hys2f1(c - a, b, c, -x / s);
            (s.powf(-b) * v, err)
        };
        return (v, err);
    }

    let d = c - a - b;
    let id = d.round(); // nearest integer to d

    if x > 0.9 && !(neg_int_a || neg_int_b) {
        if (d - id).abs() > MACHEP {
            // test for integer c-a-b
            // Try the power series first
            let (y, err) = hys2f1(a, b, c, x);
            if err < ETHRESH {
                return (y, err);
            }
            // If power series fails, then apply AMS55 #15.3.6
            let (mut q, mut err) = hys2f1(a, b, 1.0 - d, s);
            let mut sign = 1.0;
            let (mut w, sgngam) = lgam_sgn(d);
            sign *= sgngam;
            let (l, sgngam) = lgam_sgn(c - a);
            w -= l;
            sign *= sgngam;
            let (l, sgngam) = lgam_sgn(c - b);
            w -= l;
            sign *= sgngam;
            q *= sign * w.exp();
            let (v, err1) = hys2f1(c - a, c - b, d + 1.0, s);
            let mut r = s.powf(d) * v;
            sign = 1.0;
            let (mut w, sgngam) = lgam_sgn(-d);
            sign *= sgngam;
            let (l, sgngam) = lgam_sgn(a);
            w -= l;
            sign *= sgngam;
            let (l, sgngam) = lgam_sgn(b);
            w -= l;
            sign *= sgngam;
            r *= sign * w.exp();
            let mut y = q + r;

            q = q.abs(); // estimate cancellation error
            r = r.abs();
            if q > r {
                r = q;
            }
            err += err1 + (MACHEP * r) / y;

            y *= gamma(c);
            return (y, err);
        }

        // Psi function expansion, AMS55 #15.3.10, #15.3.11, #15.3.12
        //
        // Although AMS55 does not explicitly state it, this expansion fails
        // for negative integer a or b, since the psi and Gamma functions
        // involved have poles.
        let (e, d1, d2, aid);
        if id >= 0.0 {
            e = d;
            d1 = d;
            d2 = 0.0;
            aid = id as i32;
        } else {
            e = -d;
            d1 = 0.0;
            d2 = d;
            aid = (-id) as i32;
        }

        let ax = s.ln();

        // sum for t = 0
        let mut y = psi(1.0) + psi(1.0 + e) - psi(a + d1) - psi(b + d1) - ax;
        y *= rgamma(e + 1.0);

        let mut p = (a + d1) * (b + d1) * s * rgamma(e + 2.0); // Poch for t=1
        let mut t = 1.0;
        loop {
            let r = psi(1.0 + t) + psi(1.0 + t + e) - psi(a + t + d1) - psi(b + t + d1) - ax;
            let q = p * r;
            y += q;
            p *= s * (a + t + d1) / (t + 1.0);
            p *= (b + t + d1) / (t + 1.0 + e);
            t += 1.0;
            if t > MAXITER as f64 {
                // should never happen
                return (f64::NAN, 1.0);
            }
            if !(y == 0.0 || (q / y).abs() > EPS) {
                break;
            }
        }

        if id == 0.0 {
            y *= gamma(c) / (gamma(a) * gamma(b));
            return (y, 0.0);
        }

        let mut y1 = 1.0;

        if aid != 1 {
            let mut t = 0.0;
            let mut p = 1.0;
            for _ in 1..aid {
                let r = 1.0 - e + t;
                p *= s * (a + t + d2) * (b + t + d2) / r;
                t += 1.0;
                p /= t;
                y1 += p;
            }
        }
        let p = gamma(c);
        y1 *= gamma(e) * p * (rgamma(a + d1) * rgamma(b + d1));

        y *= p * (rgamma(a + d2) * rgamma(b + d2));
        if (aid & 1) != 0 {
            y = -y;
        }

        let q = s.powf(id); // s to the id power
        if id > 0.0 {
            y *= q;
        } else {
            y1 *= q;
        }

        y += y1;
        return (y, 0.0);
    }

    // Use defining power series if no special cases
    hys2f1(a, b, c, x)
}

/// A.&S. 15.4.2.
fn hyp2f1_neg_c_equal_bc(a: f64, b: f64, x: f64) -> f64 {
    let mut collector = 1.0;
    let mut sum = 1.0;
    let mut collector_max: f64 = 1.0;

    if b.is_nan() || b.abs() >= 1e5 {
        return f64::NAN;
    }

    let mut k = 1.0;
    while k <= -b {
        collector *= (a + k - 1.0) * x / k;
        collector_max = collector.abs().max(collector_max);
        sum += collector;
        k += 1.0;
    }

    if 1e-16 * (1.0 + collector_max / sum.abs()) > 1e-7 {
        return f64::NAN;
    }

    sum
}

/// Evaluate the hypergeometric function by a two-term recurrence in `a`.
///
/// This avoids some of the loss of precision in the strongly alternating
/// hypergeometric series and reduces the parameter `a` to a smaller value.
/// AMS55 #15.2.10.
fn hyp2f1ra(a: f64, b: f64, c: f64, x: f64) -> (f64, f64) {
    // Don't cross c or zero
    let da = if (c < 0.0 && a <= c) || (c >= 0.0 && a >= c) {
        (a - c).round()
    } else {
        a.round()
    };
    let mut t = a - da;

    let mut loss = 0.0;

    if da.abs() > MAXITER as f64 {
        // Too expensive to compute this value, so give up
        return (f64::NAN, 1.0);
    }

    let mut f2;
    let mut f1;
    let mut f0;
    if da < 0.0 {
        // Recurse down
        let (v1, e1) = hys2f1(t, b, c, x);
        f1 = v1;
        loss += e1;
        let (v0, e0) = hys2f1(t - 1.0, b, c, x);
        f0 = v0;
        loss += e0;
        t -= 1.0;
        let mut n = 1.0;
        while n < -da {
            f2 = f1;
            f1 = f0;
            f0 = -(2.0 * t - c - t * x + b * x) / (c - t) * f1 - t * (x - 1.0) / (c - t) * f2;
            t -= 1.0;
            n += 1.0;
        }
    } else {
        // Recurse up
        let (v1, e1) = hys2f1(t, b, c, x);
        f1 = v1;
        loss += e1;
        let (v0, e0) = hys2f1(t + 1.0, b, c, x);
        f0 = v0;
        loss += e0;
        t += 1.0;
        let mut n = 1.0;
        while n < da {
            f2 = f1;
            f1 = f0;
            f0 = -((2.0 * t - c - t * x + b * x) * f1 + (c - t) * f2) / (t * (x - 1.0));
            t += 1.0;
            n += 1.0;
        }
    }

    (f0, loss)
}

/// Where the main routine continues, after the `goto` labels of the C code.
enum Exit {
    /// `hypdon`: return the value.
    Done(f64),
    /// `hypf`: the transformation for negative integer `c - a` or `c - b`,
    /// AMS55 #15.3.3.
    Hypf,
    /// `hypok`: the transformations of `hyt2f1`.
    Hypok,
    /// `hypdiv`: the series diverges.
    Hypdiv,
}

/// `2F1(a, b; c; x)` for real arguments.
pub(crate) fn hyp2f1(a: f64, b: f64, c: f64, x: f64) -> f64 {
    let ax = x.abs();
    let s = 1.0 - x;
    let mut ia = a.round(); // nearest integer to a
    let ib = b.round();

    if x == 0.0 {
        return 1.0;
    }

    let d = c - a - b;
    let id = d.round();

    if (a == 0.0 || b == 0.0) && c != 0.0 {
        return 1.0;
    }

    if a == -1.0 && c != 0.0 {
        let result = 1.0 - (b / c) * x; // hyp2f1(-1, b; c; x) = 1 - (b/c)*x, c != 0
        if result.abs() < 1e-15 {
            return 0.0;
        }
        return result;
    }

    let neg_int_a = a <= 0.0 && (a - ia).abs() < EPS; // a is a negative integer
    let neg_int_b = b <= 0.0 && (b - ib).abs() < EPS; // b is a negative integer

    if d <= -1.0 && !((d - id).abs() > EPS && s < 0.0) && !(neg_int_a || neg_int_b) {
        return s.powf(d) * hyp2f1(c - a, c - b, c, x);
    }

    let exit = 'exit: {
        if d <= 0.0 && x == 1.0 && !(neg_int_a || neg_int_b) {
            break 'exit Exit::Hypdiv;
        }

        if ax < 1.0 || x == -1.0 {
            // 2F1(a,b;b;x) = (1-x)**(-a)
            if (b - c).abs() < EPS {
                // b = c
                let y = if neg_int_b {
                    hyp2f1_neg_c_equal_bc(a, b, x)
                } else {
                    s.powf(-a) // s to the -a power
                };
                break 'exit Exit::Done(y);
            }
            if (a - c).abs() < EPS {
                // a = c
                break 'exit Exit::Done(s.powf(-b)); // s to the -b power
            }
        }

        if c <= 0.0 {
            let ic = c.round(); // nearest integer to c
            if (c - ic).abs() < EPS {
                // c is a negative integer
                // check if termination before explosion
                if neg_int_a && (ia > ic) {
                    break 'exit Exit::Hypok;
                }
                if neg_int_b && (ib > ic) {
                    break 'exit Exit::Hypok;
                }
                break 'exit Exit::Hypdiv;
            }
        }

        if neg_int_a || neg_int_b {
            // function is a polynomial
            break 'exit Exit::Hypok;
        }

        let t1 = (b - a).abs();
        if x < -2.0 && (t1 - t1.round()).abs() > EPS {
            // This transform has a pole for b-a integer, and
            // may produce large cancellation errors for |1/x| close 1
            let mut p = hyp2f1(a, 1.0 - c + a, 1.0 - b + a, 1.0 / x);
            let mut q = hyp2f1(b, 1.0 - c + b, 1.0 - a + b, 1.0 / x);
            p *= (-x).powf(-a);
            q *= (-x).powf(-b);
            let t1 = gamma(c);
            let s = t1 * gamma(b - a) * (rgamma(b) * rgamma(c - a));
            let y = t1 * gamma(a - b) * (rgamma(a) * rgamma(c - b));
            return s * p + y * q;
        } else if x < -1.0 {
            if a.abs() < b.abs() {
                return s.powf(-a) * hyp2f1(a, c - b, c, x / (x - 1.0));
            } else {
                return s.powf(-b) * hyp2f1(b, c - a, c, x / (x - 1.0));
            }
        }

        if ax > 1.0 {
            // series diverges
            break 'exit Exit::Hypdiv;
        }

        let p = c - a;
        ia = p.round(); // nearest integer to c-a
        let mut neg_int_ca_or_cb = false;
        if ia <= 0.0 && (p - ia).abs() < EPS {
            // negative int c - a
            neg_int_ca_or_cb = true;
        }

        let r = c - b;
        let ib = r.round(); // nearest integer to c-b
        if ib <= 0.0 && (r - ib).abs() < EPS {
            // negative int c - b
            neg_int_ca_or_cb = true;
        }

        let id = d.round(); // nearest integer to d

        // Thanks to Christian Burger <BURGER@DMRHRZ11.HRZ.Uni-Marburg.DE>
        // for reporting a bug here.
        if (ax - 1.0).abs() < EPS {
            // |x| == 1.0
            if x > 0.0 {
                if neg_int_ca_or_cb {
                    if d >= 0.0 {
                        break 'exit Exit::Hypf;
                    } else {
                        break 'exit Exit::Hypdiv;
                    }
                }
                if d <= 0.0 {
                    break 'exit Exit::Hypdiv;
                }
                break 'exit Exit::Done(gamma(c) * gamma(d) * (rgamma(p) * rgamma(r)));
            }
            if d <= -1.0 {
                break 'exit Exit::Hypdiv;
            }
        }

        // Conditionally make d > 0 by recurrence on c
        // AMS55 #15.2.27
        if d < 0.0 {
            // Try the power series first
            let (y, err) = hyt2f1(a, b, c, x);
            if err < ETHRESH {
                break 'exit Exit::Done(y);
            }
            // Apply the recurrence if power series fails
            let aid = (2.0 - id) as i32;
            let mut e = c + f64::from(aid);
            let mut d2 = hyp2f1(a, b, e, x);
            let mut d1 = hyp2f1(a, b, e + 1.0, x);
            let q = a + b + 1.0;
            let mut y = 0.0;
            for _ in 0..aid {
                let r = e - 1.0;
                y = (e * (r - (2.0 * e - q) * x) * d2 + (e - a) * (e - b) * x * d1) / (e * r * s);
                e = r;
                d1 = d2;
                d2 = y;
            }
            break 'exit Exit::Done(y);
        }

        if neg_int_ca_or_cb {
            break 'exit Exit::Hypf; // negative integer c-a or c-b
        }

        Exit::Hypok
    };

    match exit {
        Exit::Done(y) => y,
        // The transformation for c-a or c-b negative integer, AMS55 #15.3.3
        Exit::Hypf => s.powf(d) * hys2f1(c - a, c - b, c, x).0,
        Exit::Hypok => hyt2f1(a, b, c, x).0,
        // The alarm exit
        Exit::Hypdiv => f64::INFINITY,
    }
}
