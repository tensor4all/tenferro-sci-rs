//! Classical orthogonal polynomials of integer degree, precision-adaptive.
//!
//! Written from the three-term recurrences of the NIST Digital Library of
//! Mathematical Functions (DLMF), chapter 18 (Table 18.9.1 and the
//! relations of Chebyshev, Legendre and Hermite polynomials in 18.7 and
//! 18.9); no code was translated. Every function is a forward recurrence in
//! the degree: the polynomial is the dominant solution for an argument beyond
//! its zeros, and inside them both solutions have comparable size, so the
//! recurrence loses no relative accuracy away from the zeros.
//!
//! The terms are kept as a mantissa and a binary exponent ([`Pair`]), so an
//! intermediate term cannot overflow or underflow while the result is
//! representable; only the final scaling can.
//!
//! * Jacobi `P_n^(a,b)`, Gegenbauer `C_n^(a)`, Legendre `P_n`, Chebyshev
//!   `T_n` and `U_n`, Laguerre `L_n^(a)` and Hermite `H_n` and `He_n`: the
//!   recurrences of DLMF 18.9, from `p_0 = 1` and the explicit `p_1`.
//! * The rescaled and shifted Chebyshev and Legendre polynomials
//!   `C_n(x) = 2 T_n(x/2)`, `S_n(x) = U_n(x/2)`, `T*_n(x) = T_n(2x - 1)`,
//!   `U*_n(x) = U_n(2x - 1)` and `P*_n(x) = P_n(2x - 1)` evaluate the base
//!   polynomial at the transformed argument.
//! * SciPy's shifted Jacobi polynomial is `G_n(p, q, x) =
//!   P_n^(p-q, q-1)(2x - 1) / C(2n + p - 1, n)`; the binomial coefficient is the
//!   product `prod_(j=1..n) (n + p - 1 + j) / j`.
//! * At `a = -1` the first coefficient of the Jacobi recurrence vanishes; the
//!   relation `P_n^(-1,b)(x) = (n + b) / (2n) (x - 1) P_(n-1)^(1,b)(x)` (Szego,
//!   Orthogonal Polynomials, 4.22.2) replaces it, and `P_n^(a,-1)(x) = (-1)^n
//!   P_n^(-1,a)(-x)` handles `b = -1`.
//!
//! The Jacobi recurrence divides by `2 (j + 1)(j + a + b + 1)(2j + a + b)`;
//! where that is zero (`a + b` a negative integer below `-1`) the result is
//! not finite.

use super::{ldexp, mul};
use crate::real::{k, Real};

/// Two consecutive terms `(a, b)` of a recurrence, with the common factor
/// `2^e`.
pub(crate) struct Pair<T> {
    pub(crate) a: T,
    pub(crate) b: T,
    pub(crate) e: i64,
}

impl<T: Real> Pair<T> {
    pub(crate) fn new(a: T, b: T) -> Self {
        let mut p = Pair { a, b, e: 0 };
        p.renorm();
        p
    }

    /// Shift the pair and keep the larger term's exponent near zero.
    pub(crate) fn push(&mut self, next: T) {
        self.a = self.b;
        self.b = next;
        self.renorm();
    }

    fn renorm(&mut self) {
        let big = self.a.abs().max(self.b.abs()).approx_f64();
        if big.is_finite() && big != 0.0 && !(2.0_f64.powi(-200)..=2.0_f64.powi(200)).contains(&big)
        {
            let ex = big.log2().floor() as i64;
            let s = (-ex) as i32;
            self.a = ldexp(self.a, s);
            self.b = ldexp(self.b, s);
            self.e += ex;
        }
    }
}

/// `m 2^e`.
#[derive(Clone, Copy)]
pub(crate) struct Sc<T> {
    pub(crate) m: T,
    pub(crate) e: i64,
}

impl<T: Real> Sc<T> {
    pub(crate) fn value(self) -> T {
        ldexp(self.m, self.e.clamp(-3000, 3000) as i32)
    }
}

/// `p_n` of the recurrence `p_(j+1) = step(j, p_(j-1), p_j)` from `p_0`, `p_1`.
fn recur<T: Real>(n: i64, p0: T, p1: T, step: impl Fn(T, T, T) -> T) -> Sc<T> {
    if n == 0 {
        return Sc { m: p0, e: 0 };
    }
    if n == 1 {
        return Sc { m: p1, e: 0 };
    }
    let mut s = Pair::new(p0, p1);
    for j in 1..n {
        let next = step(k::<T>(j as f64), s.a, s.b);
        s.push(next);
    }
    Sc { m: s.b, e: s.e }
}

fn two<T: Real>() -> T {
    k::<T>(2.0)
}

/// `P_n^(α,β)(x)` as a scaled value, `n >= 0`.
fn jacobi_scaled<T: Real>(n: i64, alpha: T, beta: T, x: T) -> Sc<T> {
    let one = T::one();
    if n == 0 {
        return Sc { m: one, e: 0 };
    }
    // Where the first coefficient of the recurrence vanishes (DLMF 18.9.2 at
    // alpha = -1) or the weight has a singular endpoint.
    if alpha == -one {
        let q = jacobi_scaled(n - 1, one, beta, x);
        let f = (k::<T>(n as f64) + beta) / (two::<T>() * k::<T>(n as f64)) * (x - one);
        return Sc {
            m: mul(f, q.m),
            e: q.e,
        };
    }
    if beta == -one {
        let q = jacobi_scaled(n, beta, alpha, -x);
        return Sc {
            m: if n % 2 == 0 { q.m } else { -q.m },
            e: q.e,
        };
    }
    let a1 = alpha + one;
    let p1 = (two::<T>() * a1 + (alpha + beta + two::<T>()) * (x - one)) / two::<T>();
    let ab = alpha + beta;
    let a2b2 = (alpha - beta) * ab;
    recur(n, one, p1, |j, pm, p| {
        let t = two::<T>() * j + ab;
        let den = two::<T>() * (j + one) * (j + ab + one) * t;
        let num_p = (t + one) * ((t + two::<T>()) * t * x + a2b2);
        let num_m = two::<T>() * (j + alpha) * (j + beta) * (t + two::<T>());
        (mul(num_p, p) - mul(num_m, pm)) / den
    })
}

/// `P_n^(α,β)(x)`.
pub(crate) fn jacobi<T: Real>(n: i64, alpha: T, beta: T, x: T) -> T {
    jacobi_scaled(n, alpha, beta, x).value()
}

/// `C(2n + p - 1, n) = prod_(j=1..n) (n + p - 1 + j) / j`, as a scaled value.
fn binom_2n<T: Real>(n: i64, p: T) -> Sc<T> {
    let mut s = Sc { m: T::one(), e: 0 };
    let base = k::<T>(n as f64) + p - T::one();
    for j in 1..=n {
        let jf = k::<T>(j as f64);
        s.m = mul(s.m, (base + jf) / jf);
        let big = s.m.abs().approx_f64();
        if big.is_finite() && big != 0.0 && !(2.0_f64.powi(-200)..=2.0_f64.powi(200)).contains(&big)
        {
            let ex = big.log2().floor() as i64;
            s.m = ldexp(s.m, (-ex) as i32);
            s.e += ex;
        }
    }
    s
}

/// The shifted Jacobi polynomial `G_n(p, q, x) = P_n^(p-q, q-1)(2x - 1) / C(2n + p - 1, n)`.
pub(crate) fn sh_jacobi<T: Real>(n: i64, p: T, q: T, x: T) -> T {
    let num = jacobi_scaled(n, p - q, q - T::one(), two::<T>() * x - T::one());
    let den = binom_2n(n, p);
    let r = super::div(num.m, den.m);
    Sc {
        m: r,
        e: num.e - den.e,
    }
    .value()
}

/// `C_n^(α)(x)`; `0` for `α = 0` and for `n < 0`, as xsf.
pub(crate) fn gegenbauer<T: Real>(n: i64, alpha: T, x: T) -> T {
    if alpha == T::zero() || n < 0 {
        return T::zero();
    }
    let one = T::one();
    recur(n, one, two::<T>() * alpha * x, |j, pm, p| {
        let a = mul(two::<T>() * (j + alpha), x);
        (mul(a, p) - mul(j + two::<T>() * alpha - one, pm)) / (j + one)
    })
    .value()
}

/// `P_n(x)`, with `P_(-n-1) = P_n`.
pub(crate) fn legendre<T: Real>(n: i64, x: T) -> T {
    let n = if n < 0 { -(n + 1) } else { n };
    let one = T::one();
    recur(n, one, x, |j, pm, p| {
        let a = mul(two::<T>() * j + one, x);
        (mul(a, p) - mul(j, pm)) / (j + one)
    })
    .value()
}

/// `T_n(x)`, with `T_(-n) = T_n`.
pub(crate) fn chebyt<T: Real>(n: i64, x: T) -> T {
    let n = n.abs();
    recur(n, T::one(), x, |_, pm, p| mul(two::<T>() * x, p) - pm).value()
}

/// `U_n(x)`, with `U_(-1) = 0` and `U_(-n-2) = -U_n`.
pub(crate) fn chebyu<T: Real>(n: i64, x: T) -> T {
    if n == -1 {
        return T::zero();
    }
    let (n, neg) = if n < -1 { (-n - 2, true) } else { (n, false) };
    let v = recur(n, T::one(), two::<T>() * x, |_, pm, p| {
        mul(two::<T>() * x, p) - pm
    })
    .value();
    if neg {
        -v
    } else {
        v
    }
}

/// `C_n(x) = 2 T_n(x/2)`.
pub(crate) fn chebyc<T: Real>(n: i64, x: T) -> T {
    two::<T>() * chebyt(n, x / two::<T>())
}

/// `S_n(x) = U_n(x/2)`.
pub(crate) fn chebys<T: Real>(n: i64, x: T) -> T {
    chebyu(n, x / two::<T>())
}

/// `T_n(2x - 1)`.
pub(crate) fn sh_chebyt<T: Real>(n: i64, x: T) -> T {
    chebyt(n, two::<T>() * x - T::one())
}

/// `U_n(2x - 1)`.
pub(crate) fn sh_chebyu<T: Real>(n: i64, x: T) -> T {
    chebyu(n, two::<T>() * x - T::one())
}

/// `P_n(2x - 1)`.
pub(crate) fn sh_legendre<T: Real>(n: i64, x: T) -> T {
    legendre(n, two::<T>() * x - T::one())
}

/// `L_n^(α)(x)` for `α > -1`: NaN for `α <= -1` and `0` for `n < 0`, as xsf.
pub(crate) fn genlaguerre<T: Real>(n: i64, alpha: T, x: T) -> T {
    let one = T::one();
    if alpha <= -one {
        return T::nan();
    }
    if n < 0 {
        return T::zero();
    }
    recur(n, one, one + alpha - x, |j, pm, p| {
        let a = two::<T>() * j + one + alpha - x;
        (mul(a, p) - mul(j + alpha, pm)) / (j + one)
    })
    .value()
}

/// Probabilists' Hermite polynomial `He_n(x)`; NaN for `n < 0`.
pub(crate) fn hermitenorm<T: Real>(n: i64, x: T) -> T {
    if n < 0 {
        return T::nan();
    }
    recur(n, T::one(), x, |j, pm, p| mul(x, p) - mul(j, pm)).value()
}

/// Physicists' Hermite polynomial `H_n(x)`; NaN for `n < 0`.
pub(crate) fn hermite<T: Real>(n: i64, x: T) -> T {
    if n < 0 {
        return T::nan();
    }
    recur(n, T::one(), two::<T>() * x, |j, pm, p| {
        mul(two::<T>() * x, p) - mul(two::<T>() * j, pm)
    })
    .value()
}

/// The `f64` value of an integer-degree kernel where a double-double result
/// is not finite: an overflow or an infinite argument, whose classification
/// (sign of the infinity, NaN) follows xsf.
pub(crate) fn finite_or<T: Real>(r: T, f64_kernel: impl FnOnce() -> f64) -> T {
    if r.approx_f64().is_finite() {
        r
    } else {
        T::from_dd(f64_kernel(), 0.0)
    }
}
