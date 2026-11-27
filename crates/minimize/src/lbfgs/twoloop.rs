//! The L-BFGS two-loop recursion `out = H g`.
//!
//! Written independently for tensor4all/tenferro-sci-rs#40 from the published recursion (J. Nocedal, Math. Comp. 35(151),
//! 773–782, 1980, doi:10.1090/S0025-5718-1980-0572855-7); no source code was translated.
//!
//! **Red stage:** the stub returns `H g = g` (`H = I`), so every test that needs a non-trivial `H` fails.

use super::memory::Memory;
use crate::vector::Vector;

/// Computes `out = H g`, `H` the L-BFGS matrix of the pairs in `mem` with `H₀ = gamma · I`. `alpha` is scratch space
/// of length at least `mem.capacity()`; nothing is allocated.
pub(crate) fn twoloop<V: Vector>(
    _mem: &Memory<V>,
    ctx: &mut V::Ctx,
    _gamma: f64,
    g: &V,
    out: &mut V,
    _alpha: &mut [f64],
) {
    // Stub: H = I.
    out.clone_from(g);
    let _ = ctx;
}

#[cfg(test)]
mod tests {
    //! Validates the two-loop recursion against (1) the dense matrix it represents (the BFGS product form), (2) the secant equation of the newest pair, (3) the exact direction `−H g` recomputed in
    //! mpmath from the pairs SciPy stored (`tests/data/lbfgs/searches.csv`, `pairs.csv`), (4) the finite termination
    //! of BFGS on a quadratic with exact line searches and `H₀ = I`.
    //!
    //! Tolerances (all derived from the dot-product error bound γₙ Σ|terms|, Higham §3.1): (1), (2) `4 (2m+1) n ε S`
    //! with `S` the absolute-value recursion of the same computation (measured: 2% of the bound on the committed
    //! seeds); (3) `16 ε d_scale` (measured maximum 1.34 on 1028 searches; README); (4) `8 n κ ε ‖g₀‖` for the final
    //! gradient and `16 n κ ε` for `Hₙ A − I` (measured: 1e-6 and 1.0 of these bounds on κ = 1e2, 1e4).
    //!
    //! The dense matrices and the reference loops here are test oracles, not library kernels.
    use super::*;
    use std::collections::HashMap;

    const EPS: f64 = f64::EPSILON;

    /// Knuth's MMIX linear congruential generator; the top 53 bits give a uniform number in [0, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        // contraction-kernel-ok: test oracle on tiny vectors
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// A memory holding the pairs, fed through `push` with `dr = sᵀy`, `ddum = 0`.
    fn memory_of(pairs: &[(Vec<f64>, Vec<f64>)], m: usize) -> Memory<Vec<f64>> {
        let mut mem = Memory::new(m, &vec![0.0; pairs[0].0.len()]);
        for (s, y) in pairs {
            let dr = dot(s, y);
            assert_eq!(
                mem.push(s, y, dr, 0.0, dot(y, y)),
                super::super::memory::Push::Stored
            );
        }
        mem
    }

    fn apply(mem: &Memory<Vec<f64>>, gamma: f64, g: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; g.len()];
        let mut alpha = vec![0.0; mem.capacity()];
        twoloop(mem, &mut (), gamma, &g.to_vec(), &mut out, &mut alpha);
        out
    }

    /// The dense `H` and the absolute-value matrix `H̄` of the product form
    /// `H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ` started from `γ I`.
    fn dense(pairs: &[(Vec<f64>, Vec<f64>)], gamma: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = pairs[0].0.len();
        let eye = |c: f64| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { c } else { 0.0 }).collect())
                .collect()
        };
        let mul = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            // contraction-kernel-ok: dense test oracle, n <= 6
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| (0..n).map(|k| a[i][k] * b[k][j]).sum())
                        .collect()
                })
                .collect()
        };
        let (mut h, mut hb) = (eye(gamma), eye(gamma.abs()));
        for (s, y) in pairs {
            let rho = 1.0 / dot(y, s);
            let a1: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { 1.0 } else { 0.0 } - rho * s[i] * y[j])
                        .collect()
                })
                .collect();
            let a2: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { 1.0 } else { 0.0 } - rho * y[i] * s[j])
                        .collect()
                })
                .collect();
            let abs = |a: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
                a.iter()
                    .map(|r| r.iter().map(|t| t.abs()).collect())
                    .collect()
            };
            let mut h2 = mul(&mul(&a1, &h), &a2);
            let mut hb2 = mul(&mul(&abs(&a1), &hb), &abs(&a2));
            // contraction-kernel-ok: dense test oracle, n <= 6
            for i in 0..n {
                for j in 0..n {
                    h2[i][j] += rho * s[i] * s[j];
                    hb2[i][j] += rho.abs() * s[i].abs() * s[j].abs();
                }
            }
            h = h2;
            hb = hb2;
        }
        (h, hb)
    }

    fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter().map(|r| dot(r, x)).collect()
    }

    /// Random pairs with `sᵀy > 0` by construction: `s` has `|s|₂ ≥ 1` (first entry shifted by 2) and `y = s + w/4`,
    /// `|w|∞ ≤ 1`, so `sᵀy ≥ |s| (|s| − 0.62) > 0` for `n ≤ 6`.
    fn random_pairs(rng: &mut Lcg, n: usize, m: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        (0..m)
            .map(|_| {
                let mut s: Vec<f64> = (0..n).map(|_| 2.0 * rng.next() - 1.0).collect();
                s[0] += 2.0;
                let y: Vec<f64> = s
                    .iter()
                    .map(|t| t + 0.25 * (2.0 * rng.next() - 1.0))
                    .collect();
                assert!(dot(&s, &y) > 0.0);
                (s, y)
            })
            .collect()
    }

    #[test]
    fn two_loop_equals_the_dense_bfgs_matrix() {
        let mut rng = Lcg(12345);
        let n = 6;
        for m in [1, 3, 5] {
            for trial in 0..20 {
                let pairs = random_pairs(&mut rng, n, m);
                let mem = memory_of(&pairs, m);
                let gamma = mem.gamma();
                let g: Vec<f64> = (0..n).map(|_| 2.0 * rng.next() - 1.0).collect();
                let (h, hb) = dense(&pairs, gamma);
                let (want, scale) = (
                    matvec(&h, &g),
                    matvec(&hb, &g.iter().map(|t| t.abs()).collect::<Vec<_>>()),
                );
                let got = apply(&mem, gamma, &g);
                for i in 0..n {
                    let tol = 4.0 * (2 * m + 1) as f64 * n as f64 * EPS * scale[i];
                    assert!(
                        (got[i] - want[i]).abs() <= tol,
                        "m {m} trial {trial} component {i}: {} vs {}, tol {tol:e}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn the_matrix_satisfies_the_secant_equation_of_the_newest_pair() {
        let mut rng = Lcg(777);
        let n = 6;
        for m in [1, 3, 5] {
            for _ in 0..20 {
                let pairs = random_pairs(&mut rng, n, m);
                let mem = memory_of(&pairs, m);
                let (s, y) = pairs.last().expect("m >= 1");
                let (_, hb) = dense(&pairs, mem.gamma());
                let scale = matvec(&hb, &y.iter().map(|t| t.abs()).collect::<Vec<_>>());
                let got = apply(&mem, mem.gamma(), y);
                for i in 0..n {
                    let tol = 4.0 * (2 * m + 1) as f64 * n as f64 * EPS * scale[i];
                    assert!(
                        (got[i] - s[i]).abs() <= tol,
                        "H y = s violated at {i}: {} vs {}",
                        got[i],
                        s[i]
                    );
                }
            }
        }
    }

    #[test]
    fn h0_is_a_free_scalar_and_the_secant_equation_holds_for_any_gamma() {
        // the two-loop takes γ as an argument (ScaledGamma passes mem.gamma(), FixedIdentity passes 1)
        let mut rng = Lcg(31337);
        let pairs = random_pairs(&mut rng, 4, 3);
        let mem = memory_of(&pairs, 3);
        let (s, y) = pairs.last().expect("pairs");
        for gamma in [1.0, 0.1, 7.5] {
            let (_, hb) = dense(&pairs, gamma);
            let scale = matvec(&hb, &y.iter().map(|t| t.abs()).collect::<Vec<_>>());
            let got = apply(&mem, gamma, y);
            for i in 0..4 {
                assert!(
                    (got[i] - s[i]).abs() <= 4.0 * 7.0 * 4.0 * EPS * scale[i],
                    "γ = {gamma}, component {i}"
                );
            }
        }
        // and with no pair at all the recursion is H₀ g = γ g exactly
        let empty: Memory<Vec<f64>> = Memory::new(3, &vec![0.0; 4]);
        assert_eq!(
            apply(&empty, 2.5, &[1.0, -2.0, 0.0, 4.0]),
            vec![2.5, -5.0, 0.0, 10.0]
        );
    }

    // ------------------------------------------------------------------ the exact direction from SciPy's data

    fn csv(name: &str) -> Vec<HashMap<String, String>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/lbfgs")
            .join(name);
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty());
        let header: Vec<&str> = lines.next().expect("header").split(',').collect();
        lines
            .map(|l| {
                header
                    .iter()
                    .zip(l.split(','))
                    .map(|(h, c)| (h.to_string(), c.to_string()))
                    .collect()
            })
            .collect()
    }

    fn vecf(s: &str) -> Vec<f64> {
        s.split(';').map(|t| t.parse().expect("float")).collect()
    }

    #[test]
    fn the_float_two_loop_matches_the_exact_direction_of_scipys_pairs() {
        let mut pairs: HashMap<(String, usize), (Vec<f64>, Vec<f64>)> = HashMap::new();
        for r in csv("pairs.csv") {
            pairs.insert(
                (r["run"].clone(), r["p"].parse().expect("p")),
                (vecf(&r["s"]), vecf(&r["y"])),
            );
        }
        let (mut worst, mut checked) = (0.0_f64, 0);
        for r in csv("searches.csv")
            .iter()
            .filter(|r| r["pair_lo"] != "none")
        {
            let (lo, hi): (usize, usize) = (
                r["pair_lo"].parse().expect("lo"),
                r["pair_hi"].parse().expect("hi"),
            );
            let window: Vec<(Vec<f64>, Vec<f64>)> = (lo..=hi)
                .map(|p| pairs[&(r["run"].clone(), p)].clone())
                .collect();
            let mem = memory_of(&window, window.len());
            let h_g = apply(&mem, mem.gamma(), &vecf(&r["g_base"]));
            let (exact, scale) = (vecf(&r["d_exact"]), vecf(&r["d_scale"]));
            for i in 0..exact.len() {
                let err = (-h_g[i] - exact[i]).abs();
                let ratio = err / (EPS * scale[i]).max(f64::MIN_POSITIVE);
                worst = worst.max(if scale[i] == 0.0 { 0.0 } else { ratio });
                assert!(
                    scale[i] == 0.0 && err == 0.0 || ratio <= 16.0,
                    "{} search {} component {i}: {} vs exact {}, ratio {ratio:.1}",
                    r["run"],
                    r["j"],
                    -h_g[i],
                    exact[i]
                );
            }
            checked += 1;
        }
        assert!(checked >= 900, "only {checked} searches with a memory");
        eprintln!("two-loop vs exact direction: largest |d - d_exact| / (eps d_scale) = {worst:.2} (bound 16)");
    }

    // ------------------------------------------------------------------ n-step termination

    /// `A = P diag(λ) P`, `P = I − 2uuᵀ`, the matrix of the generator's quadratics, formed densely (n ≤ 5).
    fn quadratic_matrix(n: usize, cond: f64) -> Vec<Vec<f64>> {
        let v: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let nv = dot(&v, &v).sqrt();
        let u: Vec<f64> = v.iter().map(|t| t / nv).collect();
        let lam: Vec<f64> = (0..n)
            .map(|i| cond.powf(i as f64 / (n - 1) as f64))
            .collect();
        let p = |i: usize, j: usize| if i == j { 1.0 } else { 0.0 } - 2.0 * u[i] * u[j];
        // contraction-kernel-ok: dense test oracle, n <= 5
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..n).map(|k| p(i, k) * lam[k] * p(k, j)).sum())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bfgs_with_fixed_identity_and_exact_line_searches_terminates_in_n_steps() {
        let n = 5;
        for cond in [1e2, 1e4] {
            let a = quadratic_matrix(n, cond);
            let xs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let mut x = vec![0.0; n];
            let grad = |x: &[f64]| {
                matvec(
                    &a,
                    &x.iter().zip(&xs).map(|(p, q)| p - q).collect::<Vec<_>>(),
                )
            };
            let mut g = grad(&x);
            let g0 = dot(&g, &g).sqrt();
            let mut mem: Memory<Vec<f64>> = Memory::new(n, &vec![0.0; n]);
            for _ in 0..n {
                // d = −H g with H₀ = I (H0::FixedIdentity), exact step α = −gᵀd / dᵀAd
                let d: Vec<f64> = apply(&mem, 1.0, &g).iter().map(|t| -t).collect();
                let alpha = -dot(&g, &d) / dot(&d, &matvec(&a, &d));
                let s: Vec<f64> = d.iter().map(|t| alpha * t).collect();
                x.iter_mut().zip(&s).for_each(|(p, q)| *p += q);
                let g_new = grad(&x);
                let y: Vec<f64> = g_new.iter().zip(&g).map(|(p, q)| p - q).collect();
                // with an exact line search g_newᵀs = 0, so yᵀs = −g_oldᵀs: far above the skip threshold
                assert_eq!(
                    mem.push(&s, &y, dot(&y, &s), -dot(&g, &s), dot(&y, &y)),
                    super::super::memory::Push::Stored
                );
                g = g_new;
            }
            let bound = 8.0 * n as f64 * cond * EPS * g0;
            assert!(
                dot(&g, &g).sqrt() <= bound,
                "κ = {cond}: |g_n| = {:e} > {bound:e}",
                dot(&g, &g).sqrt()
            );
            assert_eq!(mem.len(), n);
            // Hₙ = A⁻¹ after n conjugate steps: H A e_i = e_i
            for i in 0..n {
                let col: Vec<f64> = a.iter().map(|r| r[i]).collect();
                let got = apply(&mem, 1.0, &col);
                for (k, t) in got.iter().enumerate() {
                    let want = if k == i { 1.0 } else { 0.0 };
                    assert!(
                        (t - want).abs() <= 16.0 * n as f64 * cond * EPS,
                        "κ = {cond}: (H A)[{k}][{i}] = {t}"
                    );
                }
            }
        }
    }
}
