//! Properties of BFGS that do not depend on SciPy: the update itself (secant equation, symmetry,
//! positive definiteness, n-step termination with exact line searches) and the iteration (first
//! step, descent, the strong Wolfe conditions and the curvature bound they give).
//!
//! Written from the mathematics: Nocedal & Wright, *Numerical Optimization*, the BFGS inverse update
//! `H' = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ`, `ρ = 1/(yᵀs)`; the strong Wolfe conditions
//! `f(x + αp) ≤ f(x) + c₁ α gᵀp`, `|g(x + αp)ᵀp| ≤ c₂ |gᵀp|`. The two-line proofs are in the test
//! comments. Each test says whether its claim is **exact** (an inequality or equality of the
//! mathematics, evaluated without slack) or **tol** (a rounding bound, stated and derived).
//!
//! | Property | Test | Label |
//! |---|---|---|
//! | secant equation `H'y = s` | `update_satisfies_the_secant_equation` | tol: 4 n ε · scale, scale = (\|H\| + \|c\|\|s\|\|sᵀ\| + ρ(\|u\|\|sᵀ\| + \|s\|\|uᵀ\|))\|y\| + \|s\| |
//! | `H'` symmetric | `update_keeps_symmetry` | tol: 4 ε · entry scale (not bitwise: a gemm may fuse or reassociate) |
//! | `H'` positive definite while `yᵀs > 0` | `update_keeps_positive_definiteness` | exact (Cholesky succeeds) |
//! | n steps with exact line searches | `exact_line_search_terminates_in_n_steps` | tol: n ε κ̂ multiples, derived in the test |
//! | `H₀ = I`, first step `min(1, 1.01/‖g‖₂)` | `first_iteration_is_steepest_descent_with_scipys_first_step` | exact direction, tol on the point |
//! | descent `gᵀp < 0` | `every_iteration_descends_and_satisfies_strong_wolfe` | exact |
//! | strong Wolfe at every accepted step | same | tol: 8 n ε · Σ\|terms\| |
//! | `yᵀs ≥ (1 − c₂)(−gᵀs) > 0` | same | tol, same |
//! | `H` after an iteration is the update of the previous `H` | `state_applies_the_update_it_exposes` | tol |
//!
//! Not claimed: scale invariance `f ↦ 2ᵏ f` for BFGS with `H₀ = I` (it does not hold: the first step
//! and the trial steps depend on the scale of `f`; the `γ`-scaled L-BFGS is invariant, BFGS is not);
//! no allocation per iteration (colmat's gemm may allocate scratch; to be measured with the
//! implementation).
//!
//! Status in the red stage: every test here fails (the stub's `inverse_update` leaves `H` unchanged, which
//! the secant, symmetry and positive-definiteness tests each detect by also requiring that `H` moved).

mod bfgs_common;

use bfgs_common::*;
use colmat::Mat;
use minimize::bfgs::{inverse_update, BfgsConfig, BfgsState, Step};

// ---- deterministic test data (no RNG) ---------------------------------------------------------

/// A symmetric, strictly diagonally dominant (hence positive definite) matrix.
fn spd(n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            n as f64 + 1.0
        } else {
            1.0 / (1.0 + i.abs_diff(j) as f64)
        }
    })
}

/// A pair `(s, y)` with `yᵀs > 0`, deterministic in `t`: `y = D s + 0.3 Σs + small perturbation`.
fn pair(t: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let s: Vec<f64> = (0..n)
        .map(|i| ((t * 31 + i * 17 + 3) % 23) as f64 / 11.0 - 1.0)
        .collect();
    let sum: f64 = s.iter().sum();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            (i + 1) as f64 * s[i] + 0.3 * sum + 0.05 * (((t * 7 + i * 5) % 9) as f64 - 4.0) / 4.0
        })
        .collect();
    (s, y)
}

fn rows(h: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..h.nrows())
        .map(|i| (0..h.ncols()).map(|j| h[(i, j)]).collect())
        .collect()
}

/// The absolute-value scale of one update, entrywise: `|H| + |c||s||sᵀ| + ρ(|u||sᵀ| + |s||uᵀ|)`,
/// `u = H y`, `c = ρ + ρ² yᵀu`: the terms the symmetric form adds. (SciPy's form has a smaller one;
/// the bound is for either.)
fn update_scale(h: &[Vec<f64>], s: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    let n = s.len();
    let rho = 1.0 / dot(s, y);
    let u: Vec<f64> = (0..n).map(|i| dot(&h[i], y)).collect();
    let c = rho + rho * rho * dot(y, &u);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    h[i][j].abs()
                        + c.abs() * (s[i] * s[j]).abs()
                        + rho.abs() * ((u[i] * s[j]).abs() + (s[i] * u[j]).abs())
                })
                .collect()
        })
        .collect()
}

// ---- the update -------------------------------------------------------------------------------

/// Secant equation (exact in exact arithmetic): `H' y = (I − ρ s yᵀ) H (y − ρ y (sᵀy)) + ρ s (sᵀy) = s`,
/// since `y − ρ y (sᵀy) = 0` kills the first term and `ρ sᵀy = 1`. In floating point the error of
/// `(H'y)_i` is at most the matrix–vector bound `γₙ Σⱼ |H'ᵢⱼ||yⱼ|` plus the rounding of the entries of `H'`
/// (each a sum of the terms of `update_scale`, at most 3 additions and 4 products): `4 n ε` times the
/// scale below covers both, measured ≤ 0.14 of the bound on these pairs for n = 5.
#[test]
fn update_satisfies_the_secant_equation() {
    for n in [2usize, 5, 8] {
        for t in 0..20 {
            let (s, y) = pair(t, n);
            assert!(
                dot(&s, &y) > 0.05 * norm2(&s) * norm2(&y),
                "test data: yᵀs > 0"
            );
            let mut h = spd(n);
            let scale = update_scale(&rows(&h), &s, &y);
            inverse_update(&mut h, &s, &y).expect("yᵀs > 0");
            let hn = rows(&h);
            for i in 0..n {
                let hy = dot(&hn[i], &y);
                let sc: f64 = (0..n).map(|j| scale[i][j] * y[j].abs()).sum::<f64>() + s[i].abs();
                assert!(
                    (hy - s[i]).abs() <= 4.0 * n as f64 * EPS * sc,
                    "n = {n}, pair {t}, row {i}: (H'y)ᵢ = {hy:e}, sᵢ = {:e}",
                    s[i]
                );
            }
        }
    }
}

/// `H'` is symmetric if `H` is: `H + c s sᵀ − ρ(u sᵀ + s uᵀ)` is symmetric term by term. Computed with a
/// matrix product the entries `(i, j)` and `(j, i)` may be rounded differently (FMA, reassociation), so the
/// claim is the entrywise bound `|H'ᵢⱼ − H'ⱼᵢ| ≤ 4 ε` · (the entry's scale), not bitwise equality; an
/// implementation that symmetrizes may be bitwise symmetric. The update must also have changed `H`.
#[test]
fn update_keeps_symmetry() {
    for n in [2usize, 5, 8] {
        for t in 0..20 {
            let (s, y) = pair(t, n);
            let mut h = spd(n);
            let before = rows(&h);
            let scale = update_scale(&before, &s, &y);
            inverse_update(&mut h, &s, &y).expect("yᵀs > 0");
            let hn = rows(&h);
            let mut moved = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    moved = moved.max((hn[i][j] - before[i][j]).abs());
                    assert!(
                        (hn[i][j] - hn[j][i]).abs() <= 4.0 * EPS * scale[i][j],
                        "n = {n}, pair {t}: H'[{i}][{j}] = {:e} vs H'[{j}][{i}] = {:e}",
                        hn[i][j],
                        hn[j][i]
                    );
                }
            }
            assert!(
                moved > 1e-3,
                "the update left H unchanged (max change {moved:e})"
            );
        }
    }
}

/// Positive definiteness (own proof): for `z ≠ 0` put `w = z − ρ y (sᵀz)`. Then `zᵀH'z = wᵀHw + ρ(sᵀz)²`,
/// which is `≥ 0`, and `= 0` only if `w = 0` and `sᵀz = 0`; but `w = 0` gives `z = ρ y (sᵀz) = 0`. So `H' ≻ 0` if
/// `H ≻ 0` and `ρ > 0`. Exact: Cholesky of the computed `H'` must succeed, for a chain of 20 updates.
#[test]
fn update_keeps_positive_definiteness() {
    let n = 5;
    let mut h = spd(n);
    let before = rows(&h);
    for t in 0..20 {
        let (s, y) = pair(t, n);
        inverse_update(&mut h, &s, &y).expect("yᵀs > 0");
        let flat: Vec<f64> = rows(&h).into_iter().flatten().collect();
        assert!(
            is_positive_definite(&flat, n),
            "H lost positive definiteness after update {t}"
        );
    }
    let flat: Vec<f64> = rows(&h).into_iter().flatten().collect();
    let first: Vec<f64> = before.into_iter().flatten().collect();
    assert_ne!(flat, first, "twenty updates left H unchanged");
}

/// BFGS with exact line searches on `f = ½xᵀAx − bᵀx` (A symmetric positive definite) finishes in `n` steps:
/// the directions are `A`-conjugate (a classical property of BFGS with exact line searches, proved by induction
/// over the iterations), so `gₙ = 0` and `Hₙ = A⁻¹`.
/// The step `α = −gᵀp / pᵀAp` is exact; the iteration is written here, the update is `inverse_update`.
///
/// Tolerances: in floating point the errors grow with the condition number `κ̂ = ‖A‖_F / μ ≥ κ(A)`
/// (`μ` = λmin, `problems.csv`). Measured on this problem with a plain implementation of the update (both
/// the symmetric form and SciPy's): `‖gₙ‖∞ ≈ 0.08 ε κ ‖g₀‖∞`, `‖HₙA − I‖ ≈ 5–8 ε κ²`, conjugacy
/// `|pᵢᵀApⱼ| / (‖pᵢ‖‖Apⱼ‖) ≈ 14–18 ε κ`. The bounds are `16 n ε κ̂ ‖g₀‖∞`, `16 n ε κ̂²` and `16 n ε κ̂`: 4×
/// or more above the measurements.
#[test]
fn exact_line_search_terminates_in_n_steps() {
    let p = load_problems().remove("quad5").expect("quad5");
    let (a, b) = (p.a.clone().expect("A"), p.b.clone().expect("b"));
    let (n, mu) = (p.n, p.mu.expect("mu"));
    let frob = a.iter().map(|t| t * t).sum::<f64>().sqrt();
    let kappa = frob / mu;
    let mut h = Mat::<f64>::identity(n, n);
    let mut x = vec![0.0; n];
    let mut g: Vec<f64> = b.iter().map(|t| -t).collect(); // A·0 − b
    let g0 = norm_inf(&g);
    let mut dirs: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n {
        let hr = rows(&h);
        let pdir: Vec<f64> = (0..n).map(|i| -dot(&hr[i], &g)).collect();
        let ap = matvec(&a, n, &pdir);
        let alpha = -dot(&g, &pdir) / dot(&pdir, &ap);
        let s: Vec<f64> = pdir.iter().map(|t| alpha * t).collect();
        for i in 0..n {
            x[i] += s[i];
        }
        let g_new: Vec<f64> = matvec(&a, n, &x)
            .iter()
            .zip(&b)
            .map(|(u, v)| u - v)
            .collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(u, v)| u - v).collect();
        dirs.push(pdir);
        g = g_new;
        if norm_inf(&g) == 0.0 {
            break;
        }
        inverse_update(&mut h, &s, &y).expect("exact line search gives yᵀs > 0");
    }
    let nf = n as f64;
    assert!(
        norm_inf(&g) <= 16.0 * nf * EPS * kappa * g0,
        "‖gₙ‖∞ = {:e}, bound {:e}",
        norm_inf(&g),
        16.0 * nf * EPS * kappa * g0
    );
    // Hₙ = A⁻¹: ‖Hₙ A − I‖ small. Hₙ contains all n updates; the n-th pair has yᵀs = sᵀAs > 0 although gₙ ≈ 0.
    let hr = rows(&h);
    let a_rows: Vec<Vec<f64>> = (0..n).map(|i| a[i * n..(i + 1) * n].to_vec()).collect();
    let mut worst = 0.0_f64;
    for (i, hi) in hr.iter().enumerate() {
        for j in 0..n {
            let hij: f64 = hi.iter().zip(&a_rows).map(|(h, row)| h * row[j]).sum();
            worst = worst.max((hij - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    assert!(
        worst <= 16.0 * nf * EPS * kappa * kappa,
        "‖HA − I‖ = {worst:e}"
    );
    // conjugacy of the directions
    for i in 0..n {
        for j in 0..i {
            let apj = matvec(&a, n, &dirs[j]);
            let c = dot(&dirs[i], &apj).abs() / (norm2(&dirs[i]) * norm2(&apj));
            assert!(c <= 16.0 * nf * EPS * kappa, "p{i}ᵀA p{j} relative {c:e}");
        }
    }
}

// ---- the iteration ----------------------------------------------------------------------------

/// `H₀ = I` is never rescaled and the first trial step is SciPy's `min(1, 1.01·2(f₀ − f_old)/g₀ᵀp₀)` with
/// `f_old = f₀ + ‖g₀‖₂/2`, which is `min(1, 1.01/‖g₀‖₂)` for `p₀ = −g₀`. So after the first `tell`: `H = I`,
/// `p = −g` exactly, and the next point asked is `x₀ − α₁ g₀`.
#[test]
fn first_iteration_is_steepest_descent_with_scipys_first_step() {
    for (id, x0) in [
        ("rosen2", vec![-1.2, 1.0]),
        ("quad5", vec![0.0; 5]),
        ("beale", vec![1.0, 1.0]),
    ] {
        let obj = objective(id);
        let n = x0.len();
        let cfg = BfgsConfig::scipy_default(n);
        let mut state = BfgsState::new(&cfg, &mut (), x0.clone());
        // evaluation 0 at x0
        let Step::Evaluate { x } = state.ask() else {
            panic!("{id}: the state did not ask for x0");
        };
        assert_eq!(x, &x0, "{id}: the first request is x0");
        let (f0, g0) = obj(&x0);
        state.tell(&cfg, &mut (), f0, &g0);
        // after the first tell: H₀ = I exactly and p₀ = −g₀ exactly
        let mut identity = vec![0.0; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        assert_eq!(matrix_of(&state), identity, "{id}: H₀ = I");
        let neg_g: Vec<f64> = g0.iter().map(|v| -v).collect();
        assert_eq!(state.direction(), &neg_g, "{id}: p₀ = −g₀ exactly");
        // the first trial point is x₀ − α₁ g₀
        let Step::Evaluate { x: trial } = state.ask() else {
            panic!("{id}: the state did not ask for a first trial step");
        };
        let alpha = 1.0_f64.min(1.01 / norm2(&g0));
        let want: Vec<f64> = x0.iter().zip(&g0).map(|(x, g)| x - alpha * g).collect();
        let scale: Vec<f64> = x0
            .iter()
            .zip(&g0)
            .map(|(x, g)| x.abs() + alpha * g.abs())
            .collect();
        let m = mismatch(&format!("{id}: first trial"), trial, &want, &scale, 1.0, n);
        assert!(m.is_none(), "{}", m.unwrap_or_default());
    }
}

/// Per accepted iteration, on five problems:
/// - descent (exact): `gₖᵀpₖ < 0` before the line search (`H ≻ 0`: `gᵀp = −gᵀHg < 0`),
/// - the strong Wolfe conditions at the accepted step `s = x₊ − x`, with `gₖᵀs = α gₖᵀp`:
///   `f₊ ≤ f + c₁ gᵀs` and `|g₊ᵀs| ≤ c₂ |gᵀs|`, each with the rounding slack `8 n ε Σ|terms|`,
/// - the curvature bound from the lower half of the second condition: `g₊ᵀs ≥ c₂ gᵀs`, hence
///   `yᵀs = (g₊ − g)ᵀs ≥ (c₂ − 1) gᵀs = (1 − c₂)|gᵀs| > 0` (the update is valid).
#[test]
fn every_iteration_descends_and_satisfies_strong_wolfe() {
    for id in ["rosen2_a", "rosen5", "himmelblau", "beale", "quad5"] {
        let run = run_by_id(id);
        let t = drive(
            &run.config(),
            &run.x0,
            &*objective(&run.problem),
            10 * run.maxiter,
        );
        let n = run.n;
        assert!(
            t.snaps.len() >= 3,
            "{id}: iterations happened ({} snapshots)",
            t.snaps.len()
        );
        let nf = n as f64;
        for w in t.snaps.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let gp = dot(&a.g, &a.p);
            assert!(
                gp < 0.0,
                "{id}: iteration {}: gᵀp = {gp:e} is not a descent direction",
                a.iterations
            );
            let s: Vec<f64> = b.x.iter().zip(&a.x).map(|(u, v)| u - v).collect();
            let gs = dot(&a.g, &s);
            let g_plus_s = dot(&b.g, &s);
            let abs_terms = |g: &[f64]| g.iter().zip(&s).map(|(u, v)| (u * v).abs()).sum::<f64>();
            let slack_a = 8.0 * nf * EPS * abs_terms(&a.g);
            let slack_b = 8.0 * nf * EPS * abs_terms(&b.g);
            // sufficient decrease
            let slack_f = 8.0 * EPS * (a.f.abs() + b.f.abs() + run.c1 * gs.abs()) + slack_a;
            assert!(
                b.f <= a.f + run.c1 * gs + slack_f,
                "{id}: iteration {}: f₊ = {:e} > f + c₁ gᵀs = {:e}",
                a.iterations,
                b.f,
                a.f + run.c1 * gs
            );
            // strong curvature
            assert!(
                g_plus_s.abs() <= run.c2 * gs.abs() + slack_a + slack_b,
                "{id}: iteration {}: |g₊ᵀs| = {:e} > c₂|gᵀs| = {:e}",
                a.iterations,
                g_plus_s.abs(),
                run.c2 * gs.abs()
            );
            // yᵀs ≥ (1 − c₂)(−gᵀs) > 0
            let y: Vec<f64> = b.g.iter().zip(&a.g).map(|(u, v)| u - v).collect();
            let ys = dot(&y, &s);
            assert!(
                ys >= (1.0 - run.c2) * (-gs) - slack_a - slack_b,
                "{id}: iteration {}: yᵀs = {ys:e} < (1 − c₂)(−gᵀs) = {:e}",
                a.iterations,
                (1.0 - run.c2) * (-gs)
            );
            assert!(
                ys > 0.0,
                "{id}: iteration {}: yᵀs = {ys:e} ≤ 0",
                a.iterations
            );
        }
    }
}

/// The state applies the update it exposes: after an accepted iteration that did not end the run,
/// `H` equals `inverse_update(H_before, s, y)` with `s = x₊ − x`, `y = g₊ − g`, within the bound of the
/// update (`4 n ε`·scale per entry, plus the rounding of `s = α p` against `x₊ − x`: `ε(|x| + |x₊|)` per
/// component of `s`, whose effect on `H'` is `|H'|` times that relative to `s`, covered by a factor 8).
#[test]
fn state_applies_the_update_it_exposes() {
    let run = run_by_id("rosen2_a");
    let t = drive(&run.config(), &run.x0, &*objective("rosen2"), 400);
    assert!(t.snaps.len() >= 5, "iterations happened");
    let n = run.n;
    let mut checked = 0;
    for w in t.snaps.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        if !b.running {
            continue; // the run ended at this iterate: SciPy breaks before the update
        }
        let s: Vec<f64> = b.x.iter().zip(&a.x).map(|(u, v)| u - v).collect();
        let y: Vec<f64> = b.g.iter().zip(&a.g).map(|(u, v)| u - v).collect();
        let mut h = Mat::from_fn(n, n, |i, j| a.h[i * n + j]);
        let scale = update_scale(&rows(&h), &s, &y);
        inverse_update(&mut h, &s, &y).expect("yᵀs > 0");
        let want = rows(&h);
        let s_round = (0..n)
            .map(|i| EPS * (a.x[i].abs() + b.x[i].abs()))
            .fold(0.0_f64, f64::max);
        let s_rel = s_round / norm_inf(&s).max(f64::MIN_POSITIVE);
        for i in 0..n {
            for j in 0..n {
                let tol = 8.0 * (4.0 * n as f64 * EPS + s_rel) * scale[i][j];
                assert!(
                    (b.h[i * n + j] - want[i][j]).abs() <= tol,
                    "iteration {}: H[{i}][{j}] = {:e}, update of the previous H gives {:e} (tol {tol:e})",
                    b.iterations,
                    b.h[i * n + j],
                    want[i][j]
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 3, "only {checked} updates were compared");
}
