//! BFGS and L-BFGS agree on the direction when they hold the same pairs and the same fixed `H₀ = I`.
//!
//! The design sketch's "with memory at least the number of iterations, both take the same path" is false for SciPy
//! (decision 4 on #40): SciPy's BFGS and L-BFGS-B differ in `H₀` (the identity vs `γ = sᵀy/yᵀy`), in the first trial
//! step (`min(1, 1.01·2Δf/φ')` vs `1/‖d‖` once, then 1) and in the line search (c₁ = 1e-4, xtol = 1e-14 vs 1e-3, 0.1), so
//! two full runs do not coincide and nothing here claims they do. What is true, and exact in exact arithmetic, is that
//! the L-BFGS two-loop recursion with `m ≥ k` stored pairs and a fixed `H₀ = I` is the product form of the BFGS update
//! applied `k` times to `I` (unrolling the update): the same `(s_j, y_j)` give the same `−H g`.
//!
//! Test: run L-BFGS with `H0::FixedIdentity` and `m` larger than the run; at every iteration `k ≥ 1` its first trial point
//! of the next line search is `x_k + d_k` (the initial step is 1 for `iter > 0`), so `d_k = trial − x_k` is the direction
//! its two-loop recursion computed. Build the dense `H_k` with [`minimize::bfgs::inverse_update`] from the pairs of that
//! same run (`s_j = x_{j+1} − x_j`, `y_j = g_{j+1} − g_j`) and compare `d_k` with `−H_k g_k` within the README's
//! tolerance (`bfgs_common::mismatch`, which prints the observed maxima and the bounds), the scale being the absolute
//! terms of the dense product plus the rounding of `trial − x`.
//!
//! Status in the red stage: L-BFGS is real, `inverse_update` is a stub that leaves `H` unchanged, so the comparison
//! fails until the BFGS update is implemented.

mod bfgs_common;

use bfgs_common::*;
use colmat::Mat;
use minimize::bfgs::inverse_update;
use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step, H0};

/// `(x_k, g_k, d_k)` for every iteration `k ≥ 1` of an L-BFGS run with unbounded memory, `H₀ = I`.
fn lbfgs_run(problem: &str, x0: &[f64], iters: usize) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let cfg =
        LbfgsConfig::new(1000, 0.0, 0.0, 10_000, 10_000, 20, H0::FixedIdentity).expect("valid");
    let obj = objective(problem);
    let mut state = LbfgsState::new(&cfg, &mut (), x0.to_vec());
    let mut out = Vec::new();
    let mut last_iter = usize::MAX;
    let mut evals = 0;
    while let Step::Evaluate { x } = state.ask() {
        let x = x.clone();
        evals += 1;
        assert!(evals < 10_000, "the run does not terminate");
        // the point asked right after an acceptance (iteration k ≥ 1) is x_k + d_k
        if state.iterations() >= 1 && state.iterations() != last_iter {
            last_iter = state.iterations();
            let d: Vec<f64> = x.iter().zip(state.x()).map(|(t, xk)| t - xk).collect();
            out.push((state.x().clone(), state.grad().clone(), d));
            assert_eq!(
                state.restarts(),
                0,
                "a restart empties the memory: the comparison is for pure runs"
            );
            assert_eq!(
                state.stored_pairs(),
                state.iterations(),
                "a skipped update changes the pair history"
            );
            if out.len() >= iters {
                break;
            }
        }
        let (f, g) = obj(&x);
        state.tell(&cfg, &mut (), f, &g);
    }
    out
}

fn check(problem: &str, x0: &[f64], iters: usize) {
    let n = x0.len();
    let run = lbfgs_run(problem, x0, iters);
    assert!(run.len() >= 3, "{problem}: only {} iterations", run.len());
    // L-BFGS's own iterates: x_0 = x0 is not recorded (k ≥ 1 only), so rebuild the pairs from consecutive records,
    // starting with the pair (x_1 − x_0, g_1 − g_0)
    let (f0, g0) = objective(problem)(x0);
    let _ = f0;
    let mut xs = vec![x0.to_vec()];
    let mut gs = vec![g0];
    for (x, g, _) in &run {
        xs.push(x.clone());
        gs.push(g.clone());
    }
    let mut h = Mat::<f64>::identity(n, n);
    for (idx, (x, g, d)) in run.iter().enumerate() {
        let k = idx + 1; // iteration number: k pairs are stored
        let s: Vec<f64> = xs[k].iter().zip(&xs[k - 1]).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gs[k].iter().zip(&gs[k - 1]).map(|(a, b)| a - b).collect();
        inverse_update(&mut h, &s, &y).unwrap_or_else(|e| panic!("{problem} pair {k}: {e}"));
        let hr: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| h[(i, j)]).collect())
            .collect();
        let want: Vec<f64> = hr.iter().map(|row| -dot(row, g)).collect();
        let scale: Vec<f64> = (0..n)
            .map(|i| {
                x[i].abs()
                    + d[i].abs()
                    + hr[i].iter().zip(g).map(|(a, b)| (a * b).abs()).sum::<f64>()
            })
            .collect();
        let m = mismatch(
            &format!("{problem} iteration {k}: L-BFGS d vs dense −H g"),
            d,
            &want,
            &scale,
            norm_inf(&want),
            n,
        );
        assert!(m.is_none(), "{}", m.unwrap_or_default());
    }
}

#[test]
fn lbfgs_direction_with_unbounded_memory_and_fixed_identity_equals_the_dense_bfgs_direction() {
    check("rosen2", &[-1.2, 1.0], 12);
    check("quad5", &[0.0; 5], 4);
    check("beale", &[1.0, 1.0], 8);
}
