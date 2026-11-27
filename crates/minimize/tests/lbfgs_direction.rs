//! The search direction `d = −H g` of SciPy's L-BFGS-B against the exact one, from the recorded data.
//!
//! Validates (data only, so these tests pass before the implementation exists): (1) SciPy's direction (the compact
//! representation, `subsm`) equals the exact direction recomputed in mpmath (60 digits) from the pairs SciPy stored,
//! within `C_DIR · ε · d_scale` (the README derives the bound and the observed maximum); (2) the facts of
//! `__lbfgsb.c` the implementation must reproduce: `H₀ = γ I` with `γ = sᵀy/yᵀy` of the newest pair, `d = −g` with an
//! empty memory, the first step `1/‖d‖` at iteration 0 and 1 afterwards.
//!
//! The implementation's two-loop recursion is checked against the same exact directions in the unit tests of
//! `src/lbfgs/twoloop.rs` (the recursion is `pub(crate)`).
//!
//! Source: SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34) `__lbfgsb.c` `mainlb` (LINE555),
//! `lnsrlb` (stp), `matupd` (theta); the generator `scripts/gen_minimize_lbfgs_reference.py`.

mod lbfgs_common;

use lbfgs_common::*;

#[test]
fn scipys_direction_is_within_the_derived_bound_of_the_exact_direction() {
    let mut worst = 0.0_f64;
    for (run, ss) in load_searches() {
        for s in &ss {
            for (c, ((d, e), sc)) in s.d.iter().zip(&s.d_exact).zip(&s.d_scale).enumerate() {
                let err = (d - e).abs();
                if *sc == 0.0 {
                    assert_eq!(err, 0.0, "{run} search {} component {c}", s.j);
                    continue;
                }
                let ratio = err / (EPS * sc);
                worst = worst.max(ratio);
                assert!(
                    ratio <= C_DIR,
                    "{run} search {} component {c}: scipy {d:e}, exact {e:e}, ratio {ratio:.1} > {C_DIR}",
                    s.j
                );
            }
        }
    }
    eprintln!("scipy direction vs exact: largest |d - d_exact| / (eps d_scale) = {worst:.1} (bound {C_DIR})");
}

#[test]
fn with_an_empty_memory_the_direction_is_exactly_minus_the_gradient() {
    let mut n = 0;
    for (run, ss) in load_searches() {
        for s in ss.iter().filter(|s| s.col == 0) {
            let minus_g: Vec<f64> = s.g_base.iter().map(|g| -g).collect();
            assert_eq!(s.d, minus_g, "{run} search {}", s.j);
            assert_eq!(s.d_exact, minus_g, "{run} search {}", s.j);
            n += 1;
        }
    }
    assert!(n >= 40, "only {n} searches with an empty memory");
}

#[test]
fn h0_is_gamma_times_identity_with_gamma_s_dot_y_over_y_dot_y() {
    // θ = yᵀy / dr (matupd) is the reciprocal of γ; SciPy forms dr = (gd − gdold) α, so it differs from the exact
    // sᵀy / yᵀy by the rounding of that difference: bounded by 64 ε in the committed runs (README).
    let mut worst = 0.0_f64;
    let mut n = 0;
    for (run, ss) in load_searches() {
        for s in ss.iter().filter(|s| s.col > 0) {
            let rel = (s.gamma_scipy / s.gamma_exact - 1.0).abs();
            worst = worst.max(rel);
            assert!(
                rel <= 64.0 * EPS,
                "{run} search {}: γ differs by {rel:e}",
                s.j
            );
            n += 1;
        }
    }
    assert!(n >= 500);
    eprintln!(
        "gamma: largest relative difference scipy vs exact = {worst:e} = {:.1} eps",
        worst / EPS
    );
}

#[test]
fn the_first_trial_step_is_one_over_norm_d_at_iteration_zero_and_one_after() {
    let mut zero = 0;
    for (run, ss) in load_searches() {
        for s in &ss {
            if s.iter == 0 {
                let norm = s.d.iter().map(|t| t * t).sum::<f64>().sqrt();
                let want = (1.0 / norm).min(1e10);
                assert!(
                    (s.stp0 - want).abs() <= 4.0 * EPS * want,
                    "{run}: stp0 {} but 1/‖d‖ = {want}",
                    s.stp0
                );
                zero += 1;
            } else {
                assert_eq!(s.stp0, 1.0, "{run} search {} (iter {})", s.j, s.iter);
            }
        }
    }
    assert!(zero >= 40);
}
