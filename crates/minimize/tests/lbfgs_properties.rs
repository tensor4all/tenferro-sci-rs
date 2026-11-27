//! Properties of the L-BFGS iteration that do not depend on SciPy.
//!
//! Validates, through the public ask/tell API: power-of-two scale invariance of `f` (bitwise: with `H₀ = γ I`
//! the scale of `γ` cancels in `H g`, and Moré–Thuente is invariant under `φ ↦ 2ᵏ φ`); translation covariance;
//! the effect of `H0::FixedIdentity` (it changes the second iterate, `ScaledGamma` does not use `I`); the contract of
//! `evaluations()` (one per `tell`) and of `ask()` after termination.
//!
//! The properties of the internal pieces (secant equation, two-loop = dense matrix, ring buffer, skip rule, γ,
//! n-step termination with exact line searches) are unit tests in `src/lbfgs/{memory,twoloop}.rs`; the descent
//! property, the restart rule and the pair count are checked on every recorded run in `lbfgs_replay.rs`.
//!
//! Tolerance policy: scale invariance exact (bitwise); translation covariance "tol", measured on a float64 mirror of the algorithm at 2.7e-15 over 8
//! iterations with `|x|∞ = 5`; the test allows `64 ε (|x|∞ + |c|)`, i.e. 26× the observation (measured, not derived).

mod lbfgs_common;

use lbfgs_common::*;
use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step, Termination, H0};

fn fixed_iterations(m: usize, iters: usize, h0: H0) -> LbfgsConfig {
    // gtol = ftol = 0: neither tolerance test is scale invariant ((max(|f|, 1)) and an absolute ‖g‖∞), so the run stops
    // on maxiter only
    LbfgsConfig::new(m, 0.0, 0.0, iters, 15000, 20, h0).expect("valid")
}

#[test]
fn scaling_f_by_a_power_of_two_leaves_the_iterates_bitwise_identical() {
    let x0 = vec![-1.2, 1.0, -1.2, 1.0, -1.2];
    let cfg = fixed_iterations(10, 15, H0::ScaledGamma);
    let reference = {
        let (state, t, requested) = drive(&cfg, &x0, |x, g| {
            let mut gv = g.to_vec();
            let f = rosen(&mut (), &x.to_vec(), &mut gv);
            g.copy_from_slice(&gv);
            f
        });
        (state.x().clone(), t, requested)
    };
    assert_eq!(reference.1, Termination::MaxIter);
    assert_eq!(
        reference.2.len(),
        23,
        "the 15-iteration run evaluates 23 points (SciPy-equal count)"
    );
    for k in [10, -10, 40] {
        let s = 2.0_f64.powi(k);
        let (state, t, requested) = drive(&cfg, &x0, |x, g| {
            let mut gv = g.to_vec();
            let f = rosen(&mut (), &x.to_vec(), &mut gv);
            g.iter_mut().zip(&gv).for_each(|(a, b)| *a = b * s);
            f * s
        });
        assert_eq!(t, Termination::MaxIter, "2^{k}");
        assert_eq!(requested, reference.2, "2^{k}: the requested points differ");
        assert_eq!(state.x(), &reference.0, "2^{k}");
    }
}

#[test]
fn translating_the_problem_translates_the_iterates() {
    let q = Quadratic::new(5, 1e2);
    let cfg = fixed_iterations(10, 8, H0::ScaledGamma);
    let run = |shift: f64| {
        let x0 = vec![shift; 5];
        let (state, t, _) = drive(&cfg, &x0, |x, g| q.eval(x, shift, g));
        assert_eq!(t, Termination::MaxIter);
        state.x().clone()
    };
    let base = run(0.0);
    let xmax = base.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
    for c in [0.5, 3.0] {
        let moved = run(c);
        for (a, b) in moved.iter().zip(&base) {
            let tol = 64.0 * EPS * (xmax + c);
            assert!(
                (a - c - b).abs() <= tol,
                "c = {c}: {} vs {b}, tol {tol:e}",
                a - c
            );
        }
    }
}

#[test]
fn fixed_identity_and_scaled_gamma_differ_after_the_first_pair() {
    // A = Householder quadratic, κ = 100: the first step is identical (empty memory: d = −g in both), the second
    // direction is not, because γ ≠ 1.
    let q = Quadratic::new(5, 1e2);
    let go = |h0| {
        let cfg = fixed_iterations(10, 2, h0);
        drive(&cfg, &[0.0; 5], |x, g| q.eval(x, 0.0, g)).2
    };
    let (a, b) = (go(H0::ScaledGamma), go(H0::FixedIdentity));
    assert!(a.len() >= 3 && b.len() >= 3);
    assert_eq!(a[..2], b[..2], "the first search does not depend on H0");
    assert_ne!(a[2], b[2], "the second search must use H0");
}

#[test]
fn evaluations_count_the_tell_calls() {
    let q = Quadratic::new(5, 1e2);
    let cfg = LbfgsConfig::scipy_default().expect("valid");
    let mut state = LbfgsState::new(&cfg, &mut (), vec![0.0; 5]);
    let mut g = vec![0.0; 5];
    let mut told = 0;
    while let Step::Evaluate { x } = state.ask() {
        let f = q.eval(x, 0.0, &mut g);
        state.tell(&cfg, &mut (), f, &g);
        told += 1;
        assert_eq!(state.evaluations(), told);
    }
    assert!(told > 5);
    // after termination, ask keeps answering the same, and tell changes nothing
    let (x, it, ev) = (state.x().clone(), state.iterations(), state.evaluations());
    let first = match state.ask() {
        Step::Converged(t) => t,
        Step::Evaluate { .. } => panic!("still asking"),
    };
    state.tell(&cfg, &mut (), -1.0, &g);
    assert!(matches!(state.ask(), Step::Converged(t) if t == first));
    assert_eq!(
        (state.x(), state.iterations(), state.evaluations()),
        (&x, it, ev)
    );
}
