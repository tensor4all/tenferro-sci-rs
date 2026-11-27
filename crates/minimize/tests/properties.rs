//! Properties of the Moré–Thuente line search that follow from the paper's theory, independent of
//! SciPy: Moré & Thuente, "Line Search Algorithms with Guaranteed Sufficient Decrease", ACM TOMS
//! 20(3):286-307, 1994. Equation, theorem and page numbers below are the paper's.
//!
//! Notation: mu = ftol, eta = gtol, psi(a) = phi(a) - phi(0) - mu a phi'(0) (p. 290), (1.1)
//! sufficient decrease `phi(a) <= phi(0) + mu a phi'(0)` and (1.2) `|phi'(a)| <= eta |phi'(0)|`.
//! Sufficient decrease is evaluated the way DCSRCH evaluates it, `phi0 + a * (mu * dphi0)`, so the
//! tests and the code round alike (`common::sufficient_decrease`); all strong-Wolfe, sign and
//! ordering claims are asserted exactly.
//!
//! Some properties are checked in `common::drive` after *every* tell of *every* run of the whole
//! suite (they cost nothing extra): P1 (every `Converged` result satisfies (1.1) and (1.2)), P6a
//! (every trial lies in [stpmin, stpmax]), P9 (the best point and the bracket invariant), P10 (the
//! stage switch), P11 (the 0.66 interval shrink), P12 (`best` on termination). This file adds the
//! pinned examples and the properties that need a specific run. Runs per test are stated in each
//! comment; the file performs about 100 line-search runs of at most ~100 evaluations each.
//!
//! Where the code deliberately differs from the paper's prose (case 3 extrapolation, the stage
//! switch `phi' >= 0`), it follows MINPACK, whose code produced the paper's tables.

mod common;

use common::*;
use minimize::linesearch::{MoreThuenteConfig, Reason, Stage};

type Fun = fn(f64) -> (f64, f64);

fn run_fn(
    setup: &MoreThuenteConfig,
    f: Fun,
    alpha0: f64,
) -> (minimize::linesearch::Termination, Vec<Trial>) {
    run_traced(setup, |a| f(a).0, |a| f(a).1, alpha0)
}

fn full_run(setup: &MoreThuenteConfig, f: Fun, alpha0: f64) -> Run {
    drive(
        setup,
        |a| f(a).0,
        |a| f(a).1,
        f(0.0).0,
        f(0.0).1,
        alpha0,
        1000,
    )
    .expect("starts")
}

const MT_SETUPS: [(&str, f64, f64); 6] = [
    ("mt1", 1e-3, 1e-1),
    ("mt2", 1e-1, 1e-1),
    ("mt3", 1e-1, 1e-1),
    ("mt4", 1e-3, 1e-3),
    ("mt5", 1e-3, 1e-3),
    ("mt6", 1e-3, 1e-3),
];
const A0S: [f64; 4] = [1e-3, 1e-1, 1e1, 1e3];

/// Example (3.1), p. 295: C^1, |phi'| >= sigma everywhere, so no point satisfies (1.2) if eta < sigma.
const SIGMA: f64 = 0.1;
fn ex31(a: f64) -> (f64, f64) {
    if a <= 1.0 {
        (0.5 * (1.0 - SIGMA) * a * a - a, (1.0 - SIGMA) * a - 1.0)
    } else {
        (0.5 * (SIGMA - 1.0) - SIGMA * a, -SIGMA)
    }
}

/// Example (3.3), p. 296: phi(a) = -3a/(a^2 + 2) - mu a with mu = 0.03.
const MU33: f64 = 0.03;
fn ex33(a: f64) -> (f64, f64) {
    (
        -3.0 * a / (a * a + 2.0) - MU33 * a,
        3.0 * (a * a - 2.0) / ((a * a + 2.0) * (a * a + 2.0)) - MU33,
    )
}

fn quad1(a: f64) -> (f64, f64) {
    ((a - 1.0) * (a - 1.0), 2.0 * (a - 1.0))
}

/// psi <= 0 and phi' < mu phi'(0): the "no bracket yet" trial of (2.3).
fn extrapolating(setup: &MoreThuenteConfig, phi0: f64, dphi0: f64, t: Trial) -> bool {
    sufficient_decrease(phi0, dphi0, setup.ftol(), t.0, t.1) && t.2 < setup.ftol() * dphi0
}

// ---- P1, P2, P12: the paper's six functions ---------------------------------------------------

#[test]
fn p2_finite_termination_on_the_test_set() {
    // Theorem 2.3 (p. 293-295): mu <= eta, phi bounded below, alpha_min = 0, alpha_max = 1e10 give
    // finite termination at a point satisfying (1.1) and (1.2). MT1-MT6 at the four alpha0 with the
    // table's (mu, eta): converged within 30 trials. 24 runs.
    // P1 (the termination test (1.1)/(1.2) holds for every Converged result) is asserted inside
    // `drive`; P12 (p. 290, invariant (2.1)): `best` satisfies (1.1) on termination, asserted there
    // too and explicitly here. MINPACK returns stp before dcstep updates stx on convergence, so
    // `best.alpha == alpha` is NOT claimed: the converged point need not be the best point yet.
    for (name, mu, eta) in MT_SETUPS {
        for a0 in A0S {
            let s = cfg(mu, eta);
            let (t, trace) = run_named(&s, name, &[], a0);
            assert!(
                t.reason.is_converged(),
                "{name} alpha0 = {a0:e}: {:?}",
                t.reason
            );
            assert!(
                trace.len() <= 30,
                "{name} alpha0 = {a0:e}: {} trials",
                trace.len()
            );
            let (phi0, dphi0) = eval(name, &[], 0.0);
            assert_strong_wolfe(phi0, dphi0, mu, eta, t.alpha, t.phi, t.dphi, name);
            assert!(
                sufficient_decrease(phi0, dphi0, mu, t.best.alpha, t.best.phi),
                "{name}: best (1.1)"
            );
        }
    }
}

// ---- Example (3.1): T(mu) and the existence of Wolfe points -----------------------------------

#[test]
fn example_3_1_converges_inside_t_of_mu() {
    // p. 295: for mu < 1/2, T(mu) = [(1-mu)/(1-sigma), (1-sigma)/(2(mu-sigma))]; with mu = 0.25 that is
    // [5/6, 3]. For eta in {0.25, 0.5} (mu <= eta, Theorem 2.3) the search converges inside T(mu)
    // from every alpha0. 8 runs.
    let mu = 0.25;
    let (lo, hi) = (
        (1.0 - mu) / (1.0 - SIGMA),
        (1.0 - SIGMA) / (2.0 * (mu - SIGMA)),
    );
    assert!((lo - 5.0 / 6.0).abs() < 1e-15 && (hi - 3.0).abs() < 1e-15);
    for eta in [0.25, 0.5] {
        for a0 in A0S {
            let (t, _) = run_fn(&cfg(mu, eta), ex31, a0);
            assert!(
                t.reason.is_converged(),
                "eta = {eta}, alpha0 = {a0:e}: {:?}",
                t.reason
            );
            assert!(
                t.alpha >= lo && t.alpha <= hi,
                "eta = {eta}, alpha0 = {a0:e}: {} not in [{lo}, {hi}]",
                t.alpha
            );
        }
    }
}

#[test]
fn p3_no_strong_wolfe_point_exists() {
    // Section 3 (p. 295): with eta = 0.05 < sigma, |phi'| >= sigma > eta everywhere, so (1.2) is
    // impossible. Every alpha0: terminates, never Converged. On the warnings that report the best
    // point (rounding errors, xtol, stpmax) the returned alpha satisfies (1.1): MINPACK has reset
    // stp to stx, which satisfies (1.1) by invariant (2.1). 4 runs.
    let s = cfg(0.25, 0.05);
    let (lo, hi) = (5.0 / 6.0, 3.0);
    for a0 in A0S {
        let (t, trace) = run_fn(&s, ex31, a0);
        assert!(!t.reason.is_converged(), "alpha0 = {a0:e}");
        let (phi0, dphi0) = (ex31(0.0).0, ex31(0.0).1);
        if matches!(
            t.reason,
            Reason::RoundingErrors | Reason::XtolSatisfied | Reason::StpMax
        ) {
            assert!(
                sufficient_decrease(phi0, dphi0, s.ftol(), t.alpha, ex31(t.alpha).0),
                "alpha0 = {a0:e}: returned alpha {:e} violates (1.1) on {:?}",
                t.alpha,
                t.reason
            );
        }
        // P9a on the best point, whatever the reason (also checked in `drive`)
        assert!(sufficient_decrease(
            phi0,
            dphi0,
            s.ftol(),
            t.best.alpha,
            t.best.phi
        ));
        if a0 == 1e-3 {
            // SciPy ends this run with its bare non-finite-step WARN (last trial 5.3467, psi > 0)
            // on a function that is finite everywhere: the cubic through two collinear points on
            // the linear piece has q = 0 (0/0). So the non-finite-step exit is NOT only about
            // non-finite phi. The last trial violates (1.1) but `best` satisfies it (P9a).
            assert!(matches!(t.reason, Reason::NonFiniteStep), "{:?}", t.reason);
            let &(a, f, g) = trace.last().unwrap();
            // Termination.alpha/phi/dphi are the last evaluated trial (finite), bitwise
            assert_eq!(
                (t.alpha.to_bits(), t.phi.to_bits(), t.dphi.to_bits()),
                (a.to_bits(), f.to_bits(), g.to_bits())
            );
            assert!(
                !sufficient_decrease(phi0, dphi0, s.ftol(), a, f),
                "last trial {a:e} must violate (1.1)"
            );
            assert!(sufficient_decrease(
                phi0,
                dphi0,
                s.ftol(),
                t.best.alpha,
                t.best.phi
            ));
            assert!(t.best.alpha > 0.0, "best moved off the start");
            // SciPy's best point here (the last decrease point, in T(mu)) is within [5/6, 3]
            assert!(
                t.best.alpha >= lo && t.best.alpha <= hi,
                "best {:e}",
                t.best.alpha
            );
        }
    }
}

// ---- Theorem 3.3: eta < mu ---------------------------------------------------------------------

#[test]
fn p4_eta_below_mu_converges_after_the_stage_switch() {
    // Theorem 3.3 (p. 296-297): if an iterate has psi(a_k) <= 0 and phi'(a_k) > 0 the search ends
    // Converged with (1.1) and (1.2), although eta < mu. Example (3.3), mu = 0.03. SciPy
    // converges at alpha ~ 1.47-1.50 in these cases. Each case states whether the premise holds
    // on its trace; alpha0 = 0.1 with eta = 0.01 converges before any iterate satisfies it (no
    // claim of the theorem; recorded). 4 runs.
    let cases = [
        (0.01, 1e-3, true),
        (0.01, 1e-1, false),
        (1e-3, 1e-3, true),
        (1e-3, 1e-1, true),
    ];
    for (eta, a0, premise_expected) in cases {
        let s = cfg(MU33, eta);
        let run = full_run(&s, ex33, a0);
        let (phi0, dphi0) = (ex33(0.0).0, ex33(0.0).1);
        let premise = run
            .trace
            .iter()
            .any(|&(a, f, g)| sufficient_decrease(phi0, dphi0, MU33, a, f) && g > 0.0);
        assert_eq!(
            premise, premise_expected,
            "eta = {eta}, alpha0 = {a0:e}: premise"
        );
        let Ended::Done(t) = run.ended else {
            panic!("no termination")
        };
        assert!(
            t.reason.is_converged(),
            "eta = {eta}, alpha0 = {a0:e}: {:?}",
            t.reason
        );
        let (phi_a, dphi_a) = ex33(t.alpha);
        assert_strong_wolfe(
            phi0,
            dphi0,
            MU33,
            eta,
            t.alpha,
            phi_a,
            dphi_a,
            "example (3.3)",
        );
        if premise {
            // P10 (p. 297-298): the stage switch happened before the converging tell
            let before = run.snaps[..run.snaps.len() - 1]
                .iter()
                .any(|s| s.stage == Stage::Function);
            assert!(
                before,
                "eta = {eta}, alpha0 = {a0:e}: switched to phi only at the last tell"
            );
        }
    }
}

#[test]
fn p4_documented_non_case() {
    // eta = 0.001 from alpha0 = 10 and 1e3: no iterate has psi <= 0 and phi' > 0, so the theorem
    // promises nothing (SciPy: its bare non-finite-step WARN). Assert only that the search
    // terminates and is not Converged, and that the premise indeed never holds. 2 runs.
    for a0 in [1e1, 1e3] {
        let s = cfg(MU33, 1e-3);
        let run = full_run(&s, ex33, a0);
        let (phi0, dphi0) = (ex33(0.0).0, ex33(0.0).1);
        assert!(!run
            .trace
            .iter()
            .any(|&(a, f, g)| sufficient_decrease(phi0, dphi0, MU33, a, f) && g > 0.0));
        match run.ended {
            Ended::Done(t) => assert!(!t.reason.is_converged(), "alpha0 = {a0:e}: {:?}", t.reason),
            Ended::MaxIter => panic!("did not terminate"),
        }
    }
}

// ---- P5: termination at a bound (Theorem 2.2) --------------------------------------------------

#[test]
fn p5_stp_max_on_a_monotone_function() {
    // Theorem 2.2 / equation (2.3) (p. 292): phi = -a is unbounded below, so every trial has
    // psi <= 0 and phi' < mu phi'(0), the trials increase strictly, and the search ends at
    // alpha_max = 50 with psi <= 0 and phi' <= mu phi'(0). SciPy's trials: 1, 5, 21, 50. 1 run.
    let s = cfg_with(1e-4, 0.9, 1e-14, 1e-8, 50.0);
    let (t, trace) = run_fn(&s, |a| (-a, -1.0), 1.0);
    assert!(matches!(t.reason, Reason::StpMax), "{:?}", t.reason);
    assert_eq!(t.alpha, 50.0);
    let (phi0, dphi0) = (0.0, -1.0);
    assert!(
        sufficient_decrease(phi0, dphi0, s.ftol(), t.alpha, t.phi) && t.dphi <= s.ftol() * dphi0
    );
    for w in trace.windows(2) {
        assert!(w[0].0 < w[1].0, "trials must increase strictly");
    }
    for &tr in &trace {
        assert!(
            extrapolating(&s, phi0, dphi0, tr),
            "(2.3) at trial {:?}",
            tr
        );
    }
    assert_eq!(
        trace.iter().map(|t| t.0).collect::<Vec<_>>(),
        vec![1.0, 5.0, 21.0, 50.0]
    );
}

#[test]
fn p5_stp_min_on_a_steeply_rising_function() {
    // Theorem 2.2 / equation (2.4) (p. 292): phi = -a + 1e6 a^2 rises steeply; every trial has
    // psi > 0 or phi' >= mu phi'(0), the trials decrease strictly, and the search ends at
    // alpha_min = 1e-3 where (1.1) fails or phi' >= mu phi'(0). 1 run.
    let s = cfg_with(1e-4, 0.9, 1e-14, 1e-3, 50.0);
    let f: Fun = |a| (-a + 1e6 * a * a, -1.0 + 2e6 * a);
    let (t, trace) = run_fn(&s, f, 1.0);
    assert!(matches!(t.reason, Reason::StpMin), "{:?}", t.reason);
    assert_eq!(t.alpha, 1e-3);
    let (phi0, dphi0) = (0.0, -1.0);
    assert!(
        !sufficient_decrease(phi0, dphi0, s.ftol(), t.alpha, t.phi) || t.dphi >= s.ftol() * dphi0
    );
    for w in trace.windows(2) {
        assert!(w[0].0 > w[1].0, "trials must decrease strictly");
    }
    for &(a, f, g) in &trace {
        assert!(
            !sufficient_decrease(phi0, dphi0, s.ftol(), a, f) || g >= s.ftol() * dphi0,
            "(2.4) at {a:e}"
        );
    }
}

// ---- P6: safeguards (section 2 and 4) ----------------------------------------------------------

/// Eq. (2.2), delta in [1.1, 4]: while no bracket has appeared (every trial so far had psi <= 0 and
/// phi' < mu phi'(0)), the next trial lies in [a + 1.1 (a - a_l), min(a + 4 (a - a_l), a_max)], with
/// a_l the previous best point (0 for the first step; each such trial becomes the new best).
fn check_extrapolation(setup: &MoreThuenteConfig, phi0: f64, dphi0: f64, trace: &[Trial]) -> usize {
    let mut al = 0.0;
    let mut checked = 0;
    for w in trace.windows(2) {
        let (aj, next) = (w[0].0, w[1].0);
        if !extrapolating(setup, phi0, dphi0, w[0]) {
            break;
        }
        let lo = aj + 1.1 * (aj - al);
        let hi = (aj + 4.0 * (aj - al)).min(setup.stpmax());
        assert!(
            next >= lo && next <= hi,
            "(2.2): {next:e} outside [{lo:e}, {hi:e}] after {aj:e} (a_l = {al:e})"
        );
        al = aj;
        checked += 1;
    }
    checked
}

#[test]
fn p6b_extrapolation_bounds() {
    // 2 runs: phi = -a (alpha_max = 50), and MT1 from alpha0 = 1e-3 with the paper's iterates.
    let s = cfg_with(1e-4, 0.9, 1e-14, 1e-8, 50.0);
    let (_, trace) = run_fn(&s, |a| (-a, -1.0), 1.0);
    assert_eq!(check_extrapolation(&s, 0.0, -1.0, &trace), 3);

    let s = cfg(1e-3, 1e-1);
    let (t, trace) = run_named(&s, "mt1", &[], 1e-3);
    let (phi0, dphi0) = eval("mt1", &[], 0.0);
    assert!(check_extrapolation(&s, phi0, dphi0, &trace) >= 4);
    // p. 304: delta = 4 gives 0.005, 0.021, 0.085, 0.341, 1.365 "until the
    // minimizer is bracketed or one of these iterates satisfies the termination conditions"
    assert_eq!(trace[0].0, 1e-3);
    for (got, printed) in trace[1..]
        .iter()
        .map(|t| t.0)
        .zip([0.005, 0.021, 0.085, 0.341, 1.365])
    {
        assert!(
            (got - printed).abs() <= 0.5e-3 + 1e-9,
            "{got} vs printed {printed}"
        );
    }
    assert_eq!(t.evaluations, 6);
}

#[test]
fn p6c_case_1_bound() {
    // p. 299 (Case 1, interval rule U1): if psi(a_t) > psi(a_l), with a_l the best point (least
    // psi so far, 0 at the start), the next trial satisfies |a+ - a_l| <= 7/12 |a_t - a_l| unless
    // it was clamped to alpha_min (bisection gives 1/2 and also passes). 3 runs.
    let cases: [(Fun, f64, f64, f64); 3] = [
        (|a| eval("mt1", &[], a), 1e-3, 1e-1, 1e3),
        (|a| eval("mt2", &[], a), 1e-1, 1e-1, 1e3),
        (quad1, 1e-4, 0.9, 10.0),
    ];
    for (f, mu, eta, a0) in cases {
        let s = cfg(mu, eta);
        let (_, trace) = run_fn(&s, f, a0);
        let (phi0, dphi0) = f(0.0);
        let gtest = mu * dphi0;
        let (mut best, mut best_psi) = (0.0_f64, 0.0_f64);
        let mut applied = 0;
        for w in trace.windows(2) {
            let (a, fa, _) = w[0];
            let psi = fa - phi0 - a * gtest;
            if psi > best_psi && w[1].0 != s.stpmin() {
                assert!(
                    (w[1].0 - best).abs() <= 7.0 / 12.0 * (a - best).abs(),
                    "Case 1 bound: a_t = {a:e}, a_l = {best:e}, next = {:e}",
                    w[1].0
                );
                applied += 1;
            }
            if psi < best_psi {
                best = a;
                best_psi = psi;
            }
        }
        assert!(
            applied >= 1,
            "the Case 1 situation never arose from alpha0 = {a0:e}"
        );
    }
}

// ---- P7: exact interpolation on a quadratic ----------------------------------------------------

#[test]
fn p7_quadratic_is_solved_by_the_second_trial() {
    // Section 4 (p. 298-300): on phi = (a - 1)^2 the cubic, the quadratic and the secant
    // interpolants are exact, so the second trial is the minimizer a = 1: from alpha0 = 3 (Case 1:
    // phi(3) > phi(0)) and from alpha0 = 1.5 (Case 2: lower phi, and phi'(1.5) > 0 > phi'(0) have
    // opposite signs). eta = 1e-3 so that the first trial does not already satisfy (1.2)
    // (with eta = 0.9 alpha0 = 1.5 is accepted at once). 2 runs. Also the property later BFGS
    // tests rely on: an exact line search on quadratics.
    for a0 in [3.0, 1.5] {
        let (t, trace) = run_fn(&cfg(1e-4, 1e-3), quad1, a0);
        assert!(t.reason.is_converged(), "alpha0 = {a0}");
        assert_eq!(t.evaluations, 2, "alpha0 = {a0}");
        assert!(
            (trace[1].0 - 1.0).abs() <= 4.0 * spacing(1.0),
            "alpha0 = {a0}: second trial {:e}",
            trace[1].0
        );
        assert!(t.dphi.abs() <= 1e-15, "alpha0 = {a0}: phi' = {:e}", t.dphi);
    }
}

// ---- P8: scale invariance, exact ---------------------------------------------------------------

#[test]
fn p8_scale_invariance_is_exact() {
    // (1.1) and (1.2) are invariant under phi -> c phi (c > 0) and under a -> s a, and every constant
    // of the algorithm (0.5, 0.66, 1.1, 4, the relative xtol) commutes with multiplication by a power
    // of two. So with c = 2^k and s = 2^k (k = +-10): (a) phi -> 2^k phi gives the bitwise identical
    // trial sequence and reason; (b) phi(a) -> phi(2^k a), with alpha0, alpha_min, alpha_max divided
    // by 2^k, gives trials exactly 2^-k times the originals, with the same reason. Exact `==`. MT1-MT6
    // at alpha0 = 0.1 and example (3.3): 7 functions x (1 + 2 x 2) = 35 runs.
    let mut funcs: Vec<(String, Fun, f64, f64)> = MT_SETUPS
        .iter()
        .map(|&(n, mu, eta)| {
            let f: Fun = match n {
                "mt1" => |a| eval("mt1", &[], a),
                "mt2" => |a| eval("mt2", &[], a),
                "mt3" => |a| eval("mt3", &[], a),
                "mt4" => |a| eval("mt4", &[], a),
                "mt5" => |a| eval("mt5", &[], a),
                _ => |a| eval("mt6", &[], a),
            };
            (n.to_string(), f, mu, eta)
        })
        .collect();
    funcs.push(("ex33".into(), ex33 as Fun, MU33, 0.01));
    let a0 = 0.1;
    for (name, f, mu, eta) in funcs {
        let s = cfg(mu, eta);
        let (t0, tr0) = run_fn(&s, f, a0);
        for k in [10_i32, -10] {
            let c = 2f64.powi(k);
            // (a) values scaled
            let (t1, tr1) = run_traced(&s, |a| c * f(a).0, |a| c * f(a).1, a0);
            let alphas0: Vec<f64> = tr0.iter().map(|t| t.0).collect();
            let alphas1: Vec<f64> = tr1.iter().map(|t| t.0).collect();
            assert_eq!(
                alphas0, alphas1,
                "{name}, k = {k}: phi -> 2^k phi changed the trials"
            );
            assert_eq!(
                reason_name(&t0.reason),
                reason_name(&t1.reason),
                "{name}, k = {k}"
            );
            // (b) argument scaled
            let sb = cfg_with(s.ftol(), s.gtol(), s.xtol(), s.stpmin(), s.stpmax() / c);
            let (t2, tr2) = run_traced(&sb, |a| f(c * a).0, |a| c * f(c * a).1, a0 / c);
            let scaled: Vec<f64> = tr0.iter().map(|t| t.0 / c).collect();
            let alphas2: Vec<f64> = tr2.iter().map(|t| t.0).collect();
            assert_eq!(
                scaled, alphas2,
                "{name}, k = {k}: a -> 2^k a changed the trials"
            );
            assert_eq!(
                reason_name(&t0.reason),
                reason_name(&t2.reason),
                "{name}, k = {k}"
            );
        }
    }
}

// ---- P9-P12 pinned on named runs ---------------------------------------------------------------

#[test]
fn p11_bracketing_terminates_quickly_on_the_hard_problems() {
    // Section 2 (p. 292-293): the interval shrinks by delta = 0.66 every two tells (checked after
    // every tell in `drive`, P11). Hence the number of bracketed tells is bounded by
    // 2 ceil(log(w_first / (xtol hi)) / log(1/0.66)) + 4, hi the upper end of the last bracket, xtol
    // the relative width of the xtol test. MT2 and MT3, the paper's hard problems (p. 305): 8 runs.
    for (name, mu, eta) in [("mt2", 1e-1, 1e-1), ("mt3", 1e-1, 1e-1)] {
        for a0 in A0S {
            let s = cfg(mu, eta);
            let f: Fun = if name == "mt2" {
                |a| eval("mt2", &[], a)
            } else {
                |a| eval("mt3", &[], a)
            };
            let run = full_run(&s, f, a0);
            let bracketed: Vec<(f64, f64)> = run.snaps.iter().filter_map(|s| s.bracket).collect();
            if let (Some(&(l0, h0)), Some(&(_, hl))) = (bracketed.first(), bracketed.last()) {
                let w_first = h0 - l0;
                let k = ((w_first / (s.xtol() * hl)).ln() / (1.0 / 0.66f64).ln())
                    .ceil()
                    .max(0.0) as usize;
                let bound = 2 * k + 4;
                assert!(
                    bracketed.len() <= bound,
                    "{name} alpha0 = {a0:e}: {} bracketed tells > {bound}",
                    bracketed.len()
                );
            }
        }
    }
}

#[test]
fn p9_best_never_worsens_and_states_start_clean() {
    // P9a-c are asserted after every tell in `drive`. Here the start state itself: right after
    // start, best() = (0, phi(0), phi'(0)), stage() = Auxiliary, bracket() = None (asserted in
    // `drive`), and on a run that converges at the first trial `best` is still the start point
    // (MINPACK returns before dcstep updates stx): MT1 from alpha0 = 10 (Table I: m = 1). 1 run.
    let s = cfg(1e-3, 1e-1);
    let run = full_run(&s, |a| eval("mt1", &[], a), 10.0);
    let Ended::Done(t) = run.ended else { panic!() };
    assert!(t.reason.is_converged());
    assert_eq!(t.evaluations, 1);
    assert_eq!(t.best.alpha, 0.0);
    assert_eq!(t.best.phi, run.phi0);
    assert_eq!(t.alpha, 10.0, "the converged point is the trial, not best");
}
