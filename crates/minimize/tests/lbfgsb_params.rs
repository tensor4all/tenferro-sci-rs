//! The core takes L-BFGS-B's own line-search numbers and matches SciPy on them.
//!
//! SciPy's L-BFGS-B (`scipy/optimize/__lbfgsb.c`, v1.16.3, `lnsrlb`, line 2578) calls `dcsrch`
//! with `ftol = 1e-3`, `gtol = 0.9`, `xtol = 0.1`, `stpmin = 0` (line 2651) and `stpmax = 1e10`
//! when the iterate is unconstrained (line 2585). These are not the numbers of
//! `scalar_search_wolfe1` (1e-4, 0.9, xtol 1e-14), and they are deliberately nowhere in `src/`:
//! `MoreThuenteConfig` validates DCSRCH's own conditions only (the `0 < c1 < c2 < 1` check and the
//! defaults live in the driver), so a caller such as a future L-BFGS-B passes its own numbers.
//!
//! The data (group `lbfgsb_params`, 24 runs: MT1-MT6 x alpha0 in {1e-3, 1e-1, 1e1, 1e3}) was
//! recorded from `DCSRCH` directly, not through `scalar_search_wolfe1`. Each run is replayed
//! (SciPy's recorded `(phi, phi')` fed to `tell`) and run live on the closed-form function:
//! steps within 1e-12 relative, evaluation counts and reasons equal. `xtol = 0.1` is operative:
//! MT2 and MT3 end with `XtolSatisfied`.

mod common;

use common::*;
use minimize::linesearch::{MoreThuente, MoreThuenteConfig, Reason, Step};

const STEP_TOL: f64 = 1e-12;

fn runs() -> Vec<Case> {
    let v: Vec<Case> = load_cases()
        .into_iter()
        .filter(|c| c.group == "lbfgsb_params")
        .collect();
    assert_eq!(v.len(), 24, "6 functions x 4 alpha0");
    v
}

#[test]
fn the_core_accepts_lbfgsb_numbers() {
    // ftol = 1e-3 is not SciPy's driver default (1e-4); xtol = 0.1 is far from 1e-14
    let cfg = MoreThuenteConfig::new(1e-3, 0.9, 0.1, 0.0, 1e10).expect("valid core config");
    assert_eq!(
        (
            cfg.ftol(),
            cfg.gtol(),
            cfg.xtol(),
            cfg.stpmin(),
            cfg.stpmax()
        ),
        (1e-3, 0.9, 0.1, 0.0, 1e10)
    );
    for c in runs() {
        assert_eq!(
            (c.ftol, c.gtol, c.xtol, c.stpmin, c.stpmax),
            (1e-3, 0.9, 0.1, 0.0, 1e10),
            "{}",
            c.id
        );
        assert_eq!(
            c.kind, "dcsrch",
            "{}: recorded through DCSRCH, not the driver",
            c.id
        );
        assert_eq!(c.config(), cfg, "{}", c.id);
    }
}

#[test]
fn replay_matches_scipy() {
    for case in runs() {
        let cfg = case.config();
        let (mut ls, mut step) = MoreThuente::start(&cfg, case.phi0, case.dphi0, case.alpha1)
            .unwrap_or_else(|e| panic!("{}: {e}", case.id));
        for (k, &(alpha, phi, dphi)) in case.trials.iter().enumerate() {
            let Step::Evaluate { alpha: asked } = step else {
                panic!(
                    "{}: finished after {k} trials, SciPy ran {}",
                    case.id, case.n_trials
                )
            };
            assert!(
                rel_close(asked, alpha, STEP_TOL),
                "{}: trial {k}: {asked:e} vs {alpha:e}",
                case.id
            );
            step = ls.tell(&cfg, phi, dphi);
        }
        let Step::Done(t) = step else {
            panic!("{}: still asking after {} trials", case.id, case.n_trials)
        };
        assert_eq!(
            reason_name(&t.reason),
            case.reason,
            "{}: {}",
            case.id,
            case.task
        );
        assert_eq!(t.evaluations, case.n_trials, "{}", case.id);
        if let Some(stp) = case.stp {
            assert!(
                rel_close(t.alpha, stp, STEP_TOL),
                "{}: {:e} vs {stp:e}",
                case.id,
                t.alpha
            );
        } else {
            assert!(!t.reason.is_converged(), "{}", case.id);
        }
    }
}

#[test]
fn live_matches_scipy_and_satisfies_wolfe() {
    let mut xtol_runs = 0;
    for case in runs() {
        let (name, p) = (case.func.as_str(), case.p.as_slice());
        let (t, trace) = run_named(&case.config(), name, p, case.alpha1);
        assert_eq!(trace.len(), case.n_trials, "{}: evaluations", case.id);
        assert_eq!(reason_name(&t.reason), case.reason, "{}", case.id);
        // every trial matches the recorded one (the same arithmetic as the generator's)
        for (k, (&(a, ..), &(ra, ..))) in trace.iter().zip(&case.trials).enumerate() {
            assert!(
                rel_close(a, ra, STEP_TOL),
                "{}: trial {k}: {a:e} vs {ra:e}",
                case.id
            );
        }
        if t.reason.is_converged() {
            let stp = case.stp.expect("SciPy converged");
            assert!(rel_close(t.alpha, stp, STEP_TOL), "{}", case.id);
            // (1.1) and (1.2) with L-BFGS-B's (mu, eta), exactly
            assert_strong_wolfe(
                case.phi0, case.dphi0, 1e-3, 0.9, t.alpha, t.phi, t.dphi, &case.id,
            );
        }
        if matches!(t.reason, Reason::XtolSatisfied) {
            xtol_runs += 1;
        }
    }
    assert!(
        xtol_runs >= 2,
        "xtol = 0.1 is operative on MT2 and MT3 ({xtol_runs} runs)"
    );
}
