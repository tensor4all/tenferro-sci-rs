//! Replay of the recorded SciPy runs through `minimize::lbfgs::LbfgsState`: the primary parity claim.
//!
//! Validates: for every run of `tests/data/lbfgs/runs.csv` (SciPy 1.16.3 `minimize(method="L-BFGS-B",
//! bounds=None)`), feeding SciPy's `(f, g)` of the i-th evaluation back, the state asks for the same point as
//! SciPy's i-th request within the derived tolerance; asks exactly as often; ends with the same termination, the
//! same `nit`, the same `nfev`; has as many stored pairs as SciPy's `col` at every line search; takes the unit
//! steepest-descent step from an empty memory after a failed search; and starts every search with a descent
//! direction.
//!
//! Source: SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34) `__lbfgsb.c` (`mainlb`, `lnsrlb`) and
//! `_lbfgsb_py.py`. Tolerance policy: README.md of the data directory. Evaluation counts and terminations are exact;
//! a mismatch is a finding to explain, not a tolerance to widen.

mod lbfgs_common;

use lbfgs_common::*;

fn replays() -> Vec<Replay> {
    let runs = load_runs();
    let evals = load_evals();
    let searches = load_searches();
    runs.iter()
        .map(|r| {
            // a run that stops at the start has no line search and no row in searches.csv
            let ss = searches.get(&r.id).map_or(&[][..], |v| v.as_slice());
            replay(r, &evals[&r.id], ss)
        })
        .collect()
}

/// Every line search of the data must have been reached: a replay that stops early checks nothing.
fn assert_all_searches_seen(reps: &[Replay]) {
    let expected: usize = load_searches().values().map(Vec::len).sum();
    let seen: usize = reps.iter().map(|r| r.searches_seen).sum();
    assert_eq!(
        seen, expected,
        "the replay reached {seen} of {expected} recorded line searches"
    );
}

fn assert_none(what: &str, mut bad: Vec<String>) {
    if !bad.is_empty() {
        let n = bad.len();
        bad.truncate(12);
        panic!("{what}: {n} problems, first ones:\n{}", bad.join("\n"));
    }
}

#[test]
fn replay_reproduces_every_requested_point_count_and_termination() {
    let reps = replays();
    let worst = reps.iter().map(|r| r.max_ratio).fold(0.0, f64::max);
    eprintln!("replay: largest (|dx| - slack) / (eps x_scale) over all runs = {worst:.2} (tolerance {C_X})");
    assert_none(
        "path",
        reps.iter()
            .flat_map(|r| r.path_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn replay_ends_with_scipys_final_point_value_and_gradient() {
    assert_none(
        "final state",
        replays()
            .iter()
            .flat_map(|r| r.final_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn stored_pairs_follow_scipys_col_at_every_search() {
    // the ring buffer (m = 1, 3, 10, 50), the reset after a failed search and the skip rule, observed through `col`
    let reps = replays();
    assert_all_searches_seen(&reps);
    assert_none(
        "stored pairs",
        reps.iter()
            .flat_map(|r| r.col_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn every_search_starts_with_a_descent_direction() {
    let reps = replays();
    assert_all_searches_seen(&reps);
    assert_none(
        "descent",
        reps.iter()
            .flat_map(|r| r.descent_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn after_a_failed_search_the_retry_is_steepest_descent_with_unit_step() {
    // SciPy (lnsrlb): stp = 1/‖d‖ only when iter == 0, else 1. After a reset at iter > 0 the first trial is the full −g.
    let reps = replays();
    assert_all_searches_seen(&reps);
    let restarts: usize = reps.iter().map(|r| r.restarts).sum();
    assert!(
        restarts >= 3,
        "the data must contain restarts, found {restarts}"
    );
    assert_none(
        "restart",
        reps.iter()
            .flat_map(|r| r.restart_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn the_data_covers_every_termination_and_the_counting_quirks() {
    // Data only: this passes against the stubs. It fails when the generator stops recording a case the tests name.
    let runs = load_runs();
    let terms: std::collections::HashSet<&str> =
        runs.iter().map(|r| r.termination.as_str()).collect();
    for t in [
        "grad_tol",
        "f_tol",
        "max_iter",
        "max_eval",
        "line_search_failed",
    ] {
        assert!(terms.contains(t), "no recorded run ends with {t}");
    }
    let by = |id: &str| {
        runs.iter()
            .find(|r| r.id == id)
            .unwrap_or_else(|| panic!("run {id}"))
    };
    // the first search fails after maxls evaluations: nfev = 1 + maxls, not 2 + maxls (the (maxls+1)-th trial is never evaluated)
    assert_eq!(by("maxls1_rosen2_first").nfev, 2);
    // maxfun is tested only after an accepted iteration: the run overshoots (nfev 3 > maxfun 1)
    assert_eq!(
        (by("maxfun1_rosen2").nit, by("maxfun1_rosen2").nfev),
        (1, 3)
    );
    // maxfun is a strict '>' : nfev = maxfun + 1 when a search ends exactly on the budget
    assert_eq!(by("maxfun5_rosen5").nfev, 6);
    // maxiter before the gradient and function tests
    assert_eq!(by("maxiter_eq_k_rosen5").termination, "max_iter");
    assert_eq!(by("maxiter_eq_k1_rosen5").termination, "grad_tol");
    assert_eq!(by("maxiter_eq_k_ftol_rosen5").termination, "max_iter");
    assert_eq!(
        (by("stationary_start").nit, by("stationary_start").nfev),
        (0, 1)
    );
    // SciPy quirk: after an ABNORMAL end `res.fun` is the f of the last *trial*, not of the restored `res.x`
    // (the driver rebinds `f`, so the restore in `mainlb` never reaches it)
    let evals = load_evals();
    for r in runs
        .iter()
        .filter(|r| r.termination == "line_search_failed")
    {
        let e = &evals[&r.id];
        let last_trial = e
            .last()
            .unwrap_or_else(|| panic!("{}: no evaluations", r.id));
        let last_accepted = e.iter().rfind(|x| x.accepted).unwrap_or(&e[0]);
        assert_eq!(r.f, last_trial.f, "{}: res.fun is the last trial's f", r.id);
        assert_ne!(
            r.f, last_accepted.f,
            "{}: and differs from f at the restored x",
            r.id
        );
        assert_eq!(r.x, last_accepted.x, "{}: res.x is restored", r.id);
    }
    // every call has bounds=None and names every option (the protocol)
    assert!(runs.iter().all(|r| r.call.contains("method='L-BFGS-B'")
        && r.call.contains("bounds=None")
        && r.call.contains("'maxls'")
        && r.call.contains("'maxcor'")));
}
