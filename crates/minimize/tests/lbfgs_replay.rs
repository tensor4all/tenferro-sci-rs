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
    let runs: Vec<Run> = load_runs().into_iter().filter(|r| r.replay).collect();
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
    eprintln!("replay: largest (|dx| - 2 x_slack) / (eps x_scale) over the compared prefixes = {worst:.2} (tolerance {C_X})");
    assert_none(
        &format!("path (largest observed ratio {worst:.2}, bound C_X = {C_X})"),
        reps.iter()
            .flat_map(|r| r.path_errors.iter().map(|e| format!("{}: {e}", r.run)))
            .collect(),
    );
}

#[test]
fn replay_ends_with_scipys_final_point_value_and_gradient() {
    let reps = replays();
    let worst = reps.iter().map(|r| r.max_ratio).fold(0.0, f64::max);
    assert_none(
        &format!("final state (largest observed ratio {worst:.2}, bound C_X = {C_X})"),
        reps.iter()
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

#[test]
fn only_a_recorded_cap_shortens_a_run() {
    // Data only. A run whose `maxiter` is below SciPy's default is either one of the tests of the iteration limit
    // itself (groups `maxiter_order`, `termination`) or a capped run: then `maxiter_cap` is recorded, equals `maxiter`
    // and the run ends with MaxIter. The replay compares every run to its recorded termination, so nothing else can
    // be cut short unnoticed.
    let mut capped = 0;
    for r in load_runs() {
        match r.maxiter_cap {
            Some(cap) => {
                assert_eq!(
                    r.maxiter, cap,
                    "{}: maxiter must equal the recorded cap",
                    r.id
                );
                assert_eq!(
                    (r.termination.as_str(), r.nit),
                    ("max_iter", cap),
                    "{}",
                    r.id
                );
                assert_eq!(
                    r.group, "memory",
                    "{}: only the memory group is capped",
                    r.id
                );
                capped += 1;
            }
            None if r.maxiter != 15000 => assert!(
                matches!(r.group.as_str(), "maxiter_order" | "termination"),
                "{}: maxiter = {} is neither a recorded cap nor a test of the iteration limit",
                r.id,
                r.maxiter
            ),
            None => {}
        }
    }
    assert_eq!(
        capped, 5,
        "the five memory runs that leave the bounds are capped"
    );
}

#[test]
fn caps_cannot_hide_a_disagreement() {
    // The run length of a capped row (`maxiter_cap`) is fixed by SciPy's own direction error: every recorded line search
    // before the cap has direction error <= C_DIR/2 (the minimum over the checked platforms of the largest such prefix;
    // CI runs 46132168523, 46133315711). The float comparison prefix `cmp_nsearch` is separate and shorter or equal:
    // the first line search whose float64-mirror ratio exceeds C_X/4. For each capped row: (1) the replay agrees on
    // every request of the prefix; (2) every search before the cap has direction error inside the bound, and the
    // generating platform's first search outside it (`cap_first_bad`) is at or after the cap and really is outside;
    // (3) the recorded mirror series is inside C_X/4 for every search of the comparison prefix, and the prefix ends at
    // the cap or at a search that exceeds C_X/4: a prefix is never shortened silently. Rows between the prefix and the
    // cap are compared by counts and termination only.
    //
    // Monotonicity of the per-search ratio does NOT hold in the data (it goes up and down); only its running maximum is
    // monotone, which proves nothing (decreases printed below).
    let (runs, evals, searches, caps) = (load_runs(), load_evals(), load_searches(), load_caps());
    for r in runs.iter().filter(|r| r.maxiter_cap.is_some()) {
        let cap = r.maxiter_cap.expect("filtered");
        let cmp = r
            .cmp_nsearch
            .expect("a replayed row has a comparison prefix");
        let bad = r
            .cap_first_bad
            .expect("a capped row records where the bounds were first exceeded");
        let rep = replay(r, &evals[&r.id], &searches[&r.id]);
        assert!(
            rep.path_errors.is_empty(),
            "{}: disagreement within the prefix: {:?}",
            r.id,
            &rep.path_errors[..1.min(rep.path_errors.len())]
        );
        assert!(
            rep.max_ratio <= C_X,
            "{}: observed ratio {:.2} exceeds C_X = {C_X}",
            r.id,
            rep.max_ratio
        );
        let series = &caps[&r.id];
        assert!(
            cap <= bad && cmp <= cap,
            "{}: cmp {cmp} <= cap {cap} <= first search outside {bad} must hold",
            r.id
        );
        assert!(
            series.len() > bad,
            "{}: the uncapped series must continue past {bad}",
            r.id
        );
        for c in &series[..cap] {
            assert!(
                c.dir_ratio <= CAP_DIR_BOUND,
                "{}: line search {} inside the cap has direction error {:.2}",
                r.id,
                c.j,
                c.dir_ratio
            );
        }
        assert!(
            series[bad].dir_ratio > CAP_DIR_BOUND,
            "{}: cap_first_bad {bad} is inside the direction bound",
            r.id
        );
        for c in &series[..cmp] {
            assert!(
                c.mirror_ratio <= CAP_MIRROR_BOUND,
                "{}: line search {} inside the prefix has mirror ratio {:.2}",
                r.id,
                c.j,
                c.mirror_ratio
            );
        }
        // the margin rule: where the mirror bound is exceeded at all, the prefix stays PREFIX_MARGIN searches below it
        if let Some(out) = r.first_outside {
            assert!(
                series[out].mirror_ratio > CAP_MIRROR_BOUND,
                "{}: first_outside {out} is inside the mirror bound",
                r.id
            );
            assert!(cmp + PREFIX_MARGIN <= out, "{}: prefix {cmp} is closer than {PREFIX_MARGIN} searches to the first search outside ({out})", r.id);
        }
        let decreases = series[1..cap]
            .windows(2)
            .filter(|w| w[1].mirror_ratio < w[0].mirror_ratio)
            .count();
        eprintln!(
            "{}: cap {cap}, comparison prefix {cmp}, first search outside the direction bound {bad} (direction {:.2}); {decreases} decreases of the mirror ratio before the cap",
            r.id,
            series[bad].dir_ratio
        );
    }
}

#[test]
fn rows_past_the_comparison_prefix_are_compared_by_counts_only() {
    // Data only. A run whose comparison prefix ends before its last line search (the Rosenbrock problems: rounding
    // differences are amplified by the problem) still replays to the same counts and termination (the replay tests
    // check that for every run); only the requested points past the prefix are not compared.
    let runs = load_runs();
    let short: Vec<&Run> = runs
        .iter()
        .filter(|r| r.cmp_nsearch.is_some_and(|c| c < r.nsearch))
        .collect();
    assert!(!short.is_empty());
    for r in runs.iter().filter(|r| r.replay) {
        let c = r.cmp_nsearch.expect("replayed rows have a prefix");
        assert!(c <= r.nsearch, "{}", r.id);
    }
    eprintln!(
        "{} of {} replayed runs have a comparison prefix shorter than the run",
        short.len(),
        runs.iter().filter(|r| r.replay).count()
    );
}

#[test]
fn only_the_live_group_is_counts_only() {
    // Data only. Rows outside the replay bounds (`replay` = 0) have no recorded evaluations; the live tests compare
    // their counts with them. Nothing else may be skipped by the replay.
    let evals = load_evals();
    let skipped: Vec<String> = load_runs()
        .into_iter()
        .filter(|r| !r.replay)
        .map(|r| {
            assert_eq!(r.group, "live", "{}: only live rows are counts-only", r.id);
            assert!(
                !evals.contains_key(&r.id),
                "{}: a counts-only row has no evaluations",
                r.id
            );
            r.id
        })
        .collect();
    assert!(!skipped.is_empty() && skipped.len() <= 2, "{skipped:?}");
}

#[test]
fn prefixes_and_caps_keep_the_margin_rule() {
    // Data only. Every committed cap and comparison prefix is at least PREFIX_MARGIN line searches shorter than the
    // smallest first-outside index observed on any checked platform: macOS (the `first_outside`, `cap_first_bad` columns)
    // and the Ubuntu CI runs 46132168523, 46133315711 (caps: first outside m1_rosen5 7, m1_quad10_k1e2 16, m1_powell4 15,
    // m3_rosen5 25, m3_quad10_k1e2 26), 46134596827 and 46135341702 (quad5_k1e4 comparison prefix: 14 where macOS has 15).
    let linux_caps = [
        ("m1_rosen5", 7),
        ("m1_quad10_k1e2", 16),
        ("m1_powell4", 15),
        ("m3_rosen5", 25),
        ("m3_quad10_k1e2", 26),
    ];
    let runs = load_runs();
    let get = |id: &str| {
        runs.iter()
            .find(|r| r.id == id)
            .unwrap_or_else(|| panic!("run {id}"))
    };
    for (id, linux) in linux_caps {
        let r = get(id);
        let cap = r.maxiter_cap.expect("capped");
        let mac = r.cap_first_bad.expect("recorded");
        // macOS evidence for the cap: the direction criterion (first_bad) and the mirror criterion of the earlier selection
        // (25, 15, 15, 26, 26); Linux as listed
        let mac_old = [
            ("m1_rosen5", 25),
            ("m1_quad10_k1e2", 15),
            ("m1_powell4", 15),
            ("m3_rosen5", 26),
            ("m3_quad10_k1e2", 26),
        ]
        .iter()
        .find(|(i, _)| *i == id)
        .map(|(_, v)| *v)
        .expect("listed");
        let smallest = mac.min(mac_old).min(linux);
        assert!(cap + PREFIX_MARGIN <= smallest, "{id}: cap {cap} needs {PREFIX_MARGIN} searches below the smallest first-outside index {smallest}");
    }
    for id in ["default_quad5_k1e4", "tight_quad5_k1e4", "live_quad5_k1e4"] {
        let r = get(id);
        let cmp = r.cmp_nsearch.expect("replayed");
        assert!(
            cmp + PREFIX_MARGIN <= 14,
            "{id}: prefix {cmp} must be <= 12 (Linux first outside: 14)"
        );
    }
    for r in runs.iter().filter(|r| r.replay) {
        if let Some(out) = r.first_outside {
            assert!(
                r.cmp_nsearch.expect("replayed") + PREFIX_MARGIN <= out,
                "{}: prefix {:?}, first outside {out}",
                r.id,
                r.cmp_nsearch
            );
        }
    }
}
