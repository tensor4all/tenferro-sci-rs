//! Shared helpers of the `minimize::lbfgs` integration tests: the reader of the SciPy reference data
//! (`tests/data/lbfgs/`, produced by `scripts/gen_minimize_lbfgs_reference.py`), the closed-form test problems and
//! the replay driver.
//!
//! The replay feeds SciPy's recorded `(f, g)` to the ask/tell state and compares the points the state asks for with
//! the points SciPy evaluated. The tolerance is derived in `tests/data/lbfgs/README.md` ("How the tolerance was
//! derived"): per component `|x_rust − x_scipy| ≤ C_X · ε · x_scale + x_slack`, with the recorded `x_scale` (the
//! absolute-value scale of the computation that produced the point, Higham §3.1) and `x_slack` (SciPy's own measured
//! deviation from the exact direction, accumulated).
//!
//! Written for tensor4all/tenferro-sci-rs#40. SciPy v1.16.3, commit b9105ccc2237f57acb1060202cd77f6dd264fb34,
//! `scipy/optimize/_lbfgsb_py.py` and `__lbfgsb.c` are the behaviour the data records.
#![allow(dead_code)] // each test binary uses a different part

use std::collections::HashMap;
use std::path::PathBuf;

use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step, Termination, H0};

/// ε = 2⁻⁵².
pub const EPS: f64 = f64::EPSILON;
/// The replay tolerance multiplier (README: observed maximum on the committed runs ≤ C_X / 2).
pub const C_X: f64 = 32.0;
/// The bound on SciPy's own direction error against the exact direction (README: observed maximum ≤ C_DIR / 2).
pub const C_DIR: f64 = 128.0;
/// The bound on the direction of a float64 two-loop recursion against the exact one (README: observed ≤ 1.4).
pub const C_TWOLOOP: f64 = 16.0;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/lbfgs")
}

/// Reads a CSV of the data directory: `#` lines are comments, the first other line is the header. The format is
/// unquoted; a quote or a backslash in a cell means a broken file.
pub fn read_csv(name: &str) -> Vec<HashMap<String, String>> {
    let path = data_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("empty csv").split(',').collect();
    lines
        .map(|l| {
            assert!(
                !l.contains('"') && !l.contains('\\'),
                "quote or escape in {name}: {l}"
            );
            let cells: Vec<&str> = l.split(',').collect();
            assert_eq!(cells.len(), header.len(), "bad row in {name}: {l}");
            header
                .iter()
                .zip(cells)
                .map(|(h, c)| (h.to_string(), c.to_string()))
                .collect()
        })
        .collect()
}

fn num(row: &HashMap<String, String>, key: &str) -> f64 {
    row[key]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{key}={:?}: {e}", row[key]))
}

fn int(row: &HashMap<String, String>, key: &str) -> usize {
    row[key]
        .parse()
        .unwrap_or_else(|e| panic!("{key}={:?}: {e}", row[key]))
}

fn vecf(row: &HashMap<String, String>, key: &str) -> Vec<f64> {
    row[key]
        .split(';')
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|e| panic!("{key}: {t:?}: {e}"))
        })
        .collect()
}

/// One row of `runs.csv`: a SciPy `minimize(..., method="L-BFGS-B", bounds=None, ...)` run.
#[derive(Clone, Debug)]
pub struct Run {
    pub id: String,
    pub group: String,
    pub problem: String,
    pub x0: Vec<f64>,
    pub maxcor: usize,
    pub ftol: f64,
    pub gtol: f64,
    pub maxfun: usize,
    pub maxiter: usize,
    pub maxls: usize,
    /// The exact SciPy call (the protocol).
    pub call: String,
    pub termination: String,
    pub nit: usize,
    pub nfev: usize,
    pub nskip: usize,
    pub nrestart: usize,
    pub f: f64,
    pub x: Vec<f64>,
}

/// One row of `evals.csv`: an evaluation SciPy requested.
#[derive(Clone, Debug)]
pub struct Eval {
    pub k: usize,
    /// −1 for the start.
    pub search: i64,
    pub ifun: usize,
    pub x: Vec<f64>,
    pub f: f64,
    pub g: Vec<f64>,
    pub accepted: bool,
    pub x_scale: Vec<f64>,
    pub x_slack: Vec<f64>,
}

/// One row of `searches.csv`: a line search with SciPy's direction and the exact one.
#[derive(Clone, Debug)]
pub struct Search {
    pub run: String,
    pub j: usize,
    pub k0: usize,
    pub iter: usize,
    pub col: usize,
    pub restart: bool,
    pub gamma_scipy: f64,
    pub gamma_exact: f64,
    pub stp0: f64,
    pub x_base: Vec<f64>,
    pub g_base: Vec<f64>,
    pub d: Vec<f64>,
    pub d_exact: Vec<f64>,
    pub d_scale: Vec<f64>,
    /// Ids (within the run, `pairs.csv`) of the stored pairs, oldest to newest.
    pub pairs: std::ops::Range<usize>,
}

pub fn load_runs() -> Vec<Run> {
    read_csv("runs.csv")
        .iter()
        .map(|r| Run {
            id: r["id"].clone(),
            group: r["group"].clone(),
            problem: r["problem"].clone(),
            x0: vecf(r, "x0"),
            maxcor: int(r, "maxcor"),
            ftol: num(r, "ftol"),
            gtol: num(r, "gtol"),
            maxfun: int(r, "maxfun"),
            maxiter: int(r, "maxiter"),
            maxls: int(r, "maxls"),
            call: r["call"].clone(),
            termination: r["termination"].clone(),
            nit: int(r, "nit"),
            nfev: int(r, "nfev"),
            nskip: int(r, "nskip"),
            nrestart: int(r, "nrestart"),
            f: num(r, "f"),
            x: vecf(r, "x"),
        })
        .collect()
}

pub fn load_evals() -> HashMap<String, Vec<Eval>> {
    let mut by: HashMap<String, Vec<Eval>> = HashMap::new();
    for r in read_csv("evals.csv") {
        by.entry(r["run"].clone()).or_default().push(Eval {
            k: int(&r, "k"),
            search: r["search"].parse().expect("search"),
            ifun: int(&r, "ifun"),
            x: vecf(&r, "x"),
            f: num(&r, "f"),
            g: vecf(&r, "g"),
            accepted: int(&r, "accepted") == 1,
            x_scale: vecf(&r, "x_scale"),
            x_slack: vecf(&r, "x_slack"),
        });
    }
    for v in by.values_mut() {
        v.sort_by_key(|e| e.k);
    }
    by
}

pub fn load_searches() -> HashMap<String, Vec<Search>> {
    let mut by: HashMap<String, Vec<Search>> = HashMap::new();
    for r in read_csv("searches.csv") {
        let pairs = if r["pair_lo"] == "none" {
            0..0
        } else {
            int(&r, "pair_lo")..int(&r, "pair_hi") + 1
        };
        by.entry(r["run"].clone()).or_default().push(Search {
            run: r["run"].clone(),
            j: int(&r, "j"),
            k0: int(&r, "k0"),
            iter: int(&r, "iter"),
            col: int(&r, "col"),
            restart: int(&r, "restart") == 1,
            gamma_scipy: num(&r, "gamma_scipy"),
            gamma_exact: num(&r, "gamma_exact"),
            stp0: num(&r, "stp0"),
            x_base: vecf(&r, "x_base"),
            g_base: vecf(&r, "g_base"),
            d: vecf(&r, "d"),
            d_exact: vecf(&r, "d_exact"),
            d_scale: vecf(&r, "d_scale"),
            pairs,
        });
    }
    for v in by.values_mut() {
        v.sort_by_key(|s| s.j);
    }
    by
}

/// `(s, y)` pairs of a run, by id.
/// The `(s, y)` pairs of one run.
pub type PairList = Vec<(Vec<f64>, Vec<f64>)>;

pub fn load_pairs() -> HashMap<String, PairList> {
    type Numbered = Vec<(usize, Vec<f64>, Vec<f64>)>;
    let mut by: HashMap<String, Numbered> = HashMap::new();
    for r in read_csv("pairs.csv") {
        by.entry(r["run"].clone())
            .or_default()
            .push((int(&r, "p"), vecf(&r, "s"), vecf(&r, "y")));
    }
    by.into_iter()
        .map(|(k, mut v)| {
            v.sort_by_key(|t| t.0);
            assert!(
                v.iter().enumerate().all(|(i, t)| t.0 == i),
                "pair ids of {k}"
            );
            (k, v.into_iter().map(|(_, s, y)| (s, y)).collect())
        })
        .collect()
}

/// The Rust `Termination` a `runs.csv` termination name stands for.
pub fn termination_of(name: &str) -> Termination {
    match name {
        "grad_tol" => Termination::GradTol,
        "f_tol" => Termination::FTol,
        "max_iter" => Termination::MaxIter,
        "max_eval" => Termination::MaxEval,
        "line_search_failed" => Termination::LineSearchFailed,
        other => panic!("unknown termination {other:?}"),
    }
}

/// The configuration of a recorded run: exactly the options of its SciPy call.
pub fn config_of(run: &Run) -> LbfgsConfig {
    LbfgsConfig::new(
        run.maxcor,
        run.gtol,
        run.ftol,
        run.maxiter,
        run.maxfun,
        run.maxls,
        H0::ScaledGamma,
    )
    .unwrap_or_else(|e| panic!("{}: {e}", run.id))
}

/// What one replay of a recorded run observed.
#[derive(Debug, Default)]
pub struct Replay {
    pub run: String,
    /// Largest `(|Δx| − x_slack) / (ε x_scale)` over every request (the tolerance is `C_X`).
    pub max_ratio: f64,
    /// Requests that left the tolerance, or other path differences (counts, termination).
    pub path_errors: Vec<String>,
    /// At a line search start the number of stored pairs differed from SciPy's `col`.
    pub col_errors: Vec<String>,
    /// `gᵀ(x_trial − x)` was not negative at the first trial of a search.
    pub descent_errors: Vec<String>,
    /// A restarted search did not start with the unit steepest-descent step from an empty memory.
    pub restart_errors: Vec<String>,
    /// The final state differs from what SciPy returned.
    pub final_errors: Vec<String>,
    pub terminated: Option<Termination>,
    pub requests: usize,
    /// Line searches whose start was checked (the data has one row per search).
    pub searches_seen: usize,
    pub iterations: usize,
    pub evaluations: usize,
    pub skipped: usize,
    pub restarts: usize,
}

/// Replays a recorded run through [`LbfgsState`].
pub fn replay(run: &Run, evals: &[Eval], searches: &[Search]) -> Replay {
    let cfg = config_of(run);
    let mut out = Replay {
        run: run.id.clone(),
        ..Replay::default()
    };
    let mut state = LbfgsState::new(&cfg, &mut (), run.x0.clone());
    let mut next_search = 0;
    let mut i = 0;
    let terminated = loop {
        let requested = match state.ask() {
            Step::Evaluate { x } => x.clone(),
            Step::Converged(t) => break t,
        };
        let Some(e) = evals.get(i) else {
            out.path_errors.push(format!(
                "request {i} is beyond SciPy's last evaluation ({} recorded)",
                evals.len()
            ));
            break Termination::MaxEval; // not used: path_errors is non-empty
        };
        // 1. the requested point against SciPy's, within the derived tolerance
        for (c, ((xr, xs), (sc, sl))) in requested
            .iter()
            .zip(&e.x)
            .zip(e.x_scale.iter().zip(&e.x_slack))
            .enumerate()
        {
            let over = ((xr - xs).abs() - sl).max(0.0);
            let ratio = over / (EPS * sc).max(f64::MIN_POSITIVE);
            out.max_ratio = out.max_ratio.max(ratio);
            if ratio.is_nan() || ratio > C_X {
                out.path_errors.push(format!(
                    "request {i} component {c}: rust {xr:e}, scipy {xs:e}, ratio {ratio:.1} > {C_X}"
                ));
            }
        }
        // 2. what the state must look like when a line search starts
        if let Some(s) = searches.get(next_search).filter(|s| s.k0 == i) {
            next_search += 1;
            out.searches_seen += 1;
            if state.stored_pairs() != s.col {
                out.col_errors.push(format!(
                    "search {}: {} stored pairs, SciPy's col = {}",
                    s.j,
                    state.stored_pairs(),
                    s.col
                ));
            }
            let slope: f64 = requested
                .iter()
                .zip(state.x())
                .zip(state.grad())
                .map(|((t, x), g)| (t - x) * g)
                .sum();
            if slope.is_nan() || slope >= 0.0 {
                out.descent_errors.push(format!(
                    "search {}: gᵀ(x_trial − x) = {slope:e} is not negative",
                    s.j
                ));
            }
            if s.restart {
                // After a failed search SciPy retries steepest descent with the unit step: x_trial = x − g, taken
                // from the state's own x and gradient (they differ from SciPy's by the accumulated rounding, which
                // the point comparison above bounds).
                let ok = requested
                    .iter()
                    .zip(state.x().iter().zip(state.grad()))
                    .all(|(r, (x, g))| (r - (x - g)).abs() <= 2.0 * EPS * (x.abs() + g.abs()));
                if !ok || state.stored_pairs() != 0 {
                    out.restart_errors.push(format!(
                        "search {}: first trial {requested:?}, expected x − g = {:?}, {} stored pairs",
                        s.j,
                        state
                            .x()
                            .iter()
                            .zip(state.grad())
                            .map(|(x, g)| x - g)
                            .collect::<Vec<_>>(),
                        state.stored_pairs()
                    ));
                }
            }
        }
        i += 1;
        state.tell(&cfg, &mut (), e.f, &e.g);
    };
    out.terminated = Some(terminated);
    out.requests = i;
    out.iterations = state.iterations();
    out.evaluations = state.evaluations();
    out.skipped = state.skipped_updates();
    out.restarts = state.restarts();

    // 3. the end of the run
    if i != evals.len() {
        out.path_errors.push(format!(
            "the state stopped after {i} requests, SciPy evaluated {}",
            evals.len()
        ));
    }
    if terminated != termination_of(&run.termination) {
        out.path_errors.push(format!(
            "terminated with {terminated:?}, SciPy: {}",
            run.termination
        ));
    }
    if out.iterations != run.nit {
        out.path_errors.push(format!(
            "{} iterations, SciPy's nit = {}",
            out.iterations, run.nit
        ));
    }
    if out.evaluations != run.nfev {
        out.path_errors.push(format!(
            "{} evaluations, SciPy's nfev = {}",
            out.evaluations, run.nfev
        ));
    }
    // the final iterate is the last accepted one (or x0); f and g are the values SciPy handed over there
    let last = evals.iter().rfind(|e| e.accepted).unwrap_or(&evals[0]);
    for (c, ((xr, xs), (sc, sl))) in state
        .x()
        .iter()
        .zip(&run.x)
        .zip(last.x_scale.iter().zip(&last.x_slack))
        .enumerate()
    {
        let over = ((xr - xs).abs() - sl).max(0.0);
        if over.is_nan() || over > C_X * EPS * sc {
            out.final_errors
                .push(format!("final x[{c}]: rust {xr:e}, scipy {xs:e}"));
        }
    }
    // SciPy's `res.fun` after an ABNORMAL end is the last *trial's* f, not f at the restored x (the driver rebinds `f`
    // to the float it got from the objective, so the restore of `mainlb` does not reach it; `res.x` and `res.jac`
    // are restored). The state keeps f consistent with x: the last accepted evaluation's f, which is SciPy's `fun`
    // on every other termination.
    if state.f() != last.f {
        out.final_errors.push(format!(
            "final f: rust {:e}, last accepted {:e}",
            state.f(),
            last.f
        ));
    }
    if terminated != Termination::LineSearchFailed && state.f() != run.f {
        out.final_errors
            .push(format!("final f: rust {:e}, scipy {:e}", state.f(), run.f));
    }
    if state.grad() != &last.g {
        out.final_errors.push("final gradient differs".into());
    }
    if out.skipped != run.nskip {
        out.final_errors.push(format!(
            "{} skipped updates, SciPy's nskip = {}",
            out.skipped, run.nskip
        ));
    }
    if out.restarts != run.nrestart {
        out.final_errors.push(format!(
            "{} restarts, SciPy: {}",
            out.restarts, run.nrestart
        ));
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// Closed-form problems. They mirror the generator's (`scripts/gen_minimize_lbfgs_reference.py`); the live tests
// check solutions, not paths, so the operation order need not match.
// ---------------------------------------------------------------------------------------------------------------

// The `&Vec` / `&mut Vec` parameters are dictated by `F: FnMut(&mut V::Ctx, &V, &mut V) -> f64` with `V = Vec<f64>`.

/// Rosenbrock's function Σ 100 (x_{i+1} − x_i²)² + (1 − x_i)²; minimizer (1, …, 1), value 0.
#[allow(clippy::ptr_arg)]
pub fn rosen(_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>) -> f64 {
    let n = x.len();
    g.iter_mut().for_each(|t| *t = 0.0);
    let mut f = 0.0;
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    f
}

/// Beale's function; minimizer (3, 0.5), value 0.
#[allow(clippy::ptr_arg)]
pub fn beale(_: &mut (), x: &Vec<f64>, g: &mut Vec<f64>) -> f64 {
    let (a, b) = (x[0], x[1]);
    let t1 = 1.5 - a + a * b;
    let t2 = 2.25 - a + a * b * b;
    let t3 = 2.625 - a + a * b * b * b;
    g[0] = 2.0 * t1 * (b - 1.0) + 2.0 * t2 * (b * b - 1.0) + 2.0 * t3 * (b * b * b - 1.0);
    g[1] = 2.0 * t1 * a + 2.0 * t2 * 2.0 * a * b + 2.0 * t3 * 3.0 * a * b * b;
    t1 * t1 + t2 * t2 + t3 * t3
}

/// `f(x) = ½ (x − xs)ᵀ A (x − xs)` with `A = P diag(λ) P`, `P = I − 2 u uᵀ`, `u = v/|v|`, `v = (1, …, n)`,
/// `λᵢ = cond^(i/(n−1))` (smallest eigenvalue 1, largest `cond`) and `xs = (1, …, n)`. `A` is applied without being
/// formed: `A y = λ∘y − 2 u (λu·y) − 2 (λu)(u·y) + 4 (u·λu) u (u·y)`.
#[derive(Clone, Debug)]
pub struct Quadratic {
    u: Vec<f64>,
    lam: Vec<f64>,
    lu: Vec<f64>,
    c: f64,
    pub xs: Vec<f64>,
}

impl Quadratic {
    pub fn new(n: usize, cond: f64) -> Self {
        let v: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let nv = v.iter().map(|t| t * t).sum::<f64>().sqrt();
        let u: Vec<f64> = v.iter().map(|t| t / nv).collect();
        let lam: Vec<f64> = (0..n)
            .map(|i| cond.powf(i as f64 / (n.max(2) - 1) as f64))
            .collect();
        let lu: Vec<f64> = lam.iter().zip(&u).map(|(l, t)| l * t).collect();
        let c = u.iter().zip(&lu).map(|(a, b)| a * b).sum();
        Self {
            u,
            lam,
            lu,
            c,
            xs: v,
        }
    }

    /// `A y`.
    pub fn apply(&self, y: &[f64]) -> Vec<f64> {
        let uy: f64 = self.u.iter().zip(y).map(|(a, b)| a * b).sum();
        let luy: f64 = self.lu.iter().zip(y).map(|(a, b)| a * b).sum();
        (0..y.len())
            .map(|i| {
                self.lam[i] * y[i] - 2.0 * self.u[i] * luy - 2.0 * self.lu[i] * uy
                    + 4.0 * self.c * self.u[i] * uy
            })
            .collect()
    }

    /// `f` and `∇f` at `x − shift` (a translated copy of the problem; `shift = 0` is the problem itself).
    pub fn eval(&self, x: &[f64], shift: f64, g: &mut [f64]) -> f64 {
        let d: Vec<f64> = x.iter().zip(&self.xs).map(|(a, b)| a - shift - b).collect();
        let ad = self.apply(&d);
        g.copy_from_slice(&ad);
        0.5 * d.iter().zip(&ad).map(|(a, b)| a * b).sum::<f64>()
    }
}

/// Runs `f` through the ask/tell state and records every requested point.
pub fn drive<F>(
    cfg: &LbfgsConfig,
    x0: &[f64],
    mut f: F,
) -> (LbfgsState<Vec<f64>>, Termination, Vec<Vec<f64>>)
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let x0 = x0.to_vec();
    let mut state = LbfgsState::new(cfg, &mut (), x0.clone());
    let mut g = x0.clone();
    let mut requested = Vec::new();
    let t = loop {
        let fx = match state.ask() {
            Step::Evaluate { x } => {
                requested.push(x.clone());
                f(x, &mut g)
            }
            Step::Converged(t) => break t,
        };
        state.tell(cfg, &mut (), fx, &g);
    };
    (state, t, requested)
}
