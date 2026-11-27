//! Shared helpers of the `minimize::bfgs` integration tests: the reader for the SciPy reference
//! data (`tests/data/bfgs`, see its README), the closed-form objectives, the tolerance rule, and the
//! ask/tell drivers (live and replay).
//!
//! The objectives mirror `scripts/gen_minimize_bfgs_reference.py` mathematically (same formulas);
//! live tests assert solutions and invariants, not paths, so the operation order need not match.
//! Quadratics read `A` and `b` from `problems.csv`.
//!
//! Written for this repository; the formulas are the textbook ones (Rosenbrock; Himmelblau;
//! Beale; Moré–Garbow–Hillstrom no. 13, Powell's singular function). Not a port.

#![allow(dead_code)]

use minimize::bfgs::{BfgsConfig, BfgsState, Step, Termination};
use std::collections::HashMap;
use std::path::PathBuf;

/// ε = 2⁻⁵², the unit the README's tolerances are written in.
pub const EPS: f64 = f64::EPSILON;
/// The constant C of the strict (local rounding) term of the replay tolerance.
pub const REPLAY_C: f64 = 64.0;
/// The relative propagation envelope of the replay tolerance.
pub const ENVELOPE: f64 = 1e-8;

// ----------------------------------------------------------------------------------------------
// Data
// ----------------------------------------------------------------------------------------------

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/bfgs")
}

/// Rows of a CSV file of the data directory as column-name maps. `#` lines are comments; the
/// format is unquoted by construction, so a quote or backslash means a broken file and panics.
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

type Row = HashMap<String, String>;

fn num(row: &Row, key: &str) -> f64 {
    row[key]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{key}={:?}: {e}", row[key]))
}

fn count(row: &Row, key: &str) -> usize {
    row[key]
        .parse()
        .unwrap_or_else(|e| panic!("{key}={:?}: {e}", row[key]))
}

fn vecs(row: &Row, key: &str) -> Vec<f64> {
    row[key]
        .split(';')
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|e| panic!("{key}: {t:?}: {e}"))
        })
        .collect()
}

fn opt_vecs(row: &Row, key: &str) -> Option<Vec<f64>> {
    (row[key] != "none").then(|| vecs(row, key))
}

/// One row of `problems.csv`.
#[derive(Clone, Debug)]
pub struct ProblemData {
    pub id: String,
    pub n: usize,
    /// Row-major `A` of a quadratic `½xᵀAx − bᵀx`.
    pub a: Option<Vec<f64>>,
    pub b: Option<Vec<f64>>,
    pub x_star: Option<Vec<f64>>,
    pub f_star: f64,
    /// A strong-convexity bound valid near `x*` (README of the data).
    pub mu: Option<f64>,
}

pub fn load_problems() -> HashMap<String, ProblemData> {
    read_csv("problems.csv")
        .into_iter()
        .map(|r| {
            let p = ProblemData {
                id: r["id"].clone(),
                n: count(&r, "n"),
                a: opt_vecs(&r, "A"),
                b: opt_vecs(&r, "b"),
                x_star: opt_vecs(&r, "x_star"),
                f_star: if r["f_star"] == "none" {
                    f64::NAN
                } else {
                    num(&r, "f_star")
                },
                mu: (r["mu"] != "none").then(|| num(&r, "mu")),
            };
            (p.id.clone(), p)
        })
        .collect()
}

/// One recorded SciPy evaluation `(x, f, g)` with the rounding scale of the requested point.
#[derive(Clone, Debug)]
pub struct Eval {
    pub x: Vec<f64>,
    pub f: f64,
    pub g: Vec<f64>,
    pub x_scale: Option<Vec<f64>>,
}

/// One recorded line search (one iteration of `_minimize_bfgs`).
#[derive(Clone, Debug)]
pub struct Iter {
    pub k: usize,
    pub x: Vec<f64>,
    pub g: Vec<f64>,
    pub p: Vec<f64>,
    pub d_exact: Option<Vec<f64>>,
    pub d_scale: Option<Vec<f64>>,
    pub alpha: Option<f64>,
}

/// SciPy's final inverse Hessian with the exact matrix and the scale (row-major).
#[derive(Clone, Debug)]
pub struct Hess {
    pub n_updates: usize,
    pub h_scipy: Vec<f64>,
    pub h_exact: Vec<f64>,
    pub h_scale: Vec<f64>,
}

/// One run of `runs.csv` with its evaluations, line searches and final matrix.
#[derive(Clone, Debug)]
pub struct Run {
    pub id: String,
    pub group: String,
    pub problem: String,
    pub n: usize,
    pub x0: Vec<f64>,
    pub gtol: f64,
    pub xrtol: f64,
    pub c1: f64,
    pub c2: f64,
    pub maxiter: usize,
    pub status: i32,
    pub message: String,
    pub nit: usize,
    pub nfev: usize,
    pub wolfe2_calls: usize,
    pub wolfe2_first_eval: Option<usize>,
    pub wolfe1_fail: String,
    pub termination: String,
    pub parity: bool,
    pub final_x: Vec<f64>,
    pub final_f: f64,
    pub evals: Vec<Eval>,
    pub iters: Vec<Iter>,
    pub hess: Option<Hess>,
}

impl Run {
    pub fn config(&self) -> BfgsConfig {
        BfgsConfig::new(self.gtol, self.xrtol, self.c1, self.c2, self.maxiter)
            .unwrap_or_else(|e| panic!("{}: recorded options are invalid: {e}", self.id))
    }

    pub fn expected_termination(&self) -> Termination {
        parse_termination(&self.termination)
    }
}

pub fn parse_termination(name: &str) -> Termination {
    match name {
        "GradTol" => Termination::GradTol,
        "StepTol" => Termination::StepTol,
        "MaxIter" => Termination::MaxIter,
        "LineSearchFailed" => Termination::LineSearchFailed,
        "NonFiniteValue" => Termination::NonFiniteValue,
        other => panic!("unknown termination {other:?} in the data"),
    }
}

pub fn load_runs() -> Vec<Run> {
    let mut evals: HashMap<String, Vec<(usize, Eval)>> = HashMap::new();
    for r in read_csv("evals.csv") {
        evals.entry(r["run"].clone()).or_default().push((
            count(&r, "i"),
            Eval {
                x: vecs(&r, "x"),
                f: num(&r, "f"),
                g: vecs(&r, "g"),
                x_scale: opt_vecs(&r, "x_scale"),
            },
        ));
    }
    let mut iters: HashMap<String, Vec<Iter>> = HashMap::new();
    for r in read_csv("iters.csv") {
        iters.entry(r["run"].clone()).or_default().push(Iter {
            k: count(&r, "k"),
            x: vecs(&r, "x"),
            g: vecs(&r, "g"),
            p: vecs(&r, "p"),
            d_exact: opt_vecs(&r, "d_exact"),
            d_scale: opt_vecs(&r, "d_scale"),
            alpha: (r["alpha"] != "none").then(|| num(&r, "alpha")),
        });
    }
    let mut hess: HashMap<String, Hess> = HashMap::new();
    for r in read_csv("hess_final.csv") {
        hess.insert(
            r["run"].clone(),
            Hess {
                n_updates: count(&r, "n_updates"),
                h_scipy: vecs(&r, "h_scipy"),
                h_exact: vecs(&r, "h_exact"),
                h_scale: vecs(&r, "h_scale"),
            },
        );
    }
    read_csv("runs.csv")
        .into_iter()
        .map(|r| {
            let id = r["id"].clone();
            let mut ev = evals.remove(&id).unwrap_or_default();
            ev.sort_by_key(|(i, _)| *i);
            for (pos, (i, _)) in ev.iter().enumerate() {
                assert_eq!(pos, *i, "{id}: evaluations are not 0..n in order");
            }
            let mut it = iters.remove(&id).unwrap_or_default();
            it.sort_by_key(|i| i.k);
            Run {
                n: count(&r, "n"),
                x0: vecs(&r, "x0"),
                gtol: num(&r, "gtol"),
                xrtol: num(&r, "xrtol"),
                c1: num(&r, "c1"),
                c2: num(&r, "c2"),
                maxiter: count(&r, "maxiter"),
                status: r["status"].parse().expect("status"),
                message: r["message"].clone(),
                nit: count(&r, "nit"),
                nfev: count(&r, "nfev"),
                wolfe2_calls: count(&r, "wolfe2_calls"),
                wolfe2_first_eval: (r["wolfe2_first_eval"] != "none")
                    .then(|| count(&r, "wolfe2_first_eval")),
                wolfe1_fail: r["wolfe1_fail"].clone(),
                termination: r["termination"].clone(),
                parity: r["parity"] == "true",
                final_x: vecs(&r, "final_x"),
                final_f: num(&r, "final_f"),
                group: r["group"].clone(),
                problem: r["problem"].clone(),
                hess: hess.remove(&id),
                evals: ev.into_iter().map(|(_, e)| e).collect(),
                iters: it,
                id,
            }
        })
        .collect()
}

pub fn run_by_id(id: &str) -> Run {
    load_runs()
        .into_iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("no run {id} in the data"))
}

// ----------------------------------------------------------------------------------------------
// The tolerance rule of the data README
// ----------------------------------------------------------------------------------------------

/// `min(C·n·ε·scale_j, ENVELOPE·env)` for component `j`: the README's bound (strict local-rounding
/// term, capped by the propagation envelope). `env` is `1 + ‖want‖∞` for iterates and `‖want‖∞`
/// for directions and matrices.
pub fn tol(scale: &[f64], j: usize, n: usize, env: f64) -> f64 {
    (REPLAY_C * n as f64 * EPS * scale[j]).min(ENVELOPE * env)
}

/// `None` if `got` is within the README's bound of `want` componentwise, else a message.
pub fn mismatch(
    label: &str,
    got: &[f64],
    want: &[f64],
    scale: &[f64],
    env: f64,
    n: usize,
) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!("{label}: length {} vs {}", got.len(), want.len()));
    }
    for j in 0..want.len() {
        let t = tol(scale, j, n, env);
        let d = (got[j] - want[j]).abs();
        if d.is_nan() || d > t {
            return Some(format!(
                "{label}[{j}]: got {:e}, want {:e}, |diff| {:e} > tol {:e}",
                got[j], want[j], d, t
            ));
        }
    }
    None
}

pub fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, t| m.max(t.abs()))
}

pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ----------------------------------------------------------------------------------------------
// Closed-form objectives
// ----------------------------------------------------------------------------------------------

/// `x ↦ (f(x), ∇f(x))`.
pub type ObjFn = dyn Fn(&[f64]) -> (f64, Vec<f64>);

pub type Objective = Box<ObjFn>;

pub fn rosenbrock(x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    let mut f = 0.0;
    let mut g = vec![0.0; n];
    for i in 0..n - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    (f, g)
}

pub fn himmelblau(x: &[f64]) -> (f64, Vec<f64>) {
    let a = x[0] * x[0] + x[1] - 11.0;
    let b = x[0] + x[1] * x[1] - 7.0;
    (
        a * a + b * b,
        vec![4.0 * a * x[0] + 2.0 * b, 2.0 * a + 4.0 * b * x[1]],
    )
}

pub fn beale(x: &[f64]) -> (f64, Vec<f64>) {
    let c = [1.5, 2.25, 2.625];
    let (mut f, mut g0, mut g1) = (0.0, 0.0, 0.0);
    for (i, ci) in c.iter().enumerate() {
        let k = (i + 1) as i32;
        let r = ci - x[0] * (1.0 - x[1].powi(k));
        f += r * r;
        g0 += 2.0 * r * -(1.0 - x[1].powi(k));
        g1 += 2.0 * r * (x[0] * k as f64 * x[1].powi(k - 1));
    }
    (f, vec![g0, g1])
}

pub fn powell(x: &[f64]) -> (f64, Vec<f64>) {
    let f = (x[0] + 10.0 * x[1]).powi(2)
        + 5.0 * (x[2] - x[3]).powi(2)
        + (x[1] - 2.0 * x[2]).powi(4)
        + 10.0 * (x[0] - x[3]).powi(4);
    let g = vec![
        2.0 * (x[0] + 10.0 * x[1]) + 40.0 * (x[0] - x[3]).powi(3),
        20.0 * (x[0] + 10.0 * x[1]) + 4.0 * (x[1] - 2.0 * x[2]).powi(3),
        10.0 * (x[2] - x[3]) - 8.0 * (x[1] - 2.0 * x[2]).powi(3),
        -10.0 * (x[2] - x[3]) - 40.0 * (x[0] - x[3]).powi(3),
    ];
    (f, g)
}

/// `A x` for a row-major `A`.
pub fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

pub fn quadratic(a: Vec<f64>, b: Vec<f64>) -> Objective {
    let n = b.len();
    Box::new(move |x| {
        let ax = matvec(&a, n, x);
        let f = 0.5 * dot(x, &ax) - dot(&b, x);
        let g = ax.iter().zip(&b).map(|(u, v)| u - v).collect();
        (f, g)
    })
}

/// The objective named like a `problem` of the data (`problems.csv`).
pub fn objective(id: &str) -> Objective {
    match id {
        "rosen2" | "rosen5" => Box::new(rosenbrock),
        "himmelblau" => Box::new(himmelblau),
        "beale" => Box::new(beale),
        "powell" => Box::new(powell),
        "linear_unbounded" => Box::new(|x| (-x[0], vec![-1.0, 0.0])),
        "rosen2_neggrad" => Box::new(|x| {
            let (f, g) = rosenbrock(x);
            (f, g.into_iter().map(|t| -t).collect())
        }),
        "rosen2_nan" => Box::new(|x| {
            if x[0] > 0.5 {
                (f64::NAN, vec![f64::NAN; x.len()])
            } else {
                rosenbrock(x)
            }
        }),
        "rosen2_inf" => Box::new(|x| {
            if x[0] > 0.5 {
                (f64::INFINITY, vec![f64::INFINITY; x.len()])
            } else {
                rosenbrock(x)
            }
        }),
        other => {
            let p = load_problems()
                .remove(other)
                .unwrap_or_else(|| panic!("no problem {other}"));
            quadratic(
                p.a.unwrap_or_else(|| panic!("{other} is not a quadratic")),
                p.b.expect("b"),
            )
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Drivers
// ----------------------------------------------------------------------------------------------

/// The state at a moment the iteration count changed (and after the first evaluation).
#[derive(Clone, Debug)]
pub struct Snap {
    pub iterations: usize,
    pub nfev: usize,
    pub x: Vec<f64>,
    pub f: f64,
    pub g: Vec<f64>,
    /// The direction the next line search uses (meaningful only if `running`).
    pub p: Vec<f64>,
    /// Row-major copy of the inverse Hessian approximation.
    pub h: Vec<f64>,
    /// The state asked for another evaluation after this `tell`.
    pub running: bool,
}

/// A live run: every evaluation asked, the snapshots, and how it ended.
pub struct Trace {
    /// `(x asked, f, g)` as the objective returned them.
    pub evals: Vec<(Vec<f64>, f64, Vec<f64>)>,
    pub snaps: Vec<Snap>,
    pub termination: Termination,
    pub state: BfgsState<Vec<f64>>,
}

pub fn matrix_of(state: &BfgsState<Vec<f64>>) -> Vec<f64> {
    let h = state.inverse_hessian();
    let n = h.nrows();
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            out.push(h[(i, j)]);
        }
    }
    out
}

fn snap(state: &BfgsState<Vec<f64>>) -> Snap {
    Snap {
        iterations: state.iterations(),
        nfev: state.nfev(),
        x: state.x().clone(),
        f: state.f(),
        g: state.g().clone(),
        p: state.direction().clone(),
        h: matrix_of(state),
        running: matches!(state.ask(), Step::Evaluate { .. }),
    }
}

/// Runs the state against a live objective until it says `Converged`; panics after `max_evals`.
pub fn drive(cfg: &BfgsConfig, x0: &[f64], obj: &ObjFn, max_evals: usize) -> Trace {
    let mut state = BfgsState::new(cfg, &mut (), x0.to_vec());
    let mut evals = Vec::new();
    let mut snaps: Vec<Snap> = Vec::new();
    let termination = loop {
        match state.ask() {
            Step::Evaluate { x } => {
                assert!(
                    evals.len() < max_evals,
                    "more than {max_evals} evaluations: the run does not terminate"
                );
                let xr = x.clone();
                let (f, g) = obj(&xr);
                state.tell(cfg, &mut (), f, &g);
                evals.push((xr, f, g));
                if snaps.last().map(|s| s.iterations) != Some(state.iterations()) {
                    snaps.push(snap(&state));
                }
            }
            Step::Converged(t) => break t,
        }
    };
    Trace {
        evals,
        snaps,
        termination,
        state,
    }
}

/// The outcome of feeding a recorded run through the state.
pub struct Replay {
    /// The points the state asked for, in order.
    pub asked: Vec<Vec<f64>>,
    /// How it ended; `None` if it asked for more evaluations than SciPy made.
    pub termination: Option<Termination>,
    /// `(k, direction)` each time iteration `k` started.
    pub directions: Vec<(usize, Vec<f64>)>,
    pub state: BfgsState<Vec<f64>>,
}

/// Feeds the recorded `(f, g)` to the state. With `stop_after`, stops after that many `tell`s.
pub fn replay(run: &Run, stop_after: Option<usize>) -> Replay {
    let cfg = run.config();
    let mut state = BfgsState::new(&cfg, &mut (), run.x0.clone());
    let mut asked = Vec::new();
    let mut directions = Vec::new();
    let mut seen_iter: Option<usize> = None;
    let termination = loop {
        match state.ask() {
            Step::Evaluate { x } => {
                let i = asked.len();
                if i >= run.evals.len() || Some(i) == stop_after {
                    break None;
                }
                asked.push(x.clone());
                let e = &run.evals[i];
                state.tell(&cfg, &mut (), e.f, &e.g);
                if matches!(state.ask(), Step::Evaluate { .. })
                    && seen_iter != Some(state.iterations())
                {
                    seen_iter = Some(state.iterations());
                    directions.push((state.iterations(), state.direction().clone()));
                }
            }
            Step::Converged(t) => break Some(t),
        }
    };
    Replay {
        asked,
        termination,
        directions,
        state,
    }
}

// ----------------------------------------------------------------------------------------------
// Small dense helpers for the property tests (test code, not library)
// ----------------------------------------------------------------------------------------------

/// Cholesky factorization of a symmetric row-major matrix; `false` if a pivot is not positive.
pub fn is_positive_definite(a: &[f64], n: usize) -> bool {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    true
}
