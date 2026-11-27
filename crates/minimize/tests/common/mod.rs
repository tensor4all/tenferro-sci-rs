//! Shared helpers of the `minimize::linesearch` integration tests: the CSV reader for the
//! SciPy reference data, the closed-form test functions, the exact strong-Wolfe checks and the
//! ask/tell drivers.
//!
//! The test functions mirror `scripts/gen_minimize_linesearch_reference.py` operation for
//! operation (same expression order, `powf` where Python has `**` with a non-trivial power, no
//! FMA), so a live run evaluates the same arithmetic as the recorded `closed` runs.
//!
//! Sources: SciPy v1.16.3 (commit b9105ccc2237f57acb1060202cd77f6dd264fb34),
//! `scipy/optimize/tests/test_linesearch.py` for `assert_wolfe` and `assert_fp_equal`;
//! Moré & Thuente, ACM TOMS 20(3), 1994, section 5 for the six test functions.

#![allow(dead_code)]

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::PathBuf;

use minimize::linesearch::{
    MoreThuente, MoreThuenteConfig, Point, Reason, Stage, StartError, Step, Termination,
    Wolfe1Outcome,
};

// ---------------------------------------------------------------------------------------------
// Reference data
// ---------------------------------------------------------------------------------------------

/// One recorded trial of SciPy: `(alpha, phi, dphi)`.
pub type Trial = (f64, f64, f64);

/// One row of `cases.csv` with its trials attached.
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    /// `scipy` (SciPy's own test functions) or `closed` (closed-form scalar functions).
    pub run: String,
    /// `wolfe1` (through `scalar_search_wolfe1`, maxiter 100) or `dcsrch` (`DCSRCH` directly).
    pub kind: String,
    pub group: String,
    pub func: String,
    pub p: Vec<f64>,
    pub phi0: f64,
    pub old_phi0: Option<f64>,
    pub dphi0: f64,
    pub phi0_given: bool,
    pub dphi0_given: bool,
    pub alpha1: f64,
    pub ftol: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub stpmin: f64,
    pub stpmax: f64,
    pub maxiter: usize,
    /// converged, rounding, xtol, stpmax, stpmin, nonfinite_step, maxiter, err_below_stpmin,
    /// err_above_stpmax or err_not_descent.
    pub reason: String,
    pub task: String,
    /// SciPy's returned step; `None` exactly where SciPy returns `None`.
    pub stp: Option<f64>,
    pub phi_evals: usize,
    pub dphi_evals: usize,
    pub n_trials: usize,
    pub trials: Vec<Trial>,
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/linesearch")
}

/// Read a CSV file of the data directory: `#` lines are comments, the first other line is the
/// header. Returns the rows as column-name maps.
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

fn f(row: &HashMap<String, String>, key: &str) -> f64 {
    row[key]
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{key}={:?}: {e}", row[key]))
}

fn opt_f(row: &HashMap<String, String>, key: &str) -> Option<f64> {
    if row[key] == "none" {
        None
    } else {
        Some(f(row, key))
    }
}

fn u(row: &HashMap<String, String>, key: &str) -> usize {
    row[key].parse().unwrap_or_else(|e| panic!("{key}: {e}"))
}

/// All runs of `cases.csv`, each with its trials from `trials.csv` attached in order.
pub fn load_cases() -> Vec<Case> {
    let mut trials: HashMap<String, Vec<(usize, Trial)>> = HashMap::new();
    for row in read_csv("trials.csv") {
        trials.entry(row["id"].clone()).or_default().push((
            u(&row, "k"),
            (f(&row, "alpha"), f(&row, "phi"), f(&row, "dphi")),
        ));
    }
    read_csv("cases.csv")
        .into_iter()
        .map(|row| {
            let id = row["id"].clone();
            let mut t = trials.remove(&id).unwrap_or_default();
            t.sort_by_key(|(k, _)| *k);
            assert!(
                t.iter().enumerate().all(|(i, (k, _))| i == *k),
                "{id}: trial gap"
            );
            let case = Case {
                run: row["run"].clone(),
                kind: row["kind"].clone(),
                group: row["group"].clone(),
                func: row["fn"].clone(),
                p: ["p0", "p1", "p2"]
                    .iter()
                    .filter_map(|k| opt_f(&row, k))
                    .collect(),
                phi0: f(&row, "phi0"),
                old_phi0: opt_f(&row, "old_phi0"),
                dphi0: f(&row, "dphi0"),
                phi0_given: u(&row, "phi0_given") == 1,
                dphi0_given: u(&row, "dphi0_given") == 1,
                alpha1: f(&row, "alpha1"),
                ftol: f(&row, "ftol"),
                gtol: f(&row, "gtol"),
                xtol: f(&row, "xtol"),
                stpmin: f(&row, "stpmin"),
                stpmax: f(&row, "stpmax"),
                maxiter: u(&row, "maxiter"),
                reason: row["reason"].clone(),
                task: row["task"].clone(),
                stp: opt_f(&row, "stp"),
                phi_evals: u(&row, "phi_evals"),
                dphi_evals: u(&row, "dphi_evals"),
                n_trials: u(&row, "n_trials"),
                trials: t.into_iter().map(|(_, x)| x).collect(),
                id,
            };
            assert_eq!(case.trials.len(), case.n_trials, "{}: trial count", case.id);
            case
        })
        .collect()
}

pub fn case_by_id(id: &str) -> Case {
    load_cases()
        .into_iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("no case {id} in cases.csv"))
}

impl Case {
    pub fn setup(&self) -> Setup {
        Setup {
            ftol: self.ftol,
            gtol: self.gtol,
            xtol: self.xtol,
            stpmin: self.stpmin,
            stpmax: self.stpmax,
        }
    }

    /// Whether the open non-finite decision (issue #38, review 9.1) applies to this run.
    pub fn is_nonfinite_group(&self) -> bool {
        self.group == "nonfinite"
    }
}

// ---------------------------------------------------------------------------------------------
// Test functions (closed form)
// ---------------------------------------------------------------------------------------------

const MT1_B: f64 = 2.0;
const MT2_B: f64 = 0.004;
const MT3_B: f64 = 0.01;
const MT3_L: f64 = 39.0;

fn mt3_p0(a: f64) -> f64 {
    if a <= 1.0 - MT3_B {
        1.0 - a
    } else if a >= 1.0 + MT3_B {
        a - 1.0
    } else {
        (a - 1.0) * (a - 1.0) / (2.0 * MT3_B) + MT3_B / 2.0
    }
}

fn mt3_d0(a: f64) -> f64 {
    if a <= 1.0 - MT3_B {
        -1.0
    } else if a >= 1.0 + MT3_B {
        1.0
    } else {
        (a - 1.0) / MT3_B
    }
}

fn yanai(b1: f64, b2: f64, a: f64) -> (f64, f64) {
    let g = |b: f64| (1.0 + b * b).sqrt() - b;
    let r1 = ((1.0 - a) * (1.0 - a) + b2 * b2).sqrt();
    let r2 = (a * a + b1 * b1).sqrt();
    (
        g(b1) * r1 + g(b2) * r2,
        g(b1) * (a - 1.0) / r1 + g(b2) * a / r2,
    )
}

/// `(phi(a), phi'(a))` of the named function with parameters `p`.
pub fn eval(name: &str, p: &[f64], a: f64) -> (f64, f64) {
    match name {
        // SciPy's scalar test functions (test_linesearch.py)
        "sf1" => (
            -a - a.powf(3.0) + a.powf(4.0),
            -1.0 - 3.0 * a.powf(2.0) + 4.0 * a.powf(3.0),
        ),
        "sf2" => {
            let e = (-4.0 * a).exp();
            (e + a * a, -4.0 * e + 2.0 * a)
        }
        "sf3" => (-(10.0 * a).sin(), -10.0 * (10.0 * a).cos()),
        // test_wolfe_terminate
        "wt" => (-a + 0.05 * (a * a), -1.0 + 0.05 * 2.0 * a),
        // A a^2 + B a + C, the line functions of test_line_search_wolfe1 along x + a p
        "quad" => (p[0] * a * a + p[1] * a + p[2], 2.0 * p[0] * a + p[1]),
        "shiftq" => ((a - p[0]) * (a - p[0]), 2.0 * (a - p[0])),
        "lin_neg" => (-a, -1.0),
        "lin_pos" => (a, 1.0),
        "inf_half" => {
            if a < 0.5 {
                ((a - 1.0) * (a - 1.0), 2.0 * (a - 1.0))
            } else {
                (f64::INFINITY, f64::INFINITY)
            }
        }
        "nan_half" => {
            if a < 0.5 {
                ((a - 1.0) * (a - 1.0), 2.0 * (a - 1.0))
            } else {
                (f64::NAN, f64::NAN)
            }
        }
        "nan_dphi" => (
            (a - 1.0) * (a - 1.0),
            if a == 0.0 { 2.0 * (a - 1.0) } else { f64::NAN },
        ),
        "big_half" => {
            if a < 0.5 {
                ((a - 1.0) * (a - 1.0), 2.0 * (a - 1.0))
            } else {
                (1e308, 1.0)
            }
        }
        // Moré-Thuente (1994) section 5, equations (5.1)-(5.4)
        "mt1" => (
            -a / (a * a + MT1_B),
            (a * a - MT1_B) / ((a * a + MT1_B) * (a * a + MT1_B)),
        ),
        "mt2" => {
            let x = a + MT2_B;
            (
                x.powf(5.0) - 2.0 * x.powf(4.0),
                5.0 * x.powf(4.0) - 8.0 * x.powf(3.0),
            )
        }
        "mt3" => {
            let arg = MT3_L * PI * a / 2.0;
            (
                mt3_p0(a) + 2.0 * (1.0 - MT3_B) / (MT3_L * PI) * arg.sin(),
                mt3_d0(a) + (1.0 - MT3_B) * arg.cos(),
            )
        }
        "mt4" => yanai(1e-3, 1e-3, a),
        "mt5" => yanai(1e-2, 1e-3, a),
        "mt6" => yanai(1e-3, 1e-2, a),
        other => panic!("unknown test function {other}"),
    }
}

pub fn phi_of(name: &str, p: &[f64], a: f64) -> f64 {
    eval(name, p, a).0
}

pub fn dphi_of(name: &str, p: &[f64], a: f64) -> f64 {
    eval(name, p, a).1
}

// ---------------------------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------------------------

/// `|a - b| <= tol * max(|a|, |b|)`, true for two equal values (including two zeros).
pub fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * a.abs().max(b.abs())
}

/// `numpy.spacing(x)` for finite `x`: the distance to the next float away from zero.
pub fn spacing(x: f64) -> f64 {
    let x = x.abs();
    f64::from_bits(x.to_bits() + 1) - x
}

/// Port of `assert_fp_equal` of SciPy's test_linesearch.py (numpy's
/// `assert_array_almost_equal_nulp`): `|x - y| <= nulp * spacing(max(|x|, |y|))`, nulp = 50.
pub fn assert_fp_equal(x: f64, y: f64, msg: &str) {
    let r = 50.0 * spacing(x.abs().max(y.abs()));
    assert!(
        (x - y).abs() <= r,
        "{msg}: {x:e} vs {y:e} differ by more than 50 ulp"
    );
}

/// `assert_wolfe` of SciPy's test_linesearch.py, ported literally (its inequalities are
/// written `phi0 + c1*s*derphi0` and `abs(derphi1) <= abs(c2*derphi0)`).
pub fn assert_wolfe_scipy(
    s: f64,
    phi: &dyn Fn(f64) -> f64,
    dphi: &dyn Fn(f64) -> f64,
    c1: f64,
    c2: f64,
    msg: &str,
) {
    let (phi1, phi0, dphi0, dphi1) = (phi(s), phi(0.0), dphi(0.0), dphi(s));
    let m = format!(
        "s = {s}; phi(0) = {phi0}; phi(s) = {phi1}; phi'(0) = {dphi0}; phi'(s) = {dphi1}; {msg}"
    );
    assert!(phi1 <= phi0 + c1 * s * dphi0, "Wolfe 1 failed: {m}");
    assert!(dphi1.abs() <= (c2 * dphi0).abs(), "Wolfe 2 failed: {m}");
}

/// (1.1) sufficient decrease, written the way DCSRCH evaluates it
/// (`ftest = finit + stp * (ftol * ginit)`), so the test and the code round alike.
pub fn sufficient_decrease(phi0: f64, dphi0: f64, mu: f64, a: f64, phi_a: f64) -> bool {
    phi_a <= phi0 + a * (mu * dphi0)
}

/// (1.2) strong curvature condition, as DCSRCH writes it (`abs(g) <= gtol * -ginit`, ginit < 0).
pub fn curvature(dphi0: f64, eta: f64, dphi_a: f64) -> bool {
    dphi_a.abs() <= eta * -dphi0
}

/// Both strong-Wolfe inequalities, exactly (no tolerance).
pub fn assert_strong_wolfe(
    phi0: f64,
    dphi0: f64,
    mu: f64,
    eta: f64,
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    msg: &str,
) {
    assert!(
        sufficient_decrease(phi0, dphi0, mu, a, phi_a),
        "(1.1) sufficient decrease fails at alpha = {a:e}: phi = {phi_a:e}, phi0 = {phi0:e}, \
         phi'(0) = {dphi0:e}, mu = {mu:e}; {msg}"
    );
    assert!(
        curvature(dphi0, eta, dphi_a),
        "(1.2) curvature fails at alpha = {a:e}: phi' = {dphi_a:e}, phi'(0) = {dphi0:e}, \
         eta = {eta:e}; {msg}"
    );
}

// ---------------------------------------------------------------------------------------------
// Names (exhaustive matches: they also pin the variant lists of the planned API)
// ---------------------------------------------------------------------------------------------

pub fn reason_name(r: &Reason) -> &'static str {
    match r {
        Reason::Converged => "converged",
        Reason::RoundingErrors => "rounding",
        Reason::XtolSatisfied => "xtol",
        Reason::StpMax => "stpmax",
        Reason::StpMin => "stpmin",
        Reason::NonFiniteStep => "nonfinite_step",
    }
}

pub fn start_error_name(e: &StartError) -> &'static str {
    match e {
        StartError::StepBelowStpmin => "err_below_stpmin",
        StartError::StepAboveStpmax => "err_above_stpmax",
        StartError::NotDescent => "err_not_descent",
    }
}

/// The `reason` column of the data for a driver outcome.
pub fn outcome_name(o: &Wolfe1Outcome) -> &'static str {
    match o {
        Wolfe1Outcome::Converged => "converged",
        Wolfe1Outcome::Warning(r) => reason_name(r),
        Wolfe1Outcome::MaxIter => "maxiter",
        Wolfe1Outcome::Error(e) => start_error_name(e),
    }
}

// ---------------------------------------------------------------------------------------------
// Ask/tell drivers
// ---------------------------------------------------------------------------------------------

/// The five numbers of `MoreThuenteConfig::new`, kept readable so a test can check against them.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    pub ftol: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub stpmin: f64,
    pub stpmax: f64,
}

impl Setup {
    /// (mu, eta) with the settings of the paper's section 5 that the paper does not state and
    /// this suite assumes: `xtol = 1e-14`, `stpmin = 0`, `stpmax = 1e10`.
    pub fn new(mu: f64, eta: f64) -> Self {
        Setup {
            ftol: mu,
            gtol: eta,
            xtol: 1e-14,
            stpmin: 0.0,
            stpmax: 1e10,
        }
    }

    pub fn config(&self) -> MoreThuenteConfig {
        MoreThuenteConfig::new(self.ftol, self.gtol, self.xtol, self.stpmin, self.stpmax)
            .expect("test configuration must be valid")
    }
}

/// How a driven search ended.
#[derive(Debug)]
pub enum Ended {
    Done(Termination),
    /// SciPy's `maxiter` ran out: the last evaluated point was never told.
    MaxIter,
}

/// Bitwise equality of two points (`==` is false for NaN, which non-finite runs can produce).
pub fn same_point(a: Point, b: Point) -> bool {
    a.alpha.to_bits() == b.alpha.to_bits()
        && a.phi.to_bits() == b.phi.to_bits()
        && a.dphi.to_bits() == b.dphi.to_bits()
}

/// The observable state of a `MoreThuente` after a `tell`: `best()`, `bracket()`, `stage()`.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    pub best: Point,
    pub bracket: Option<(f64, f64)>,
    pub stage: Stage,
}

fn snap(ls: &MoreThuente) -> Snap {
    Snap {
        best: ls.best(),
        bracket: ls.bracket(),
        stage: ls.stage(),
    }
}

/// A driven search with its full trace.
#[derive(Debug)]
pub struct Run {
    pub ended: Ended,
    /// Every evaluated trial `(alpha, phi, phi')` in order, including an untested last one.
    pub trace: Vec<Trial>,
    /// State right after `start`.
    pub start: Snap,
    /// `snaps[k]` is the state after the tell of `trace[k]` (one fewer than `trace` on `MaxIter`).
    pub snaps: Vec<Snap>,
    pub phi0: f64,
    pub dphi0: f64,
}

/// Invariants every run of the suite must satisfy after every `tell`, checked by `drive`
/// (P9, P10, P11 of `properties.rs`; they are skipped once a trial value is not finite, because
/// the handling of non-finite phi is an open decision, issue #38 review 9.1).
///
/// Sources (Moré & Thuente, ACM TOMS 20(3), 1994): invariant (2.1) and Theorem 2.1 (p. 290),
/// Theorem 3.2 for the second stage (p. 297), the stage switch (p. 297-298), the 0.66 interval
/// shrink rule (p. 292-293).
struct Checker {
    mu: f64,
    eta: f64,
    stpmin: f64,
    stpmax: f64,
    phi0: f64,
    dphi0: f64,
    prev_best_phi: f64,
    switched: bool,
    finite: bool,
    widths: Vec<Option<f64>>,
}

impl Checker {
    fn new(setup: &Setup, phi0: f64, dphi0: f64) -> Self {
        Checker {
            mu: setup.ftol,
            eta: setup.gtol,
            stpmin: setup.stpmin,
            stpmax: setup.stpmax,
            phi0,
            dphi0,
            prev_best_phi: phi0,
            switched: false,
            finite: true,
            widths: Vec::new(),
        }
    }

    /// `trace` holds the trial just told as its last entry; `snap` is the state after that tell.
    /// `next` is the alpha asked next (`None` if the tell ended the search).
    fn after_tell(&mut self, trace: &[Trial], snap: &Snap, next: Option<f64>) {
        let k = trace.len() - 1;
        let (a, f, g) = trace[k];
        self.finite &= f.is_finite() && g.is_finite();
        if !self.finite {
            self.widths.push(None);
            return;
        }
        let (mu, phi0, dphi0) = (self.mu, self.phi0, self.dphi0);
        let gtest = mu * dphi0;
        let best = snap.best;

        // P9a: best satisfies (1.1), also in every Termination (Thm 2.1, invariant (2.1), p. 290)
        assert!(
            sufficient_decrease(phi0, dphi0, mu, best.alpha, best.phi),
            "P9a: best {best:?} violates (1.1) after tell {k}"
        );
        // P9b: best.phi never increases (MINPACK moves stx only when phi(stp) <= phi(stx))
        assert!(
            best.phi <= self.prev_best_phi,
            "P9b: best.phi rose from {:e} to {:e} at tell {k}",
            self.prev_best_phi,
            best.phi
        );
        self.prev_best_phi = best.phi;
        // P9c: best is (0, phi0, phi'0) or exactly one of the traced triples
        let start = Point {
            alpha: 0.0,
            phi: phi0,
            dphi: dphi0,
        };
        assert!(
            best == start
                || trace.iter().any(|&(a, f, g)| best
                    == Point {
                        alpha: a,
                        phi: f,
                        dphi: g
                    }),
            "P9c: best {best:?} is neither the start nor a traced trial"
        );
        // P10: stage switch exactly at the first tell with psi <= 0 and phi' >= 0 (p. 297-298)
        if !self.switched && sufficient_decrease(phi0, dphi0, mu, a, f) && g >= 0.0 {
            self.switched = true;
        }
        assert_eq!(
            snap.stage == Stage::Function,
            self.switched,
            "P10: stage {:?} after tell {k} (trial {a:e}); switch condition seen = {}",
            snap.stage,
            self.switched
        );

        let terminal = next.is_none();
        match snap.bracket {
            Some((lo, hi)) if !terminal => {
                self.widths.push(Some(hi - lo));
                // P9d: invariant (2.1) of the bracket [best, u]. A terminating tell returns before
                // dcstep updates the interval (while the stage may already have switched), so it
                // is excluded. In the auxiliary stage the psi form holds in Thm 2.1's setting
                // mu <= eta; for eta < mu before the switch it does not (counterexample: example
                // (3.1) with mu = 0.25, eta = 0.05, where psi'(best) > 0 with u > best), and the
                // paper claims it only for the second stage there (Thm 3.2).
                let u = if best.alpha == lo { hi } else { lo };
                assert!(
                    best.alpha == lo || best.alpha == hi,
                    "best is not a bracket end"
                );
                let (fu, _gu) = if u == 0.0 {
                    (phi0, dphi0)
                } else {
                    let &(_, fu, gu) = trace
                        .iter()
                        .find(|t| t.0 == u)
                        .unwrap_or_else(|| panic!("P9d: bracket end {u:e} is not a traced trial"));
                    (fu, gu)
                };
                if snap.stage == Stage::Function {
                    assert!(best.phi <= fu, "P9d: phi(best) > phi(u) in stage 2");
                    assert!(
                        best.dphi * (u - best.alpha) < 0.0,
                        "P9d: phi'(best)(u - best) >= 0 in stage 2"
                    );
                } else if mu <= self.eta {
                    assert!(
                        best.phi - best.alpha * gtest <= fu - u * gtest,
                        "P9d: psi(best) > psi(u) in stage 1"
                    );
                    assert!(
                        (best.dphi - gtest) * (u - best.alpha) < 0.0,
                        "P9d: psi'(best)(u - best) >= 0 in stage 1"
                    );
                }
                // P11: the width shrinks by 0.66 every two tells, or the next point is the bisection
                // point (_dcsrch.py: `abs(sty - stx) >= p66 * width1` => midpoint). Other legitimate
                // exits: the step reset to stx before a warning, and the clamp to [stpmin, stpmax].
                let w = hi - lo;
                if k >= 2 {
                    if let Some(w2) = self.widths[k - 2] {
                        if w > 0.66 * w2 {
                            let nxt = next.expect("non-terminal");
                            let bis = lo + 0.5 * (hi - lo);
                            let near = (nxt - bis).abs() <= 4.0 * spacing(nxt.abs().max(bis.abs()));
                            assert!(
                                near || nxt == best.alpha || nxt == self.stpmin || nxt == self.stpmax,
                                "P11: width {w:e} > 0.66 * {w2:e} at tell {k} but next {nxt:e} is not the \
                                 bisection point {bis:e}"
                            );
                        }
                    }
                }
            }
            _ => self.widths.push(None),
        }
    }
}

/// Drive `MoreThuente` with SciPy's `maxiter` counting (`DCSRCH.__call__`): `maxiter` iterations
/// include START, so at most `maxiter` trials are evaluated and only the first `maxiter - 1` are
/// told; the last evaluated point is never tested.
///
/// Properties of the paper's theory that must hold in every test of the suite are checked on every
/// call: every asked trial lies in `[stpmin, stpmax]` (P6a), every `Converged` result satisfies
/// (1.1) and (1.2) with the configured (mu, eta) exactly (P1), and after every tell the invariants
/// of `Checker` (P9-P11) and the termination claims (P12) hold.
pub fn drive(
    setup: &Setup,
    mut phi: impl FnMut(f64) -> f64,
    mut dphi: impl FnMut(f64) -> f64,
    phi0: f64,
    dphi0: f64,
    alpha0: f64,
    maxiter: usize,
) -> Result<Run, StartError> {
    let cfg = setup.config();
    let (mut ls, mut step) = MoreThuente::start(&cfg, phi0, dphi0, alpha0)?;
    let start = snap(&ls);
    assert_eq!(
        start.best,
        Point {
            alpha: 0.0,
            phi: phi0,
            dphi: dphi0
        },
        "best() right after start is (0, phi(0), phi'(0))"
    );
    assert_eq!(
        start.stage,
        Stage::Auxiliary,
        "the search starts in the auxiliary stage"
    );
    assert!(
        start.bracket.is_none(),
        "nothing is bracketed right after start"
    );
    let mut checker = Checker::new(setup, phi0, dphi0);
    let mut trace: Vec<Trial> = Vec::new();
    let mut snaps: Vec<Snap> = Vec::new();
    loop {
        match step {
            Step::Done(t) => {
                assert_eq!(
                    t.evaluations,
                    trace.len(),
                    "Termination.evaluations counts tell() calls"
                );
                let s = snap(&ls);
                assert!(
                    same_point(t.best, s.best),
                    "Termination.best is the state's best()"
                );
                snaps.push(s);
                checker.after_tell(&trace, &s, None);
                // P12 (see properties.rs): on every termination `best` satisfies (1.1), checked in
                // `after_tell` above (P9a); on Converged the returned alpha satisfies (1.1) and
                // (1.2). MINPACK returns stp *before* dcstep updates stx, so `best.alpha == alpha`
                // does not hold in general and is not asserted.
                if matches!(t.reason, Reason::Converged) {
                    assert_strong_wolfe(
                        phi0,
                        dphi0,
                        setup.ftol,
                        setup.gtol,
                        t.alpha,
                        t.phi,
                        t.dphi,
                        "Converged result (P1, P12)",
                    );
                }
                return Ok(Run {
                    ended: Ended::Done(t),
                    trace,
                    start,
                    snaps,
                    phi0,
                    dphi0,
                });
            }
            Step::Evaluate { alpha } => {
                assert!(
                    alpha >= setup.stpmin && alpha <= setup.stpmax,
                    "trial {alpha:e} outside [{:e}, {:e}] (P6a)",
                    setup.stpmin,
                    setup.stpmax
                );
                if let Some(&s) = snaps.last() {
                    checker.after_tell(&trace, &s, Some(alpha));
                }
                let (f, g) = (phi(alpha), dphi(alpha));
                trace.push((alpha, f, g));
                if trace.len() >= maxiter {
                    return Ok(Run {
                        ended: Ended::MaxIter,
                        trace,
                        start,
                        snaps,
                        phi0,
                        dphi0,
                    });
                }
                step = ls.tell(&cfg, f, g);
                snaps.push(snap(&ls));
            }
        }
    }
}

/// `run_traced(cfg, phi, dphi, alpha0)`: run to termination and return the termination with the
/// full trial trace. Evaluates `phi(0)` and `dphi(0)` itself. Panics on a start error or if the
/// search does not terminate within 1000 evaluations.
pub fn run_traced(
    setup: &Setup,
    mut phi: impl FnMut(f64) -> f64,
    mut dphi: impl FnMut(f64) -> f64,
    alpha0: f64,
) -> (Termination, Vec<Trial>) {
    let (phi0, dphi0) = (phi(0.0), dphi(0.0));
    let run = drive(setup, phi, dphi, phi0, dphi0, alpha0, 1000)
        .unwrap_or_else(|e| panic!("start error {}", start_error_name(&e)));
    match run.ended {
        Ended::Done(t) => (t, run.trace),
        Ended::MaxIter => panic!("no termination within 1000 evaluations"),
    }
}

/// `run_traced` for a named function of `eval`.
pub fn run_named(setup: &Setup, name: &str, p: &[f64], alpha0: f64) -> (Termination, Vec<Trial>) {
    run_traced(
        setup,
        |a| phi_of(name, p, a),
        |a| dphi_of(name, p, a),
        alpha0,
    )
}
