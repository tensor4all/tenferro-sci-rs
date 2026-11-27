//! The numerical results of Moré & Thuente, "Line Search Algorithms with Guaranteed Sufficient
//! Decrease", ACM TOMS 20(3):286-307, 1994, section 5, Tables I-VI, as ground truth, plus the
//! variants its text states. Independent of SciPy: the numbers below are the paper's own.
//!
//! The paper text extracted for this suite is `mt.txt` (line numbers below refer to that
//! extraction): Table I at line 1047, II 1061, III 1075, IV 1091, V 1107, VI 1125. The functions
//! are (5.1) -(5.4); the search runs `MoreThuente` with the table's (mu, eta) = (ftol, gtol) and
//! alpha0 = 10^-3, 10^-1, 10^1, 10^3. The paper does not state stpmin, stpmax or xtol; this suite
//! assumes stpmin = 0, stpmax = 1e10, xtol = 1e-14 (the results do not depend on them).
//!
//! Each row asserts: `Reason::Converged` (Info = 1), the number of function evaluations m,
//! alpha_m equal to the printed value at its printed precision (half a unit of the last printed
//! digit), phi'(alpha_m) equal to the printed value to its two significant digits (half a unit of
//! the second digit, sign included), and the strong Wolfe conditions (1.1), (1.2) exactly. The
//! 24 rows were checked against SciPy 1.16.3's DCSRCH when the data was generated: 23 agree on
//! all of m, alpha_m and phi'(alpha_m); the one difference is Table II, alpha0 = 1e-3, below.
//!
//! The Table values are transcribed once, in `TABLES`. They were first read from the OCR text
//! extraction `mt.txt` and the two differences from SciPy below were then checked against the
//! rendered PDF pages: both are genuine differences, not OCR slips. 23 of the 24 rows match SciPy on m,
//! alpha_m and phi'(alpha_m) including signs; the other is Table II, alpha0 = 1e-3, whose printed
//! phi' = 7.1e-9 (p. 304) differs from SciPy's 3.8e-9. The second difference is in the text: p. 303
//! says 8 evaluations for function 1 with mu = 0.1, eta = 0.001, alpha0 = 1e-3, SciPy needs 9
//! (`function_1_with_eta_0_001`).

mod common;

use common::*;

/// (mu, eta, function, [(m, alpha_m printed, decimals printed, phi'(alpha_m) printed)] per alpha0)
type Row = (usize, f64, i32, f64);
struct Table {
    name: &'static str,
    line: usize,
    func: &'static str,
    mu: f64,
    eta: f64,
    rows: [Row; 4],
}

const A0S: [f64; 4] = [1e-3, 1e-1, 1e1, 1e3];

#[rustfmt::skip]
const TABLES: [Table; 6] = [
    Table { name: "Table I (mt1, (5.1) beta = 2)", line: 1047, func: "mt1", mu: 1e-3, eta: 1e-1,
        rows: [(6, 1.4, 1, -9.2e-3), (3, 1.4, 1, 4.7e-3), (1, 10.0, 0, 9.4e-3), (4, 37.0, 0, 7.3e-4)] },
    Table { name: "Table II (mt2, (5.2) beta = 0.004)", line: 1061, func: "mt2", mu: 1e-1, eta: 1e-1,
        rows: [(12, 1.6, 1, 7.1e-9), (8, 1.6, 1, 1.0e-10), (8, 1.6, 1, -5.0e-9), (11, 1.6, 1, -2.3e-8)] },
    Table { name: "Table III (mt3, (5.3) beta = 0.01, l = 39)", line: 1075, func: "mt3", mu: 1e-1, eta: 1e-1,
        rows: [(12, 1.0, 1, -5.1e-5), (12, 1.0, 1, -1.9e-4), (10, 1.0, 1, -2.0e-6), (13, 1.0, 1, -1.6e-5)] },
    Table { name: "Table IV (mt4, (5.4) beta1 = beta2 = 0.001)", line: 1091, func: "mt4", mu: 1e-3, eta: 1e-3,
        rows: [(4, 0.08, 2, -6.9e-5), (1, 0.10, 2, -4.9e-5), (3, 0.35, 2, -2.9e-6), (4, 0.83, 2, 1.6e-5)] },
    Table { name: "Table V (mt5, (5.4) beta1 = 0.01, beta2 = 0.001)", line: 1107, func: "mt5", mu: 1e-3, eta: 1e-3,
        rows: [(6, 0.075, 3, 1.9e-4), (3, 0.078, 3, 7.4e-4), (7, 0.073, 3, -2.6e-4), (8, 0.076, 3, 4.5e-4)] },
    Table { name: "Table VI (mt6, (5.4) beta1 = 0.001, beta2 = 0.01)", line: 1125, func: "mt6", mu: 1e-3, eta: 1e-3,
        rows: [(13, 0.93, 2, 5.2e-4), (11, 0.93, 2, 8.4e-5), (8, 0.92, 2, -2.4e-4), (11, 0.92, 2, -3.2e-4)] },
];

/// Half a unit of the last printed digit, plus 1e-9 for the exact-tie case (see Table IV).
fn alpha_tol(decimals: i32) -> f64 {
    0.5 * 10f64.powi(-decimals) + 1e-9
}

/// Half a unit of the second significant digit of a value printed with two digits.
fn dphi_tol(printed: f64) -> f64 {
    // decimal exponent from the scientific notation, not from log10 (floor(log10(1e-10)) is fragile)
    let exp: i32 = format!("{:e}", printed.abs())
        .split('e')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    0.05 * 10f64.powi(exp)
}

#[test]
fn tables_i_to_vi() {
    for t in &TABLES {
        let setup = cfg(t.mu, t.eta);
        for (a0, &(m, alpha_p, decimals, dphi_p)) in A0S.iter().zip(&t.rows) {
            let ctx = format!("{} (mt.txt line {}), alpha0 = {a0:e}", t.name, t.line);
            let (term, trace) = run_named(&setup, t.func, &[], *a0);
            assert!(
                term.reason.is_converged(),
                "{ctx}: Info must be 1, got {:?}",
                term.reason
            );
            assert_eq!(
                term.evaluations, m,
                "{ctx}: number of function evaluations m"
            );
            assert_eq!(trace.len(), m, "{ctx}");
            assert!(
                (term.alpha - alpha_p).abs() <= alpha_tol(decimals),
                "{ctx}: alpha_m = {} vs printed {alpha_p}",
                term.alpha
            );
            // phi'(alpha_m): printed value to two significant digits, sign included. The result's
            // phi' must also be phi' at alpha_m.
            assert_eq!(
                term.dphi,
                dphi_of(t.func, &[], term.alpha),
                "{ctx}: Termination.dphi"
            );
            if t.func == "mt2" && *a0 == 1e-3 {
                // Table II, alpha0 = 1e-3 prints phi'(alpha_m) = 7.1e-9; SciPy 1.16.3 (and this
                // code, which replays SciPy) gives 3.8e-9. The printed value is genuine
                // (checked on the rendered page, p. 304), not an OCR slip, and is not reproduced there. The set of acceptable alpha for this function is an interval of length
                // about 2.5e-9 around 1.596 (the paper, p. 305, "approximate length 2.5 10^-9"),
                // across which phi' changes by about 5e-8, so a last interpolation step that
                // differs in the last digits lands at a different phi'; both values satisfy (1.2)
                // by a wide margin (eta |phi'(0)| is about 5e-8). The cause of the difference
                // (SciPy's MINPACK-2 vs the authors' code) is not known.
                // Asserted instead: the robust claim (|phi'| <= eta |phi'(0)|, sign, order of magnitude).
                let dphi0 = dphi_of(t.func, &[], 0.0);
                assert!(term.dphi.abs() <= t.eta * dphi0.abs(), "{ctx}: (1.2)");
                assert_eq!(
                    term.dphi.signum(),
                    dphi_p.signum(),
                    "{ctx}: sign of phi'(alpha_m)"
                );
                let ratio = term.dphi.abs() / dphi_p.abs();
                assert!(
                    (0.1..10.0).contains(&ratio),
                    "{ctx}: order of magnitude, ratio {ratio}"
                );
            } else {
                assert!(
                    (term.dphi - dphi_p).abs() <= dphi_tol(dphi_p),
                    "{ctx}: phi'(alpha_m) = {:e} vs printed {dphi_p:e}",
                    term.dphi
                );
            }
            // strong Wolfe, (1.1) and (1.2), exactly, with the table's (mu, eta)
            let (phi0, dphi0) = eval(t.func, &[], 0.0);
            let (phi_a, dphi_a) = eval(t.func, &[], term.alpha);
            assert_strong_wolfe(phi0, dphi0, t.mu, t.eta, term.alpha, phi_a, dphi_a, &ctx);
        }
    }
}

#[test]
fn table_iv_alpha_is_a_half_unit_tie() {
    // Table IV, alpha0 = 1e-3 prints alpha_m = 0.08, but the iterates are 0.005, 0.021, 0.085 (the
    // extrapolation sequence quoted on p. 304, mt.txt line 1143) and the 4th evaluation is accepted at 0.085:
    // exactly half a unit of the printed digit. The paper rounded 0.085 down; `alpha_tol` allows the tie.
    let (term, trace) = run_named(&cfg(1e-3, 1e-3), "mt4", &[], 1e-3);
    assert_eq!(term.evaluations, 4);
    assert!((term.alpha - 0.085).abs() < 1e-12, "{}", term.alpha);
    assert!((trace[2].0 - 0.021).abs() < 5e-4);
}

// ---- the paper's text: variants of Table I, II and VI ----------------------------------------

#[test]
fn function_1_with_mu_eta_equal_to_0_1() {
    // p. 302 (mt.txt line 978): with mu = eta = 0.1 the search "terminates with alpha_m = 1.6 when
    // alpha0 = 10 and with alpha_m = 1.6 when alpha0 = 10^3. There is no change in the behavior of
    // the algorithm from the other two starting points", i.e. Table I's m = 6 and 3, alpha_m = 1.4.
    // SciPy: alpha_m = 1.6496 (m = 3) and 1.6278 (m = 7), within the printed one decimal
    // (|alpha - 1.6| < 0.05).
    let s = cfg(0.1, 0.1);
    for a0 in [10.0, 1e3] {
        let (t, _) = run_named(&s, "mt1", &[], a0);
        assert!(t.reason.is_converged());
        assert!(
            (t.alpha - 1.6).abs() < 0.05,
            "alpha0 = {a0:e}: alpha_m = {}",
            t.alpha
        );
    }
    for (a0, m) in [(1e-3, 6), (1e-1, 3)] {
        let (t, _) = run_named(&s, "mt1", &[], a0);
        assert_eq!(
            t.evaluations, m,
            "unchanged from Table I at alpha0 = {a0:e}"
        );
        assert!(
            (t.alpha - 1.4).abs() < 0.05,
            "alpha0 = {a0:e}: alpha_m = {}",
            t.alpha
        );
    }
}

#[test]
fn function_1_with_eta_0_001() {
    // p. 303 (mt.txt lines 1034-1038): mu = 0.1, eta = 0.001: "the final iterate alpha_m is near
    // alpha* for all starting points. ... 6 function evaluations for alpha0 = 10 and ten for
    // alpha0 = 10^3. The number ... for alpha0 = 10^-3 and 10^-1 is, respectively, 8 and 4."
    // alpha* = sqrt(2) = 1.4142 (the minimizer of (5.1) with beta = 2).
    // mu = 0.1 > eta = 0.001: the case of Theorem 3.3 (the search switches to phi, stage 2).
    let s = cfg(0.1, 1e-3);
    for (a0, m) in [(1e-1, 4), (1e1, 6), (1e3, 10)] {
        let (t, _) = run_named(&s, "mt1", &[], a0);
        assert!(t.reason.is_converged(), "alpha0 = {a0:e}");
        assert_eq!(t.evaluations, m, "alpha0 = {a0:e}");
        assert!(
            (t.alpha - 2f64.sqrt()).abs() < 0.01,
            "alpha0 = {a0:e}: {}",
            t.alpha
        );
    }
    // alpha0 = 1e-3: the paper counts 8 (p. 303, checked on the rendered page); SciPy 1.16.3 gives 9. The difference of one evaluation is
    // unexplained (the authors' code vs SciPy's MINPACK-2 port); the port must give SciPy's 9
    // (also checked by scipy_replay.rs), so that is what is asserted, and the paper's 8 is
    // recorded here as the documented difference.
    let (t, _) = run_named(&s, "mt1", &[], 1e-3);
    assert!(t.reason.is_converged());
    assert_eq!(t.evaluations, 9, "SciPy's count; the paper says 8");
    assert!((t.alpha - 2f64.sqrt()).abs() < 0.01);
}

#[test]
fn function_6_with_mu_0_001_and_eta_0_1() {
    // p. 306 (mt.txt lines 1204-1207): "the number of function evaluations needed to obtain an
    // acceptable alpha from the starting points alpha0 = 10^i for i = -3, -1, 1, and 3 would be,
    // respectively, 2, 1, 3, and 4" (function 6 = Figure 8, mu = 0.001, eta = 0.1).
    let s = cfg(1e-3, 1e-1);
    for (a0, m) in [(1e-3, 2), (1e-1, 1), (1e1, 3), (1e3, 4)] {
        let (t, _) = run_named(&s, "mt6", &[], a0);
        assert!(t.reason.is_converged(), "alpha0 = {a0:e}");
        assert_eq!(t.evaluations, m, "alpha0 = {a0:e}");
    }
}

#[test]
fn function_2_is_unchanged_for_any_mu_below_eta() {
    // p. 305 (mt.txt line 1161): "these results remain unchanged if we set eta = 0.1 and choose any
    // mu < eta", i.e. Table II (m = 12, 8, 8, 11 and alpha_m = 1.6) for mu = 0.01 and 0.001.
    // SciPy 1.16.3 reproduces the counts and steps (1.596) for both.
    let ms = [12, 8, 8, 11];
    for mu in [1e-2, 1e-3] {
        let s = cfg(mu, 1e-1);
        for (a0, m) in A0S.iter().zip(ms) {
            let (t, _) = run_named(&s, "mt2", &[], *a0);
            assert!(t.reason.is_converged(), "mu = {mu:e}, alpha0 = {a0:e}");
            assert_eq!(t.evaluations, m, "mu = {mu:e}, alpha0 = {a0:e}");
            assert!(
                (t.alpha - 1.6).abs() < 0.05,
                "mu = {mu:e}, alpha0 = {a0:e}: {}",
                t.alpha
            );
        }
    }
}
