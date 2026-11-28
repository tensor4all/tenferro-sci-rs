//! Per-step replay: the primary parity claim of BFGS.
//!
//! Dense BFGS carries an `n × n` matrix forward, so BLAS rounding differences between platforms and implementations accumulate
//! from the first update and no *trajectory* bound can hold for long (the first Linux run of the data showed 6 of 16 runs
//! inside a trajectory prefix). One *step* has no such history. For every iteration `k` of every parity run, `steps.csv` holds
//! SciPy's own state (`H_k` before the update, the pair `s_k = α p_k`, `y_k = g_{k+1} − g_k`, SciPy's `H_{k+1}` and `g_{k+1}`,
//! SciPy's next direction `p_{k+1}`) and the exact (60-digit mpmath) results from the same inputs. This test feeds SciPy's
//! `(H_k, s_k, y_k)` to [`minimize::bfgs::inverse_update`] and SciPy's `(H_{k+1}, g_{k+1})` to
//! [`minimize::bfgs::search_direction`], and asserts, for both, the result against SciPy's and against the exact one, within the
//! **derived single-step bound** of the data README (a standard rank-2 update error analysis, `bfgs_common::step_bounds`; no
//! constant is fitted; the generator asserts that SciPy itself satisfies it against the exact result on every step, with the
//! observed ratios printed). Every iteration with an update is covered (the converging step ends the run before its update);
//! there is no prefix.
//!
//! | Claim | Bound |
//! |---|---|
//! | Rust `H_{k+1}` vs the exact update (60 digits) | `(2κ+4) γ_{n+6} M + |ρ||s||w|ᵀ` |
//! | Rust `H_{k+1}` vs SciPy's | that plus SciPy's own `(2κ+4) γ_{2n+6} S` |
//! | Rust `p_{k+1}` vs the exact `−H g` of SciPy's `H_{k+1}` | `γ_n |H||g|` |
//! | Rust `p_{k+1}` vs SciPy's | `2 γ_n |H||g|` |
//!
//! Status in the red stage: the stubs leave `H` unchanged and return a zero direction, so every replay fails; the coverage
//! test passes (it reads the data).

mod bfgs_common;

use bfgs_common::*;
use colmat::Mat;
use minimize::bfgs::{inverse_update, search_direction};

fn mat(h: &[f64], n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| h[i * n + j])
}

fn flat(m: &Mat<f64>, n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            v.push(m[(i, j)]);
        }
    }
    v
}

fn check_run(id: &str) {
    let run = run_by_id(id);
    assert!(run.parity, "{id}: not a parity run");
    assert_eq!(
        run.steps.len(),
        run.nit
            .saturating_sub(usize::from(run.steps.len() < run.nit && run.nit > 0))
            .max(run.steps.len())
            .min(run.nit),
        "{id}: a step per iteration with an update"
    );
    let mut errors: Vec<String> = Vec::new();
    for st in &run.steps {
        let n = st.n;
        let (tol_m, tol_s, _kappa) = step_bounds(&st.h, &st.s, &st.y);
        // the update
        let mut h = mat(&st.h, n);
        if let Err(e) = inverse_update(&mut h, &st.s, &st.y) {
            errors.push(format!(
                "{id} k={}: inverse_update refused SciPy's pair: {e}",
                st.k
            ));
            continue;
        }
        let got = flat(&h, n);
        let both: Vec<f64> = tol_m.iter().zip(&tol_s).map(|(a, b)| a + b).collect();
        if let Some(m) = within(
            &format!("{id} k={}: H_(k+1) vs SciPy", st.k),
            &got,
            &st.h_next,
            &both,
            false,
        ) {
            errors.push(m);
        }
        if let Some(m) = within(
            &format!("{id} k={}: H_(k+1) vs exact", st.k),
            &got,
            &st.h_exact,
            &tol_m,
            true,
        ) {
            errors.push(m);
        }
        // the direction
        let hn = mat(&st.h_next, n);
        let p =
            search_direction(&hn, &st.g_next).unwrap_or_else(|e| panic!("{id} k={}: {e}", st.k));
        let abs_h: Vec<f64> = st.h_next.iter().map(|t| t.abs()).collect();
        let abs_g: Vec<f64> = st.g_next.iter().map(|t| t.abs()).collect();
        let scale = matvec(&abs_h, n, &abs_g);
        let tol_p: Vec<f64> = scale.iter().map(|t| gamma(n) * t).collect();
        let tol_p2: Vec<f64> = tol_p.iter().map(|t| 2.0 * t).collect();
        if let Some(m) = within(
            &format!("{id} k={}: p_(k+1) vs SciPy", st.k),
            &p,
            &st.p_next,
            &tol_p2,
            false,
        ) {
            errors.push(m);
        }
        if let Some(m) = within(
            &format!("{id} k={}: p_(k+1) vs exact", st.k),
            &p,
            &st.p_exact,
            &tol_p,
            true,
        ) {
            errors.push(m);
        }
    }
    eprintln!("{id}: {} iterations replayed per step", run.steps.len());
    assert!(
        errors.is_empty(),
        "{id}: {} failing check(s); first:\n  {}",
        errors.len(),
        errors
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

macro_rules! step_tests {
    ($($name:ident),* $(,)?) => {
        $( #[test] fn $name() { check_run(stringify!($name)); } )*
    };
}

step_tests!(
    rosen2_a,
    rosen2_b,
    rosen2_c,
    rosen2_tight,
    rosen5,
    himmelblau,
    beale,
    powell,
    quad5,
    quad10_k1e2,
    quad10_k1e4,
    stationary,
    rosen2_xrtol,
    rosen2_maxiter_eq_nit,
    rosen2_maxiter1,
    rosen2_maxiter5,
);

/// Every iteration with an update of every parity run is in the data, once, in order; the data is what the README says.
#[test]
fn every_iteration_with_an_update_is_covered() {
    let mut total = 0;
    for r in load_runs().into_iter().filter(|r| r.parity) {
        assert_eq!(
            r.steps.len(),
            r.hess.as_ref().map_or(0, |h| h.n_updates),
            "{}: one step per applied update",
            r.id
        );
        for (i, st) in r.steps.iter().enumerate() {
            assert_eq!(
                st.k, i,
                "{}: steps are the iterations 0, 1, ... in order",
                r.id
            );
            assert_eq!(st.n, r.n);
            assert_eq!(st.h.len(), r.n * r.n);
        }
        total += r.steps.len();
    }
    assert!(total >= 300, "{total} iterations covered");
}
