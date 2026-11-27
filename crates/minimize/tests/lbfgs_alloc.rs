//! An L-BFGS iteration allocates nothing after the state is created (design requirement: workspace is allocated by
//! cloning `x0` in `LbfgsState::new`, `clone_from` reuses it).
//!
//! Validates: with a counting global allocator, the allocations made by the ask/tell loop of a Rosenbrock run on
//! `Vec<f64>` after `new` are exactly zero, over at least 20 iterations with the memory full
//! (m = 3) and at least one line search with more than one trial. The test lives in its own binary because a
//! `#[global_allocator]` is per binary; the counter only counts the thread that switches it on.
//!
//! Source: tensor4all-meta `docs/design/tenferro-sci.md` § Optimizers ("an iteration allocates nothing").

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use minimize::lbfgs::{LbfgsConfig, LbfgsState, Step, H0};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
}

fn note() {
    // `try_with`: the thread-local may be gone during thread teardown
    if ON.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
    }
}

// SAFETY: forwards every call to `System` unchanged; the counter has no effect on the memory handed out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Rosenbrock without allocating: writes the gradient into `g`.
fn rosen(x: &[f64], g: &mut [f64]) -> f64 {
    g.iter_mut().for_each(|t| *t = 0.0);
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let t = x[i + 1] - x[i] * x[i];
        f += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
        g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
        g[i + 1] += 200.0 * t;
    }
    f
}

#[test]
fn iterations_allocate_nothing_after_construction() {
    let cfg = LbfgsConfig::new(3, 1e-8, 0.0, 200, 400, 20, H0::ScaledGamma).expect("valid");
    let x0 = vec![-1.2, 1.0, -1.2, 1.0, -1.2];
    let mut g = vec![0.0; 5];
    let mut state = LbfgsState::new(&cfg, &mut (), x0.clone());

    ON.with(|c| c.set(true));
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let mut told = 0;
    while let Step::Evaluate { x } = state.ask() {
        let f = rosen(x, &mut g);
        state.tell(&cfg, &mut (), f, &g);
        told += 1;
    }
    let after = ALLOCATIONS.load(Ordering::SeqCst);
    ON.with(|c| c.set(false));

    assert!(
        state.iterations() >= 20,
        "only {} iterations: the loop must run",
        state.iterations()
    );
    assert!(
        told > state.iterations() + 1,
        "at least one line search needed more than one trial"
    );
    assert_eq!(
        after - before,
        0,
        "{} allocations in {told} evaluations",
        after - before
    );
}
