//! Limited-memory BFGS: the unconstrained path of SciPy's `L-BFGS-B` (`bounds=None`), as an ask/tell state machine.
//!
//! **Red stage (tensor4all/tenferro-sci-rs#40).** This module holds the public types and stubs the tests are
//! written against; the iteration itself is not implemented yet, so the behavioural tests fail by design.
//!
//! # Scalar type
//!
//! Everything is `f64` for now, and `LbfgsConfig` is not generic. This is provisional: it becomes generic over the
//! scalar when the scalar trait of #38 is public and a second precision has parity data; the migration is replacing
//! `f64` by `V::Real` in the signatures of [`LbfgsConfig`], [`LbfgsState`] and [`minimize`]. See also the contracts
//! of [`Vector`](crate::Vector).
//!
//! # Provenance
//!
//! *Written independently from the published algorithm:* the two-loop recursion and the ring buffer of the stored
//! pairs follow J. Nocedal, *Updating quasi-Newton matrices with limited storage*, Math. Comp. 35(151), 773–782, 1980
//! ([doi:10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7)), and chapter 7 of
//! J. Nocedal and S. J. Wright, *Numerical Optimization*, Springer, 2006 ("Large-Scale Unconstrained Optimization";
//! doi:10.1007/978-0-387-40065-5). No source code was translated for them.
//!
//! *Derived from SciPy by reading its source:* the driver semantics (the order of the stopping tests, the counting of
//! evaluations, the restart after a failed line search, the scaling `γ = sᵀy / yᵀy`, the first trial step) were read
//! from SciPy v1.16.3 (commit `b9105ccc2237f57acb1060202cd77f6dd264fb34`), `scipy/optimize/__lbfgsb.c` (`mainlb`,
//! `lnsrlb`) and `scipy/optimize/_lbfgsb_py.py` (`_minimize_lbfgsb`), and the code is validated against recorded runs
//! of that version. The Rust control flow follows SciPy's, but it is not a line-by-line translation: SciPy works in the
//! compact representation with bounds, this is the two-loop recursion on the unconstrained path. SciPy is BSD-3-Clause
//! and the L-BFGS-B code it translates is by Zhu, Byrd, Lu, Nocedal and Morales; the NOTICE for that chain is added
//! with the implementation. Optim.jl and LineSearches.jl were read as references only; nothing is ported.

mod config;
mod driver;
// Red stage: the stubs of `memory` and `twoloop` are not called by the (still stubbed) state machine yet.
#[allow(dead_code)]
pub(crate) mod memory;
mod state;
#[allow(dead_code)]
pub(crate) mod twoloop;

pub use config::{ConfigError, LbfgsConfig, H0};
pub use driver::{minimize, Report};
pub use state::{LbfgsState, Step, Termination};
