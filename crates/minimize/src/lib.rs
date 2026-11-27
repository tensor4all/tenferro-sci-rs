//! Scalar line searches in pure Rust.
//!
//! This crate holds `minimize::linesearch`, a port of SciPy's Moré–Thuente
//! strong-Wolfe line search (`scipy.optimize._dcsrch.DCSRCH` and
//! `scipy.optimize.scalar_search_wolfe1`).

pub mod linesearch;
mod real;
