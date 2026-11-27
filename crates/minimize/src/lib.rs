//! Scalar line searches in pure Rust.
//!
//! This crate will hold `minimize::linesearch`, a port of SciPy's Moré–Thuente
//! strong-Wolfe line search (`scipy.optimize._dcsrch.DCSRCH` and
//! `scipy.optimize.scalar_search_wolfe1`). The implementation follows in this
//! PR; the integration tests in `tests/` are written first, against the planned
//! API, and fail to compile until it lands.
