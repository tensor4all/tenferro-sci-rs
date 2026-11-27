//! `dcstep`: the safeguarded step and interval update of Moré–Thuente §4.
//!
//! Stub: the step computation lands next.

#![allow(dead_code)] // stub commit; removed when dcstep lands

use crate::real::Real;

/// A point `(α, φ(α), φ'(α))` in the units of the caller (also the ψ form of §3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pt<T> {
    pub a: T,
    pub f: T,
    pub g: T,
}

/// Which branch of §4 (p. 299–300) produced the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Case {
    /// Case 1: a higher function value.
    HigherValue,
    /// Case 2: a lower value and derivatives of opposite sign.
    OppositeSlopes,
    /// Case 3: a lower value, same sign, the magnitude of the derivative decreases.
    DecreasingSlope,
    /// Case 4: a lower value, same sign, the magnitude does not decrease.
    SteepSlope,
}

/// Result of [`dcstep`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct DcStep<T> {
    pub x: Pt<T>,
    pub y: Pt<T>,
    pub stp: T,
    pub brackt: bool,
    pub case: Case,
}

/// One safeguarded step (stub: returns its inputs).
pub(crate) fn dcstep<T: Real>(
    x: Pt<T>,
    y: Pt<T>,
    t: Pt<T>,
    brackt: bool,
    _stpmin: T,
    _stpmax: T,
) -> DcStep<T> {
    DcStep {
        x,
        y,
        stp: t.a,
        brackt,
        case: Case::SteepSlope,
    }
}
