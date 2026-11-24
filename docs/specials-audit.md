# Numerical audit guide for `specials`

Apply this guide when implementing or reviewing a change to a special-function
kernel, its public dispatch, its reference tests, or a dependency whose
functions the crate re-exports. The repository's
[numerical validation rules](../REPOSITORY_RULES.md#numerical-validation)
also apply.

## Map the public paths

Start at the public API and record the supported domain, result conventions,
accuracy target, implementation branches, and upstream source revision. Follow
the dispatched `f64` and `Df64` paths through shared helpers, dependency calls,
and fallbacks. Check the actual feature configuration. A test of the adaptive
kernel instantiated with `f64` exercises a different path from the public
`f64` dispatch. For a re-export, exercise the public entry point through
`specials` and record the dependency revision.

## Pick inputs at the boundaries

Keep a small set of representative inputs for each regime. For each applicable
boundary, also include the following.

| Boundary | Inputs and properties to check |
| --- | --- |
| Algorithm switch | Test the closest representable input below, at, and above the switch. In `Df64`, include a normalized pair for which changing only `lo` crosses the switch. For a predicate involving multiple arguments, vary their ratio or product across the boundary and repeat at different scales. |
| Domain, poles and zeros | Test poles and zeros, their nearest neighbors on both sides, signed zeros, infinities, NaN, and points just outside the supported domain. For complex domains include both sides of a branch cut when applicable. Verify the sign, classification or branch value specified by the API. |
| Numeric range | Span ordinary values, large exponent spread, min normal, min subnormal, rounding to zero, and the finite-to-infinite transition. Vary overall scale and exponent spread independently. Include cases where intermediates overflow or underflow while the final value remains representable. |
| Transformations and symmetries | Exercise permutations, reflection, recurrence, periodicity, and homogeneity where they apply. For a principal value, test configurations with strong cancellation as well as typical inputs, including the same ratios at different scales. |
| Termination | Exercise easy and difficult convergence, the iteration cap and the fallback or reported error. Include inputs so large that an update such as `x + 1` can round back to `x`. |
| Coupled regimes | Vary parameters together when the expansion depends on a combined variable. For example, sweep the Jacobi phase as well as the parameter near its endpoints. Check identities between the returned components alongside an independent reference. |

Use a fixed-seed exponent or branch sweep for discovery. Keep the smallest
deterministic reproducer for each defect and preserve neighboring passing cases
where they pin a boundary. A random sweep complements this set.

## Verify the reference and the error metric

- Reconstruct the actual binary input in the oracle. For `Df64`, retain both
  words and use enough working precision to add them exactly. Preserve signed
  zeros and input bits in reproductions where they matter.
- Recompute a reference at higher working precision and check agreement at
  the resolution needed by the assertion. When the oracle is ill-conditioned,
  use a second formula or an exact identity. Examples include an integer
  recurrence for Stirling numbers, a direct finite power sum for a short
  generalized harmonic sum, and a high-precision transformation for a Carlson
  principal value. A cancellation-prone difference in an oracle needs its own
  validation.
- Check result classification before computing an error. Verify signed
  zeros, NaN and infinities according to the documented contract. For a
  representable subnormal use error in units of the smallest subnormal;
  a loose underflow bound can hide a zero or the wrong sign. For normal
  `Df64` results compare `hi + lo` and account for the spacing of a
  subnormal low word when it limits accuracy.
- Use relative error or ULPs for well-scaled results, with a justified
  absolute or condition-aware bound at zeros and sensitive regimes. Record
  the bound with the assertion. Keep mathematical accuracy and SciPy/xsf
  parity independently visible. Identify known upstream disagreements in
  [the upstream findings log](upstream-issues.md).
- Store fixtures and versioned regeneration scripts with the tests. A
  regression test for a fix must fail on the old behavior. Record skipped
  cases and why they are skipped. Keep public `f64` tests runnable without
  `xprec`; put the feature gate on assertions that need `Df64` and confirm
  the relevant test names run in both configurations.

## Check numerical range and computational work

Trace exponent bounds through the evaluated expressions, including products,
squares, reciprocals, log/exp pairs, sums of opposite sign, and the final
rescaling. Check before the first unsafe intermediate. Use exact power-of-two
scaling or a stable equivalent formula when it preserves the relevant bits.
Inspect dependence on argument spread, subnormal classification, loss of
phase, and platform libm. Check an error or nonconvergence path separately
from genuine overflow of the final result.

For each important path, record how work grows with the inputs. Inspect
iteration bounds, costly primitives, allocations, and duplicate evaluations
on fallback paths. Compare ordinary, boundary and worst supported regimes.
Use deterministic work counts or a progress test for termination regressions.
For timing claims, follow the shared
[benchmark rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/benchmark/index.md).
Pin the revision, build mode and case matrix. Verify the timed results against
the same contract. Report source-level cost findings and timeouts with their
limits when an isolated timing comparison is unavailable.

## Record review evidence

For each finding, give the public API, scalar type and feature set, exact
input, observed value or status, reference and tolerance, relevant source
location, and a command or committed test that reproduces it. State the
tested revision, dependency and oracle versions, and host for platform-sensitive
results. A reviewer should check the smallest failing case and the neighboring
regime, and confirm that the test would reject a bad result as well as a bad
classification. Use the existing contribution rules for PR evidence and keep
larger diagnostic sweeps reproducible outside the fast test suite.
