# Numerical audit guide for `specials`

When this guide applies is stated once, in the repository's
[numerical validation rules](../REPOSITORY_RULES.md#numerical-validation),
which also apply.

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
  cases and why they are skipped.
- Check which configuration actually runs a test before citing it. Until
  #30, the five suites with the public-API `f64` reference and parity rows
  (`crates/specials/tests/reference.rs`, `xsref.rs`, `xsf_regressions.rs`,
  `review_regressions.rs`, `carlson_scaling.rs`) were gated as a whole file by
  `#![cfg(feature = "xprec")]` and compiled to zero tests under
  `cargo test -p specials --no-default-features`. They now gate only their
  `Df64` items, so their `f64` rows run in both configurations (counts at
  #30: 64 tests in those five files without `xprec`, 73 with it). The tests
  that still need `xprec`, and why:

  | File | Tests that need `xprec` | Why |
  | --- | --- | --- |
  | `reference.rs` | `df64_matches_mpmath`, `stirling2_df64_beyond_the_work_limit_is_exact`, `df64_elliprj_ratio_limit_is_fail_fast` | Their tolerances (1e-29 to 1e-30) and the `R_J` ratio limit are statements about `Df64`. `special_values_match_scipy` runs its `f64` half without the feature. |
  | `xsref.rs` | `df64_matches_mpmath_on_xsref_points`, `df64_matches_xsref_where_mpmath_is_undefined` | They check the double-double results; `f64_matches_xsref` covers every xsref row in `f64`. |
  | `xsf_regressions.rs` | none | Every test runs its `f64` case; `for_both!` adds the `Df64` case with the feature. |
  | `review_regressions.rs` | `n4_digammainv_df64_holds_1e_30_below_minus_1e154`, `n7_stirling2_df64_keeps_the_low_word_of_large_arguments`, `n8_poch_df64_subnormal_a`, `strategies_agree_on_extreme_arguments` | `n4`, `n7` and `n8` are `Df64`-only findings with 1e-29/1e-30 tolerances; the agreement grid compares `f64` with `Df64`. The other seven tests run their `f64` assertions. |
  | `carlson_scaling.rs` | none | Every test checks `f64`; the `Df64` half of each needs the feature. |
  | `carlson_ratchet.rs` | `df64_homogeneity`, `df64_homogeneity_beyond_spread_600`, `df64_reference_table` | `Df64` only, as before #30. |

  A test name that says `df64` but runs without `xprec` (for example
  `n1_zetac_df64_is_minus_infinity_where_f64_is`) checks only its `f64`
  assertions there. The unit test `near_diagonal_matches_the_recurrence` is
  also gated. The doc-test count is 32 in both configurations, because
  examples such as the double-double one in `src/lib.rs` and a line in
  `src/real.rs` sit behind `# #[cfg(feature = "xprec")]` and assert nothing
  without the feature, so that count does not show what was checked. Check the
  `running N tests` lines, and which assertions a passing test contains,
  rather than assuming a file ran.
- To make a new test cover `f64` without `xprec`, gate only its `Df64` items
  (`#[cfg(feature = "xprec")]` on the item, or on a block inside the test),
  never the whole file, and show its test names, with a nonzero count, in the
  `--no-default-features` output.
- CI (`.github/workflows/checks.yml`) builds and tests only the default
  features, which include `xprec`. No lane runs
  `cargo test -p specials --no-default-features` yet (#30 tracks adding one),
  so that configuration is checked only when someone runs it locally, with
  the per-file test counts above.

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
[benchmark rules](https://github.com/tensor4all/tensor4all-agent-rules/blob/7e0caf09745b422610aaf45710301d59a0ef3514/rules/benchmark/index.md)
(pinned to the revision this guide was written against; check `main` of
`tensor4all-agent-rules` for later changes).
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
