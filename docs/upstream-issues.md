# Known upstream issues

Bugs found in libraries and reference data this repository uses or validates
against. Each entry records the evidence, what this repository does about it,
and whether it has been reported upstream. Not reporting is a valid state:
SciPy's AI policy asks that issues there be written by a human.

## mpmath: Hurwitz zeta loses relative accuracy for tiny values

- **Affects:** mpmath 1.3.0, 1.4.1 and 1.5.0a2, `mp.zeta(s, a)` for real `s > 1`, `a > 1`
  when the value is far below `2^-prec`.
- **Example:** `zeta(32.539320257871985, 522.7366740445999)` ≈ 6.0449e-88 is off
  by 8.5e-11 (relative) at 256 bits, and by 5e-9 at the default precision; it
  is correct from 512 bits on, and in SciPy.
- **Upstream:** reported in [mpmath/mpmath#1198](https://github.com/mpmath/mpmath/issues/1198),
  fixed by [mpmath/mpmath#1199](https://github.com/mpmath/mpmath/pull/1199)
  (merged 2026-11-01, in master, not yet released). Checked: master gives the
  right value.
- **Here:** `scripts/gen_specials_reference.py` and `scripts/import_xsref.py`
  compute every reference at two working precisions that must agree
  (`stable()`). Keep this after upgrading mpmath; it guards against the same
  class of bug in other functions.

## scipy/xsref: wrong `beta` / `betaln` references when `a + b` is not exact at dps = 100

- **Affects:** xsref `c5fde949`, `tables/scipy_special_tests/beta/*_d_d-d.parquet`
  (36 rows) and `betaln/*_d_d-d.parquet` (18 rows).
- **Cause:** the reference implementations evaluate `mp.beta(a, b)` at the
  framework's default `dps=100` (~332 bits). With one huge and one small
  argument, `a + b` rounds to the huge one, Γ(b)/Γ(a+b) becomes 1, and the
  "reference" is Γ(a) (or worse).
- **Impact:** SciPy is right on all 54 rows (within 5e-15 / 2e-16 of mpmath at
  sufficient precision). But xsref's `Err_*` tolerances were fitted to the
  wrong references, so these rows tolerate errors up to `1.0` or `inf`, and
  a regression there would not be caught.
- **Reproducer** (mpmath 1.3.0, SciPy 1.16.3):

  ```python
  import mpmath as mp, scipy.special as sc

  a, b = -8.571428571428571, 8.98846567431105e307
  with mp.workdps(100):
      print(mp.beta(a, b))     # -2.30815842586817708e-5 == gamma(a): the value in Out_d_d-d.parquet
  with mp.workprec(1200):      # enough bits for a + b to be exact
      A, B = mp.mpf(a), mp.mpf(b)
      print(mp.loggamma(A).real + mp.loggamma(B) - mp.loggamma(A + B))   # 6067.23...: |B| overflows
  print(sc.beta(a, b))         # -inf (correct)
  ```

- **Fix upstream would be:** raise the working precision in the `beta` and
  `betaln` reference implementations by the exponent gap of the arguments
  (e.g. `mp.workprec(mp.prec + abs(mp.mag(a) - mp.mag(b)) + 10)`), then
  regenerate `Out` and `Err` for both.
- **Upstream:** not reported.
- **Here:** `scripts/import_xsref.py` raises the precision by the exponent gap
  (`extra_bits`) for its own double-double values, so the `Df64` checks in
  `crates/specials/tests/xsref.rs` use correct references on these rows. The
  `f64` check uses xsref's `Out`/`Err` as xsf does, so on these rows it is as
  weak as xsf's own test.
- **Affected argument pairs** (each also appears with the arguments swapped;
  "mpmath" computed with enough precision and confirmed at a higher one;
  "rel. error" is xsf's `extended_relative_error(Out, mpmath)`):

  `beta` (18 distinct argument pairs; each also appears with the arguments swapped):

  | a | b | `Out` (xsref) | mpmath, enough precision | rel. error of `Out` | `Err` gcc-linux-x86_64 |
  | --- | --- | --- | --- | --- | --- |
  | -8.571428571428571 | 8.98846567431105e+307 | -2.308158425868177e-05 | -inf | inf | inf |
  | 10.0 | 8.98846567431105e+307 | 362880.0 | 0.0 | inf | 1.0e+00 |
  | 0.1 | 1e+200 | 4.632769299667098e-20 | 9.513507698668707e-20 | 5.1e-01 | 1.1e+00 |
  | 8.98846567431105e+307 | 5.0 | 24.0 | 0.0 | inf | 1.0e+00 |
  | 1e+200 | 0.46415888336127786 | 1.2390072516906996e-92 | 2.8105879242763837e-93 | 3.4e+00 | 7.7e-01 |
  | 1e+200 | 1e-10 | 9999999999.422785 | 9999999538.905775 | 4.6e-08 | 4.6e-08 |
  | 6.666666666666667 | 8.98846567431105e+307 | 389.0349262461803 | 0.0 | inf | 1.0e+00 |
  | 1e+200 | 0.0004641588833612782 | 2153.8579332506947 | 1739.343175448155 | 2.4e-01 | 1.9e-01 |
  | 8.98846567431105e+307 | 8.333333333333334 | 9940.0909844927 | 0.0 | inf | 1.0e+00 |
  | 8.98846567431105e+307 | -4.285714285714286 | -0.1079187243207444 | -inf | inf | inf |
  | -1.2684342882037117e-10 | 8.98846567431105e+307 | -7883735163.682481 | -7883735872.7720785 | 9.0e-08 | 9.0e-08 |
  | -5.714285714285714 | 8.98846567431105e+307 | 0.009469447626296845 | inf | inf | inf |
  | -1.4285714285714286 | 8.98846567431105e+307 | 2.545682353740709 | inf | inf | inf |
  | 1.2909944487358053e-15 | 8.98846567431105e+307 | 774596669241483.0 | 774596669240773.9 | 9.2e-13 | 9.2e-13 |
  | -7.142857142857143 | 8.98846567431105e+307 | 0.0010757344772795688 | inf | inf | inf |
  | -2.857142857142857 | 8.98846567431105e+307 | -1.4397343516057721 | -inf | inf | inf |
  | 8.98846567431105e+307 | 3.3333333333333335 | 2.7781584804376647 | 0.0 | inf | 1.0e+00 |
  | 1.6666666666666667 | 8.98846567431105e+307 | 0.9027452929509336 | 0.0 | inf | 1.0e+00 |

  `betaln` (9 distinct argument pairs; each also appears with the arguments swapped):

  | a | b | `Out` (xsref) | mpmath, enough precision | rel. error of `Out` | `Err` gcc-linux-x86_64 |
  | --- | --- | --- | --- | --- | --- |
  | 1e+200 | -10003.4 | 4524596.17546318 | 4524596.895021021 | 1.6e-07 | 1.6e-07 |
  | -100.3 | 1e+200 | 45824.651643594756 | 45826.09075927788 | 3.1e-05 | 3.1e-05 |
  | 1e+200 | 2154.4346900318865 | -977776.5902751234 | -977775.2962369854 | 1.3e-06 | 1.3e-06 |
  | 1e-10 | 1e+200 | 23.025850929882736 | 23.025850883831033 | 2.0e-09 | 2.0e-09 |
  | 2.1544346900318834 | 1e+200 | -992.9170749058193 | -992.0810937714695 | 8.4e-04 | 8.4e-04 |
  | 1e+200 | 0.1 | -44.518547049707344 | -43.79898920814671 | 1.6e-02 | 1.6e-02 |
  | -2.3 | 1e+200 | 1058.1195937575947 | 1059.558709440716 | 1.4e-03 | 1.4e-03 |
  | 0.46415888336127786 | 1e+200 | -211.62351809996463 | -213.10701996126411 | 7.0e-03 | 7.0e-03 |
  | 0.0004641588833612782 | 1e+200 | 7.6750159006901 | 7.461262835568412 | 2.9e-02 | 2.8e-02 |

## scipy/xsf: `binom(n, k)` applies its large-`k` expansion when `k` is not large

- **Affects:** xsf `31177fd9` (`include/xsf/binom.h`), and SciPy 1.16.3's
  `scipy.special.binom`, which gives the same values.
- **Cause:** the branch "avoid loss of precision (DLMF 5.11.13)" is guarded by
  `k > 1e8 * abs(n)`. That holds for a moderate `k` when `n` is tiny, but the
  expansion `Γ(1+n)/|k| (1 + n(n+1)/(2k))` is a series in `1/k` and is not
  accurate for `k` of order 1. Example: `binom(1e-9, 0.5)`.
- **Evidence** (mpmath 1.3.0): the true value is `0.63661977325012374`; xsf and
  SciPy give `0.6366197730780055`, relative error 2.7e-10 (the generic formula
  `1/(n+1)/beta(1+n-k, 1+k)` is accurate to 4e-16 at the same point).
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port, so it has the same error; the
  `f64` check in `tests/reference.rs` allows 1e-9 for exactly this class of
  arguments (`k > 1e8 |n|`, `k < 1e6`) and states why. The `Df64` path uses the
  exact reflection formula there and is accurate to 5e-29.

## scipy: SciPy 1.16.3 predates xsf fixes that the pinned xsf has

Not a bug in this repository or in xsf. The SciPy 1.16.3 used for the parity
column of `tests/data/*.csv` was released before these xsf fixes, so its values
differ from the pinned xsf (and from mpmath) on the affected rows; the parity
check in `tests/reference.rs` skips them (`scipy_comparable`) and the rows are
still checked against mpmath and, for `f64`, against xsf's own regression
cases:

| function | xsf issue | SciPy 1.16.3 behavior |
| --- | --- | --- |
| `lgam1p` | gh-348 | Taylor series stopped after 41 terms near `abs(x) = 1/2`: relative error 3e-14 at `x = 0.5` |
| `zetac` | gh-346 | returns 0 for `127 <= x < 1075` |
| `binom` | gh-352 | wrong for `k > 1e8 abs(n)` (4e-5 at `binom(-10.5, 500000000000.25)`) |
| `stirling2` | gh-378 | Temme approximation for `n > 50`: 2e-5 at `S(51, 7)` |
| `ellipj` | (xsf transformation for `m < 0`) | NaN in all four outputs for `m < 0`, e.g. `ellipj(0.0, -2.0)`; xsf evaluates `m < 0` through DLMF 22.17 |
| `elliprf` | gh-467 | NaN or `-inf` if `x` is 0 or subnormal and `y * z` underflows, e.g. `elliprf(0, 1e-300, 1e-30)` (xsf: 3.1223528191531603e17) |
| `sindg`, `tandg`, `cotdg` | scipy/scipy#20731 (xsf `tests/xsf_tests/test_trig.cpp`) | signed zeros and infinities: `sindg(-0) = +0`, `tandg(-0) = +0`, `tandg(-90) = +inf`, `tandg(540) = +0`, `cotdg(-0) = +inf`, `cotdg(270) = +0`, ... (18 rows of `tests/data/elementary_special_values.csv`, whose `expected` column follows xsf's tests and whose `scipy` column keeps SciPy's value) |


## xprec: `Df64::ln` and `ln_1p` return NaN for `x >= 2^1023`

- **Affects:** xprec 0.2.3, `Df64::ln` and `Df64::ln_1p` for finite arguments
  from 2^1023 ≈ 8.988e307 up to `f64::MAX` (8e307 is still correct).
- **Example:** `Df64::from(1e308).ln()` and `Df64::from(f64::MAX).ln()` are
  `(NaN, NaN)`; the values are 709.196… and 709.782….
- **Upstream:** not reported.
- **Here:** `generic::ln` scales arguments above 2^1000 by the exact 2^-600
  first (it already scaled subnormal arguments up). `gen_harmonic` uses it for
  `ln i`, `ln n` and `ln(n/N)`, so that `gen_harmonic(f64::MAX, a)` is finite in
  both strategies. Other callers of the raw `ln` on huge double-double
  arguments, for example `poch(f64::MAX, m)` with tiny `m`, still give NaN.

## Apple libm: `log` is not correctly rounded, which moves one xsref `binom` row

- **Affects:** macOS (aarch64) libm `log`, through Rust's `f64::ln`.
- **Example:** `ln(403.1393536837051)` is 5.999282292953255 on macOS and
  5.9992822929532545 with glibc. glibc's value is the correctly rounded one
  (mpmath at 300 bits), so the macOS value is 1 ulp away.
  `binom(458.13935368370517, 56.000000000000064)` (xsref `binom.csv` line 2030)
  evaluates `exp` of a difference of ln Γ values of size 2e3 through xsf's
  `lgam`, so that ulp becomes a relative error of 4.3956e-13. That is exactly
  xsf's own error in the `clang-darwin-aarch64` column, and 7.2× the
  `gcc-linux-x86_64` bound (4 · 1.523e-14). It is the only `binom` row that
  exceeds that bound on macOS; the two `pmv` rows in the Apple libm `pmv` entry
  below are the others.
- **Upstream:** not reported (not a bug: C does not require `log` to be
  correctly rounded).
- **Here:** the gcc-linux-x86_64 tolerances are asserted on every platform, as
  issue #8 requires, except that on macOS this row is checked against an
  allowance of 4.4e-13, its observed error rounded up (`APPLE_LIBM_ALLOWANCES`
  in `crates/specials/tests/xsref.rs`; `apple_libm_allowances_are_needed` fails
  once the row no longer needs it). The row passes the gcc-linux-x86_64 bound on
  Linux x86-64 (CI) and Linux aarch64 (glibc, checked in Docker).
  Making the `f64` port independent of the platform libm (a correctly rounded
  `ln` in pure Rust) would remove the deviation; that is a separate change.

## scipy/xsf: `ellipj` is wrong or NaN for `m >= 0.9999999999` outside the validity of its expansion

- **Affects:** xsf `31177fd9` (`include/xsf/cephes/ellpj.h`), and SciPy 1.16.3's
  `scipy.special.ellipj`, which gives the same values.
- **Cause:** for `m >= 0.9999999999` the code uses an expansion in `1 - m` that
  is valid only while `(1 - m) e^(2|u|)` is small (the Cephes header: "the
  approximation applies only for phi < pi/2"). Beyond that it returns wrong
  values, and for `|u| > 710`, where `cosh(u)` overflows, `ellipj` returns NaN.
- **Evidence** (mpmath 1.3.0): `ellipj(100, 0.99999999999)` gives
  `cn = -3.36e31`, `dn = 3.36e31`, `ph = 3.36e31` in xsf; the true values are
  `cn = 7.899e-6`, `dn = 8.508e-6`, `ph = 10.9956`. `ellipj(1000, 0.99999999999)` and
  `ellipj(1e5, 1.0)` give NaN (true `sn = -0.99999999985` and `1`).
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port, so it has the same values;
  `tests/reference.rs` does not compare `f64` with mpmath for these
  arguments (`ellipj_expansion_valid`). The `Df64` path uses the AGM
  descent for every `m < 1` and the closed forms at `m = 1`, and is checked at
  these points.

## scipy/xsf: `elliprj` is NaN for one huge `x`, `y` or `z`

- **Affects:** xsf `31177fd9` (`include/xsf/ellint_carlson/_rj.hh`) and SciPy 1.18.1
  (checked), `elliprj(x, y, z, p)` when one of `x`, `y`, `z` exceeds about
  3.7e156 (10^156.57, in the `f64` path of this crate: 1e156 is finite, 1e157 is
  NaN) with the others of order 1.
- **Example:** `elliprj(1, 2, z, 4)` is `5.97353250106694e-79` for `z = 1e156`
  and NaN for `z = 1e157, 1e160, 1e170, 1e180, 1e300` (the true value at
  `z = 1e180` is `5.97353250106694e-91`, which `Df64` returns). A huge `p` is
  not affected: `elliprj(1, 2, 3, 1e300)` is `2.1808378064067243e-300`.
- **Cause:** not investigated further; the duplication overflows in an
  intermediate product of the large argument.
- **Neighbors:** with one huge argument, `elliprf` has no such limit (finite
  up to `f64::MAX`). `elliprd`, `elliprg` (one huge argument in third
  position) and `elliprc(1, y)` are NaN only within a factor of about 3 of
  `f64::MAX` (above 6e307, 6e307 and 9e307), where the result is far below the
  smallest subnormal anyway for `elliprd` and `elliprc`
  (`elliprd(1, 2, 1e300)` is `0.0`). With several arguments in the top two
  binades, R_C and R_F overflow in a sum of the arguments although the result
  is about 1e-154: `elliprc(2^1022, -2^1022)` is NaN (mpmath
  9.2964523864998226e-155), `elliprc(2^1020, -2^1022)` is 0.0 (mpmath
  6.420273070929153e-155), and `elliprf(2^1022, 2^1023, 1.5 2^1023)` is NaN
  (mpmath 1.084362095977639e-154). SciPy 1.16.3 gives the same values, and so
  does the `f64` path; the `Df64` path is accurate there to 3e-32
  (`crates/specials/tests/carlson_ratchet.rs`, `F64_TOP_BITS`; found for
  tensor4all/tenferro-sci-rs#27).
- **Upstream:** not reported.
- **Here:** since tensor4all/tenferro-sci-rs#26 the `f64` path moves arguments
  that are at most 2^600 apart into 2^±300 (`Scaling::Window`), which removes
  these NaNs for `elliprj`: `elliprj(1, 2, 1e157, 4)` is
  1.888996838041373e-79 (mpmath 1.8889968380413732e-79; SciPy 1.16.3 NaN).
  Farther apart, the `f64` path keeps xsf's value and its NaN
  (`elliprj(1, 2, 1e300, 4)`). `carlson_ratchet.rs` checks rows with one
  argument 2^500 to 2^600 above the others (regime `xsf_one_huge`). The `Df64`
  path is accurate (checked to 5e-32 at ratios up to 1e180) and returns NaN,
  by design, when the largest and smallest nonzero argument differ by more
  than 2^600 (`elliprj` in `lib.rs`; `df64_elliprj_ratio_limit_is_fail_fast`).

## scipy/xsf: `ellipkinc(phi, 1)` is `+inf` at the `f64` value of pi/2, where `F` is finite

- **Affects:** xsf `31177fd9` (`cephes/ellik.h`), `abs(phi) >= (double)M_PI_2`.
- **Example:** `ellipkinc(1.5707963267948966, 1.0)` is `inf` in xsf and SciPy;
  the `f64` number 1.5707963267948966 is 6.1e-17 below pi/2, so
  `F = asinh(tan(phi))` is finite, 38.025003373828866 (mpmath 1.3.0). The result
  is also `+inf` for `phi <= -pi/2` (the sign of the integral is lost).
- **Upstream:** not reported.
- **Evidence:** the xsref `ellipkinc` rows `(+-1.5707963267948966, 1.0)` record
  the finite value `+-38.025003373828866` with tolerance `inf` on every
  platform, i.e. xsref knows xsf disagrees; the exact value
  `atanh(sin(1.5707963267948966))` is 38.02500337382886806180240516117
  (mpmath 1.3.0, 60 digits).
- **Here:** `f64` is the faithful port, so it returns `inf` (the xsref check is
  vacuous on those two rows because of the infinite tolerance). `Df64` compares
  with the double-double pi/2 and returns the finite value, which agrees with
  the exact one to 1.5e-31 (relative); `df64_matches_mpmath_on_xsref_points`
  checks it. The special-value table does not contain this row.

## mpmath: `ellipf(phi, 1)` loses accuracy near `phi = pi/2` at working precision

- **Affects:** mpmath 1.3.0, `mp.ellipf(phi, 1)` for `phi` within a few
  working-precision units of pi/2 (F(phi | 1) = atanh(sin phi) diverges at pi/2,
  so `pi/2 - phi` cancels).
- **Examples:** at 256 bits `mp.ellipf(mp.pi/2, 1)` returns 179.286 (the
  argument is 5.4846e-78 below pi/2, and the exact value, computed at 2000 bits
  for the same rounded argument, is 178.59284260210710691: 0.4% off). At 15
  digits `mp.ellipf(mpf(1.5707963267948966), 1)` returns 38.718, where the
  exact value is 38.0250033738 (50 and 1200 digits agree).
- **Upstream:** not reported.
- **Here:** the reference scripts compute every value at two working precisions
  that must agree (`stable()`), and an `f64` argument such as
  1.5707963267948966 is 6.1e-17 below pi/2, far above the 256-bit resolution,
  so it is evaluated correctly (38.0250033738...). The scripts give no mpmath
  value for `m = 1` and `abs(phi) >= pi/2` (computed at the working precision),
  where F is infinite.

## scipy/xsref: wrong references in the `log_expit` and `cbrt` tables

- **Affects:** xsref `c5fde949` `tables/scipy_special_tests/log_expit/*_d-d.parquet`
  and `cbrt/*_d-d.parquet`.
- **Example:** `log_expit` at `x = 500, 710, 725, 735` has the reference `0`;
  the value is `-e^-x` (`-7.1245764067412860e-218` at 500, mpmath at 256 bits),
  which xsf returns. `cbrt` at `x = -10` and `-1` has the reference NaN; the
  real cube root is `-2.154434690031884` and `-1`, which xsf returns. xsref
  records the resulting error as the tolerance (up to 1.4e106, and `inf` for
  `cbrt`), so the rows pass xsf's criterion without checking anything.
- **Upstream:** not reported.
- **Here:** `tests/elementary.rs` keeps the rows with their recorded tolerance
  but leaves them out of the reported worst errors (`BAD_REFERENCE`); the
  importer adds mpmath double-double values at these points, which the `Df64`
  path is checked against, and `tests/data/log_expit.csv` and `cbrt.csv`
  check both types at such arguments against mpmath.

## scipy/xsref: the degree-function tables are SciPy values, not references

- **Affects:** xsref `c5fde949` `sindg`, `cosdg`, `tandg` and `cotdg` (every row
  has `fallback = True`: the "reference" is SciPy's own output).
- **Example:** `cotdg(1)` is `57.289961630759144` in the table (Cephes'
  `tan(89 π/180)`); the value is `57.28996163075942` (relative difference
  4.8e-15, which is Cephes' error: rounding `89 π/180` costs the condition
  number of `tan`, about 90 there). `sindg(30)` is `0.49999999999999994`.
- **Upstream:** not reported (xsref marks the rows as fallback).
- **Here:** the tables pin parity with xsf only; `f64` accuracy is checked
  against mpmath in `tests/data/{sindg,cosdg,tandg,cotdg}.csv` with a
  tolerance that grows with the condition number of `tan`
  (`tests/elementary.rs`, `f64_tolerance`), and `Df64` is checked to 1e-30.

## scipy/xsf: `cotdg(x)` is `±inf` for `0 < |x| < 1e-14`

- **Affects:** xsf `31177fd9` (`cephes/tandg.h`, `tancot`) and SciPy 1.16.3.
- **Example:** `cotdg(1e-300)` is `+inf`; the value is `5.729577951308232e301`.
  `tancot` folds `x` to `90 - x`, which rounds to `90` for `|x|` below
  `45 ε ≈ 1e-14`, and returns the pole. The same happens to `tandg` within
  `1e-14` of an odd multiple of 90°.
- **Upstream:** not reported.
- **Here:** the `f64` path keeps xsf's value (parity); `tests/elementary.rs`
  checks those rows against SciPy only (`f64_known_limit`). The `Df64` path
  computes the two complementary angles separately and returns the finite
  value (checked against mpmath).

## xprec: defects in elementary functions of `Df64`

- **Affects:** xprec 0.2.3.
- **Examples:**
  - `Df64::from(-0.0).exp_m1()` and `.ln_1p()` are `+0` (should be `-0`).
  - `Df64::new_full(-1.0, 1e-25).ln_1p()` is `-inf`: it starts from
    `ln_1p` of the high word, `-1`. The value is `ln(1e-25) = -57.56...`.
  - `Df64::exp2` is `exp(x ln 2)`: NaN at `±inf`, and its relative error grows
    with `|x|` (1.9e-30 at `x = 1023.9`). `Df64::from(10.0).powf(x)` likewise
    (2.8e-30 at `x = 307.9`).
  - `Float::cbrt` for `Df64` is `todo!()` and panics.
  - `Df64::sin` and `cos` panic (an `assert!` in the kernel) for NaN and
    `±inf`.
  - `Df64::ln` loses accuracy below about 1e-290, where `exp` in its Newton
    step has a subnormal low word: `Df64::from(1e-300).ln()` is off by 1.2e-27
    (relative).
  - A product that is zero is `+0` whatever the signs:
    `Df64::from(-10.0) * Df64::from(0.0)` is `+0`.
- **Upstream:** not reported.
- **Here:** `generic/elementary.rs` handles each (signed zeros returned
  directly, `1 + x` evaluated exactly near `-1`, `exp2`/`exp10` with an exact
  integer split, `cbrt` by a Newton step from the `f64` root, non-finite
  arguments returned before `sin`/`cos`, zero products signed from the `f64`
  product), and `generic::ln` now rescales arguments below 2^-900 (it
  previously did so only below `f64::MIN_POSITIVE`).
  `tests/elementary.rs::df64::xprec_defects_are_handled` checks them.

## xprec: `Df64::sqrt` loses digits below about 1e-292

- **Affects:** xprec 0.2.3.
- **Example:** `Df64::from(4.450147717014403e-308).sqrt()` differs from the
  value computed after an exact scaling by 2^1000 by 1.3e-17 (relative), and
  `Df64::from(1e-300).sqrt()` by 6.7e-25.
- **Upstream:** not reported.
- **Here:** `generic::sqrt` scales arguments below 2^-900 by an exact power of
  four; the elliptic algorithms use it. This is the square-root counterpart of
  the subnormal low word noted in `generic/mod.rs`.

## scipy/xsf: `elliprj` and `elliprg` are NaN or 0 for large or small arguments

- **Affects:** xsf `31177fd9` (`include/xsf/ellint_carlson/_rj.hh`, `_rg.hh`),
  and SciPy 1.18.1's `scipy.special.elliprj` and `elliprg`, which give the same
  values.
- **Cause:** the arguments are never scaled, and R_J forms
  `(p - x)(p - y)(p - z)` (and `x y` in the principal value for `p < 0`,
  Carlson 1995 Eq. 2.26), R_G `z (x + y)`, before anything else. From about
  2^±341 (R_J) and 2^±512 (R_G) on the products leave the `f64` range,
  although the argument ratios are 1 and the integrals finite.
- **Example:** `elliprj(2**520, 2**520, 2**520, -2**520)`,
  `elliprj(2**344, 2**345, 3*2**344, 2**346)` and `elliprg(2**520, 2**520, 2**520)`
  are NaN; `elliprg(2**-544, 2**-543, 3*2**-544)` is 0. The values are
  `2^-780 R_J(1, 1, 1, -1) = -8.887435110666537e-236`,
  `2^-516 R_J(1, 2, 3, 4) = 1.118043e-156`, `2^260 = 1.85e78` and
  `2^-272 R_G(1, 2, 3) = 1.847e-82` (homogeneity, DLMF 19.16.2 and 19.16.3).
- **Upstream:** not reported.
- **Here:** the `f64` wrappers `xsf::carlson::elliprj` and `elliprg` scale the
  arguments by an exact power of four when one lies beyond 2^±300, none is
  subnormal and they are at most 2^600 apart (a deviation from the port, stated
  in its header): they are moved by the power of four that brings the outlying
  argument just inside 2^±300, which leaves all of them within 2^±301, so that
  every product of three stays in range, and xsf is evaluated both unscaled and
  scaled: its value depends on the scale (underflow, and absolute thresholds of
  its asymptotic expansions such as `y <= 1e-26`, across which it moves by up to
  2e-6), so the unscaled value, which is what the port returned before, is kept
  unless it is NaN, infinite or 0, or the precision-adaptive algorithm evaluated
  in `f64` shows it to be off by more than 1e-10 and the scaled value is closer:
  a scaled evaluation never replaces a value that independent evaluation agrees
  with. Arguments farther apart are not scaled: no power of four brings them
  within range, and evaluating xsf with only the largest moved into range gave
  values off by up to 13 orders of magnitude where xsf returns NaN. At such
  spreads xsf's own value can also be wrong but finite, as on `main`:
  `elliprg(6.271122910098695e-280, 8.009645382365165e-60,
  2.6711411019225204e-264)` is 1.3071070935084508e-30 (SciPy 1.18.1 and the
  `f64` path) for 1.4150658449666896e-30 (mpmath at 250 bits, and the `Df64`
  path). Arguments within 2^±300 take xsf's path unchanged, and so does xsf's
  treatment of a subnormal argument as zero. Beyond 2^±300 xsf can also return a
  wrong finite value, e.g. `elliprj(3e-100, 1, 2e80, -1e-70)` times 2^-392 is
  -2.1e137 instead of 3.72e157; the scaling corrects it.
  `crates/specials/tests/carlson_scaling.rs` checks both strategies at 2^±520
  and 2^±600 (tensor4all/tenferro-sci-rs#25).

## scipy/xsf: `elliprj` for `p < 0` cancels catastrophically when `|p|` is far below `y`

- **Affects:** xsf `31177fd9` (`_rj.hh`, the principal value of Carlson 1995
  Eq. 2.26), SciPy 1.18.1's `scipy.special.elliprj`, and this repository's own
  `Df64` path (DLMF 19.20.14), at every scale.
- **Cause:** both transformations write the principal value as a combination
  of three terms (R_J with a positive fourth argument, R_F, R_C) that cancel.
  With `x << |p| << y << z` and a wide spread their condition number reaches
  1e45 to 1e91 (mpmath 1.3.0 at 400 bits).
- **Example:** `elliprj(2^-275, 2^-75, 2^225, -2^-175)` is
  `-3.972563343495336e-100` (DLMF 19.20.14 and Eq. 2.26 in mpmath with 400
  extra bits, agreeing to 1e-40); SciPy and the `f64` path give
  `6.462348535570529e-27`, the `Df64` path `7.174648137343064e-43`.
- **Upstream:** not reported.
- **Here:** where a path scales the arguments (tensor4all/tenferro-sci-rs#25;
  the `f64` path when one lies beyond 2^±300 and they are at most 2^600 apart,
  the `Df64` path also when their center lies beyond 2^±100), it computes the
  condition number of the combination it evaluates (Eq. 2.26 for `f64`, DLMF
  19.20.14 for `Df64`; the two can differ by orders of magnitude) and returns
  NaN when it can cost more than a relative 1e-10 (above about 4.5e5 for `f64`,
  4.1e21 for `Df64`), instead of a value without correct digits. Elsewhere, as
  in the example, both paths return the value as computed, as before. A remedy
  for all scales (a transformation that does not cancel there) is separate work.

## mpmath: `elliprj` for `p < 0` loses its real part at large or widely spread arguments

- **Affects:** mpmath 1.3.0, `mp.elliprj(x, y, z, p)` for real `p < 0`, whose
  real part is the Cauchy principal value at ordinary arguments.
- **Example:** at 200 bits, `re(elliprj(2^520, 2^520, 2^520, -2^520)) · 2^780`
  is 76.496… instead of `R_J(1, 1, 1, -1) = -0.56516…`; at
  `(2^-201, 1, 2^399, -2^399)` the real part is -7.9e-179 at 200 bits,
  -1.58e-178 at 300 bits and -2.84332914131e-178 at 600 bits, against
  -2.843329141308721e-178 from DLMF 19.20.14 (stable at 200 and 300 bits).
  At `2^-520` times `(1, 1, 1, -1)` it did not return within two minutes.
- **Upstream:** not reported.
- **Here:** `scripts/gen_carlson_scaling.py` computes the principal value from
  DLMF 19.20.14 (R_J with a positive fourth argument, R_F, R_C), at base points
  of ordinary size, where it checks it against mpmath's real part, and the test
  scales by the exact power of two of the homogeneity.

## scipy/xsf: `elliprj` uses its small-`y` expansion when `y / z` is not small

- **Affects:** xsf `31177fd9` (`include/xsf/ellint_carlson/_rj.hh`,
  `rj_asym_conf`, case `tiny_y`) and SciPy 1.16.3, `elliprj(x, y, z, p)` with
  `p > 0`, sorted `x <= y <= z`.
- **Cause:** the expansion of Carlson & Gustafson (1994), Proposition J3, is
  selected when `y <= 1e-26` (an absolute threshold) and
  `(a/z + a/p) |ln(p/a)| <= 1` with `a = (x + y)/2`. Its truncation error is
  of order `y / z`, and nothing bounds `y / z` itself: with `p >> z` the
  selection allows `y / z` up to about `1 / |ln(p / y)|`.
- **Example** (mpmath 1.3.0 at 200 bits; SciPy and the `f64` path give the
  same values):

  | `(x, y, z, p)` | xsf | mpmath | relative error |
  | --- | --- | --- | --- |
  | `(0, 2^-88, 2^-70, 2^-28)` | 2.1097463894034245e20 | 2.1097424673320438e20 | 1.9e-6 |
  | `(0, 2^-88, 2^-65, 2^-28)` | 4.577149050077143e19 | 4.5771488586721284e19 | 4.2e-8 |
  | `(0, 2^-88, 2^-60, 2^-28)` | 9.589710776101472e18 | 9.5897107669232288e18 | 9.6e-10 |

  The error falls with `y / z` (2^-18, 2^-23, 2^-28). With `z` of order 1,
  or `p` small enough that the expansion is not selected, xsf is accurate to
  about 1e-16 at the same `y`.
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port and has the same values.
  The `f64` window of #26 moved some arguments into this case: the review of
  0873683 found `elliprj(2^-322, 2^-110, 2^-92, 2^-50)` 1.9e-6 off after
  scaling, while the unscaled value is right. The arbiter keeps that unscaled
  value. `crates/specials/tests/carlson_ratchet.rs` pins the three points above
  (regime `xsf_tiny_y`), against mpmath with the tolerance `y / z` and against
  SciPy for parity. It also pins the window's image of the review's point,
  `(2^-300, 2^-88, 2^-70, 2^-28)`, the same value 1.9e-6 off. The `Df64` path is
  accurate there (2.2e-32).

## scipy/xsf: `elliprj` loses accuracy just above its `p <= 1e-9` switch

- **Affects:** xsf `31177fd9` (`_rj.hh`) and SciPy 1.16.3, `elliprj(x, y, z, p)`
  with `p` just above 1e-9 and `x, y, z` of order 1.
- **Cause:** for `p <= 1e-9` (an absolute threshold, `abs_close_zero`) xsf
  uses Gustafson's identity (1982, Theorem 11), which is exact. Just above
  it, the duplication runs with `p / x` near 1e-9 and loses digits.
- **Example** (mpmath 1.3.0 at 200 bits; SciPy and the `f64` path give the
  same values): `elliprj(1, 1, 1, 1.0000000000000003e-9)` is
  30.16434034308821 for 30.164340343095963 (2.6e-13);
  `elliprj(1, 2, 3, 2e-9)` is 12.217688234897604 for 12.217688234898564
  (7.9e-14). At `p = 1e-9` itself both are accurate to 2e-16. The loss
  persists to `p = 1e-6` (2.3e-14 at `(1, 2, 3, 1e-6)`) and is below 2e-15 at
  `p = 1e-4` (the `threshold_p` rows of `carlson_ratchet.csv`).
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port. `carlson_ratchet.rs` checks
  `p` from 5e-10 to 1e-4 on both sides of the switch (regime `threshold_p`),
  with an `f64` tolerance of 1e-12 above the switch and the ordinary 2e-14 at
  and below it, and against SciPy for parity. The `Df64` path is accurate
  there (6.4e-32).

## Apple libm: two xsref `pmv` rows differ from the `gcc-linux-x86_64` bound

- **Affects:** macOS (aarch64) libm `sin`, `cos`, `tan` or `log` (the one that
  differs was not isolated), through the digamma expansion of xsf's `lpmv0`
  (`x < -0.35`, non-integer degree).
- **Evidence:** xsref `pmv.csv` lines 587 and 1412, `pmv(±5, -11.3, -0.557)`:
  the `f64` port gives `5.844786490795273e-7` and `-8643.566755327178`, and
  xsref (and xsf built with gcc on x86-64, tolerance 0) has `5.844786490795288e-7` and
  `-8643.5667553272`. Errors 2.5e-15 against the bound 8.9e-16. Replacing every
  `sin`, `cos`, `tan` and `log` of a Python transcription of `lpmv0` by the correctly
  rounded value (mpmath) gives `-8643.5667553272` exactly; the platform `math`
  functions give `-8643.566755327178`, the port's value. xsf compiled here with
  `clang++ -ffp-contract=off` gives the port's value bit for bit on all 1625 rows
  (and with FMA contraction a third value, inside xsref's `clang-darwin-aarch64` bound).
- **Upstream:** not reported (not a bug).
- **Here:** like the `binom` row above, these two rows keep the gcc-linux-x86_64
  bound on every other platform and are checked against an allowance of 2.6e-15,
  their observed error rounded up, on macOS (`APPLE_LIBM_ALLOWANCES`). Linux
  x86-64 is covered by CI; Linux aarch64 was not checked.

## scipy/xsf: `eval_jacobi` (integer degree) is NaN at `alpha = -1` unless `abs(beta) = 1`, and loses all digits within 1e-6 of `-1`

- **Affects:** xsf `31177fd9` (`include/xsf/orthogonal_eval.h`, `eval_jacobi_l`),
  `eval_jacobi` and `eval_sh_jacobi` of an integer degree, and SciPy 1.16.3.
- **Evidence** (xsf built locally, mpmath 1.3.0): `eval_jacobi(2, -1, 0.5, -0.75)` is
  NaN and `eval_jacobi(2, -1 + 2^-52, 0.5, -0.75)` is `0`; the values are
  `1.162109375` in both cases (mpmath, and `P_n^(-1,b) = (n + b)/(2n) (x - 1) P_(n-1)^(1,b)`).
- **Cause:** the recurrence is written in differences `d_k` normalized by
  `binom(k + alpha, k)` and starts from `d = (alpha + beta + 2)(x - 1) / (2 (alpha + 1))`.
  At `alpha = -1` that is `0/0` or `inf`; for `alpha + 1` of order 1e-16 the
  normalized values are 1e16 times the result and `p` cancels. Only `alpha = -1`,
  `abs(beta) = 1` has a special case.
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port and `f64_limit` of
  `tests/orthogonal.rs` skips those rows; the `Df64` recurrence is the one of DLMF
  18.9.2, which does not divide by `alpha + 1`, with the relation above at `alpha = -1`
  and its mirror at `beta = -1`, and passes them.

## scipy/xsf: `eval_jacobi` and `eval_genlaguerre` (integer degree) are not finite where `binom(n + alpha, n)` overflows

- **Affects:** xsf `31177fd9` (`eval_jacobi_l`, `eval_genlaguerre_l`) and SciPy 1.16.3.
- **Evidence** (xsf built locally, mpmath 1.3.0): `eval_genlaguerre(1000, 1000, 1000)`
  is `inf` and `eval_jacobi(1000, 1000, 0, -1)` is NaN; the values are
  `1.9004077401575e299` and `1` (`P_n^(a,0)(-1) = (-1)^n`).
- **Cause:** both return `binom(n + alpha, n) * p` with `p` the recurrence normalized
  by that binomial coefficient; `binom(2000, 1000) = 2.05e600` overflows while
  `p` is 1e-383 (underflow to 0, or `inf * 0 = NaN`), although the result is representable.
- **Upstream:** not reported.
- **Here:** `f64` keeps the behavior (never a wrong finite value: the result is not
  finite there); `tests/orthogonal.rs` requires exactly that of rows where the binomial
  coefficient overflows. `Df64` runs the unnormalized recurrence with a binary exponent,
  and returns the value.

## scipy/xsf: `sph_legendre_p` and the normalized `assoc_legendre_p` are NaN from degree 646

- **Affects:** xsf `31177fd9` (`include/xsf/cpu/legendre.h`, `sph_legendre_p_recurrence_n`,
  `assoc_legendre_p_recurrence_n` with the normalized policy) and SciPy 1.16.3.
- **Evidence** (xsf built locally; mpmath 1.3.0): `sph_legendre_p(n, 0, 0.3)` is finite
  up to `n = 645` and NaN from `n = 646` (`sph_legendre_p(1200, 0, 0.3)`: NaN, true
  `0.19973130740583738`); the same for the normalized `assoc_legendre_p`
  (`n = 1500`, `m = 0`, `z = 0.3`: NaN, true `0.073116429152240806`).
- **Cause:** the coefficients are `sqrt(value_type((2n + 1) * (4 (n - 1)^2 - 1)) / ...)`,
  with the product formed in `int`: `2 (2n + 1) (n - 1)^2 * 2 > 2^31` from `n = 646`
  (signed overflow, undefined behavior; a negative value under the square root).
- **Upstream:** not reported.
- **Here:** the `f64` port forms these products in `f64` (a documented deviation in
  `src/xsf/legendre.rs`); `tests/data/sph_legendre_p.csv` and `assoc_legendre_p*.csv`
  have rows at `n = 645`, `646`, `1025`, `1200`, `1500` and `2000`.

## scipy/xsf: `lqmn` returns zeros for `-1.1 <= x <= -1.0001` and is accurate to only 1e-8 for `1.0001 <= abs(x) <= 1.1`

- **Affects:** xsf `31177fd9` (`include/xsf/cpu/legendre.h`, `lqmn`, real argument)
  and SciPy 1.16.3 (`lqmn(1, 2, -1.05)` has zeros for `n >= 1`).
- **Evidence** (xsf built locally; mpmath 1.3.0): `lqmn(1, 3, -1.0001)` has `Q_1^0 = Q_3^0 = Q_3^1 = 0`;
  the values are `3.9522639525`, `3.1209068500` and `70.4017735162`. For `x = +1.0001` the
  result is `Q_3^0 = 3.1209068417` against `3.1209068500`, a relative error of 2.6e-9.
- **Cause:** the start index of the backward recurrence is `(40 + m + n) * (int)(-1 - 1.8 log(x - 1))`:
  for negative `x` the logarithm is NaN and the cast is undefined behavior (a huge negative
  index, so no iteration runs). For positive `x` the start index leaves a truncation error
  `exp(-2 k acosh x)` of 1e-9 at `x = 1.0001`.
- **Upstream:** not reported.
- **Here:** the port uses `log(abs(x) - 1)` (so the negative arguments agree with the
  positive ones, up to the odd/even symmetry) and keeps the start index; the tolerance
  of the `f64` rows in this range follows that error (`q_f64_tolerance` in `tests/orthogonal.rs`).
  The `Df64` path uses Miller's algorithm from `n + 40 / acosh abs(x)`.

## scipy/xsf: `lqn` loses accuracy for large `n` just above `abs(x) = 1`

- **Affects:** xsf `31177fd9` (`lqn`): for `1 < abs(x) <= 1.021` it takes the forward
  recurrence, whose error grows like `exp(2 n acosh abs(x))`.
- **Evidence** (mpmath 1.3.0): `lqn(80, 1.021)` derivative `Q_80'` has a relative error
  of 1.9e-8 (`-8.659303999403373e-6` against `-8.660096298438902e-6`).
- **Upstream:** not reported.
- **Here:** the port is unchanged; the test tolerance grows like that factor.
  `Df64` takes the backward recurrence when `n acosh abs(x) > 2`.

## scipy/xsf: `pmv` loses its digits next to a zero of an integer-degree function

- **Affects:** xsf `31177fd9` (`lpmv0`, integer degree) and SciPy 1.16.3.
- **Evidence** (mpmath 1.3.0, xsf built locally): `pmv(0, 1, 1e-15)` is `1.1102230246251565e-15`
  (true `1e-15`, relative error 11%), `pmv(0, 1, 1e-30)` is `-0` (true `1e-30`); the xsref
  table has these rows (`pmv.csv` lines with `x = ±1e-30`, `desired = -0.0`).
- **Cause:** `P_1(x)` is formed as `1 + r` with `r = -(1 + x)`, which is `-1` for tiny `x`; the sum of
  the series for an integer degree has an absolute error of `eps` where the polynomial is near
  zero.
- **Upstream:** not reported.
- **Here:** `f64` is the port (xsref's own tolerances accept it, since the references are
  SciPy's values); `Df64` evaluates `P_n^m` by its recurrence, which is exact at such `x`, and
  is compared with mpmath on those rows (`xsref.rs`, `pmv_int`).

## scipy/xsf: the hypergeometric kernels of the real-degree polynomials are inaccurate at large degree

- **Affects:** xsf `31177fd9`: `hyp2f1` (Cephes) behind the real-degree `eval_jacobi`,
  `eval_gegenbauer`, `eval_chebyt`, `eval_chebyu`, `eval_chebyc`, `eval_chebys`, `eval_legendre` and their shifted
  forms, and `hyp1f1` (`chgm`) behind `eval_laguerre` and `eval_genlaguerre`; SciPy 1.16.3 has similar errors.
- **Evidence** (xsf built locally; mpmath 1.3.0): `eval_jacobi(1.3, 5, 3, -0.9)` is `13.510458555221577`
  (true `14.807967691314438`); `eval_chebyt(12.3, -0.9)` is `-0.10705593185349507`
  (true `-0.10703157601721262`); `eval_chebyc(20.2, -1.8)` is `923217.27698217402` (true `-1.1757693567334924`);
  `eval_laguerre(60.25, 10)` is `-21.757075325080386` (true `-8.579814496933425`). Only the Jacobi value was compared with SciPy 1.16.3 (`13.25`).
- **Cause:** not investigated further. The failures are at `z = (1 - x)/2 > 0.9` with `abs(n) > 5`
  (the transformations for `x` near 1 of `hyt2f1`, including the digamma expansion for an integer
  `c - a - b`, which is `-beta` for the Jacobi polynomial), and the power series of `chgm` for large `n` and `x`
  (cancellation of terms up to 1e17).
- **Upstream:** not reported.
- **Here:** the `f64` path is the port; `real_limit` in `tests/orthogonal.rs` lists the rows
  excluded from the comparison with mpmath (they are the regimes above), and the other rows of
  the 13 real-degree files agree to 1e-9. `Df64` takes the recurrence for a non-negative
  integer-valued degree, so it is exact there; any other degree has `f64` accuracy.

## scipy/xsf: `legendre_p(n < 0, z)` returns `z`, and `assoc_legendre_p(n < 0, ...)` returns `0`

- **Affects:** xsf `31177fd9` (`legendre_p`, `assoc_legendre_p`) and SciPy 1.16.3
  (`scipy.special.legendre_p(-1, 0.3)` is `0.3`); the SciPy docstring requires `n >= 0`.
- **Cause:** the recurrence has no iteration for a negative degree and returns its first state.
- **Upstream:** not reported.
- **Here:** both return NaN for `n < 0` (fail fast); `eval_legendre_int` keeps the
  `P_(-n-1) = P_n` convention of `eval_legendre`.

## scipy/xsf: `lqn` and `lqmn` return `1e300` at `x = ±1`

- **Affects:** xsf `31177fd9` (`lqn`, `lqmn`) and SciPy 1.16.3 (`scipy.special.lqn(2, 1.0)` is `1e300` for every
  entry, derivatives included). `pmv` converts the same stand-in to `±inf` (`SPECFUN_CONVINF`); these routines do not.
- **Correct values:** `Q_k(1) = +∞` and `Q_k(-1) = (-1)^(k+1) ∞` (from `Q_k = P_k ln((1+x)/(1-x))/2 - W_(k-1)` and
  `Q_k(-x) = (-1)^(k+1) Q_k(x)`); `Q_k^i(1) = (-1)^i ∞` and `Q_k^i(-1) = (-1)^(k+1) ∞` (DLMF 14.6.2 and 14.6.4: `Q_k^i`
  behaves like `(-1)^i (1 - x)^(-i/2)` on both sides of 1, and `Q_k^i(-x) = (-1)^(k+i+1) Q_k^i(x)` for both branches).
  The derivatives tend to opposite infinities from `|x| < 1` and `|x| > 1`, so they have no value: NaN.
- **Upstream:** not reported.
- **Here:** a deliberate deviation of the `f64` port: both strategies return those infinities and NaN
  derivatives (`second_kind_at_the_endpoints`, and the `lqn`/`lqmn` rows of `orthogonal_special_values.csv`, whose
  `scipy` column keeps `1e300`).

## scipy/xsf: Chebyshev, Hermite, Laguerre and other recurrences return NaN instead of `±inf` when they overflow

- **Affects:** xsf `31177fd9` (`include/xsf/orthogonal_eval.h`) and SciPy 1.16.3.
- **Evidence** (mpmath 1.3.0; xsf built locally and SciPy 1.16.3 give NaN for the first four):
  `eval_hermite(660, -14.568126673025812)` (true `+5.1e932`), `eval_hermite(284, -16.46689288168518)` (`+2.7e388`),
  `eval_hermitenorm(995, 48.980587028696874)` (`-1.1e1536`), `eval_laguerre(2000, 1e5)` (`+5.6e4246`).
  A sweep of integer degrees 50 to 4000 and `x` from `-300` to `1e5` where the true result overflows gives NaN
  from the port (which is bit-identical to xsf there) for every overflowing Chebyshev `T` and `U`, Hermite `H` and `He` point
  (39 to 41 points of each) and 5 of 9 generalized Laguerre points; Legendre and Jacobi at 17 of 39,
  `eval_gegenbauer_int` in none (it returns the infinity).
- **Cause:** a recurrence value becomes `inf` and the next step forms `inf - inf` (or `0 * inf`). Threshold
  (xsf built locally): the first overflowing degree is the correct infinity, `T_112(-300)` and
  `H_235(-14.568126673025812)` are `inf`, and the degrees above it are NaN (`H_656` to `H_669` checked).
- **Upstream:** not reported.
- **Here:** `f64` keeps xsf's NaN (faithful port). `Df64` returns the infinity with the sign of the
  polynomial: the recurrence carries a binary exponent, so the overflow occurs only in the final scaling; for
  `abs(x) > 2^30` where a step itself overflows, the sign and the growth are read from the same recurrence at
  `x = ±2^30` (`resolve` in `src/generic/orthogonal.rs`). A NaN of finite arguments (the singular Jacobi
  recurrence, `alpha <= -1`, `n < 0` for Hermite) stays NaN. A non-finite argument takes xsf's value.
  `overflow_is_a_signed_infinity_in_df64_and_xsfs_nan_in_f64` pins both.

## scipy/xsf: integer-degree Chebyshev `T` overflows when the result is above about half of `f64::MAX`

- **Affects:** xsf `31177fd9` (`eval_chebyt(long, double)` in `include/xsf/orthogonal_eval.h`) and SciPy 1.16.3,
  and `eval_sh_chebyt`, which calls it.
- **Evidence:** `eval_chebyt(2, 9.4e153)` is `inf` (true `1.7672e308`) and `eval_chebyt(3, 3.107e102)` is `inf`
  (true `1.1997e308`); SciPy 1.16.3 gives `inf` for `eval_chebyt(2, 9e153)` (true `1.62e308`);
  `eval_sh_chebyt(2, 4.6e153)` is `inf` (true `1.6928e308`). `eval_chebyt(2, 6.5e153)` (`8.45e307`) is exact.
- **Cause:** the kernel runs the recurrence of `U_n` and returns `T_n = (U_n - U_(n-2)) / 2`. For `abs(x) > 1`,
  `abs(U_n)` is about twice `abs(T_n)`, so `U_n` overflows while `T_n` is still representable: every `T_n` with
  `f64::MAX / 2 <~ abs(T_n) < f64::MAX` is `inf`. `eval_chebyu` and `eval_chebyc` return their recurrence
  value directly and do not have this band.
- **Upstream:** not reported.
- **Here:** `f64` keeps xsf's value (faithful port). `Df64` returns the finite value
  (`eval_chebyt_int(2, 9.4e153)` = `1.7672e308`).

## scipy/xsf and SciPy 1.16.3: values the pinned xsf fixed

SciPy 1.16.3 bundles an older copy of this code. At the pinned xsf commit these differ, and the
special-value file (`tests/data/orthogonal_special_values.csv`, `expected`) follows xsf's own tests:
`eval_legendre(n, NaN)` is NaN for every `n` (SciPy: `1` for `n <= 0`); `eval_gegenbauer(0, 0, x)` is `0`
(SciPy: `1`); `eval_gegenbauer(n, alpha, ±inf)` is `±inf` (SciPy: NaN); `pmv(0, 1.5, -1)` is `+inf`
(SciPy: `-inf`); `assoc_legendre_p(3, 0, -1)` is `-1` (SciPy: `1`, scipy/gh-23101);
`eval_genlaguerre(n, alpha, x)` for `alpha` within 1e-14 of `-1` is accurate (SciPy: 0, gh-457).
The `scipy` column of the reference files is SciPy's value, and the parity comparisons skip those rows.

## scipy: `_dcsrch.py` compares with `<=` where MINPACK-2, the paper and SciPy's own C copy have `<`

- **Affects:** SciPy v1.16.3 (`b9105ccc2237f57acb1060202cd77f6dd264fb34`),
  `scipy/optimize/_dcsrch.py`, `dcstep`, first case (`fp > fx`), line 615.
- **Evidence:** the step is the cubic step `stpc` when it is closer to `stx` than the quadratic
  step `stpq`, else their average. Moré & Thuente (ACM TOMS 20(3), 1994, p. 299) say "closer",
  MINPACK-2 `dcstep.f:120` has `abs(stpc-stx) .lt. abs(stpq-stx)`, and SciPy's own C
  translation in `scipy/optimize/__lbfgsb.c:3626` has `fabs(stpc - *stx) < fabs(stpq - *stx)`.
  Only `_dcsrch.py` has `abs(stpc - stx) <= abs(stpq - stx)`.
- **Impact:** the two differ only on an exact tie `|stpc - stx| == |stpq - stx|` with
  `stpc != stpq`, that is when the two steps are equidistant on opposite sides of `stx`. For a
  valid state (negative slope at `stx` in the direction of the step) `stpq` lies on the side of
  `stp`, so the tie needs a cubic step that extrapolates backwards by exactly the quadratic
  step's distance. No run in the recorded reference data (SciPy's own test problems, the paper's
  tables, the edge cases) depends on it: they all replay identically with either comparison.
  No valid MINPACK state is known to reach it. A tie exists for an input that is not a valid
  state (`stx = 0`, `fx = 0`, `dx = +0.25`, `stp = 1`, `fp = 0.5`, `dp = 2.75` gives
  `stpc = 0.5`, `stpq = -0.5`), and `dcstep.rs` pins the operator with it
  (`case_one_tie_takes_the_average`: the strict `<` returns the average 0, `<=` would return
  0.5); that test fails if the comparison is changed to `<=`.
- **Upstream:** not reported.
- **Here:** `crates/minimize/src/linesearch/dcstep.rs` uses the strict `<` of the paper and the
  Fortran. A run that hits the tie would take the averaged step here and the cubic step in
  SciPy; the evaluation counts of such a run could differ by that one step.

## scipy: after `ABNORMAL` (a failed line search), `L-BFGS-B` returns `fun` of the last trial point, not of `x`

- **Affects:** SciPy v1.16.3 (`b9105ccc2237f57acb1060202cd77f6dd264fb34`),
  `scipy/optimize/_lbfgsb_py.py`, `_minimize_lbfgsb`, the loop at lines 455-485 and the result at 507,
  with `scipy/optimize/__lbfgsb.c`, `mainlb`, lines 918-935.
- **Evidence:** when a line search fails (`maxls` trials, or a direction that is not a descent
  direction) with an empty memory, `mainlb` restores `x` and `g` to the start of the search and sets
  `*f = fold`, then returns `ABNORMAL`. The driver passes `f` to `setulb` as a Python float, which
  the C code cannot write back, and every `task[0] == 3` request rebinds it
  (`f, g = func_and_grad(x)`), so the restore of `f` never reaches `OptimizeResult`. `x` and `g` are
  numpy arrays updated in place and do arrive. Reproducer (SciPy 1.16.3, the run
  `maxls1_rosen2_first` of `crates/minimize/tests/data/lbfgs/runs.csv`): Rosenbrock from
  `x0 = (-1.2, 1)` with `maxls = 1` gives `res.x = [-1.2, 1.]` and `res.jac` the gradient there,
  but `res.fun = 171.33595917774778`, the value at the rejected trial
  `(-0.27415235630480134, 1.3778969974266115)`, while `f(x0) = 24.199999999999996`; `res.message` is
  `ABNORMAL: ` and `res.nfev = 2`. All recorded runs that end in `ABNORMAL` show the same
  (`lbfgs_replay.rs::the_data_covers_every_termination_and_the_counting_quirks` asserts `res.fun`
  is the last trial's `f` and differs from `f(res.x)` on each of them).
- **Impact:** `res.fun`, `res.x` and `res.jac` are mutually inconsistent exactly when the run failed,
  so a caller who reads `fun` of a failed run gets a value that is not attained at the returned `x`
  (here it is 7 times larger).
- **Upstream:** not reported.
- **Here:** `LbfgsState::f()` and `Report::f` are `f(x)` at the returned point after every
  termination, including `LineSearchFailed`; the replay pins SciPy's value in the data
  (`runs.csv`, column `f`) and Rust's consistent one separately.
