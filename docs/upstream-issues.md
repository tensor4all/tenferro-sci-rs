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
  `gcc-linux-x86_64` bound (4 · 1.523e-14). It is the only xsref row of the
  twelve tables that fails on macOS.
- **Upstream:** not reported (not a bug: C does not require `log` to be
  correctly rounded).
- **Here:** the gcc-linux-x86_64 tolerances are asserted on every platform, as
  issue #8 requires, so `f64_matches_xsref` fails on this one row on macOS. It
  passes on Linux x86-64 (CI) and Linux aarch64 (glibc, checked in Docker).
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
- **Neighbors:** `elliprf` has no such limit (finite up to `f64::MAX`). `elliprd`,
  `elliprg` (one huge argument in third position) and `elliprc(1, y)` are NaN
  only within a factor of about 3 of `f64::MAX` (above 6e307, 6e307 and 9e307),
  where the result is far below the smallest subnormal anyway for `elliprd` and
  `elliprc` (`elliprd(1, 2, 1e300)` is `0.0`).
- **Upstream:** not reported.
- **Here:** the `f64` path is the faithful port and has the same NaNs. No
  xsref table exists for `elliprj`, and the largest argument of any `elliprj`
  reference row is 1e90, so no test row is affected. The `Df64` path is
  accurate (checked to 5e-32 at ratios up to 1e180) and returns NaN, by design,
  when the largest and smallest nonzero argument differ by more than 2^600
  (`elliprj` in `lib.rs`; `df64_elliprj_ratio_limit_is_fail_fast`).

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
  arguments by an exact power of four when one lies beyond 2^±300 and none is
  subnormal (a deviation from the port, stated in its header): arguments at most
  2^600 apart are moved by the smallest power of four that brings them within
  2^±300, so that every product of three stays in range, and xsf's unscaled
  value is kept where it agrees with the scaled one to 1e-10 (xsf picks some
  asymptotic expansions by absolute thresholds, so its value moves with the
  scale by about 1e-11); arguments farther apart are scaled only when
  xsf's unscaled value is NaN, infinite or 0, because scaling the largest into
  range can then make products of the smallest underflow. Arguments within
  2^±300 take xsf's path unchanged, and so does xsf's treatment of a subnormal
  argument as zero. Beyond 2^±300 xsf can also return a wrong finite value, e.g.
  `elliprj(3e-100, 1, 2e80, -1e-70)` times 2^-392 is -2.1e137 instead of
  3.72e157; the scaling corrects it. `crates/specials/tests/carlson_scaling.rs`
  checks both strategies at 2^±520 and 2^±600 (tensor4all/tenferro-sci-rs#25).

## scipy/xsf: `elliprj` for `p < 0` cancels catastrophically when `|p|` is far below `y`

- **Affects:** xsf `31177fd9` (`_rj.hh`, the principal value of Carlson 1995
  Eq. 2.26), SciPy 1.18.1's `scipy.special.elliprj`, and this repository's own
  `Df64` path (DLMF 19.20.14), at every scale.
- **Cause:** both transformations write the principal value as a combination
  of three terms (R_J with a positive fourth argument, R_F, R_C) that cancel.
  With `x << |p| << y << z` and a wide spread their condition number reaches
  1e45 to 1e91 (mpmath 1.3.0 at 400 bits).
- **Example:** `elliprj(2^-175, 2^25, 2^325, -2^-75)` is `-2.78337e-145`
  (DLMF 19.20.14 and Eq. 2.26 in mpmath, agreeing at 400 and 800 bits); SciPy
  and the `f64` path give `2.263919769706678e-72`, the `Df64` path
  `5.026911708464872e-88`.
- **Upstream:** not reported.
- **Here:** at arguments within 2^±300 both paths return the value as
  computed, as before. Where they scale the arguments
  (tensor4all/tenferro-sci-rs#25) they compute the condition number of the
  DLMF 19.20.14 combination and return NaN above 1e6, as `main` and SciPy did
  at those scales, instead of a value without correct digits. A remedy for
  all scales (a transformation that does not cancel there) is separate work.

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
