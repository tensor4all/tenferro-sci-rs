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
