# Reference data of `minimize::bfgs`

SciPy 1.16.3 runs of `scipy.optimize.minimize(fun, x0, jac=True, method="BFGS", bounds=None, options=...)`
(`_minimize_bfgs`, `_line_search_wolfe12`), recorded for the replay, live and property tests
(`../../bfgs_replay.rs`, `../../bfgs_live.rs`, ...). SciPy v1.16.3 is commit
`b9105ccc2237f57acb1060202cd77f6dd264fb34` (the commit tag `v1.16.3` points to; the tag object itself is
`56e567e70ccad213d04ab04e7cfe7365562a71f1`).

| File | Contents |
|---|---|
| `problems.csv` | the test functions: dimension, quadratic data `A`, `b` (numpy's generator is not mirrored in Rust), minimizer, a strong-convexity bound `mu` |
| `runs.csv` | one row per run (with the comparison prefix `cmp_nsearch` and the observed maxima `x_ratio_max`, `dir_ratio_max` of the replay mirror, and `repro`): the exact SciPy call and every option, `status`, `message`, `nit`, `nfev`, `njev`, `wolfe2_calls`, where the fallback began, whether it returned a step and whether it warned, why `wolfe1` failed, `cache_hits`, the number of updates with `yᵀs == 0` and `<= 0`, the expected Rust termination, `parity` |
| `evals.csv` | every `(x, f, g)` SciPy asked for, in order, with the rounding scale `x_scale` of the requested point |
| `iters.csv` | every line search: `x_k`, `f_k`, `g_k`, SciPy's direction `p_k`, the exact (mpmath) direction `d_exact` and its scale, the step, `yᵀs` of the pair and whether the update was applied, the evaluation range |
| `hess_final.csv` | SciPy's final `hess_inv`, the exact (mpmath) matrix from SciPy's pairs, and the scale |

Generate with `uv run scripts/gen_minimize_bfgs_reference.py` from the repository root. Regenerating is a
separate commit. Produced on macOS 15.6.1 arm64, Python 3.12.11, SciPy 1.16.3, NumPy 2.3.5, mpmath 1.3.0
(60 digits). The `#` header lines of every file record the versions and the date.

## The protocol

- **The call.** `runs.csv` has the exact call (`scipy_call`) and the options as columns: `gtol`, `norm = inf`,
  `xrtol`, `c1`, `c2`, `maxiter` (200·n unless the run says otherwise), `bounds = None`, `hess_inv0 = None`
  (H₀ = I). `_minimize_bfgs` has no `maxls`; its line search is `_line_search_wolfe12(..., amin=1e-100,
  amax=1e100, c1, c2)`: `wolfe1` (Moré–Thuente, `xtol = 1e-14`, at most 100 iterations, fixed inside
  `scalar_search_wolfe1`, column `ls_maxiter`) and, only if that returns `None`, `wolfe2` with `maxiter = 10`.
  The initial trial step of every line search is `min(1, 1.01·2(f_k − f_{k−1})/g_kᵀp_k)` (negative: 1), with
  `f_{−1} = f_0 + ‖g_0‖₂/2`.
- **What is recorded.** Hooks only observe: the objective (every call, in order), `_line_search_wolfe12`
  (arguments and result), `line_search_wolfe2` (that it was called, at which evaluation, whether it found a
  step) and `DCSRCH.__call__` (the task that made `wolfe1` give up). `nfev` equals the number of recorded
  evaluations; where no fallback fired `njev == nfev` (`wolfe1` asks for f and g together).
- **`wolfe2_calls` and `parity`.** The Rust port has no `wolfe2` (decision 3 on #40): where SciPy's `wolfe1`
  fails it stops with `LineSearchFailed` and returns no step. Parity is claimed only on runs with
  `wolfe2_calls == 0` and a finite path, `parity = true`. The other runs are data: `group = fallback`
  (`wolfe2_calls > 0`; expected termination `LineSearchFailed`) and `group = nonfinite` (expected
  `NonFiniteValue`: the #38 rule, stop at the last finite iterate; SciPy keeps going).
- **Expected termination.** SciPy's `status` 0 maps to `GradTol` (`‖g‖∞ ≤ gtol`) or `StepTol` (the `xrtol` test),
  1 to `MaxIter`, 2 to `LineSearchFailed` or `NonFiniteValue`. `MaxIter` wins over convergence on the last
  allowed iteration, as SciPy's `k >= maxiter` is tested after the loop (run `rosen2_maxiter_eq_nit`).

### The wolfe2 runs

No problem in the probed set (an uncommitted probe: Rosenbrock with gtol from 1e-5 to 1e-14 and scaled by 1e-8 to 1e12,
noisy and perturbed variants, 136 runs in all, plus Powell, Beale, Himmelblau) made `wolfe1` fail on a *smooth,
consistent, bounded-below* objective. The fallback fires only where the objective is not what SciPy's line search assumes,
so every `fallback` run is **constructed** (the problem is chosen to make `wolfe1` fail; SciPy 1.16.3 does call `wolfe2`
on each). Columns: `wolfe2_found` = `wolfe2` returned a step, `wolfe2_warned` = it did so with a `LineSearchWarning`
(`_line_search_wolfe12` silences it). **A step returned with that warning is not a Wolfe step**: when its `maxiter`
(10) is exhausted `scalar_search_wolfe2` returns its last trial (`alpha_star = alpha1`) and warns.

| Run | Why `wolfe1` gives up | `wolfe2` began at evaluation | `wolfe2` returned a step | warned |
|---|---|---|---|---|
| `w2_unbounded` | `f = −x₀` is unbounded below, so the curvature condition never holds and `wolfe1` ends at its limit of 100 trials (`dcsrch did not converge within max iterations`) on the **first** line search. Exact arithmetic (`g = (−1, 0)`), no rounding noise. `maxiter = 1` keeps the run short | 101 (x₀ and 100 trials) | yes, α = 1024 = 2¹⁰ (ten doublings) | **yes**: it gave up |
| `w2_neggrad` | the gradient has the wrong sign: steps shrink to α ≈ 10⁻¹⁹ (below one ulp of x) | 49 | no | yes |
| `w2_l1` | nonsmooth ‖x‖₁: `XTOL TEST SATISFIED` | 65 | no | yes |
| `w2_noisy` | `f` plus 1e-12·sin(1e7 x₀): `dcsrch did not converge within max iterations` | 83 | no | yes |
| `w2_pert` | the gradient is off by the constant 1e-3·(1, −1) | 55 | no | yes |

`nan_region` and `inf_region` (`f`, `g` NaN or +inf for `x₀ > 0.5`) are the only runs with `wolfe2` steps that came without
a warning (`inf_region`: four, `wolfe2_found = 11110`, `wolfe2_warned = 00001`): there the fallback genuinely succeeds.
Under the #38 rule the Rust port stops at the last finite iterate instead.

**The asserted boundary case is `w2_unbounded`**: deterministic (no summation order or libm call can change a decision), and
the place where the port and SciPy part for good: SciPy takes `wolfe2`'s last trial as the step and goes on, and the
port stops at the same evaluation with `LineSearchFailed` and no step. The test (`bfgs_termination.rs`) asserts that the
state asks for exactly the 101 evaluations SciPy made before `wolfe2`, then stops.

**The `y^T s == 0` branch of SciPy is reached here.** `g` is constant, so after that step `y = 0`, `yᵀs = 0` exactly
(`iters.csv` `ys`, `runs.csv` `n_rho1000 = 1`), and `_minimize_bfgs` applies its update with `ρ = 1000`. The branch is
reachable after a `wolfe2` step that is not a Wolfe step; after `wolfe1` (strong Wolfe) steps, which are the only steps the
port takes, `yᵀs ≥ (1 − c₂)(−gᵀs) > 0` and every update of every parity run has `yᵀs > 0` (`n_ys_nonpos = 0`, asserted by the
generator).

### Evaluations are not always one per trial

SciPy's `ScalarFunction` does not re-evaluate the objective at a point equal to the previous one. In the rounding regime
(steps below one ulp of `x`) the line search makes more trials than there are recorded evaluations (`cache_hits` in
`runs.csv` = trials − evaluations: `w2_neggrad` 49, `w2_noisy` 57, `w2_pert` 48, `inf_region` 5), so the recorded
evaluations cannot be matched one to one with trials there, and a Rust state that evaluates at every trial counts more.
This is why the fallback runs other than `w2_unbounded` cannot be replayed index by index, and why the generator
asserts `cache_hits == 0` for every parity run (all have 0).

## Two claims, two mechanisms

1. **Parity: the Rust implementation against SciPy's committed data** (the Rust replay, `tests/bfgs_replay.rs`). On each run's
   *comparison prefix* the requested points must agree within a **derived bound**, componentwise,

       |x_rust − x_SciPy|  ≤  C ε x_scale + 2 x_slack          (ε = 2⁻⁵², C = 32)

   with the **running magnitude scale** `x_scale = |x₀| + Σⱼ (|αⱼ pⱼ| + |x_{j+1}|) + |x_base| + |α p|` (the magnitudes whose
   rounding enters one fused update `x + α p`; Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1; no
   recursion) and `x_slack` = Σ|α||p_SciPy − d_exact|, SciPy's own direction error measured against the exact (mpmath) direction
   from SciPy's pairs, counted twice (each implementation deviates from the exact direction by about that). Evaluation counts,
   iteration counts, termination and the final `f` are compared **exactly on the whole run**. The final iterate and the final `H`
   are compared (same bound; `C ε max|H|` for `H`) only where the whole run is inside the prefix (`H` only for the runs flagged
   `h_compared`, 4 of 16: the error of `H` is amplified by the problem).

   **`C = 32` is calibrated, not derived from `n` or the update.** An independent float64 mirror of the algorithm (the generator's
   `mirror`: SciPy's recorded `f`, `g`, a plain-Python BFGS in the symmetric update form, summing left to right where BLAS does
   not, the step from SciPy's own `scalar_search_wolfe1`) measures, per line search, the largest `(|Δx| − 2 x_slack)/(ε x_scale)`.
   The **comparison prefix** `cmp_nsearch` of a run is the number of leading line searches for which that ratio stays at most
   `C / 4 = 8`; the factor 4 is the margin for an implementation that rounds differently from the mirror. If an implementation
   exceeds the bound, that is a finding to explain, not a reason to raise `C`: the prefix is shortened instead.

   The bound models rounding, not its *amplification* by the problem: a perturbation of an iterate moves the line search and
   `H`. So the prefix is where the claim is meaningful: 11 of the 16 parity runs are covered in full; `rosen2_b` 3 of 19,
   `rosen2_c` 7 of 38, `rosen5` 3 of 49, `powell` 6 of 35, `quad10_k1e4` 5 of 15 searches. Observed maxima inside the prefixes
   (`runs.csv` `x_ratio_max`): 7.93 (`powell`), 6.53 (`rosen2_b`), 4.99 (`rosen2_a` and three runs of the same path), 3.12, 2.98, below 1.4 elsewhere. The
   mirror's direction against the exact one reaches 9·10³ ε d_scale on `quad10_k1e4` (`dir_ratio_max`, a diagnostic of the same
   amplification; SciPy's own is in `dir_ratio_scipy`). Replay failure messages print the observed maximum next to `C`.

   **The earlier scale was replaced.** The first version of this data used the absolute-value recursion of the update
   (`|A₁| H̄ |A₂| + ρ|s||s|ᵀ`), which compounds: `x_scale` reached 10³⁶ on `rosen2_a`, so `C n ε x_scale` admitted anything after
   ten iterations, and a second, calibrated 1e-8 relative envelope did the work. Both are gone: the running scale keeps the bound
   meaningful (a few ε·scale) on the prefix, and past the prefix the replay claims counts and termination only.

2. **Reproducibility: SciPy regenerated on another platform against SciPy committed** (`--check`, CI on Ubuntu/OpenBLAS while the
   data come from macOS/Accelerate). Counts, termination, message, call, options and row keys are **exact**; SciPy's float columns
   (the *trajectory* columns below) are compared with **one named constant, `SCIPY_REPRO_RTOL = 1e-6`**, per component,
   `|a − b| ≤ SCIPY_REPRO_RTOL · max(|a|, |b|, scale)`, `scale` = the largest absolute component of that column over the run's
   compared rows (committed values), on the same comparison prefix as the replay (rows past it: exact columns only).

   `SCIPY_REPRO_RTOL` is a **calibrated reproducibility tolerance of SciPy across BLAS/libm, not derived from the algorithm.** There
   is **no Linux observation for BFGS yet**: the constant was chosen with a large margin after the L-BFGS one (observed 3.5e-8 at
   worst on Ubuntu for L-BFGS-B, constant 1e-6) and **will be confirmed (or revised) by the first CI run**, which prints the
   observed maximum per column next to the constant. On the data's own platform every compared value is identical (0.0).

   Why two mechanisms: a bound derived for "an implementation that follows the algorithm" says nothing about how SciPy's own BLAS
   and libm calls round, and a bound for SciPy's rounding is an empirical property of those libraries (L-BFGS #41 learned this on
   Ubuntu: 148 values outside a derived bound, including 1-ulp differences).

## What `--check` compares

`uv run scripts/gen_minimize_bfgs_reference.py --check` regenerates everything in memory, compares and writes nothing. It prints
every mismatch of an exact column in full (with the list of runs whose path changed), a summary per file and column (values
compared, outside, largest absolute difference, observed maximum relative difference next to the constant and the margin, rows
skipped) and, under "bounds on this platform", any committed comparison prefix that is longer than the regenerated one; every
comparison runs before the verdict. Exit 1 on any exact mismatch, trajectory value outside the constant, or bound violation.

| kind | columns | rule |
|---|---|---|
| exact | every column not listed below: ids, `group`, `problem`, options, `scipy_call`, `status`, `message`, `nit`, `nfev`, `njev`, `n_ls`, `wolfe2_*`, `wolfe1_fail`, `cache_hits`, `n_updates`, `n_rho1000`, `n_ys_nonpos`, `termination`, `parity`, `repro`, `search`, `ev_start`, `ev_end`, `applied`, `h_compared`, the keys and number of rows | string equality; a difference means the path of a recorded run changed on that platform: the run leaves `parity` with a documented reason, the check is never loosened |
| trajectory (SciPy's floats) | `evals` `x f g`; `iters` `x f g p alpha ys d_exact`; `runs` `final_x final_f gnorm_inf` (whole run in the prefix); `hess_final` `h_scipy h_exact` (`h_compared`); `problems` `A b x_star f_star mu` | `SCIPY_REPRO_RTOL` per component with the run scale, on the comparison prefix |
| diagnostic | `x_scale`, `x_slack`, `x_ratio`, `d_scale`, `dir_ratio_*`, `ratio*`, `h_scale`, `x_ratio_max`, `dir_ratio_max`, `cmp_nsearch` (checked: not shorter on the running platform), `note` | not compared; the generator re-measures them and refuses a prefix shorter than 3 searches |
| class runs (`repro = class`) | `w2_neggrad`, `w2_l1`, `w2_noisy`, `w2_pert`, `nan_region`, `inf_region` | only `id group problem n x0 options termination parity repro`: their counts depend on SciPy's evaluation cache and on the last bits of the arithmetic (`cache_hits`), which differ between platforms. `w2_unbounded` (exact arithmetic) is a full `exact` run |

The check is wired into the `minimize-reference` job of `.github/workflows/checks.yml`.

## Open points

- `SCIPY_REPRO_RTOL` and the first Ubuntu/OpenBLAS observation of the BFGS data (above); the replay bound is exercised only against
  the generator's mirror and a throwaway implementation so far, not against the real implementation.
- SciPy's `rho = 1000` branch (`yᵀs == 0`) is reached by `w2_unbounded` only (after a `wolfe2` step); no parity run
  reaches it, and no test pins what the port does with such a pair.
