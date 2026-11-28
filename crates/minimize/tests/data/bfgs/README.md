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
| `steps.csv` | **326 rows**: one per iteration with an update of every parity run: `H_k`, `s_k`, `y_k`, SciPy's `H_{k+1}`, `g_{k+1}`, `p_{k+1}`, the exact (mpmath) `H_{k+1}` and `p_{k+1}`, `κ` and the ratios of SciPy's own error to the derived bound |
| `hess_final.csv` | SciPy's final `hess_inv`, the exact (mpmath) matrix from SciPy's pairs, and the scale |

Sizes: `steps.csv` 490 KB (326 rows; `H_k`, `H_{k+1}` and the exact matrix are stored for every step, `n ≤ 10` for all parity problems), `evals.csv` 196 KB, `iters.csv` 177 KB, `hess_final.csv` 15 KB, `runs.csv` 14 KB, `problems.csv` 4 KB: 0.9 MB in all.

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

## Three claims, two mechanisms

1. **Parity, primary: the per-step replay** (`tests/bfgs_step_replay.rs`, `steps.csv`). Dense BFGS carries an `n × n` matrix
   forward, so BLAS rounding differences between platforms and implementations accumulate from the first update, and a
   *trajectory* bound cannot carry the claim for long (the first Linux run of this data: 6 of 16 runs stayed inside the
   trajectory prefix, the mirror's direction up to 1.3·10³ ε·d_scale from exact). One *step* has no history. For **every
   iteration `k` of every parity run that ends in an update** (326 iterations in 16 runs; the converging step ends the run before
   its update; `stationary` has none) `steps.csv` holds SciPy's own state: `H_k` before the update, the pair `s_k = α p_k`,
   `y_k = g_{k+1} − g_k`, SciPy's `H_{k+1}` and `g_{k+1}`, SciPy's next direction `p_{k+1}`, and the **exact results from the
   same inputs computed in mpmath (60 digits)**: the update (SciPy's formula, valid for any `H`) and `−H_{k+1} g_{k+1}`. SciPy's
   `H_k` are rebuilt with SciPy's own update code and checked bitwise against what SciPy did (every direction, the final
   `hess_inv`). The test feeds `(H_k, s_k, y_k)` to `inverse_update` and `(H_{k+1}, g_{k+1})` to `search_direction` and asserts
   each result against SciPy's and against the exact one, within the **derived single-step bound** below. No prefix, no fitted
   constant.

   **The single-step bound** (standard rank-2 update error analysis; Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed., §3.1 for `γ_k = kε/(1 − kε)`, ε = 2⁻⁵²). With `ρ = 1/(yᵀs)`, `u = H y`, `c = ρ + ρ² yᵀu`, the symmetric form the
   implementation is written in is `H' = H + c s sᵀ − ρ (u sᵀ + s uᵀ)`. Let `ū = |H||y|`, `c̄ = |ρ| + ρ²|y|ᵀū`, and the magnitude
   sum of its terms `M = |H| + c̄ |s||s|ᵀ + |ρ|(ū|s|ᵀ + |s|ūᵀ)`. The relative error of `ρ` is `γ_n κ` with the pair's
   conditioning `κ = |y|ᵀ|s| / |yᵀs|`; `ρ` enters every term (twice in `c`), `u` and `yᵀu` carry `γ_n`, and at most six more
   roundings (products, sums) follow. Hence, entrywise,

       |H'_computed − H'_exact|  ≤  (2κ + 4) γ_{n+6} M + |ρ||s||w|ᵀ                      (tol_M)

   where `w = |(Hᵀ − H) y|` accounts for the input's asymmetry (the symmetric form equals SciPy's formula only for a symmetric `H`;
   the `H_k` of the data are symmetric up to rounding). SciPy's own form `(I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ` has the magnitude
   sum `S = (I + |ρ||s||y|ᵀ) |H| (I + |ρ||y||s|ᵀ) + |ρ||s||s|ᵀ` (the terms *before* the cancellation in `I − ρ s yᵀ`) and two
   matrix products,

       |H'_SciPy − H'_exact|  ≤  (2κ + 4) γ_{2n+6} S                                    (tol_S)

   and the direction, a matrix–vector product, `|p_computed − p_exact| ≤ γ_n |H||g|` per side. The test asserts Rust vs exact
   within `tol_M` (plus `ε|exact|`, the data stores the exact value as a float), Rust vs SciPy within `tol_M + tol_S`, and for the
   direction `γ_n |H||g|` vs exact and `2 γ_n |H||g|` vs SciPy. **The generator asserts that SciPy itself satisfies tol_S and the
   direction bound against the exact result on every one of the 326 steps** and prints the largest ratio to the bound: 0.014 for
   `H`, 0.37 for `p` (a bound that SciPy violated would be a wrong derivation; the first version, with `|A₁|` instead of the
   magnitude sum before cancellation and the symmetric form as reference, was: 3.7). The ratios are diagnostics in `steps.csv`
   (`h_ratio`, `p_ratio`, `kappa`).

2. **Parity on the whole run: counts, iterations, termination, messages** (`tests/bfgs_replay.rs`): SciPy's recorded `(f, g)`
   go through the ask/tell state, which must ask for exactly SciPy's evaluations, in SciPy's number, iterations and termination
   (and the final `f`, exactly). This is exact on every run.

3. **Parity on the trajectory, secondary, only where it holds** (`tests/bfgs_replay.rs`, the comparison prefix below): requested
   points agree within `C ε x_scale + 2 x_slack` on each run's *comparison prefix*, with the **running magnitude scale**
   `x_scale = |x₀| + Σⱼ (|αⱼ pⱼ| + |x_{j+1}|) + |x_base| + |α p|` and `x_slack` = Σ|α||p_SciPy − d_exact|. `C = 32` is
   **calibrated** on an independent float64 mirror (the generator's `mirror`: SciPy's recorded `f`, `g`, a plain-Python BFGS in
   the symmetric form, summing left to right where BLAS does not): the first search whose ratio exceeds `C / 4 = 8` is the
   *first-outside index* (`cmp_first_outside`), and the committed prefix is the smallest first-outside index over the platforms
   observed (macOS arm64; Ubuntu 24.04 from CI run 46135920867, `LINUX_FIRST_OUTSIDE` in the generator) **minus `PREFIX_MARGIN = 2`,
   never longer, floored at 0**. The bound models rounding, not its amplification by the problem, so on a dense matrix the prefix
   is short: **runs with an empty or very short prefix are reported (the replay prints `n of N evaluations compared by value`) and
   are covered by claims 1 and 2 only.** Current prefixes (searches kept of the run's searches; first outside on macOS):
   `rosen2_c` 5 of 38 (7), `powell` 2 of 35 (6), `quad10_k1e4` 2 of 15 (5), and 0 (empty) for `rosen2_a`, `rosen2_b`,
   `rosen2_tight`, `rosen5`, `rosen2_xrtol`, `rosen2_maxiter_eq_nit`, `rosen2_maxiter5`; 6 runs are covered in full (`himmelblau`,
   `beale`, `quad5`, `quad10_k1e2`, `rosen2_maxiter1`, `stationary`). The final iterate and `H` are compared by value only
   for runs whose prefix is the whole run (`h_compared`). Observed maxima inside the prefixes (`x_ratio_max`): 3.12.
   There is no rule that refuses an empty prefix: the first Linux run showed that it cannot be guaranteed, and it does not need to
   be, because claim 1 does not depend on it.

   **The earlier scales were replaced.** The first version used the absolute-value recursion of the update, which compounds
   (10³⁶ on `rosen2_a`), and a calibrated 1e-8 relative envelope did the work; both are gone.

4. **Reproducibility: SciPy regenerated on another platform against SciPy committed** (`--check`, CI on Ubuntu/OpenBLAS while the
   data come from macOS/Accelerate). Counts, termination, message, call, options and row keys are **exact**; SciPy's recorded float
   states (`x f g p`, and in `steps.csv` `H_k s y H_{k+1} g_{k+1} p_{k+1}`) are compared with **one named constant,
   `SCIPY_REPRO_RTOL = 1e-6`**, per component, `|a − b| ≤ SCIPY_REPRO_RTOL · max(|a|, |b|, scale)`, `scale` = the largest absolute
   component of that column over the run's compared rows (for the final scalars of `runs.csv`, which are noise relative to
   themselves: the largest magnitude of the *same quantity over the run's evaluations*: `final_x` against the largest |x|, `final_f`
   against the largest |f|, `gnorm_inf` against the largest |g| component; CI run 46136919000 on Ubuntu had `quad5` `gnorm_inf` 7.0e-15 vs 2.5e-14), on the comparison prefix (rows past it: exact columns only), and the
   observed maximum over the rows past the prefix is *reported* per column, never failing: past the prefix SciPy's recorded
   `H_k` may differ between platforms by more than the constant, and that is exactly why the per-step replay compares Rust against
   the **committed** states, which is platform-independent by construction (the replay never regenerates SciPy).
   `SCIPY_REPRO_RTOL` is a **calibrated reproducibility tolerance of SciPy across BLAS/libm, not derived from the algorithm**;
   the only Linux observation for BFGS so far is the first CI run, which refused the old prefix design before reaching the
   float comparison, so the observed maximum per column is still to be confirmed by the next CI run. Calibration source for the
   constant: L-BFGS-B on Ubuntu (3.5e-8 at worst, constant 1e-6).

## What `--check` compares

`uv run scripts/gen_minimize_bfgs_reference.py --check` regenerates everything in memory, compares and writes nothing. It prints
every mismatch of an exact column in full (with the list of runs whose path changed), a summary per file and column (values
compared, outside, largest absolute difference, observed maximum relative difference next to the constant and the margin, rows
skipped) the margin of every prefix ("margins on this platform", smallest first) and, under "bounds on this platform", any prefix whose margin is below 2; every
comparison runs before the verdict. Exit 1 on any exact mismatch, trajectory value outside the constant, or bound violation.

| kind | columns | rule |
|---|---|---|
| exact | every column not listed below: ids, `group`, `problem`, options, `scipy_call`, `status`, `message`, `nit`, `nfev`, `njev`, `n_ls`, `wolfe2_*`, `wolfe1_fail`, `cache_hits`, `n_updates`, `n_rho1000`, `n_ys_nonpos`, `termination`, `parity`, `repro`, `search`, `ev_start`, `ev_end`, `applied`, `h_compared`, the keys and number of rows | string equality; a difference means the path of a recorded run changed on that platform: the run leaves `parity` with a documented reason, the check is never loosened |
| trajectory (SciPy's floats) | `evals` `x f g`; `iters` `x f g p alpha ys d_exact`; `steps` `h s y h_next g_next p_next` (rows with `k` inside the prefix; past it reported); `runs` `final_x final_f gnorm_inf` (whole run in the prefix); `hess_final` `h_scipy` (`h_compared`); `problems` `A b x_star f_star mu` | `SCIPY_REPRO_RTOL` per component with the run scale, on the comparison prefix |
| diagnostic | `x_scale`, `x_slack`, `x_ratio`, `d_scale`, `dir_ratio_*`, `ratio*`, `h_scale`, `x_ratio_max`, `dir_ratio_max`, `cmp_first_outside`, `cmp_nsearch` (checked: first-outside index on the running platform minus the committed prefix must be at least `PREFIX_MARGIN`), `h_exact`, `p_exact`, `kappa`, `h_ratio`, `p_ratio`, `note` | not compared (the generator re-measures them; it refuses data on which SciPy itself violates the single-step bound) |
| class runs (`repro = class`) | `w2_neggrad`, `w2_l1`, `w2_noisy`, `w2_pert`, `nan_region`, `inf_region` | only `id group problem n x0 options termination parity repro`: their counts depend on SciPy's evaluation cache and on the last bits of the arithmetic (`cache_hits`), which differ between platforms. `w2_unbounded` (exact arithmetic) is a full `exact` run |

The check is wired into the `minimize-reference` job of `.github/workflows/checks.yml`.

## Open points

- `SCIPY_REPRO_RTOL` and the first Ubuntu/OpenBLAS observation of the BFGS data (above); the replay bound is exercised only against
  the generator's mirror and a throwaway implementation so far, not against the real implementation.
- SciPy's `rho = 1000` branch (`yᵀs == 0`) is reached by `w2_unbounded` only (after a `wolfe2` step); no parity run
  reaches it, and no test pins what the port does with such a pair.
