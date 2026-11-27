# Reference data of `minimize::bfgs`

SciPy 1.16.3 runs of `scipy.optimize.minimize(fun, x0, jac=True, method="BFGS", bounds=None, options=...)`
(`_minimize_bfgs`, `_line_search_wolfe12`), recorded for the replay, live and property tests
(`../../bfgs_replay.rs`, `../../bfgs_live.rs`, ...). SciPy v1.16.3 is commit
`b9105ccc2237f57acb1060202cd77f6dd264fb34` (the commit tag `v1.16.3` points to; the tag object itself is
`56e567e70ccad213d04ab04e7cfe7365562a71f1`).

| File | Contents |
|---|---|
| `problems.csv` | the test functions: dimension, quadratic data `A`, `b` (numpy's generator is not mirrored in Rust), minimizer, a strong-convexity bound `mu` |
| `runs.csv` | one row per run: the exact SciPy call and every option, `status`, `message`, `nit`, `nfev`, `njev`, `wolfe2_calls`, where the fallback began, whether it returned a step and whether it warned, why `wolfe1` failed, `cache_hits`, the number of updates with `yᵀs == 0` and `<= 0`, the expected Rust termination, `parity` |
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

## Tolerances

Same policy as #39: mathematical parity, not bitwise agreement. Counts and decisions are exact; floats are
compared with a tolerance derived below and **confirmed on the committed runs by the generator**, which refuses to
write data that violates them with a 4× margin (and `--check` repeats the measurement).

| Quantity | Claim | Bound |
|---|---|---|
| evaluation count, iteration count, termination, `n_ls`, `cache_hits`, `n_updates`, `n_rho1000`, `n_ys_nonpos`, `wolfe2_*`, `applied` | equal to SciPy's | exact |
| `f` at every evaluation, final `f` | the replay feeds SciPy's `f`, so equal | exact |
| requested point `x_i` (evaluation `i`) | equals SciPy's `i`-th evaluation | `‖x_i − x_i^SciPy‖` componentwise ≤ min(C·n·ε·`x_scale`_i, 1e-8·(1 + ‖x_i^SciPy‖∞)), C = 64, ε = 2⁻⁵² |
| final iterate | equals SciPy's `final_x` | the same bound at the last evaluation (SciPy's accepted point is the last it evaluated) |
| direction `d_k = −H_k g_k` | equals the exact direction from SciPy's pairs, `d_exact` | ≤ min(C·n·ε·`d_scale`, 1e-8·‖`d_exact`‖∞) |
| final `H` | equals SciPy's `hess_inv`, and the exact `h_exact` | ≤ min(C·n·ε·`h_scale`, 1e-8·‖`h_exact`‖∞) |

**Derivation of the strict term.** `x = x_k + α p`, `p = −H g`. The matrix–vector product is computed to within
γₙ·|H||g| with γₙ = nε/(1 − nε) ≈ nε (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1),
and the sum with `x_k` adds one rounding; so one evaluation of `x` is right to ≈ nε·(|x_k| + α|H||g|). The
entries of `|H|` are themselves bounded by a recursion that follows the update formula: with `ρ = 1/(yᵀs)`,

- SciPy's form `H' = (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ`: `H̄' = |A₁| H̄ |A₂| + |ρ||s||s|ᵀ`;
- the symmetric form the implementation is written in, `H' = H + c s sᵀ − ρ(u sᵀ + s uᵀ)`, `u = H y`,
  `c = ρ + ρ² yᵀu`: `H̄' = H̄ + |c||s||s|ᵀ + |ρ|(|u||s|ᵀ + |s||u|ᵀ)`, `|u| = H̄|y|`.

`x_scale` is `|x_k| + α(H̄_SciPy + H̄_sym)|g|` (both forms enter: Rust and SciPy compute different expressions of the same
update), `d_scale = (H̄_SciPy + H̄_sym)|g|`, `h_scale = H̄_SciPy + H̄_sym`. C = 64 is a **measured** constant: a
pure-Python mirror of the Rust replay (the same algorithm in the symmetric form, summing left to right where BLAS
does not, feeding SciPy's recorded `f` and `g`, taking the step from SciPy's own `scalar_search_wolfe1`) has
|error| / (n ε `x_scale`) ≤ 10.3 on the 902 recorded evaluations (worst: the last evaluation of `quad10_k1e4`, κ = 10⁴),
≤ 0.24 for the directions of both the mirror and SciPy against `d_exact`, and ≤ 0.012 for the final `H`
of SciPy against `h_exact`. C = 64 is 6× the worst.

**Why there is an envelope.** The recursion above is a bound on local rounding and grows exponentially in the
number of updates (`x_scale` reaches 10³⁶ on `rosen2_a`): after about ten iterations it admits any difference.
The *propagation* of rounding differences through BFGS (a perturbed `p` moves `α`, which moves `s`, which moves `H`)
is not covered by it; measuring against the exact `|H||g|` instead of the recursion gives ratios up to 8·10⁴ (Powell), so no
constant times nεS bounds the replay uniformly, and none is claimed. The second term, 1e-8 relative, is the stated
tolerance for the replay's end-to-end difference: measured 2·10⁻¹² on the evaluations, 4.5·10⁻¹¹ on the directions
(relative to ‖d‖∞) and 9·10⁻¹⁴ on the final `H`, i.e. 200× margin, and it is far below what a wrong Wolfe
branch, a wrong step or a wrong update does (those move the next iterate by O(α‖p‖) or change the evaluation count). The README does not claim the envelope is a theorem; it is an observed envelope with a margin. **Do not loosen
either term without writing why.**

## What `--check` compares

`uv run scripts/gen_minimize_bfgs_reference.py --check` regenerates everything in memory, repeats the measurement
above, compares with the committed CSVs and writes nothing.

| Field | Rule | Why |
|---|---|---|
| ids, `group`, `problem`, `n`, options, `status`, `message`, counts, `wolfe2_*`, `wolfe1_fail`, `termination`, `parity`, `n_ls`, `cache_hits`, `n_updates`, `n_rho1000`, `n_ys_nonpos`, `applied`, `ev_start`, `ev_end`, problem data | exact | they pin the path of every run |
| `x`, `f`, `g`, `p`, `alpha`, `ys`, `d_exact`, `final_*`, `gnorm_inf`, `h_scipy`, `h_exact` | within 1e-8 relative + 1e-10 absolute | last bits of libm (`sin`, `pow`) and BLAS (`@`, `np.dot`) differ by platform and feed the next step: bitwise would pin them |
| `x_scale`, `d_scale`, `h_scale`, `x_ratio`, `dir_ratio_*`, `ratio`, `note`, `scipy_call` | not compared | derived diagnostics and text |

CI wiring of this check (the `minimize-reference` job) is not part of this stage.

## Open points

- The envelope is measured on one platform (macOS arm64, Accelerate); the platform check on ubuntu-24.04 x86_64
  (OpenBLAS) has not been run. If a count differs there, that run is chaotic and is dropped from `parity`.
- SciPy's `rho = 1000` branch (`yᵀs == 0`) is reached by `w2_unbounded` only (after a `wolfe2` step); no parity run
  reaches it, and no test pins what the port does with such a pair (open question 1 of the PR).
