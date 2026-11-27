# Reference data of `minimize::linesearch`

SciPy 1.16.3 runs of the Moré–Thuente search (`DCSRCH`, `scalar_search_wolfe1`), recorded for the
replay and live tests (`../../scipy_replay.rs`, `../../scipy_live.rs`, ...).

| File | Contents |
|---|---|
| `cases.csv` | one row per run: inputs, options, SciPy's task and step, counts |
| `trials.csv` | every trial `(alpha, phi, dphi)` SciPy evaluated, with `dphi_scale` and `dphi_n` |
| `divergences.csv` | runs where SciPy's own run and the closed-form run differ (none) |

Generate with `uv run scripts/gen_minimize_linesearch_reference.py` from the repository root.
Regenerating is a separate commit.

Produced on macOS 15.6.1 arm64, Python 3.12.11, SciPy 1.16.3, NumPy 2.3.5 (Accelerate BLAS,
Apple libm). CI checks the same data on ubuntu-24.04 x86_64 (OpenBLAS, glibc) with
`uv run scripts/gen_minimize_linesearch_reference.py --check`, which regenerates in memory,
compares, and writes nothing.

## What `--check` compares

| Field | Rule | Why |
|---|---|---|
| ids, run/kind/group/fn, `task`, `reason`, `n_trials`, `phi_evals`, `dphi_evals`, flags, `dphi_n`, None-ness of every column | exact | they pin the path of every search |
| steps (`alpha`, `stp`, `alpha1`), `phi`, inputs and coefficients | within 1e-12 relative | last bits of libm (`exp`, `sin`, `pow`) and BLAS (`np.dot`) differ by platform, and feed the next step; bitwise steps would pin them |
| `dphi` (trials) | within max(1e-12 \|dphi\|, 4 n ε `dphi_scale`), ε = 2⁻⁵² | cancellation noise, below |

The runs that converge at trial 1 (`lw1_line*`) have φ'(α) = ∇f·p ≈ 1e-15: pure cancellation noise
of a BLAS dot product, whose sign and size depend on the summation order. A sum of n terms is
computed to within γₙ Σ|terms|, γₙ = nε/(1 − nε) (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., §3.1), so `dphi_scale` is Σ|terms| of φ' at that trial and `dphi_n` the
number of terms and operations; the tolerance uses the committed values. The scale is:

- SciPy line runs: |p|ᵀ|M|(|x| + |α||p|) with M = 2I or A + Aᵀ, n = 42 (matrix-vector product, dot);
- the closed quadratic of a line: 2|α|·S_A + S_B, the absolute sums of the dot-product terms
  that form its coefficients A, B (these are dots too), n = 42;
- closed-form scalar functions: the sum of the absolute values of the terms the code adds
  (sf1: 3 terms, sf2/wt/mt2/mt3: 2, mt1: (α²+b)/(α²+b)² , mt4–mt6: 2 terms with square roots);
  where φ' is a single term (sf3, shiftq, lin_*) or non-finite helpers, |φ'|.

The Rust tests read these files by column name and replay the committed values; the convergence
tests hold for either rounding of the noise.
