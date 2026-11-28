# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["scipy==1.16.3", "numpy==2.3.5", "mpmath==1.3.0"]
# ///
"""Generate the SciPy reference data for `minimize::bfgs` (SciPy's `minimize(method="BFGS")`).

Run from the repository root:

    uv run scripts/gen_minimize_bfgs_reference.py            # regenerate the CSVs
    uv run scripts/gen_minimize_bfgs_reference.py --check    # compare with the committed CSVs, write nothing

Target (SciPy v1.16.3, commit b9105ccc2237f57acb1060202cd77f6dd264fb34, the commit the tag points to):

* ``scipy/optimize/_optimize.py``:  ``_minimize_bfgs``, ``_line_search_wolfe12``
* ``scipy/optimize/_linesearch.py``: ``line_search_wolfe1``, ``scalar_search_wolfe1`` (the Moré-Thuente
  search ``DCSRCH`` of ``_dcsrch.py``), and ``line_search_wolfe2`` for the fallback.

The exact SciPy call of every run is
``scipy.optimize.minimize(fun, x0, jac=True, method="BFGS", bounds=None, options=OPTIONS)`` with
``OPTIONS = {gtol, norm=inf, xrtol, c1, c2, maxiter}`` written into ``runs.csv`` (``scipy_call``);
``hess_inv0 = None`` (H0 = I), ``disp = False``, no callback halting. ``_minimize_bfgs`` has no ``maxls``;
its line search is ``_line_search_wolfe12(..., amin=1e-100, amax=1e100, c1, c2)``: ``wolfe1`` with
``xtol = 1e-14`` and ``maxiter = 100`` (fixed inside ``scalar_search_wolfe1``), and, if ``wolfe1`` returns None,
``wolfe2`` with ``maxiter = 10`` and the default ``c1``/``c2``/``amax`` passed through. These are recorded in
the data headers (``ls_maxiter``, ``wolfe2_maxiter``).

Hooks (they only observe; every number SciPy computes is unchanged):

* the objective is wrapped, so ``evals.csv`` holds every ``(x, f, g)`` SciPy asked for, in order;
* ``_optimize._line_search_wolfe12`` is wrapped, so ``iters.csv`` holds the iterate, the gradient, SciPy's
  direction ``p_k``, the steps and the evaluations of every line search, and whether it failed;
* ``_optimize.line_search_wolfe2`` is wrapped, so a run records ``wolfe2_calls``, the evaluation index
  ``wolfe2_first_eval`` at which the fallback began, whether it returned a step (``wolfe2_found``) and whether it
  did so with a ``LineSearchWarning`` (``wolfe2_warned``: ``scalar_search_wolfe2`` returns its last trial step,
  which need not satisfy the Wolfe conditions, when its ``maxiter`` is exhausted, and
  ``_line_search_wolfe12`` silences the warning);
* the objective calls the line searches make are counted (``cache_hits`` = trials minus evaluations): SciPy's
  ``ScalarFunction`` does not re-evaluate at a point equal to the previous one, so in the rounding regime
  (steps below one ulp of x) the recorded evaluations are fewer than the trials; parity runs must have none;
* ``y^T s`` of every accepted pair is recorded as ``_minimize_bfgs`` forms it (``ys``), with whether the update
  was applied, and ``n_rho1000`` / ``n_ys_nonpos`` count applied updates with ``y^T s == 0`` / ``<= 0``;
* ``DCSRCH.__call__`` is wrapped, so a failing ``wolfe1`` records its DCSRCH task (``wolfe1_fail``).

Outputs (committed; regenerating them is a separate commit), all in crates/minimize/tests/data/bfgs/:

* problems.csv    -- the test functions: dimension, quadratic data (A, b), minimizer, a strong-convexity bound
* runs.csv        -- one row per run: options, SciPy's status/message/counts, expected Rust termination
* evals.csv       -- every evaluation SciPy performed, with the rounding scale of the requested point
* iters.csv       -- every line search: x_k, g_k, SciPy's p_k, the exact (mpmath) direction and its scale
* hess_final.csv  -- SciPy's final ``hess_inv`` with its rounding scale and the exact (mpmath) matrix

Floats are written with ``repr(float(x))`` (shortest round-trip), vectors and matrices ``;``-separated
(row-major), ``inf``/``-inf``/``nan`` and ``none`` spelled out. No cell contains a comma or a quote.

Parity is claimed only on runs with ``wolfe2_calls == 0`` and a finite path (``parity = true``): where
SciPy's fallback fires the Rust port stops with ``LineSearchFailed`` (decision 3 on #40), and where the
objective is non-finite it stops at the last finite iterate (the #38 rule). The generator also measures, with
a pure-Python mirror of the replay that sums in a different order than BLAS, that the tolerance constant
documented in the README holds with margin (``--check`` repeats the measurement).
"""

from __future__ import annotations

import argparse
import datetime
import math
import platform
import sys
import warnings
from pathlib import Path

import mpmath
import numpy as np
import scipy
import scipy.optimize as so
import scipy.optimize._linesearch as lsm
import scipy.optimize._optimize as oo
from scipy.optimize import rosen, rosen_der
from scipy.optimize._dcsrch import DCSRCH

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "crates" / "minimize" / "tests" / "data" / "bfgs"

SCIPY_COMMIT = "b9105ccc2237f57acb1060202cd77f6dd264fb34"  # tag v1.16.3 peeled; the tag object is 56e567e
EPS = 2.0**-52
INF = math.inf
NAN = math.nan

# CALIBRATED constant of the Rust replay (not derived from n or the update): the requested point of every evaluation inside a
# run's comparison prefix satisfies |x_rust - x_SciPy| <= REPLAY_C * eps * x_scale + 2 * x_slack, componentwise (README,
# "Two claims, two mechanisms"). The generator measures the ratio of an independent float64 mirror of the algorithm and defines
# the prefix as the searches whose requested points stay <= REPLAY_C / MARGIN (directions are diagnostics: SciPy's own direction error
# enters the bound as x_slack, the mirror's amplifies with the problem and is reported, not bounded); the constant is never raised, the prefix is shortened.
REPLAY_C = 32.0
MARGIN = 4.0
# Every committed comparison prefix = (smallest first-outside index observed on any checked platform) - PREFIX_MARGIN, never longer
# (the rule of #41's L-BFGS data). Observed so far for BFGS: macOS arm64 only; the first CI run adds Linux and may only shorten.
PREFIX_MARGIN = 2
# First search outside the bound observed on Ubuntu 24.04 / OpenBLAS (CI run 46135920867; the first Linux run of the BFGS check,
# with the earlier mirror bound): there the trajectory prefix holds for only a few searches, because the n x n matrix carries BLAS
# rounding forward from the first update. The per-step replay (steps.csv) does not depend on it.
LINUX_FIRST_OUTSIDE = {"rosen2_a": 2, "rosen2_b": 0, "rosen2_c": 9, "rosen2_tight": 2, "rosen5": 1, "powell": 4, "quad10_k1e4": 4,
                       "rosen2_xrtol": 2, "rosen2_maxiter_eq_nit": 2, "rosen2_maxiter5": 2}
# CALIBRATED reproducibility tolerance of SciPy across platforms (BLAS, libm), used by --check only: per component
# |a - b| <= SCIPY_REPRO_RTOL * max(|a|, |b|, scale), scale = the largest |component| of that column over the run's compared rows.
# No Linux data for BFGS exists yet: chosen with a large margin after the L-BFGS observation (3.5e-8 at worst, constant 1e-6);
# the first CI run on Ubuntu prints the observed maxima and confirms or revises it.
SCIPY_REPRO_RTOL = 1e-6

HEADER = (
    f"# generated by scripts/gen_minimize_bfgs_reference.py on {datetime.date.today().isoformat()}\n"
    f"# scipy {scipy.__version__} ({SCIPY_COMMIT[:7]}), numpy {np.__version__}, mpmath {mpmath.__version__}, "
    f"python {platform.python_version()}\n"
    "# method=BFGS bounds=None ls_maxiter=100 wolfe1_xtol=1e-14 amin=1e-100 amax=1e100 wolfe2_maxiter=10\n"
)


# ----------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------


def fl(x) -> str:
    if x is None:
        return "none"
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return repr(x)


def vec(v) -> str:
    return ";".join(fl(t) for t in np.asarray(v, dtype=float).ravel())


def clean(s: str) -> str:
    return str(s).replace(",", " ").replace('"', "'").replace("\n", " ").strip()


# ----------------------------------------------------------------------------------------------
# Problems. The Rust mirror is tests/bfgs_common/mod.rs; quadratics are read from problems.csv.
# ----------------------------------------------------------------------------------------------


class Problem:
    def __init__(self, pid, n, f, g, *, x_star=None, f_star=None, mu=None, A=None, b=None, note=""):
        self.pid, self.n, self.f, self.g = pid, n, f, g
        self.x_star, self.f_star, self.mu, self.A, self.b, self.note = x_star, f_star, mu, A, b, note


def himmelblau(x):
    a, b = x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0
    return a * a + b * b


def himmelblau_der(x):
    a, b = x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0
    return np.array([4.0 * a * x[0] + 2.0 * b, 2.0 * a + 4.0 * b * x[1]])


_BEALE_C = (1.5, 2.25, 2.625)


def beale(x):
    return sum((c - x[0] * (1.0 - x[1] ** (i + 1))) ** 2 for i, c in enumerate(_BEALE_C))


def beale_der(x):
    g0 = g1 = 0.0
    for i, c in enumerate(_BEALE_C):
        k = i + 1
        r = c - x[0] * (1.0 - x[1] ** k)
        g0 += 2.0 * r * -(1.0 - x[1] ** k)
        g1 += 2.0 * r * (x[0] * k * x[1] ** (k - 1))
    return np.array([g0, g1])


def powell(x):  # Powell's singular function (singular Hessian at the minimizer 0)
    return (x[0] + 10 * x[1]) ** 2 + 5 * (x[2] - x[3]) ** 2 + (x[1] - 2 * x[2]) ** 4 + 10 * (x[0] - x[3]) ** 4


def powell_der(x):
    return np.array(
        [
            2 * (x[0] + 10 * x[1]) + 40 * (x[0] - x[3]) ** 3,
            20 * (x[0] + 10 * x[1]) + 4 * (x[1] - 2 * x[2]) ** 3,
            10 * (x[2] - x[3]) - 8 * (x[1] - 2 * x[2]) ** 3,
            -10 * (x[2] - x[3]) - 40 * (x[0] - x[3]) ** 3,
        ]
    )


def quadratic(A, b):
    return (lambda x: 0.5 * x @ A @ x - b @ x), (lambda x: A @ x - b)


def l1(x):
    return float(np.abs(x).sum())


def l1_der(x):
    return np.sign(x)


def noisy(x):  # constructed: Rosenbrock plus rounding-level oscillation
    return rosen(x) + 1e-12 * math.sin(1e7 * x[0])


def pert_der(x):  # constructed: an inconsistent gradient (constant offset 1e-3 * (1, -1))
    return rosen_der(x) + 1e-3 * np.array([1.0, -1.0])


def rosen_nan(x):  # constructed: non-finite region
    return NAN if x[0] > 0.5 else rosen(x)


def rosen_nan_der(x):
    return np.full(len(x), NAN) if x[0] > 0.5 else rosen_der(x)


def rosen_inf(x):
    return INF if x[0] > 0.5 else rosen(x)


def rosen_inf_der(x):
    return np.full(len(x), INF) if x[0] > 0.5 else rosen_der(x)


def _rosen_mu(n):
    """Smallest eigenvalue of the Rosenbrock Hessian at the minimizer (1,...,1), times 0.9."""
    x = np.ones(n)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += 1200 * x[i] ** 2 - 400 * x[i + 1] + 2
        H[i, i + 1] -= 400 * x[i]
        H[i + 1, i] -= 400 * x[i]
        H[i + 1, i + 1] += 200
    return 0.9 * float(np.linalg.eigvalsh(H)[0])


def _hess_mu(fder2, x, shrink=0.9):
    return shrink * float(np.linalg.eigvalsh(fder2(x))[0])


def _build_problems():
    ps = {}
    ps["rosen2"] = Problem("rosen2", 2, rosen, rosen_der, x_star=np.ones(2), f_star=0.0, mu=_rosen_mu(2),
                           note="mu: 0.9 * lambda_min of the Hessian at the minimizer; valid on the segment [x, x*] near x*")
    ps["rosen5"] = Problem("rosen5", 5, rosen, rosen_der, x_star=np.ones(5), f_star=0.0, mu=_rosen_mu(5),
                           note="as rosen2")
    ps["himmelblau"] = Problem(
        "himmelblau", 2, himmelblau, himmelblau_der, x_star=np.array([3.0, 2.0]), f_star=0.0,
        mu=_hess_mu(lambda x: np.array([[12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
                                        [4 * x[0] + 4 * x[1], 12 * x[1] ** 2 + 4 * x[0] - 26]]), np.array([3.0, 2.0])),
        note="minimizer (3,2); mu as rosen2")

    def beale_hess(x):
        h = np.zeros((2, 2))
        for i, c in enumerate(_BEALE_C):
            k = i + 1
            j = np.array([-(1.0 - x[1] ** k), x[0] * k * x[1] ** (k - 1)])
            h += 2.0 * np.outer(j, j)  # residuals vanish at the minimizer
        return h

    ps["beale"] = Problem("beale", 2, beale, beale_der, x_star=np.array([3.0, 0.5]), f_star=0.0,
                          mu=_hess_mu(beale_hess, np.array([3.0, 0.5])), note="minimizer (3, 0.5); zero residual, H = 2 J^T J")
    ps["powell"] = Problem("powell", 4, powell, powell_der, x_star=np.zeros(4), f_star=0.0, mu=None,
                           note="Hessian singular at the minimizer: no strong-convexity bound")
    # quadratics: A, b are data (numpy's generator is not mirrored in Rust)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    A = A @ A.T + 5.0 * np.eye(5)
    b = rng.standard_normal(5)
    f, g = quadratic(A, b)
    ps["quad5"] = Problem("quad5", 5, f, g, A=A, b=b, x_star=np.linalg.solve(A, b),
                          mu=float(np.linalg.eigvalsh(A)[0]),
                          note="default_rng(0): A = G G^T + 5 I with G standard normal 5x5, b standard normal; the probe on which SciPy's BFGS finishes a 5-dim SPD quadratic in n = 5 iterations")
    for kappa, name in ((1e2, "quad10_k1e2"), (1e4, "quad10_k1e4")):
        A = np.diag(np.logspace(0.0, math.log10(kappa), 10))
        b = np.ones(10)
        f, g = quadratic(A, b)
        ps[name] = Problem(name, 10, f, g, A=A, b=b, x_star=np.linalg.solve(A, b), mu=1.0,
                           note=f"A = diag(logspace(0, log10({kappa:g}), 10)), b = ones")
    A = np.eye(3)
    f, g = quadratic(A, np.zeros(3))
    ps["halfsq3"] = Problem("halfsq3", 3, f, g, A=A, b=np.zeros(3), x_star=np.zeros(3), mu=1.0, note="f = |x|^2/2")
    # constructed problems for the fallback and the non-finite runs
    ps["rosen2_neggrad"] = Problem("rosen2_neggrad", 2, rosen, lambda x: -rosen_der(x), note="constructed: the gradient has the wrong sign")
    ps["l1"] = Problem("l1", 3, l1, l1_der, note="constructed: |x|_1, nonsmooth at the minimizer")
    ps["linear_unbounded"] = Problem("linear_unbounded", 2, lambda x: -x[0], lambda x: np.array([-1.0, 0.0]),
                                     note="unbounded below: f = -x0, so wolfe1 never meets the curvature condition and ends at its 100-trial limit")
    ps["rosen2_noisy"] = Problem("rosen2_noisy", 2, noisy, rosen_der, note="constructed: rosen + 1e-12 sin(1e7 x0) with the smooth gradient")
    ps["rosen2_pert"] = Problem("rosen2_pert", 2, rosen, pert_der, note="constructed: rosen gradient plus the constant 1e-3 * (1, -1)")
    ps["rosen2_nan"] = Problem("rosen2_nan", 2, rosen_nan, rosen_nan_der, note="constructed: f and g are NaN for x0 > 0.5")
    ps["rosen2_inf"] = Problem("rosen2_inf", 2, rosen_inf, rosen_inf_der, note="constructed: f and g are +inf for x0 > 0.5")
    return ps


PROBLEMS = _build_problems()

# ----------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------

ROSEN5_X0 = [-1.2, 1.0, -1.2, 1.0, -1.2]


class Spec:
    def __init__(self, rid, group, problem, x0, *, gtol=1e-5, xrtol=0.0, c1=1e-4, c2=0.9, maxiter=None, note=""):
        self.rid, self.group, self.problem = rid, group, problem
        self.x0 = np.array(x0, dtype=float)
        self.gtol, self.xrtol, self.c1, self.c2 = gtol, xrtol, c1, c2
        self.maxiter = 200 * len(x0) if maxiter is None else maxiter
        self.note = note


def _specs():
    S = [
        Spec("rosen2_a", "finite", "rosen2", [-1.2, 1.0], note="the standard start"),
        Spec("rosen2_b", "finite", "rosen2", [0.0, 0.0]),
        Spec("rosen2_c", "finite", "rosen2", [2.0, -1.0]),
        Spec("rosen2_tight", "finite", "rosen2", [-1.2, 1.0], gtol=1e-10),
        Spec("rosen5", "finite", "rosen5", ROSEN5_X0, note="the 5-dim Rosenbrock start (-1.2, 1, -1.2, 1, -1.2)"),
        Spec("himmelblau", "finite", "himmelblau", [1.0, 1.0]),
        Spec("beale", "finite", "beale", [1.0, 1.0]),
        Spec("powell", "finite", "powell", [3.0, -1.0, 0.0, 1.0], note="singular Hessian at the minimizer"),
        Spec("quad5", "finite", "quad5", [0.0] * 5, gtol=1e-10, note="SciPy finishes this 5-dim SPD quadratic in n = 5 iterations (11 evaluations)"),
        Spec("quad10_k1e2", "finite", "quad10_k1e2", [0.0] * 10, gtol=1e-8),
        Spec("quad10_k1e4", "finite", "quad10_k1e4", [0.0] * 10, gtol=1e-8),
        Spec("stationary", "finite", "halfsq3", [0.0, 0.0, 0.0], note="x0 is stationary: GradTol after one evaluation, no line search"),
        Spec("rosen2_xrtol", "finite", "rosen2", [-1.2, 1.0], gtol=0.0, xrtol=1e-8,
             note="gtol = 0 cannot be met, so the xrtol test ends the run"),
        Spec("rosen2_maxiter1", "finite", "rosen2", [-1.2, 1.0], maxiter=1),
        Spec("rosen2_maxiter5", "finite", "rosen2", [-1.2, 1.0], maxiter=5),
        # maxiter = nit of rosen2_a is set in generate(): the converging iteration is the last allowed one
        Spec("w2_unbounded", "fallback", "linear_unbounded", [0.0, 0.0], maxiter=1,
             note="f is unbounded below: wolfe1 ends at its 100-trial limit on the first line search, wolfe2 FINDS a step, SciPy goes on; "
                  "exact arithmetic (f = -x0, g = (-1, 0)), no rounding noise"),
        Spec("w2_neggrad", "fallback", "rosen2_neggrad", [-1.2, 1.0], note="constructed: wolfe1 fails on the first line search"),
        Spec("w2_l1", "fallback", "l1", [1.0, 2.0, -3.0], note="constructed: nonsmooth objective"),
        Spec("w2_noisy", "fallback", "rosen2_noisy", [-1.2, 1.0], gtol=1e-8, note="constructed: rounding-level noise in f"),
        Spec("w2_pert", "fallback", "rosen2_pert", [-1.2, 1.0], note="constructed: inconsistent gradient"),
        Spec("nan_region", "nonfinite", "rosen2_nan", [-1.2, 1.0], note="constructed: NaN beyond x0 = 0.5"),
        Spec("inf_region", "nonfinite", "rosen2_inf", [-1.2, 1.0], note="constructed: +inf beyond x0 = 0.5"),
    ]
    return S


# ----------------------------------------------------------------------------------------------
# Running SciPy with the hooks
# ----------------------------------------------------------------------------------------------


class Log:
    def __init__(self):
        self.evals = []  # (x, f, g)
        self.ls = []  # one dict per _line_search_wolfe12 call
        self.callbacks = []  # (k, x, fun, n_evals)
        self.w2 = []  # (first_eval, found, warned)
        self.f_calls = 0  # objective calls the line searches made (trials), before ScalarFunction's cache
        self.tasks = []  # last DCSRCH task per call


def run_scipy(spec: Spec):
    prob = PROBLEMS[spec.problem]
    log = Log()

    def fg(x):
        f, g = prob.f(x), prob.g(x)
        log.evals.append((np.array(x, dtype=float), float(f), np.array(g, dtype=float)))
        return f, g

    orig12, orig2, origcall = oo._line_search_wolfe12, oo.line_search_wolfe2, DCSRCH.__call__

    def hooked12(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kw):
        rec = dict(k=len(log.ls), xk=np.array(xk, dtype=float), pk=np.array(pk, dtype=float),
                   gk=np.array(gfk, dtype=float), fk=float(old_fval), f_old=float(old_old_fval),
                   ev_start=len(log.evals), alpha=None, ev_end=None, failed=False, w2=None, w1_fail=None)
        log.ls.append(rec)
        n_w2, n_task = len(log.w2), len(log.tasks)

        def f_counted(x, *aa):
            log.f_calls += 1
            return f(x, *aa)

        try:
            ret = orig12(f_counted, fprime, xk, pk, gfk, old_fval, old_old_fval, **kw)
        except oo._LineSearchError:
            rec["failed"] = True
            rec["ev_end"] = len(log.evals)
            if len(log.w2) > n_w2:
                rec["w2"] = log.w2[n_w2]
            rec["w1_fail"] = _w1_reason(log, n_task)
            raise
        rec["alpha"] = float(ret[0])
        rec["ev_end"] = len(log.evals)
        if len(log.w2) > n_w2:
            rec["w2"] = log.w2[n_w2]
            rec["w1_fail"] = _w1_reason(log, n_task)
        return ret

    def hooked2(*a, **k):
        first = len(log.evals)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")  # inside _line_search_wolfe12's own "ignore": we see the warning
            ret = orig2(*a, **k)
        warned = any(issubclass(w.category, lsm.LineSearchWarning) for w in caught)
        log.w2.append((first, ret[0] is not None, warned))
        return ret

    def hookedcall(self, *a, **k):
        ret = origcall(self, *a, **k)
        log.tasks.append(ret[3].decode() if isinstance(ret[3], bytes) else str(ret[3]))
        return ret

    options = dict(gtol=spec.gtol, norm=INF, xrtol=spec.xrtol, c1=spec.c1, c2=spec.c2, maxiter=spec.maxiter)

    def cb(intermediate_result):
        log.callbacks.append((len(log.callbacks) + 1, np.array(intermediate_result.x, dtype=float),
                              float(intermediate_result.fun), len(log.evals)))

    oo._line_search_wolfe12, oo.line_search_wolfe2, DCSRCH.__call__ = hooked12, hooked2, hookedcall
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = so.minimize(fg, spec.x0.copy(), jac=True, method="BFGS", bounds=None, options=options, callback=cb)
    finally:
        oo._line_search_wolfe12, oo.line_search_wolfe2, DCSRCH.__call__ = orig12, orig2, origcall
    call = (f"scipy.optimize.minimize(fun, x0, jac=True, method='BFGS', bounds=None, "
            f"options={{gtol: {spec.gtol!r}, norm: inf, xrtol: {spec.xrtol!r}, c1: {spec.c1!r}, c2: {spec.c2!r}, maxiter: {spec.maxiter}}})")
    return res, log, call


def _w1_reason(log, n_task):
    """Why wolfe1 returned None: the DCSRCH task of the last wolfe1 call before the fallback."""
    tasks = log.tasks[n_task:]
    if not tasks:
        return "no_dcsrch_call"
    t = tasks[0] if len(tasks) == 1 else tasks[0]
    if t.startswith("FG"):
        return "maxiter_100"
    if t.startswith("ERROR"):
        return "start_error:" + clean(t)
    return clean(t)


# ----------------------------------------------------------------------------------------------
# Exact (mpmath) BFGS inverse Hessian from SciPy's pairs, and the absolute-value scale recursion
# ----------------------------------------------------------------------------------------------

mpmath.mp.dps = 60


def mp_updates(n, pairs):
    """H_k as mpf matrices for k = 0..len(pairs), H_0 = I, SciPy's recursion in exact arithmetic."""
    H = mpmath.eye(n)
    out = [H.copy()]
    for s, y in pairs:
        sm, ym = mpmath.matrix([mpmath.mpf(float(t)) for t in s]), mpmath.matrix([mpmath.mpf(float(t)) for t in y])
        rho = 1 / (ym.T * sm)[0]
        I = mpmath.eye(n)
        A1 = I - rho * sm * ym.T
        A2 = I - rho * ym * sm.T
        H = A1 * H * A2 + rho * sm * sm.T
        out.append(H.copy())
    return out


def gamma(k):
    """Higham's gamma_k = k eps / (1 - k eps)."""
    return k * EPS / (1.0 - k * EPS)


def step_bounds(H, s, y):
    """The derived single-step bounds (README, "Per-step replay"), entrywise, for the update H' = U(H, s, y) in floating point.

    With rho = 1/(y^T s), u = H y, c = rho + rho^2 y^T u, the symmetric form is H' = H + c s s^T - rho (u s^T + s u^T). The
    magnitude sum of its terms is M = |H| + cbar |s||s|^T + |rho| (ubar |s|^T + |s| ubar^T), ubar = |H||y|, cbar = |rho| + rho^2 |y|^T ubar.
    The relative error of rho is gamma_n kappa, kappa = |y|^T|s| / |y^T s| (the conditioning of the pair); rho enters
    every term (twice in c), u and y^T u carry gamma_n, and at most six more roundings (products, sums) follow, so

        |H'_computed - H'_exact|  <=  (2 kappa + 4) gamma_{n+6} M + |rho||s||w|^T    (tolM; any implementation of the symmetric form)

    where H'_exact is SciPy's formula in exact arithmetic (valid for any H) and w = |(H^T - H) y| is the asymmetry of the input
    matrix, by which the symmetric form differs from it (zero for a symmetric H)

    SciPy's form (I - rho s y^T) H (I - rho y s^T) + rho s s^T has the magnitude sum S = A1m |H| A2m + |rho||s||s|^T, A1m = I + |rho||s||y|^T
    (the terms before the cancellation in I - rho s y^T), with two
    matrix products (gamma_n each) and the same rho:

        |H'_SciPy - H'_exact|  <=  (2 kappa + 4) gamma_{2n+6} S               (tolS)

    The direction p = -H g is a matrix-vector product: |p_computed - p_exact| <= gamma_n |H||g| (tolp, per side).
    Returns (tolM, tolS, kappa).
    """
    n = len(s)
    sy = float(np.dot(s, y))
    rho = 1.0 / sy
    abss, absy, absH = np.abs(s), np.abs(y), np.abs(H)
    ubar = absH @ absy
    cbar = abs(rho) + rho * rho * float(absy @ ubar)
    M = absH + cbar * np.outer(abss, abss) + abs(rho) * (np.outer(ubar, abss) + np.outer(abss, ubar))
    I = np.eye(n)
    A1 = I + abs(rho) * np.outer(abss, absy)  # magnitude sums of I - rho s y^T and I - rho y s^T (before cancellation)
    A2 = I + abs(rho) * np.outer(absy, abss)
    S = A1 @ absH @ A2 + abs(rho) * np.outer(abss, abss)
    kappa = float(absy @ abss) / abs(sy)
    f = 2.0 * kappa + 4.0
    # the symmetric form equals SciPy's (and the exact general form A1 H A2 + rho s s^T) only for a symmetric H; the H the data
    # carries is symmetric up to rounding, and the two forms differ by -rho s w^T with w = (H^T - H) y
    w = np.abs((H.T - H) @ y)
    asym = abs(rho) * np.outer(abss, w)
    return f * gamma(n + 6) * M + asym, f * gamma(2 * n + 6) * S, kappa


def mp_step(H, s, y):
    """SciPy's update of float inputs in 60-digit arithmetic: (I - rho s y^T) H (I - rho y s^T) + rho s s^T, valid for any H."""
    n = len(s)
    Hm = mpmath.matrix([[mpmath.mpf(float(H[i][j])) for j in range(n)] for i in range(n)])
    sm = mpmath.matrix([mpmath.mpf(float(t)) for t in s])
    ym = mpmath.matrix([mpmath.mpf(float(t)) for t in y])
    rho = 1 / (ym.T * sm)[0]
    I = mpmath.eye(n)
    return (I - rho * sm * ym.T) * Hm * (I - rho * ym * sm.T) + rho * sm * sm.T


def scipy_update(H, s, y):
    """_minimize_bfgs's update, line for line."""
    n = len(s)
    I = np.eye(n, dtype=int)
    rhok = 1.0 / np.dot(y, s)
    A1 = I - s[:, np.newaxis] * y[np.newaxis, :] * rhok
    A2 = I - y[:, np.newaxis] * s[np.newaxis, :] * rhok
    return np.dot(A1, np.dot(H, A2)) + (rhok * s[:, np.newaxis] * s[np.newaxis, :])


# ----------------------------------------------------------------------------------------------
# The mirror of the Rust replay (different summation order than SciPy's BLAS)
# ----------------------------------------------------------------------------------------------


def _mv(H, v):  # left-to-right summation, the order BLAS does not use
    return [sum(H[i][j] * v[j] for j in range(len(v))) for i in range(len(v))]


def mirror(spec: Spec, log: Log, bases):
    """Replay SciPy's recorded (f, g) through a plain-Python BFGS, the way the Rust replay does, and measure how far its
    requested points and directions are from SciPy's / the exact ones in units of the derived bounds.

    The mirror feeds SciPy's recorded f and g, computes its own p = -H g, phi' = g . p, the step through SciPy's own
    ``scalar_search_wolfe1``, and the update (the symmetric form the Rust implementation is written in), summing left to
    right where BLAS does not. ``bases[j] = (scale_base, slack_base, x_base, p_scipy, d_exact, d_scale)`` for line search j
    (see ``generate``). Per evaluation i (search j, trial step a):

        x_scale_i = scale_base_j + |x_base_j| + |a p_j|          (a running magnitude sum, no recursion)
        x_slack_i = slack_base_j + |a| |p_scipy_j - d_exact_j|   (SciPy's own direction error, measured)
        x_ratio_i = max_c (|x_mirror - x_scipy| - 2 x_slack)_c / (eps x_scale_c)

    Returns (per-evaluation dict, per-search direction ratios, final H).
    """
    n = len(spec.x0)
    ev = log.evals
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    x = [float(t) for t in spec.x0]
    g = list(ev[0][2])
    f = ev[0][1]
    f_old = f + float(np.linalg.norm(ev[0][2])) / 2.0
    idx = 1
    per_eval = {}
    dir_ratio = []
    tiny = np.finfo(float).tiny
    for j, rec in enumerate(log.ls):
        if rec["failed"] or rec["w2"] is not None:
            break
        scale_base, slack_base, x_base, p_scipy, d_exact, d_scale = bases[j]
        p = [-t for t in _mv(H, g)]
        dir_ratio.append(float(np.max(np.abs(np.array(p) - d_exact) / (EPS * np.maximum(d_scale, tiny)))))
        xk = list(x)
        state = {"i": idx}

        def phi(a, xk=xk, p=p, state=state, j=j, scale_base=scale_base, slack_base=slack_base, x_base=x_base,
                p_scipy=p_scipy, d_exact=d_exact):
            i = state["i"]
            xr = np.array([xk[c] + a * p[c] for c in range(n)])
            scale = scale_base + np.abs(x_base) + abs(a) * np.abs(p_scipy)
            slack = slack_base + abs(a) * np.abs(p_scipy - d_exact)
            err = np.abs(xr - ev[i][0])
            ratio = float(np.max(np.maximum(err - 2 * slack, 0.0) / (EPS * np.maximum(scale, tiny))))
            per_eval[i] = dict(search=j, scale=scale, slack=slack, ratio=ratio)
            return ev[i][1]

        def dphi(a, p=p, state=state):
            i = state["i"]
            d = sum(ev[i][2][c] * p[c] for c in range(n))
            state["i"] = i + 1
            return d

        derphi0 = sum(g[c] * p[c] for c in range(n))
        alpha, phi1, _ = lsm.scalar_search_wolfe1(phi, dphi, f, f_old, derphi0, c1=spec.c1, c2=spec.c2,
                                                  amax=1e100, amin=1e-100, xtol=1e-14)
        if alpha is None:
            break
        i_last = state["i"] - 1
        s = [alpha * t for t in p]
        x = [x[c] + s[c] for c in range(n)]
        g_new = list(ev[i_last][2])
        y = [g_new[c] - g[c] for c in range(n)]
        f_old, f = f, phi1
        g = g_new
        idx = state["i"]
        gn = max(abs(t) for t in g)
        pn = math.sqrt(sum(t * t for t in p))
        xn = math.sqrt(sum(t * t for t in x))
        if gn <= spec.gtol or alpha * pn <= spec.xrtol * (spec.xrtol + xn) or not math.isfinite(f):
            break
        # update, symmetric form: H += c s s^T - rho (u s^T + s u^T), u = H y
        sy = sum(s[c] * y[c] for c in range(n))
        rho = 1.0 / sy
        u = _mv(H, y)
        c1 = rho + rho * rho * sum(y[c] * u[c] for c in range(n))
        H = [[H[r][c] + c1 * s[r] * s[c] - rho * (u[r] * s[c] + s[r] * u[c]) for c in range(n)] for r in range(n)]
    return per_eval, dir_ratio, H


# ----------------------------------------------------------------------------------------------
# Building the tables
# ----------------------------------------------------------------------------------------------


def termination_of(spec, res, log):
    """The Rust `Termination` this run is expected to map to (names of minimize::bfgs::Termination)."""
    gnorm = float(np.max(np.abs(res.jac)))
    if spec.group == "fallback":
        return "LineSearchFailed"  # the Rust port stops where SciPy's wolfe1 gives up (decision 3 on #40)
    if spec.group == "nonfinite":
        return "NonFiniteValue"  # the #38 rule: stop at the last finite iterate (SciPy differs)
    if res.status == 1:
        return "MaxIter"
    if res.status == 0:
        return "GradTol" if gnorm <= spec.gtol else "StepTol"
    if res.status == 2:
        if any(r["failed"] for r in log.ls):
            return "LineSearchFailed"
        return "NonFiniteValue"
    return "NonFiniteValue"  # status 3: NaN


def generate():
    specs = _specs()
    # the maxiter-quirk run needs nit of rosen2_a
    base = run_scipy(next(s for s in specs if s.rid == "rosen2_a"))[0]
    specs.insert(specs.index(next(s for s in specs if s.rid == "rosen2_maxiter1")),
                 Spec("rosen2_maxiter_eq_nit", "finite", "rosen2", [-1.2, 1.0], maxiter=int(base.nit),
                      note="converges on the last allowed iteration: SciPy reports status 1 (k >= maxiter is tested after the loop)"))
    out = dict(problems=[], runs=[], evals=[], iters=[], hess=[], steps=[])
    for spec in specs:
        res, log, call = run_scipy(spec)
        n = len(spec.x0)
        assert res.nfev == len(log.evals), (spec.rid, res.nfev, len(log.evals))
        if not log.w2:  # wolfe1 evaluates f and g together; wolfe2 asks for them separately
            assert res.njev == res.nfev, (spec.rid, res.nfev, res.njev)
        wolfe2_calls = len(log.w2)
        first_w2 = log.w2[0][0] if log.w2 else None
        w2_found = [found for _, found, _ in log.w2]
        w2_warned = [warned for _, _, warned in log.w2]
        cache_hits = log.f_calls - (res.nfev - 1)  # trials the line searches made minus evaluations SciPy's cache let through
        parity = spec.group == "finite" and wolfe2_calls == 0
        if spec.group == "fallback":
            assert wolfe2_calls > 0, f"{spec.rid}: no fallback fired"
        if spec.group == "finite":
            assert wolfe2_calls == 0, f"{spec.rid}: fallback fired in a parity run"
        term = termination_of(spec, res, log)
        w1_fail = next((r["w1_fail"] for r in log.ls if r["w1_fail"]), None)
        gnorm = float(np.max(np.abs(res.jac)))
        prob = PROBLEMS[spec.problem]
        if parity and prob.mu is not None and term == "GradTol":
            err = float(np.linalg.norm(res.x - prob.x_star))
            assert err <= float(np.linalg.norm(res.jac)) / prob.mu * (1 + 1e-9), (spec.rid, err)

        # pairs as SciPy formed them, and which of them were applied to H
        pairs, applied, ys_list = [], [], []
        for j, rec in enumerate(log.ls):
            if rec["alpha"] is None:
                ys_list.append(None)
                applied.append(None)
                break
            s = rec["alpha"] * rec["pk"]
            g_next = log.evals[rec["ev_end"] - 1][2]
            y = g_next - rec["gk"]
            f_next = log.evals[rec["ev_end"] - 1][1]
            x_next = rec["xk"] + s
            gn = float(np.max(np.abs(g_next)))
            stop = (gn <= spec.gtol) or (rec["alpha"] * float(np.linalg.norm(rec["pk"])) <=
                                        spec.xrtol * (spec.xrtol + float(np.linalg.norm(x_next)))) or not math.isfinite(f_next)
            if not stop:
                pairs.append((s, y))
            applied.append(not stop)
            ys_list.append(float(np.dot(y, s)))  # as _minimize_bfgs forms rhok_inv
        applied_ys = [ys for ys, ap in zip(ys_list, applied) if ap]
        n_rho1000 = sum(1 for t in applied_ys if t == 0.0)
        n_ys_nonpos = sum(1 for t in applied_ys if t <= 0.0)
        if parity:
            assert cache_hits == 0, f"{spec.rid}: SciPy's cache skipped {cache_hits} evaluations: evaluations are not one per trial"
            assert n_ys_nonpos == 0, f"{spec.rid}: a parity run applied an update with y^T s <= 0"
        mpH = mp_updates(n, pairs) if parity else None

        out["runs"].append(dict(
            id=spec.rid, group=spec.group, problem=spec.problem, n=n, x0=vec(spec.x0), gtol=fl(spec.gtol),
            xrtol=fl(spec.xrtol), c1=fl(spec.c1), c2=fl(spec.c2), maxiter=spec.maxiter, ls_maxiter=100,
            scipy_call=clean(call), status=res.status, message=clean(res.message), nit=res.nit, nfev=res.nfev,
            njev=res.njev, wolfe2_calls=wolfe2_calls, wolfe2_first_eval=("none" if first_w2 is None else first_w2),
            wolfe2_found=("none" if not w2_found else "".join("1" if t else "0" for t in w2_found)),
            wolfe2_warned=("none" if not w2_warned else "".join("1" if t else "0" for t in w2_warned)),
            cache_hits=cache_hits, n_updates=len(applied_ys), n_rho1000=n_rho1000, n_ys_nonpos=n_ys_nonpos,
            repro=("class" if spec.rid in CLASS_RUNS else "exact"),
            wolfe1_fail=("none" if w1_fail is None else clean(w1_fail)), termination=term,
            parity=("true" if parity else "false"), final_x=vec(res.x), final_f=fl(res.fun),
            gnorm_inf=fl(gnorm), n_ls=len(log.ls), cmp_nsearch="none", cmp_first_outside="none", x_ratio_max="none", dir_ratio_max="none",
            note=clean(spec.note + " | " + prob.note)))

        for i, (x, f, g) in enumerate(log.evals):
            out["evals"].append(dict(run=spec.rid, i=i, search="none", x=vec(x), f=fl(f), g=vec(g), x_scale="none",
                                     x_slack="none", x_ratio="none"))

        if parity:
            # per search: running magnitude scale, SciPy's own direction error, exact direction and its scale
            Hex_k = [np.array([[float(Hk[r, c]) for c in range(n)] for r in range(n)]) for Hk in mpH]
            bases, d_exact_all = [], []
            scale_base, slack_base = np.abs(spec.x0), np.zeros(n)
            for j, rec in enumerate(log.ls):
                k = min(j, len(mpH) - 1)
                gm = mpmath.matrix([mpmath.mpf(float(t)) for t in rec["gk"]])
                d_exact_f = np.array([float(-t) for t in (mpH[k] * gm)])
                d_scale = np.abs(Hex_k[k]) @ np.abs(rec["gk"])
                bases.append((scale_base.copy(), slack_base.copy(), rec["xk"], rec["pk"], d_exact_f, d_scale))
                x_next = log.ls[j + 1]["xk"] if j + 1 < len(log.ls) else res.x
                scale_base = scale_base + np.abs(rec["alpha"] * rec["pk"]) + np.abs(x_next)
                slack_base = slack_base + abs(rec["alpha"]) * np.abs(rec["pk"] - d_exact_f)
                d_exact_all.append(d_exact_f)
            # SciPy's own per-iteration state, rebuilt with SciPy's own update code and checked against what SciPy did
            Hs_chain = np.eye(n, dtype=int)
            for j, rec in enumerate(log.ls):
                assert np.array_equal(-np.dot(Hs_chain, rec["gk"]), rec["pk"]), (spec.rid, j, "direction")
                if applied[j] is not True:
                    continue
                s_j = rec["alpha"] * rec["pk"]
                g_next = log.evals[rec["ev_end"] - 1][2]
                y_j = g_next - rec["gk"]
                H_next = scipy_update(Hs_chain, s_j, y_j)
                p_next = -np.dot(H_next, g_next)
                if j + 1 < len(log.ls):
                    assert np.array_equal(p_next, log.ls[j + 1]["pk"]), (spec.rid, j, "next direction")
                H_k = np.array(Hs_chain, dtype=float)
                tolM, tolS, kappa = step_bounds(H_k, s_j, y_j)
                Hex = mp_step(H_k, s_j, y_j)
                H_ex = np.array([[float(Hex[a, b]) for b in range(n)] for a in range(n)])
                Hn_m = mpmath.matrix([[mpmath.mpf(float(H_next[a][b])) for b in range(n)] for a in range(n)])
                gm = mpmath.matrix([mpmath.mpf(float(t)) for t in g_next])
                p_ex = np.array([float(-t) for t in (Hn_m * gm)])
                tolp = gamma(n) * (np.abs(H_next) @ np.abs(g_next))
                Hex_m = Hex
                h_err = max(float(abs(mpmath.mpf(float(H_next[a][b])) - Hex_m[a, b])) / tolS[a, b] for a in range(n) for b in range(n))
                p_err = float(np.max(np.abs(p_next - p_ex) / np.maximum(tolp, np.finfo(float).tiny)))
                assert h_err <= 1.0 and p_err <= 1.0, (spec.rid, j, "the derived single-step bound is violated by SciPy itself", h_err, p_err)
                out["steps"].append(dict(
                    run=spec.rid, k=j, n=n, h=vec(H_k), s=vec(s_j), y=vec(y_j), h_next=vec(H_next), g_next=vec(g_next),
                    p_next=vec(p_next), h_exact=vec(H_ex), p_exact=vec(p_ex), kappa=fl(kappa), h_ratio=fl(h_err),
                    p_ratio=fl(p_err)))
                Hs_chain = H_next
            assert np.array_equal(np.array(Hs_chain, dtype=float), np.array(res.hess_inv, dtype=float)), (spec.rid, "final H")
            per_eval, dir_ratio_m, H_mirror = mirror(spec, log, bases)
            base0 = len(out["evals"]) - len(log.evals)
            out["evals"][base0].update(search=0, x_scale=vec(np.abs(spec.x0)), x_slack=vec(np.zeros(n)), x_ratio=fl(0.0))
            for i, pe in per_eval.items():
                out["evals"][base0 + i].update(search=pe["search"], x_scale=vec(pe["scale"]), x_slack=vec(pe["slack"]),
                                               x_ratio=fl(pe["ratio"]))
            # the comparison prefix: the first search whose mirror ratio exceeds C / MARGIN is the first-outside index; the
            # committed prefix keeps PREFIX_MARGIN searches below it (a runner-to-runner difference of one search was observed
            # for L-BFGS on GitHub runners of the same image), never longer. A run with no search outside is covered in full.
            first_outside = next((j for j in range(len(log.ls))
                                  if max([pe["ratio"] for pe in per_eval.values() if pe["search"] == j], default=0.0)
                                  > REPLAY_C / MARGIN), None)
            seen = [x for x in (first_outside, LINUX_FIRST_OUTSIDE.get(spec.rid)) if x is not None]
            cmp_n = len(log.ls) if not seen else max(min(seen) - PREFIX_MARGIN, 0)
            run_row = out["runs"][-1]
            run_row["cmp_nsearch"] = cmp_n
            run_row["cmp_first_outside"] = "none" if first_outside is None else first_outside
            in_prefix = [pe["ratio"] for pe in per_eval.values() if pe["search"] < cmp_n]
            run_row["x_ratio_max"] = fl(max(in_prefix, default=0.0))
            run_row["dir_ratio_max"] = fl(max(dir_ratio_m[:cmp_n], default=0.0))  # diagnostic: the amplification of H's error
            full_prefix = cmp_n == len(log.ls)
            for j, rec in enumerate(log.ls):
                d_exact_f, d_scale = d_exact_all[j], bases[j][5]
                r_scipy = float(np.max(np.abs(rec["pk"] - d_exact_f) / (EPS * np.maximum(d_scale, np.finfo(float).tiny))))
                r_mirror = dir_ratio_m[j] if j < len(dir_ratio_m) else None
                out["iters"].append(dict(
                    run=spec.rid, k=j, x=vec(rec["xk"]), f=fl(rec["fk"]), g=vec(rec["gk"]), p=vec(rec["pk"]),
                    d_exact=vec(d_exact_f), d_scale=vec(d_scale), alpha=fl(rec["alpha"]),
                    ev_start=rec["ev_start"], ev_end=rec["ev_end"], ys=fl(ys_list[j]),
                    applied=("none" if applied[j] is None else str(applied[j]).lower()),
                    dir_ratio_scipy=fl(r_scipy), dir_ratio_mirror=fl(r_mirror)))
            # final inverse Hessian: SciPy's and the mirror's vs exact, in units of eps max|H_exact|
            Hs = np.array(res.hess_inv)
            Hex = Hex_k[-1]
            hs = float(np.max(np.abs(Hex)))
            r_h = float(np.max(np.abs(Hs - Hex))) / (EPS * hs)
            r_hm = float(np.max(np.abs(np.array(H_mirror) - Hex))) / (EPS * hs)
            h_ok = full_prefix and r_hm <= REPLAY_C / MARGIN
            out["hess"].append(dict(run=spec.rid, n=n, n_updates=len(pairs), h_compared="true" if h_ok else "false",
                                    h_scipy=vec(Hs), h_exact=vec(Hex), h_scale=fl(hs), ratio=fl(r_h), ratio_mirror=fl(r_hm)))
        else:
            for j, rec in enumerate(log.ls):
                out["iters"].append(dict(
                    run=spec.rid, k=j, x=vec(rec["xk"]), f=fl(rec["fk"]), g=vec(rec["gk"]), p=vec(rec["pk"]),
                    d_exact="none", d_scale="none", alpha=fl(rec["alpha"]), ev_start=rec["ev_start"],
                    ev_end=rec["ev_end"], ys=fl(ys_list[j]),
                    applied=("none" if applied[j] is None else str(applied[j]).lower()),
                    dir_ratio_scipy="none", dir_ratio_mirror="none"))

    for p in PROBLEMS.values():
        out["problems"].append(dict(
            id=p.pid, n=p.n, A=("none" if p.A is None else vec(p.A)), b=("none" if p.b is None else vec(p.b)),
            x_star=("none" if p.x_star is None else vec(p.x_star)), f_star=fl(p.f_star), mu=fl(p.mu),
            note=clean(p.note)))
    stats = {}
    return out, stats


COLUMNS = dict(
    problems=["id", "n", "A", "b", "x_star", "f_star", "mu", "note"],
    runs=["id", "group", "problem", "n", "x0", "gtol", "xrtol", "c1", "c2", "maxiter", "ls_maxiter", "scipy_call",
          "status", "message", "nit", "nfev", "njev", "wolfe2_calls", "wolfe2_first_eval", "wolfe2_found",
          "wolfe2_warned", "cache_hits", "n_updates", "n_rho1000", "n_ys_nonpos", "repro",
          "wolfe1_fail", "termination", "parity", "final_x", "final_f", "gnorm_inf", "n_ls", "cmp_nsearch", "cmp_first_outside",
          "x_ratio_max", "dir_ratio_max", "note"],
    evals=["run", "i", "search", "x", "f", "g", "x_scale", "x_slack", "x_ratio"],
    iters=["run", "k", "x", "f", "g", "p", "d_exact", "d_scale", "alpha", "ev_start", "ev_end", "ys", "applied",
           "dir_ratio_scipy", "dir_ratio_mirror"],
    steps=["run", "k", "n", "h", "s", "y", "h_next", "g_next", "p_next", "h_exact", "p_exact", "kappa", "h_ratio", "p_ratio"],
    hess=["run", "n", "n_updates", "h_compared", "h_scipy", "h_exact", "h_scale", "ratio", "ratio_mirror"],
)
FILES = dict(problems="problems.csv", runs="runs.csv", evals="evals.csv", iters="iters.csv", hess="hess_final.csv", steps="steps.csv")

# `repro` (runs.csv, data): "exact" runs are compared by every count and by value on their comparison prefix; "class" runs
# (the fallback runs whose failure is a rounding-regime effect, and the non-finite ones) are compared by group, problem,
# options, termination and parity only, because their evaluation counts depend on SciPy's evaluation cache and on the last
# bits of the arithmetic (`cache_hits` > 0), which differ between platforms.
CLASS_RUNS = {"w2_neggrad", "w2_l1", "w2_noisy", "w2_pert", "nan_region", "inf_region"}
CLASS_EXACT = {"id", "group", "problem", "n", "x0", "gtol", "xrtol", "c1", "c2", "maxiter", "ls_maxiter", "termination",
               "parity", "repro"}


def render(out):
    texts = {}
    for key, cols in COLUMNS.items():
        lines = [HEADER.rstrip("\n"), ",".join(cols)]
        for row in out[key]:
            cells = [str(row[c]) for c in cols]
            for c in cells:
                assert "," not in c and '"' not in c, (key, c)
            lines.append(",".join(cells))
        texts[FILES[key]] = "\n".join(lines) + "\n"
    return texts


def assert_derivation(out):  # noqa
    """The README's claims, confirmed on the regenerated data: a run's comparison prefix is never empty for a run with line
    searches, and the final H of a run is compared only if the mirror agrees. Prints the observed maxima."""
    runs = [r for r in out["runs"] if r["parity"] == "true"]
    full = sum(1 for r in runs if r["cmp_nsearch"] == r["n_ls"])
    xr = max(float(r["x_ratio_max"]) for r in runs)
    dr = max(float(r["dir_ratio_max"]) for r in runs)
    print(f"comparison prefixes: {full} of {len(runs)} parity runs are covered in full; observed maxima inside the prefixes: "
          f"x {xr:.3g} (units eps x_scale); the mirror's direction vs exact reaches {dr:.3g} (units eps d_scale, a diagnostic); calibrated C = {REPLAY_C:g}, "
          f"prefix bound C/{MARGIN:g} = {REPLAY_C / MARGIN:g}")
    short = [(r["id"], r["cmp_nsearch"], r["n_ls"]) for r in runs if r["cmp_nsearch"] != r["n_ls"]]
    if short:
        print("  shorter prefixes (run, searches compared, searches):", ", ".join(f"{a} {b}/{c}" for a, b, c in short))
    print(f"  prefix margin rule: committed prefix = min(first-outside index on macOS, on Linux) - {PREFIX_MARGIN}")
    empty = [r["id"] for r in runs if r["cmp_nsearch"] < 1 and r["n_ls"] > 0]
    if empty:
        print(f"  runs with an empty trajectory prefix (counts and termination only; the per-step replay covers them): {', '.join(empty)}")
    steps = out["steps"]
    print(f"  per-step replay: {len(steps)} iterations of {len(runs)} parity runs; SciPy vs exact (mpmath), largest ratio to the derived bound: "
          f"H {max((float(t['h_ratio']) for t in steps), default=0):.3g}, p {max((float(t['p_ratio']) for t in steps), default=0):.3g}")
    for r in runs:
        assert float(r["x_ratio_max"]) <= REPLAY_C / MARGIN


# ----------------------------------------------------------------------------------------------
# --check
# ----------------------------------------------------------------------------------------------
#
# Two claims, two mechanisms (data/bfgs/README.md): the Rust replay uses the derived bound on the comparison prefix; --check asks
# whether SciPy regenerated on this platform reproduces the committed SciPy, which is an empirical property of BLAS/libm.
#
# * exact: every column that is not listed below (ids, options, call, message, status, every count, wolfe2_*, termination,
#   parity, search/ev ranges, applied, the number of rows per file); for `repro = class` runs only CLASS_EXACT;
# * trajectory (SciPy's floats): evals x f g; iters x f g p alpha ys d_exact; runs final_x final_f gnorm_inf; hess h_scipy
#   h_exact; problems A b x_star f_star mu. Per component |a - b| <= SCIPY_REPRO_RTOL * max(|a|, |b|, scale), scale = the largest
#   |component| of that column over the run's compared rows (committed values); only rows inside the committed comparison
#   prefix (evals: search < cmp_nsearch; iters: k < cmp_nsearch; runs and hess: only if the whole run is inside it);
# * diagnostic: scale, slack, ratio, *_max columns, h_scale, scipy_call, note: not compared. The committed cmp_nsearch must not be
#   longer than the regenerated one (the committed prefix would be outside the bound on this platform).

TRAJ = {"evals.csv": {"x", "f", "g"}, "iters.csv": {"x", "f", "g", "p", "alpha", "ys", "d_exact"},
        "runs.csv": {"final_x", "final_f", "gnorm_inf"}, "hess_final.csv": {"h_scipy", "h_exact"},
        "problems.csv": {"A", "b", "x_star", "f_star", "mu"},
        "steps.csv": {"h", "s", "y", "h_next", "g_next", "p_next"}}
DIAGNOSTIC = {"x_scale", "x_slack", "x_ratio", "d_scale", "dir_ratio_scipy", "dir_ratio_mirror", "ratio", "ratio_mirror",
              "h_scale", "x_ratio_max", "dir_ratio_max", "scipy_call", "note", "cmp_nsearch", "cmp_first_outside",
              "h_exact", "p_exact", "kappa", "h_ratio", "p_ratio"}


def _split_data(text):
    rows = [l for l in text.splitlines() if l and not l.startswith("#")]
    head = rows[0].split(",")
    return head, [dict(zip(head, r.split(","))) for r in rows[1:]]


def _num(t):
    try:
        return float(t)
    except ValueError:
        return None


def _run_of(name, row):
    return row.get("run") or row.get("id")


def check(data_dir: Path) -> int:
    out, _ = generate()
    assert_derivation(out)
    fresh = render(out)
    exact_bad = []  # (file, ident, column, committed, regenerated)
    info = {}  # steps.csv columns: largest relative difference on rows PAST the prefix (reported, never failing)
    traj = {}  # (file, column) -> [compared, outside, max abs, max rel]
    skipped = {}
    bounds = []
    outside_examples = []
    committed = {}
    for name in fresh:
        path = data_dir / name
        if not path.exists():
            print(f"MISSING {name}")
            exact_bad.append((name, "-", "file", "missing", "present"))
            continue
        committed[name] = _split_data(path.read_text())
    # run metadata from the committed runs.csv
    runs_c = {r["id"]: r for r in committed.get("runs.csv", ([], []))[1]}
    cmp_n = {i: (int(r["cmp_nsearch"]) if r["cmp_nsearch"] != "none" else None) for i, r in runs_c.items()}
    n_ls = {i: int(r["n_ls"]) for i, r in runs_c.items()}
    runs_f = {r["id"]: r for r in _split_data(fresh["runs.csv"])[1]}
    margins = []
    for i, r in runs_c.items():
        if r["parity"] == "true" and i in runs_f and runs_f[i]["cmp_first_outside"] != "none":
            fo = int(runs_f[i]["cmp_first_outside"])
            m = fo - int(r["cmp_nsearch"])
            margins.append((m, i, f"committed prefix {r['cmp_nsearch']} of {r['n_ls']}, first search outside the bound here {fo}"))
            if m < PREFIX_MARGIN:
                bounds.append(f"run {i}: first search outside the bound on this platform is {fo}, committed prefix {r['cmp_nsearch']}: margin {m} < {PREFIX_MARGIN}")
    for name in fresh:
        if name not in committed:
            continue
        h1, r1 = committed[name]
        h2, r2 = _split_data(fresh[name])
        if h1 != h2:
            exact_bad.append((name, "-", "columns", ",".join(h1), ",".join(h2)))
            continue
        # rows of class runs are not compared; the others row by row
        def keep(row):
            return runs_c.get(_run_of(name, row), {}).get("repro", "exact") == "exact" if name != "problems.csv" else True
        a_rows = [r for r in r1 if keep(r)]
        b_rows = [r for r in r2 if keep(r)]
        skipped[name] = len(r1) - len(a_rows)
        if len(a_rows) != len(b_rows):
            exact_bad.append((name, "-", "rows", str(len(a_rows)), str(len(b_rows))))
            continue
        runmax = {}
        for a_ in a_rows:
            for col in TRAJ.get(name, ()):
                for v in a_[col].split(";"):
                    x = _num(v)
                    if x is not None and math.isfinite(x):
                        k = (_run_of(name, a_), col)
                        runmax[k] = max(runmax.get(k, 0.0), abs(x))
        for a_, b_ in zip(a_rows, b_rows):
            rid = _run_of(name, a_)
            ident = f"{rid}" + (f" i={a_['i']}" if "i" in a_ else f" k={a_['k']}" if "k" in a_ else "")
            cn = cmp_n.get(rid)
            if name == "evals.csv":
                inside = cn is not None and (a_["search"] == "none" or int(a_["search"]) < cn)
            elif name in ("iters.csv", "steps.csv"):
                inside = cn is not None and int(a_["k"]) < cn
            elif name == "runs.csv":
                inside = cn is not None and cn == n_ls[rid]
            elif name == "hess_final.csv":
                inside = a_["h_compared"] == "true"
            else:
                inside = True
            for col in h1:
                if col in DIAGNOSTIC:
                    continue
                va, vb = a_[col], b_[col]
                if name == "runs.csv" and runs_c[rid].get("repro") == "class" and col not in CLASS_EXACT:
                    continue
                if col not in TRAJ.get(name, ()):
                    if va != vb:
                        exact_bad.append((name, ident, col, va[:70], vb[:70]))
                    continue
                st = traj.setdefault((name, col), [0, 0, 0.0, 0.0])
                la, lb = va.split(";"), vb.split(";")
                scale = runmax.get((rid, col), 0.0)
                if not inside:
                    skipped[(name, col)] = skipped.get((name, col), 0) + 1
                    if name == "steps.csv" and len(la) == len(lb):  # informational: how far SciPy's recorded states drift past the prefix
                        for u, v in zip(la, lb):
                            x, y = _num(u), _num(v)
                            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                                info[col] = max(info.get(col, 0.0), abs(x - y) / max(abs(x), abs(y), scale, 1e-300))
                    continue
                if len(la) != len(lb):
                    exact_bad.append((name, ident, col, va[:70], vb[:70]))
                    continue
                for u, v in zip(la, lb):
                    x, y = _num(u), _num(v)
                    st[0] += 1
                    if x is None or y is None or not (math.isfinite(x) and math.isfinite(y)):
                        if u != v:
                            st[1] += 1
                            outside_examples.append(f"{name} {ident} {col}: committed {u[:30]}, regenerated {v[:30]}")
                        continue
                    d = abs(x - y)
                    rel = d / max(abs(x), abs(y), scale, 1e-300)
                    st[2] = max(st[2], d)
                    st[3] = max(st[3], rel)
                    if not rel <= SCIPY_REPRO_RTOL:
                        st[1] += 1
                        outside_examples.append(f"{name} {ident} {col}: committed {u[:30]}, regenerated {v[:30]}, relative difference {rel:.3e}")
    print(f"== exact columns: {len(exact_bad)} mismatch(es)")
    for m in exact_bad:
        print("  MISMATCH", " | ".join(map(str, m)))
    if exact_bad:
        bad_runs = sorted({m[1].split()[0] for m in exact_bad if m[1] != "-"})
        print(f"  runs whose path or options changed on this platform: {', '.join(bad_runs) or '(structure)'}")
    print(f"== trajectory columns, CALIBRATED reproducibility tolerance SCIPY_REPRO_RTOL = {SCIPY_REPRO_RTOL:g}: per component "
          f"|a-b| <= rtol max(|a|, |b|, run scale), on the comparison prefix")
    worst = 0.0
    for (name, col), (n, bad, mabs, mrel) in sorted(traj.items()):
        worst = max(worst, mrel)
        marg = f"{SCIPY_REPRO_RTOL / mrel:.3g}x" if mrel > 0 else "n/a (identical)"
        print(f"  {name:14s} {col:10s} compared {n:6d}  outside {bad:5d}  largest |diff| {mabs:.3e}  observed maximum {mrel:.3e}  "
              f"constant {SCIPY_REPRO_RTOL:g}  margin {marg}  skipped {skipped.get((name, col), 0)}")
    print(f"  observed maximum over all columns {worst:.3e}; constant {SCIPY_REPRO_RTOL:g}"
          + (f" (margin {SCIPY_REPRO_RTOL / worst:.3g}x)" if worst > 0 else " (identical)"))
    if info:
        print("  steps.csv rows past the comparison prefix (reported only; the per-step replay uses the committed states): largest relative "
              "difference per column " + ", ".join(f"{c} {v:.3e}" for c, v in sorted(info.items())) + f" (constant {SCIPY_REPRO_RTOL:g})")
    print("  rows of class runs not compared (counts depend on the platform): "
          + ", ".join(f"{k} {v}" for k, v in skipped.items() if isinstance(k, str)))
    for line in outside_examples[:20]:
        print("  OUTSIDE", line)
    print(f"== margins on this platform (first search outside the bound minus the committed prefix; the rule keeps >= {PREFIX_MARGIN}): smallest first")
    for m_, rid, txt in sorted(margins):
        print(f"  margin {m_:3d}  {rid:24s} {txt}")
    print(f"  {len(margins)} prefixes with a finite first-outside index; smallest margin {min((m for m, _, _ in margins), default='n/a')}; "
          f"runs covered in full: {sum(1 for r in runs_c.values() if r['parity'] == 'true' and r['cmp_nsearch'] == r['n_ls'])}")
    if bounds:
        print("== bounds on this platform")
        for b in bounds:
            print("  BOUND", b)
    n_out = sum(v[1] for v in traj.values())
    failed = bool(exact_bad or n_out or bounds)
    print("check:", "FAILED" if failed else "ok", f"({len(exact_bad)} exact, {n_out} trajectory outside SCIPY_REPRO_RTOL, {len(bounds)} out of bounds)")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed CSVs, write nothing")
    ap.add_argument("--data", type=Path, default=DATA, help="directory of the CSVs for --check")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.data))
    out, _ = generate()
    assert_derivation(out)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, text in render(out).items():
        (DATA / name).write_text(text)
    print(f"wrote {len(out['runs'])} runs, {len(out['evals'])} evaluations, {len(out['iters'])} line searches to {DATA}")


if __name__ == "__main__":
    main()
