# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "scipy==1.16.3", "numpy==2.3.5"]
# ///
"""Generate `crates/specials/tests/data/carlson_ratchet.csv`, the reference
table of `crates/specials/tests/carlson_ratchet.rs`
(tensor4all/tenferro-sci-rs#27): Carlson's R_C, R_D, R_F, R_G and R_J in the
regimes that the scaling of #25/#26 touched.

Run from the repository root:

    uv run scripts/gen_carlson_ratchet.py

The output is committed; regenerating it is a separate commit. The script is
deterministic: the same pinned versions give the same file (except the date in
the header). It takes about two minutes.

Columns:

    function,regime,a0,a1,a2,a3,ref_hi,ref_lo,scipy,f64_nan_ok,df64_nan_ok,cond_dlmf,cond_carlson

* `a0..a3`: the arguments, as Python's shortest round-trip `repr` of the
  `f64` (Rust's `str::parse::<f64>` reads back the same value); `a3` is empty
  for the three-argument functions.
* `ref_hi,ref_lo`: the mpmath value as a double-double pair, `hi = fl(v)`,
  `lo = fl(v - hi)`.
* `scipy`: `scipy.special`'s value (SciPy 1.16.3), for parity of the `f64`
  path; `nan` where SciPy gives NaN.
* `f64_nan_ok`, `df64_nan_ok`: 1 where the documentation of the crate allows
  that strategy to return NaN instead of the value (the `Df64` R_J beyond an
  argument ratio of 2^600; the `f64` R_J and R_G beyond an exponent spread of
  600 bits, which they leave to xsf; ill-conditioned principal values). A
  finite result must still be within tolerance: what is never allowed is a
  wrong finite value.
* `cond_dlmf`, `cond_carlson`: for R_J with `p < 0`, the condition numbers
  (sum of the absolute values of the terms over the absolute value of their
  sum) of DLMF 19.20.14, which the `Df64` path evaluates, and of Carlson
  (1995) Eq. 2.26, which xsf evaluates; empty elsewhere. A principal value
  cannot be computed more accurately than its condition number times the
  working epsilon.

The regimes and their tolerances are listed in the test.

How each reference is computed:

* At 256 and at 512 bits, each plus the spread of the nonzero arguments in
  bits (mpmath's duplication loses about that many bits), which must agree to
  1e-40: mpmath 1.3.0 has precision-dependent bugs in these functions
  (`docs/upstream-issues.md`).
* At the arguments as given **and** at the arguments moved by an exact power
  of four `4^-j` that brings their largest exponent near 0, with the result
  scaled back exactly by `2^(2dj)` (`d` the degree of homogeneity, DLMF
  19.16.1-19.16.5): the two must agree to 1e-40, so a magnitude-dependent
  error of mpmath cannot enter the table.
* For `p < 0` the value of R_J is the Cauchy principal value of DLMF 19.20.14
  (R_J with a positive fourth argument, R_F and R_C), asserted equal to
  Carlson (1995) Eq. 2.26 (the transformation xsf uses) to 1e-40, both with 400
  extra bits because they cancel. mpmath's own real part is not used: it is
  wrong at large or widely spread arguments (`docs/upstream-issues.md`).
* For `y < 0` R_C is the principal value of DLMF 19.2.20,
  `R_C(x, y) = sqrt(x / (x - y)) R_C(x - y, -y)`.

Rows whose reference is not a normal `f64` above 2^-960 (where the
double-double low word stays normal) are rejected rather than written.
"""

from __future__ import annotations

import datetime
import math
import platform
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
import scipy.special as sc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates" / "specials" / "tests" / "data" / "carlson_ratchet.csv"

PRECS = (256, 512)
AGREE = mp.mpf(10) ** -40
# Extra bits for the principal-value transformations, which cancel.
PV_EXTRA = 400

# Twice the degree of homogeneity (DLMF 19.16.1-19.16.5).
TWICE_DEGREE = {"elliprc": -1, "elliprf": -1, "elliprd": -3, "elliprj": -3, "elliprg": 1}
NARGS = {"elliprc": 2, "elliprf": 3, "elliprd": 3, "elliprj": 4, "elliprg": 3}


def p2(e: int, m: float = 1.0) -> float:
    """`m * 2^e` as an exact `f64` (`m` has few bits)."""
    v = math.ldexp(m, e)
    assert v != 0.0 and math.isfinite(v) and abs(v) >= 2.0**-1022, (m, e)
    return v


# --- mpmath evaluation -------------------------------------------------------


def rc(x, y):
    """R_C; for y < 0 the principal value (DLMF 19.2.20)."""
    x, y = mp.mpf(x), mp.mpf(y)
    if y > 0:
        return mp.elliprc(x, y)
    return mp.sqrt(x / (x - y)) * mp.elliprc(x - y, -y)


def rj_dlmf(x, y, z, p):
    """DLMF 19.20.14 for p < 0, with x <= y <= z. Returns (value, condition)."""
    q = y + (z - y) * (y - x) / (y - p)
    terms = [(q - y) * mp.elliprj(x, y, z, q), -3 * mp.elliprf(x, y, z),
             3 * rc(x * z / y, p * q / y)]
    s = mp.fsum(terms)
    return s / (y - p), mp.fsum(abs(t) for t in terms) / abs(s)


def rj_carlson(x, y, z, p):
    """Carlson (1995) Eq. 2.26 for p < 0, with x <= y <= z (xsf's
    transformation). Returns (value, condition)."""
    q = (x + y - p - x * y / z) / (1 - p / z)
    pq = -p * q
    terms = [(q - z) * mp.elliprj(x, y, z, q), -3 * mp.elliprf(x, y, z),
             3 * mp.sqrt(x * y * z / (x * y + pq)) * mp.elliprc(x * y + pq, pq)]
    s = mp.fsum(terms)
    return s / (z - p), mp.fsum(abs(t) for t in terms) / abs(s)


def rj(x, y, z, p):
    if p > 0:
        return mp.elliprj(x, y, z, p)
    x, y, z = sorted((x, y, z))
    with mp.workprec(mp.mp.prec + PV_EXTRA):
        v, _ = rj_dlmf(x, y, z, p)
        w, _ = rj_carlson(x, y, z, p)
        assert abs(v - w) <= abs(v) * AGREE, ("pv", x, y, z, p, v, w)
    return +v


def evaluate(name, args):
    a = [mp.mpf(v) for v in args]
    if name == "elliprc":
        return rc(*a)
    if name == "elliprf":
        return mp.elliprf(*a)
    if name == "elliprd":
        return mp.elliprd(*a)
    if name == "elliprg":
        return mp.elliprg(*a)
    if name == "elliprj":
        return rj(*a)
    raise ValueError(name)


def pv_conditions(args):
    """Condition numbers of DLMF 19.20.14 and of Carlson Eq. 2.26."""
    x, y, z = sorted(mp.mpf(v) for v in args[:3])
    p = mp.mpf(args[3])
    with mp.workprec(PRECS[-1] + PV_EXTRA + math.ceil(spreads_of(args)[0])):
        return rj_dlmf(x, y, z, p)[1], rj_carlson(x, y, z, p)[1]


def reference(name, args):
    """mpmath's value, stable across precisions and across an exact rescaling."""
    nonzero = [abs(v) for v in args if v != 0.0]
    j = math.floor(math.log2(max(nonzero))) // 2
    # mpmath's duplication loses digits with the spread of the arguments
    # (about 136 bits at 2^640 and 256 bits): add the spread to both precisions.
    extra = math.ceil(math.log2(max(nonzero) / min(nonzero)))
    vals = []
    for prec in PRECS:
        mp.mp.prec = prec + extra
        direct = evaluate(name, args)
        moved = evaluate(name, [mp.ldexp(mp.mpf(v), -2 * j) for v in args])
        moved = mp.ldexp(moved, TWICE_DEGREE[name] * j)
        assert abs(direct - moved) <= abs(direct) * AGREE, (name, args, prec, direct, moved)
        vals.append(direct)
    mp.mp.prec = PRECS[-1] + extra
    a, b = vals
    assert abs(a - b) <= abs(b) * AGREE, (name, args, a, b)
    return b


def dd(v) -> tuple[float, float]:
    with mp.workprec(600):
        v = mp.mpf(v)
        hi = float(v)
        return hi, float(v - mp.mpf(hi))


def scipy_value(name, args) -> float:
    f = getattr(sc, name)
    with np.errstate(all="ignore"):
        return float(f(*[np.float64(v) for v in args]))


# --- the table ----------------------------------------------------------------

ROWS: list[tuple[str, str, list[float], bool, bool]] = []


def spreads_of(args):
    """(real spread in bits, floor-exponent spread) of the nonzero arguments."""
    nz = [abs(v) for v in args if v != 0.0]
    exps = [math.frexp(v)[1] - 1 for v in nz]
    return math.log2(max(nz) / min(nz)), max(exps) - min(exps)


def add(name, regime, args, f64_nan_ok=False, df64_nan_ok=False):
    """One row. On top of what the caller allows, NaN is allowed where the
    crate documents it: the `Df64` R_J for a ratio of the nonzero arguments
    above 2^600 (`RJ_RATIO_BITS`), and the `f64` R_J and R_G for exponents
    more than 600 apart, which they leave to xsf (NaN where its products leave
    the range)."""
    args = [float(v) for v in args]
    assert len(args) == NARGS[name], (name, args)
    real, floor = spreads_of(args)
    if name == "elliprj" and real > 600:
        df64_nan_ok = True
    if name in ("elliprj", "elliprg") and floor > 600:
        f64_nan_ok = True
    ROWS.append((name, regime, args, f64_nan_ok, df64_nan_ok))


def ordinary():
    for a in [(1, 2), (2, 1), (0, 1), (0.25, 1e3), (1e3, 1e-3), (1, -1), (2, -1), (1e-3, -3)]:
        add("elliprc", "ordinary", a)
    for a in [(1, 2, 3), (0, 1, 2), (2, 3, 4), (1e-3, 1, 1e3), (0.5, 0.5, 7)]:
        add("elliprf", "ordinary", a)
    for a in [(1, 2, 3), (0, 2, 1), (2, 3, 4), (1e-3, 1, 1e3), (7, 7, 0.5)]:
        add("elliprd", "ordinary", a)
    for a in [(1, 2, 3), (0, 1, 2), (0, 0, 1), (2, 3, 4), (1e-3, 1, 1e3)]:
        add("elliprg", "ordinary", a)
    for a in [(1, 2, 3, 4), (2, 3, 4, 5), (0, 1, 2, 3), (0.5, 1, 2, 1e-3), (1e-3, 1, 1e3, 7),
              (1, 1, 1, 1)]:
        add("elliprj", "ordinary", a)


def ratio_near_one():
    """Arguments equal or close to each other at 2^±520 and 2^±600, where
    products such as `x*z` in the unscaled R_J overflowed (#25)."""
    u1, u2, u3 = 1 + 2.0**-26, 1 + 2.0**-13, 1 + 2.0**-7
    shapes = {
        "elliprc": [(1, 1), (1, u2), (1, -1), (1, -u2)],
        "elliprf": [(1, 1, 1), (1, u1, u2)],
        "elliprd": [(1, 1, 1), (1, u1, u2)],
        "elliprg": [(1, 1, 1), (1, u1, u2)],
        "elliprj": [(1, 1, 1, 1), (1, 1, 1, -1), (1, u1, u2, u3), (1, u1, u2, -u3)],
    }
    for s in (-600, -520, 520, 600):
        for name, pts in shapes.items():
            for a in pts:
                # The `f64` window keeps xsf's unscaled value here (it agrees
                # with the scaled one to within `AGREE = 1e-10`), which is
                # 6.1e-13 off: the documented ceiling of `kept_xsf` applies.
                kept = name == "elliprg" and s == -520 and a == (1, u1, u2)
                add(name, "kept_xsf" if kept else "ratio_near_one",
                    [math.ldexp(v, s) for v in a])


def spreads():
    """Arguments `2^lo .. 2^hi` with `hi - lo` just under, at and just over
    `RJ_RATIO_BITS = 600` (the `Df64` R_J's ratio limit, and twice the `f64`
    window `UNSCALED_BITS = 300`), at centers inside the `f64` window (no
    scaling) and outside it (scaled)."""
    for spread in (596, 598, 599, 600, 601, 602, 640):
        over = spread > 600
        regime = "spread_over_600" if over else "spread_to_600"
        for center in (0, 150, -150, 340, -340):
            lo = center - spread // 2
            hi = lo + spread
            mid = (lo + hi) // 2
            if lo < -1022 or hi > 1023:
                continue
            rj_rows = [
                (p2(lo), p2(mid), p2(hi), p2(mid, 1.5)),
                (p2(lo), p2(mid), p2(hi), -p2(hi)),
                (p2(mid), p2(mid + 1), p2(hi), p2(lo)),
                (p2(lo), p2(mid), p2(mid + 1), p2(hi)),
                (0.0, p2(lo), p2(hi), p2(mid)),
            ]
            for a in rj_rows:
                add("elliprj", regime, a)
            for a in [(p2(lo), p2(mid), p2(hi)), (0.0, p2(lo), p2(hi))]:
                add("elliprg", regime, a)
                add("elliprf", regime, a)
            add("elliprd", regime, (p2(lo), p2(mid), p2(hi)))
            add("elliprd", regime, (p2(mid), p2(hi), p2(lo)))
            add("elliprc", regime, (p2(lo), p2(hi)))
            add("elliprc", regime, (p2(hi), p2(lo)))
    # A floor-exponent spread of 600 with a real ratio above 2^600 (600.58
    # bits): the `f64` path scales it (exponents), the `Df64` R_J may not.
    for center in (0, 150, -150):
        lo = center - 300
        a = (p2(lo), p2(center), p2(lo + 600, 1.5), p2(center, 1.25))
        add("elliprj", "spread_over_600", a)
        add("elliprj", "spread_over_600", (*a[:3], -a[2]))
        add("elliprg", "spread_over_600", a[:3])
    # One of x, y, z far above the other three arguments: xsf (and SciPy) is
    # NaN from about 2^520 on with the others of order 1
    # (`docs/upstream-issues.md`, "`elliprj` is NaN for one huge `x`, `y` or
    # `z`"); the `f64` path's window (#26) moves such arguments and must be
    # accurate up to a spread of 2^600.
    for c in (0, -300):
        for k in (500, 518, 519, 521, 540, 598, 600):
            for sign in (1, -1):
                add("elliprj", "xsf_one_huge", (p2(c), p2(c + 1), p2(c + k), sign * p2(c + 2)))
    # Review of 470803a: spreads beyond 2^600 that a retry arm turned into
    # finite garbage (7.2e54 for 1.4e41, 6.4e39 for 4.1e39, 5.1e9 for 1.0e-4,
    # 8.2231e-53 for 8.2310e-53).
    for e in [(-400, -360, 500, -500), (-400, -360, 500, -400), (-300, -260, 600, -400)]:
        add("elliprj", "spread_over_600", [p2(v) for v in e])
    add("elliprg", "spread_over_600", (1.78e-245, 2.71e-104, 2.85e-306))


def threshold_band():
    """xsf's absolute asymptotic thresholds for R_J with p > 0:
    `abs_close_zero(p)` (`p <= 1e-9`, Gustafson's identity, exact) and
    `y <= 1e-26` (Carlson & Gustafson 1994, Proposition J3, an expansion
    in y / z)."""
    one_e9 = 1e-9
    for p in (5e-10, one_e9, math.nextafter(one_e9, 1.0), 2e-9, 1e-8, 1e-6, 1e-4):
        for xyz in [(1, 2, 3), (1, 1, 1), (1e-3, 1e-2, 1e-1), (1e-12, 1e-10, 1e-8), (0, 1e-6, 1)]:
            add("elliprj", "threshold_p", (*xyz, p))
    one_e26 = 1e-26
    for y in (1e-27, one_e26, math.nextafter(one_e26, 1.0), 1e-25):
        for x, z, p in [(0, 1, 1), (y / 2, 1, 2), (0, 1e-10, 2e-10), (y / 2, 1e-16, 1e-16),
                        (0, 1e-20, 3e-20), (0, 1e3, 1e-3)]:
            add("elliprj", "threshold_y", (x, y, z, p))
    # Review of 0873683: the `f64` window moved these across `y <= 1e-26`
    # (to 2^-88), where the expansion is 1.9e-6 off, and replaced the right
    # unscaled value: 1.8122583490683579e30 for 1.8122549800346953e30.
    for e in [(-322, -110, -92, -50), (-312, -100, -82, -40)]:
        add("elliprj", "threshold_y", [p2(v) for v in e])
    # xsf's Proposition J3 is selected by `y <= 1e-26` and a bound that
    # allows `y / z` up to about 1 / |ln(p / y)|, but its truncation error is
    # of order `y / z`: at these points, inside 2^±300 (no scaling), xsf and
    # SciPy are 1.9e-6, 4.2e-8 and 9.6e-10 off. The first is the window's image
    # of the points above, as a caller would pass it.
    add("elliprj", "xsf_tiny_y", [p2(v) for v in (-300, -88, -70, -28)])
    for ez in (-70, -65, -60):
        add("elliprj", "xsf_tiny_y", (0.0, p2(-88), p2(ez), p2(-28)))


def principal_values():
    """R_J for p < 0, well conditioned (|p| not far below y), at scales from
    1 to 2^±520 (the `f64` path scales from 2^±300 on)."""
    bases = [(1, 2, 3, -4), (2, 3, 4, -0.5), (0, 1, 2, -1), (1e-3, 1, 1e3, -10),
             (1, 2, 3, -1e3), (0.5, 1, 1.5, -0.75)]
    for s in (0, -200, 200, -400, 400, -520, 520):
        for a in bases:
            args = [math.ldexp(v, s) for v in a]
            dl, ca = pv_conditions(args)
            assert dl < 1e3 and ca < 1e3, (args, dl, ca)
            add("elliprj", "principal_value", args)
    # Where xsf's unscaled value is right although an argument is far
    # beyond 2^300 (review of e3d872d: NaN on a branch).
    add("elliprj", "principal_value", (p2(-168), p2(-362), p2(-129), -p2(578)))


    # Review of 3a7bc8c: principal values beyond 2^±300 whose transformations
    # are conditioned at 4e3 to 5e6 (Eq. 2.26 / DLMF 19.20.14), and review
    # round 6 (arguments 2^396 apart, kept from xsf's unscaled evaluation).
    for a in [(8.302154427603948e-102, 5.927836371317247e-61, 1.0047079450892522e-21,
               -2.218151451072381e-81),
              (6.0170982496602805e-96, 1.1830348712965332e-95, 9.305027156391948e-60,
               -8.436824658532037e-96),
              (6.911304916951263e-92, 8.656115050435896e-92, 1.9001214587812377e-14,
               -7.738772926465598e-92),
              (2.53438683249964e-208, 2.917058701014637e-114, 3.916777135234618e-89,
               -4.5006477943934846e-101)]:
        add("elliprj", "principal_value", a)


def kept_xsf():
    """Values the `f64` window keeps from xsf's unscaled evaluation, whose
    error is xsf's own, documented up to about 1e-10 (review of b6d45b0)."""
    add("elliprg", "kept_xsf", [p2(v) for v in (-763, -368, -695)])
    add("elliprj", "kept_xsf", [p2(v) for v in (-431, -326, -330, -365)])


def top_range():
    """Arguments in the top binades: xsf's R_C and R_F overflow in a sum of
    their arguments and return NaN or 0 although the result is about 1e-154
    (`docs/upstream-issues.md`, "`elliprj` is NaN for one huge ..."), with
    neighbors one and two binades lower, which are accurate."""
    top = [("elliprc", (p2(1022), -p2(1022))), ("elliprc", (p2(1020), -p2(1022))),
           ("elliprc", (p2(1022), p2(1023))), ("elliprf", (p2(1022), p2(1023), p2(1023, 1.5))),
           ("elliprf", (p2(1021), p2(1022), p2(1022, 1.5)))]
    for name, a in top:
        add(name, "f64_top_range", a, f64_nan_ok=True)
    for name, a in [("elliprc", (p2(1021), -p2(1021))), ("elliprc", (p2(1019), -p2(1021))),
                    ("elliprc", (p2(1020), p2(1021))), ("elliprf", (p2(1020), p2(1021), p2(1021, 1.5)))]:
        add(name, "f64_top_range", a)


def ill_conditioned():
    """Principal values whose transformation cancels completely (condition
    numbers 1e45 to 1e91; reviews of 70d4940 and 62f0191): NaN or the value,
    never a wrong finite number."""
    for e in [(-675, -475, -175, -575), (-675, -625, -475, -650), (-625, -575, -425, -600),
              (-175, 25, 325, -75)]:
        args = (p2(e[0]), p2(e[1]), p2(e[2]), -p2(e[3]))
        dl, ca = pv_conditions(args)
        assert dl > 1e40 and ca > 1e40, (args, dl, ca)
        add("elliprj", "pv_ill_conditioned", args, f64_nan_ok=True, df64_nan_ok=True)


def fmt(v: float) -> str:
    return repr(v)


def main() -> None:
    ordinary()
    ratio_near_one()
    spreads()
    threshold_band()
    principal_values()
    kept_xsf()
    ill_conditioned()
    top_range()
    lines = [
        f"# generated by scripts/gen_carlson_ratchet.py on {datetime.date.today().isoformat()}",
        f"# mpmath {mp.__version__} (prec {PRECS[0]} and {PRECS[1]} plus the argument spread "
        f"in bits, +{PV_EXTRA} for p < 0), "
        f"scipy {scipy.__version__}, numpy {np.__version__}, python {platform.python_version()}",
        "# columns: function,regime,a0,a1,a2,a3,ref_hi,ref_lo,scipy,f64_nan_ok,df64_nan_ok,"
        "cond_dlmf,cond_carlson",
    ]
    seen = set()
    for name, regime, args, f64_nan_ok, df64_nan_ok in ROWS:
        key = (name, tuple(args))
        assert key not in seen, key
        seen.add(key)
        ref = reference(name, args)
        hi, lo = dd(ref)
        if not (math.isfinite(hi) and 2.0**-960 <= abs(hi) <= 2.0**1000):
            print(f"rejected (reference {hi!r} out of range): {name}{tuple(args)}")
            continue
        cols = [fmt(v) for v in args] + [""] * (4 - len(args))
        if name == "elliprj" and args[3] < 0:
            dl, ca = pv_conditions(args)
            conds = [f"{float(dl):.3e}", f"{float(ca):.3e}"]
        else:
            conds = ["", ""]
        lines.append(",".join([name, regime, *cols, fmt(hi), fmt(lo),
                               fmt(scipy_value(name, args)),
                               str(int(f64_nan_ok)), str(int(df64_nan_ok)), *conds]))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 3} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
