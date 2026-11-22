# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "pyarrow==22.0.0"]
# ///
"""Import SciPy's xsref reference tables for the `specials` crate.

xsref (https://github.com/scipy/xsref, BSD-3-Clause) holds the reference
tables that xsf's own test suite runs against: inputs, an mpmath reference
value rounded to double, a `fallback` flag (True where xsref fell back to an
older SciPy instead of an arbitrary-precision reference), and per-platform
tolerances. This script converts the real-valued tables of the functions the
crate implements to CSV, and adds an mpmath double-double value at the same
inputs, which the double-double path is checked against (xsref's values are
rounded to double, so they cannot check it).

Run from the repository root with a checkout of xsref:

    git clone https://github.com/scipy/xsref /path/to/xsref
    uv run scripts/import_xsref.py /path/to/xsref

Output: crates/specials/tests/data/xsref/<function>.csv (committed;
regenerating it is a separate commit).
"""

from __future__ import annotations

import datetime
import math
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import pyarrow.parquet as pq

BASE_PREC = 256

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates" / "specials" / "tests" / "data" / "xsref"

PLATFORMS = ["clang-darwin-aarch64", "gcc-linux-x86_64", "other"]


def dd(v) -> tuple[float, float]:
    v = mp.mpf(v)
    if mp.isnan(v):
        return (math.nan, 0.0)
    if mp.isinf(v):
        return (math.inf if v > 0 else -math.inf, 0.0)
    hi = float(v)
    if math.isinf(hi) or hi == 0.0:
        return (hi, 0.0)
    lo = float(v - mp.mpf(hi))
    return (hi, lo)


def stable(f, *args):
    """f(*args) evaluated at two working precisions that must agree.

    mpmath 1.3.0's Hurwitz zeta loses accuracy at 256 bits for some
    arguments (zeta(32.54, 522.7) is off by 9e-11), so no reference value
    is trusted from a single precision: the value is recomputed with more
    bits until two consecutive precisions agree to 2^-150.
    """
    prev = None
    for prec in (256, 640, 1600, 4000):
        with mp.workprec(prec):
            v = f(*[mp.mpf(a) if not isinstance(a, mp.mpf) else +a for a in args])
            v = +v if v is not None else None
        if v is None or isinstance(v, mp.mpc) or not mp.isfinite(v):
            return v
        if prev is not None and (v == prev or abs(v - prev) <= abs(v) * mp.mpf(2) ** -150):
            return v
        prev = v
    raise RuntimeError(f"reference did not converge: {f.__name__}{args}")


def extra_bits(*xs) -> int:
    """Bits needed so that sums of the arguments are exact."""
    exps = [mp.mpf(x).exp + mp.mpf(x).bc for x in xs if mp.isfinite(x) and x != 0]
    return min(max(exps) - min(exps), 4000) if len(exps) > 1 else 0


def is_pole(x) -> bool:
    return x <= 0 and x == math.floor(x)


def ref_gamma(x):
    if is_pole(x):
        return None
    return mp.gamma(x)


def ref_lgamma(x):
    if is_pole(x):
        return mp.inf
    return mp.re(mp.loggamma(x)) if x < 0 else mp.loggamma(x)


def ref_rgamma(x):
    return mp.rgamma(x)


def ref_digamma(x):
    if is_pole(x):
        return None
    return mp.digamma(x)


def ref_zeta(s, q):
    if q == 1.0:
        # xsf's zeta(x, 1) is the Riemann zeta function for every real x.
        return None if s == 1.0 else mp.zeta(s)
    if s <= 1 or (q <= 0 and (q == math.floor(q) or s != math.floor(s))):
        return None
    return mp.zeta(s, q)


def ref_beta(a, b):
    if is_pole(a) or is_pole(b) or is_pole(a + b):
        return None
    with mp.workprec(mp.mp.prec + extra_bits(a, b)):
        return +mp.beta(mp.mpf(a), mp.mpf(b))


def ref_lbeta(a, b):
    if is_pole(a) or is_pole(b) or is_pole(a + b):
        return None
    with mp.workprec(mp.mp.prec + extra_bits(a, b)):
        a, b = mp.mpf(a), mp.mpf(b)
        v = mp.re(mp.loggamma(a)) + mp.re(mp.loggamma(b)) - mp.re(mp.loggamma(a + b))
        return +v


# (crate name, xsref name, signature, reference)
FUNCS = [
    ("gamma", "gamma", "d-d", ref_gamma),
    ("lgamma", "gammaln", "d-d", ref_lgamma),
    ("rgamma", "rgamma", "d-d", ref_rgamma),
    ("gammasgn", "gammasgn", "d-d", None),
    ("digamma", "digamma", "d-d", ref_digamma),
    ("zeta", "zeta", "d_d-d", ref_zeta),
    ("beta", "beta", "d_d-d", ref_beta),
    ("lbeta", "betaln", "d_d-d", ref_lbeta),
]


def fmt(x: float) -> str:
    return repr(float(x))


def main(xsref: Path):
    commit = subprocess.run(
        ["git", "-C", str(xsref), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tables = xsref / "src" / "xsref" / "tables" / "scipy_special_tests"
    OUT.mkdir(parents=True, exist_ok=True)
    mp.mp.prec = BASE_PREC
    for name, xname, sig, ref in FUNCS:
        d = tables / xname
        inp = pq.read_table(d / f"In_{sig}.parquet").to_pylist()
        out = pq.read_table(d / f"Out_{sig}.parquet").to_pylist()
        tols = [pq.read_table(d / f"Err_{sig}_{p}.parquet").to_pylist() for p in PLATFORMS]
        nargs = len(inp[0])
        lines = [
            f"# imported from scipy/xsref {commit} tables/scipy_special_tests/{xname}/*_{sig}.parquet\n",
            f"# by scripts/import_xsref.py on {datetime.date.today().isoformat()}; xsref is BSD-3-Clause.\n",
            f"# dd_hi,dd_lo: mpmath {mp.__version__} at >= {BASE_PREC} bits at the same inputs (empty: pole or outside the domain).\n",
            "# columns: " + ",".join(f"a{i}" for i in range(nargs))
            + ",desired,fallback," + ",".join(f"tol_{p}" for p in PLATFORMS) + ",dd_hi,dd_lo\n",
        ]
        for i, row in enumerate(inp):
            args = [row[f"in{j}"] for j in range(nargs)]
            o = out[i]
            ddv = ("", "")
            if ref is not None and all(math.isfinite(a) for a in args):
                try:
                    v = stable(ref, *args)
                except (ValueError, ZeroDivisionError, mp.libmp.NoConvergence):
                    v = None
                if v is not None and not isinstance(v, mp.mpc):
                    ddv = tuple(fmt(c) for c in dd(v))
            cells = [fmt(a) for a in args] + [fmt(o["out0"]), str(int(o["fallback"]))]
            cells += [fmt(t[i]["err0"]) for t in tols] + list(ddv)
            lines.append(",".join(cells) + "\n")
        (OUT / f"{name}.csv").write_text("".join(lines))
        print(name, len(inp))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
