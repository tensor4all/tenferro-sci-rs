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
from fractions import Fraction
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
    # float(v) can differ from the nearest double at a tie; renormalize.
    hi2 = hi + lo
    return (hi2, (hi - hi2) + lo)


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


def ref_poch(a, m):
    with mp.workprec(mp.mp.prec + extra_bits(a, m)):
        a, m = mp.mpf(a), mp.mpf(m)
        s = a + m
        if is_pole(s) != is_pole(a):
            # A zero or a pole of (a)_m: SciPy's value is checked through xsref.
            return None
        return +mp.rf(a, m)


def ref_lgam1p(x):
    with mp.workprec(mp.mp.prec + extra_bits(1, x)):
        x = mp.mpf(x)
        if is_pole(1 + x):
            return mp.inf
        v = mp.loggamma(1 + x)
        return +(mp.re(v) if x < -1 else v)


def ref_binom(n, k):
    if n < 0 and n == math.floor(n):
        return None
    with mp.workprec(mp.mp.prec + extra_bits(n, k, 1)):
        n, k = mp.mpf(n), mp.mpf(k)
        if is_pole(n - k + 1) or is_pole(k + 1):
            return mp.mpf(0) if not is_pole(n + 1) else None
        return +(mp.gamma(n + 1) / (mp.gamma(k + 1) * mp.gamma(n - k + 1)))


def ref_zetac(x):
    if x == 1.0:
        return None
    # zeta(x, 2) = zeta(x) - 1 without cancellation.
    return mp.zeta(x, 2) if x > 1 else mp.zeta(x) - 1


def _real(v):
    return mp.re(v) if isinstance(v, mp.mpc) and abs(mp.im(v)) <= abs(v) * mp.mpf(2) ** -(mp.mp.prec - 20) else v


def ref_ellipk(m):
    if m > 1:
        return None  # xsf: NaN (K is complex)
    return mp.inf if m == 1 else mp.ellipk(m)


def ref_ellipkm1(p):
    if p < 0:
        return None
    if p == 0:
        return mp.inf
    with mp.workprec(mp.mp.prec + extra_bits(1, p)):
        return +mp.ellipk(1 - mp.mpf(p))


def ref_ellipe(m):
    if m > 1:
        return None
    return mp.mpf(1) if m == 1 else mp.ellipe(m)


def _phi_bits(phi):
    """Bits to reduce phi modulo pi exactly enough."""
    return max(0, mp.mpf(phi).exp + mp.mpf(phi).bc) + 20 if phi != 0 else 0


def ref_ellipkinc(phi, m):
    if m > 1:
        return None
    with mp.workprec(mp.mp.prec + _phi_bits(phi)):
        phi, m = mp.mpf(phi), mp.mpf(m)
        if m == 1 and abs(phi) >= mp.pi / 2:
            return None  # F is infinite at phi = +-pi/2; xsf's cutoff is the f64 value of pi/2
        return +mp.ellipf(phi, m)


def ref_ellipeinc(phi, m):
    if m > 1:
        return None
    with mp.workprec(mp.mp.prec + _phi_bits(phi)):
        return +mp.ellipe(mp.mpf(phi), mp.mpf(m))


def _jacobi(u, m):
    """(sn, cn, dn, am) of u for m <= 1, real."""
    u, m = mp.mpf(u), mp.mpf(m)
    if u == 0:
        return mp.mpf(0), mp.mpf(1), mp.mpf(1), mp.mpf(0)
    if m == 1:
        sn, cn = mp.tanh(u), 1 / mp.cosh(u)
        return sn, cn, cn, 2 * mp.atan(mp.exp(u)) - mp.pi / 2
    if m == 0:
        return mp.sin(u), mp.cos(u), mp.mpf(1), u
    K = mp.ellipk(m)
    n = int(mp.nint(u / (2 * K)))
    up = u - 2 * n * K
    sn = _real(mp.ellipfun("sn", up, m=m))
    cn = _real(mp.ellipfun("cn", up, m=m))
    dn = _real(mp.ellipfun("dn", up, m=m))
    # sn, cn flip sign with n odd; dn does not (u -> u - 2nK).
    sgn = -1 if n % 2 else 1
    return sgn * sn, sgn * cn, dn, n * mp.pi + mp.atan2(sn, cn)


def _ellipj_ref(k):
    def ref(u, m):
        if m > 1:
            return None
        with mp.workprec(mp.mp.prec + _phi_bits(u)):
            return +_jacobi(u, m)[k]

    ref.__name__ = f"ref_ellipj_{k}"
    return ref


# Elementary helpers (#19).


def ref_cosm1(x):
    # cos x - 1 = -2 sin^2(x/2), without cancellation.
    return -2 * mp.sin(mp.mpf(x) / 2) ** 2


def ref_log1pmx(x):
    if x < -1:
        return None
    if x == -1:
        return -mp.inf
    with mp.workprec(mp.mp.prec + extra_bits(1, x)):
        x = mp.mpf(x)
        return +(mp.log1p(x) - x)


def ref_xlogy(x, y):
    # x == 0 (and y not NaN) is 0 by convention, and y <= 0 is complex or
    # infinite: SciPy's value is checked through xsref.
    if x == 0 or y <= 0:
        return None
    return mp.mpf(x) * mp.log(y)


def ref_xlog1py(x, y):
    if x == 0 or y <= -1:
        return None
    with mp.workprec(mp.mp.prec + extra_bits(1, y)):
        return +(mp.mpf(x) * mp.log1p(mp.mpf(y)))


def ref_logit(x):
    if not 0 < x < 1:
        return None
    with mp.workprec(mp.mp.prec + extra_bits(1, x)):
        x = mp.mpf(x)
        return +mp.log(x / (1 - x))


def ref_expit(x):
    return 1 / (1 + mp.exp(-mp.mpf(x)))


def ref_log_expit(x):
    return -mp.log1p(mp.exp(-mp.mpf(x)))


def ref_sinpi(x):
    return mp.sinpi(x)


def ref_cospi(x):
    return mp.cospi(x)


# xsf's degree functions return 0 for |x| > 1e14 (total loss of precision);
# SciPy's value is checked through xsref there.
DG_LOSSTH = 1.0e14


def ref_sindg(x):
    return None if abs(x) > DG_LOSSTH else mp.sinpi(mp.mpf(x) / 180)


def ref_cosdg(x):
    return None if abs(x) > DG_LOSSTH else mp.cospi(mp.mpf(x) / 180)


def ref_tandg(x):
    if abs(x) > DG_LOSSTH or math.fmod(x, 90.0) == 0.0:
        return None  # exact zeros and poles: SciPy's signed values
    t = mp.mpf(x) / 180
    return mp.sinpi(t) / mp.cospi(t)


def ref_cotdg(x):
    if abs(x) > DG_LOSSTH or math.fmod(x, 90.0) == 0.0:
        return None
    t = mp.mpf(x) / 180
    return mp.cospi(t) / mp.sinpi(t)


def ref_radian(d, m, s):
    d, m, s = mp.mpf(d), mp.mpf(m), mp.mpf(s)
    return ((d * 60 + m) * 60 + s) * mp.pi / 648000


def ref_round(x):
    # Round half to even; exact.
    return mp.mpf(round(mp.mpf(x)))


def ref_exp2(x):
    return mp.power(2, mp.mpf(x))


def ref_exp10(x):
    return mp.power(10, mp.mpf(x))


def ref_cbrt(x):
    return mp.cbrt(x) if x >= 0 else -mp.cbrt(-mp.mpf(x))


def ref_expm1(x):
    return mp.expm1(x)


def ref_log1p(x):
    if x < -1:
        return None
    if x == -1:
        return -mp.inf
    with mp.workprec(mp.mp.prec + extra_bits(1, x)):
        return +mp.log1p(mp.mpf(x))


# Associated Legendre function of integer order (#16).


def _frac(v):
    """The exact value of a finite mpmath real as a Fraction."""
    p, q = mp.libmp.to_rational(mp.mpf(v)._mpf_)
    return Fraction(p, q)


def _legendre_derivative(n, m, x):
    """d^m P_n / dx^m at the exact rational x, from the explicit sum
    P_n(x) = sum_k (-1)^k (2n-2k)! / (2^n k! (n-k)! (n-2k)!) x^(n-2k)."""
    f = math.factorial
    total = Fraction(0)
    for k in range(n // 2 + 1):
        p = n - 2 * k
        if p < m:
            break
        c = Fraction((-1) ** k * f(2 * n - 2 * k), 2**n * f(k) * f(n - k) * f(p))
        total += c * Fraction(f(p), f(p - m)) * x ** (p - m)
    return total


def ref_pmv(m, v, x):
    """Ferrers function P_v^m(x), -1 < x < 1.

    Integer degree: (-1)^m (1 - x^2)^(m/2) d^m P_n / dx^m, with the
    derivative of the Legendre polynomial exact in rational arithmetic (DLMF
    14.6.1), and P_n^(-m) = (-1)^m (n-m)!/(n+m)! P_n^m (DLMF 14.9.3). Other
    degrees: mpmath's `legenp` of type 2.

    Outside that domain, at x = +-1 (xsf returns a signed infinity or a value
    that depends on its own series) and where xsf documents no result
    (non-integer m, or a negative order beyond the degree), SciPy's value is
    checked through xsref.
    """
    if m != math.floor(m) or not abs(x) < 1:
        return None
    vx = -v - 1 if v < 0 else v
    if m < 0 and not (vx + m + 1 > 0 or vx != math.floor(vx)):
        return None
    if vx == math.floor(vx):
        n, mm = int(vx), abs(int(m))
        if mm > n:
            return mp.mpf(0)
        xf = _frac(x)
        d = _legendre_derivative(n, mm, xf)
        s = mp.mpf(d.numerator) / mp.mpf(d.denominator)
        val = (-1) ** mm * mp.sqrt(1 - mp.mpf(x) ** 2) ** mm * s
        if m < 0:
            val *= (-1) ** mm * mp.factorial(n - mm) / mp.factorial(n + mm)
        return val
    return mp.legenp(mp.mpf(v), int(m), mp.mpf(x), type=2)


# (crate name, xsref name, signature, reference, output column)
FUNCS = [
    ("gamma", "gamma", "d-d", ref_gamma, 0),
    ("lgamma", "gammaln", "d-d", ref_lgamma, 0),
    ("rgamma", "rgamma", "d-d", ref_rgamma, 0),
    ("gammasgn", "gammasgn", "d-d", None, 0),
    ("digamma", "digamma", "d-d", ref_digamma, 0),
    ("zeta", "zeta", "d_d-d", ref_zeta, 0),
    ("beta", "beta", "d_d-d", ref_beta, 0),
    ("lbeta", "betaln", "d_d-d", ref_lbeta, 0),
    ("poch", "poch", "d_d-d", ref_poch, 0),
    ("lgam1p", "lgam1p", "d-d", ref_lgam1p, 0),
    ("binom", "binom", "d_d-d", ref_binom, 0),
    ("zetac", "zetac", "d-d", ref_zetac, 0),
    ("ellipk", "ellipk", "d-d", ref_ellipk, 0),
    ("ellipkm1", "ellipkm1", "d-d", ref_ellipkm1, 0),
    ("ellipe", "ellipe", "d-d", ref_ellipe, 0),
    ("ellipkinc", "ellipkinc", "d_d-d", ref_ellipkinc, 0),
    ("ellipeinc", "ellipeinc", "d_d-d", ref_ellipeinc, 0),
    # ellipj returns four values; each gets its own table with the same layout.
    ("ellipj_sn", "ellipj", "d_d-d_d_d_d", _ellipj_ref(0), 0),
    ("ellipj_cn", "ellipj", "d_d-d_d_d_d", _ellipj_ref(1), 1),
    ("ellipj_dn", "ellipj", "d_d-d_d_d_d", _ellipj_ref(2), 2),
    ("ellipj_ph", "ellipj", "d_d-d_d_d_d", _ellipj_ref(3), 3),
    # Elementary helpers (#19).
    ("cosm1", "cosm1", "d-d", ref_cosm1, 0),
    ("log1pmx", "log1pmx", "d-d", ref_log1pmx, 0),
    ("xlogy", "xlogy", "d_d-d", ref_xlogy, 0),
    ("xlog1py", "xlog1py", "d_d-d", ref_xlog1py, 0),
    ("logit", "logit", "d-d", ref_logit, 0),
    ("expit", "expit", "d-d", ref_expit, 0),
    ("log_expit", "log_expit", "d-d", ref_log_expit, 0),
    ("sinpi", "sinpi", "d-d", ref_sinpi, 0),
    ("cospi", "cospi", "d-d", ref_cospi, 0),
    ("sindg", "sindg", "d-d", ref_sindg, 0),
    ("cosdg", "cosdg", "d-d", ref_cosdg, 0),
    ("tandg", "tandg", "d-d", ref_tandg, 0),
    ("cotdg", "cotdg", "d-d", ref_cotdg, 0),
    ("radian", "radian", "d_d_d-d", ref_radian, 0),
    ("round", "round", "d-d", ref_round, 0),
    ("exp2", "exp2", "d-d", ref_exp2, 0),
    ("exp10", "exp10", "d-d", ref_exp10, 0),
    ("cbrt", "cbrt", "d-d", ref_cbrt, 0),
    ("expm1", "expm1", "d-d", ref_expm1, 0),
    ("log1p", "log1p", "d-d", ref_log1p, 0),
    # Associated Legendre function (#16).
    ("pmv", "pmv", "d_d_d-d", ref_pmv, 0),
]


def fmt(x: float) -> str:
    return repr(float(x))


def main(xsref: Path, only: list[str] | None = None):
    commit = subprocess.run(
        ["git", "-C", str(xsref), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    tables = xsref / "src" / "xsref" / "tables" / "scipy_special_tests"
    OUT.mkdir(parents=True, exist_ok=True)
    mp.mp.prec = BASE_PREC
    for name, xname, sig, ref, col in FUNCS:
        if only and name not in only:
            continue
        d = tables / xname
        inp = pq.read_table(d / f"In_{sig}.parquet").to_pylist()
        out = pq.read_table(d / f"Out_{sig}.parquet").to_pylist()
        tols = [pq.read_table(d / f"Err_{sig}_{p}.parquet").to_pylist() for p in PLATFORMS]
        nargs = len(inp[0])
        which = f" (output out{col})" if name.startswith("ellipj_") else ""
        lines = [
            f"# imported from scipy/xsref {commit} tables/scipy_special_tests/{xname}/*_{sig}.parquet{which}\n",
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
            cells = [fmt(a) for a in args] + [fmt(o[f"out{col}"]), str(int(o["fallback"]))]
            cells += [fmt(t[i][f"err{col}"]) for t in tols] + list(ddv)
            lines.append(",".join(cells) + "\n")
        (OUT / f"{name}.csv").write_text("".join(lines))
        print(name, len(inp))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2:] or None)
