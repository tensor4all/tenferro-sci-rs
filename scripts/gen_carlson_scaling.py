# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Print the reference values hard-coded in
`crates/specials/tests/carlson_scaling.rs` (tensor4all/tenferro-sci-rs#25:
Carlson's R_C, R_D, R_F, R_G and R_J at arguments near 2^±520 and 2^±600).

Run from the repository root:

    uv run scripts/gen_carlson_scaling.py

The test evaluates each function at `2^e` times a base point and compares the
result, multiplied by the exact power of two `2^(-d e)` (`d` the degree of
homogeneity), with the value printed here for the base point. The values are
computed at the base point because mpmath 1.3.0 does not evaluate R_J for
`p < 0` correctly there: its real part at `(2^520, 2^520, 2^520, -2^520)`,
rescaled, is 76.5 instead of -0.565 (see `docs/upstream-issues.md`).

Every value is computed at 200 and 300 bits, which must agree to 1e-40, and
printed as a double-double pair `(hi, lo)`. For `p < 0` the value of R_J is
the Cauchy principal value of DLMF 19.20.14, evaluated from R_J with a
positive fourth argument, R_F and R_C, and checked against the real part of
`mp.elliprj` at the base point (at the wide-ratio point it is not: mpmath's
own value there depends on the working precision).
"""

from __future__ import annotations

import datetime
import platform

import mpmath as mp


def dd(v) -> tuple[float, float]:
    mp.mp.prec = 600
    v = mp.mpf(v)
    hi = float(v)
    return hi, float(v - mp.mpf(hi))


def stable(f):
    vals = []
    for prec in (200, 300):
        mp.mp.prec = prec
        vals.append(f())
    mp.mp.prec = 300
    a, b = vals
    assert abs(a - b) <= abs(b) * mp.mpf(10) ** -40, (a, b)
    return b


def rj_principal(x, y, z, p):
    """DLMF 19.20.14 for p < 0, with x <= y <= z."""
    x, y, z = sorted(mp.mpf(v) for v in (x, y, z))
    p = mp.mpf(p)
    q = y + (z - y) * (y - x) / (y - p)
    return ((q - y) * mp.elliprj(x, y, z, q) - 3 * mp.elliprf(x, y, z)
            + 3 * rc(x * z / y, p * q / y)) / (y - p)


def rj(x, y, z, p, check=True):
    """R_J; for p < 0 the principal value, checked against mpmath's real part
    unless `check` is false (see the module docstring)."""
    if p > 0:
        return mp.elliprj(x, y, z, p)
    v = rj_principal(x, y, z, p)
    if check:
        assert abs(mp.re(mp.elliprj(x, y, z, p)) - v) <= abs(v) * mp.mpf(10) ** -40
    return v


def rc(x, y):
    """R_C; for y < 0 its Cauchy principal value, the real part of mpmath's."""
    return mp.re(mp.elliprc(x, y))


def main() -> None:
    print(f"// mpmath {mp.__version__}, Python {platform.python_version()}, "
          f"{datetime.date.today().isoformat()}")
    for a in [(1, 1, 1, -1), (1, 2, 3, 4), (1, 2, 3, -4), (0, 1, 2, -1), (0, 1, 2, 3),
              (2, 3, 4, -0.5)]:
        print("elliprj", a, dd(stable(lambda: rj(*a))))
    for name, f in [("elliprf", mp.elliprf), ("elliprd", mp.elliprd), ("elliprg", mp.elliprg)]:
        for a in [(1, 1, 1), (1, 2, 3), (0, 1, 2)]:
            print(name, a, dd(stable(lambda: f(*a))))
    for a in [(1, 1), (1, 2), (0, 1), (1, -1), (2, -1)]:
        print("elliprc", a, dd(stable(lambda: rc(*a))))
    # A ratio of 2^600, the limit of the Df64 R_J, centered at 2^99: R_J(x, y, z, z)
    # is R_D(x, y, z) (DLMF 19.20.18).
    t = 2.0**399
    a = (2.0**-201, 1.0, t, t)
    rjp = dd(stable(lambda: rj(*a)))
    assert rjp == dd(stable(lambda: mp.elliprd(*a[:3])))
    print("elliprj", "(2^-201, 1, 2^399, 2^399)", rjp)
    # mpmath's real part is wrong here at 200 and 300 bits (-7.9e-179, -1.58e-178)
    # and reaches the DLMF 19.20.14 value only at 600 bits, so it is not checked.
    print("elliprj", "(2^-201, 1, 2^399, -2^399)",
          dd(stable(lambda: rj(2.0**-201, 1.0, t, -t, check=False))))


if __name__ == "__main__":
    main()
