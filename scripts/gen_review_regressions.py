# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Print the reference values hard-coded in
`crates/specials/tests/review_regressions.rs` (the regression tests of the
dispatched review of tensor4all/tenferro-sci-rs#21).

Run from the repository root:

    uv run scripts/gen_review_regressions.py

Every mpmath value is computed at 200 and 300 bits, which must agree to 1e-40,
and printed as a double-double pair `(hi, lo)` with `hi = fl(v)` and
`lo = fl(v - hi)`. The Stirling numbers are exact Python integers: the closed
forms for `k = n - 1` and `n - 2`, and Graham, Knuth and Patashnik,
*Concrete Mathematics*, eq. (6.43) for `S(n, n - j)`, which this script first
checks against the recurrence for small `n`.
"""

from __future__ import annotations

import datetime
import platform
from math import comb

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


def digammainv(y: float):
    y = mp.mpf(y)
    x = -1 / (y + mp.euler)
    for _ in range(400):
        step = (mp.digamma(x) - y) / mp.polygamma(1, x)
        x -= step
        if abs(step) < abs(x) * mp.mpf(2) ** (-mp.mp.prec + 8):
            break
    return x


def gen_harmonic(n: float, a: float):
    return mp.zeta(mp.mpf(a)) - mp.zeta(mp.mpf(a), mp.mpf(n) + 1)


def eulerian2(j: int) -> list[int]:
    e = [[0] * (j + 1) for _ in range(j + 1)]
    e[0][0] = 1
    for m in range(1, j + 1):
        for i in range(m):
            e[m][i] = (i + 1) * e[m - 1][i] + ((2 * m - 1 - i) * e[m - 1][i - 1] if i else 0)
    return e[j]


def stirling2_near_diagonal(n: int, j: int) -> int:
    e = eulerian2(j)
    return sum(e[i] * comb(n + j - 1 - i, 2 * j) for i in range(j))


def stirling2_recurrence(n: int, k: int) -> int:
    row = [1] + [0] * k
    for i in range(1, n + 1):
        new = [0] * (k + 1)
        for jj in range(1, min(i, k) + 1):
            new[jj] = jj * row[jj] + row[jj - 1]
        row = new
    return row[k]


def main() -> None:
    print(f"// mpmath {mp.__version__}, Python {platform.python_version()}, "
          f"{datetime.date.today().isoformat()}")
    for n, a in [(1e4, -76.25), (1e4, -76.5), (1e3, -102.3),
                 (1e17, 0.5), (1e9, 0.5), (1e17, 1.0000000001)]:
        print("gen_harmonic", n, a, dd(stable(lambda: gen_harmonic(n, a))))
    print("zeta(2)", dd(stable(lambda: mp.zeta(2))))
    for x in [7.5e-155, 1e-154]:
        print("trigamma", x, dd(stable(lambda: mp.polygamma(1, mp.mpf(x)))))
    for y in [-1e155, -1e200, -1e250, -1e290]:
        print("digammainv", y, dd(stable(lambda: digammainv(y))))
    for a, m in [(1e-310, 170.5), (-1e-310, 170.5), (1e-300, 170.5), (1e-250, 3.25)]:
        print("poch", a, m, dd(stable(lambda: mp.rf(mp.mpf(a), mp.mpf(m)))))

    for n, j in [(30, 5), (50, 20), (80, 64), (100, 33)]:
        assert stirling2_near_diagonal(n, j) == stirling2_recurrence(n, n - j)
    for n, k in [(230, 3), (250, 10)]:
        print("stirling2", n, k, dd(stirling2_recurrence(n, k)))
    n = 10**17
    print("stirling2", n, "n-1", dd(n * (n - 1) // 2))
    print("stirling2", n, "n-2", dd(n * (n - 1) * (n - 2) * (3 * n - 5) // 24))
    m = 2**53 + 1
    print("stirling2", "2^53+1", "2^53-1", dd(stirling2_near_diagonal(m, 2)))
    for n, j in [(1261668211, 3), (7745630509816, 12), (10**7, 10), (10**8, 5), (2**60, 4)]:
        print("stirling2", n, n - j, dd(stirling2_near_diagonal(n, j)))


if __name__ == "__main__":
    main()
