//! Regression tests for the dispatched review of the gamma-family PR (#21,
//! findings N1-N9), plus the f64-vs-Df64 grid the review proposed.
//!
//! Reference values are mpmath 1.4.1 at 200 and 300 bits (the two must agree to
//! 1e-40), rounded to a double-double `(hi, lo)`; the Stirling numbers are exact
//! integers from Python's integer recurrence and the closed forms.

use specials::Real;
use std::sync::mpsc;
use std::time::Duration;
use xprec::Df64;

/// A function evaluated in both strategies.
type Unary = (&'static str, fn(f64) -> f64, fn(Df64) -> Df64);
type Binary = (&'static str, fn(f64, f64) -> f64, fn(Df64, Df64) -> Df64);

fn dd(hi: f64, lo: f64) -> Df64 {
    Df64::from_dd(hi, lo)
}

/// Relative error of a double-double result against a double-double reference.
fn rel(got: Df64, want: (f64, f64)) -> f64 {
    let w = dd(want.0, want.1);
    ((got - w) / w).abs().hi()
}

/// Runs `f` on a thread and fails if it has not returned after `secs` seconds,
/// so a hang (N5) is a test failure rather than a stuck test run.
fn within<R: Send + 'static>(secs: u64, what: &str, f: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("{what} did not return within {secs} s"))
}

// ------------------------------------------------------------------ N1 zetac

#[test]
fn n1_zetac_df64_is_minus_infinity_where_f64_is() {
    for s in [-261.5, -1000.5, -100000.5] {
        assert_eq!(specials::zetac(s), f64::NEG_INFINITY, "f64 zetac({s})");
        let d = specials::zetac(Df64::from(s));
        assert_eq!(
            d.hi(),
            f64::NEG_INFINITY,
            "Df64 zetac({s}) = ({}, {})",
            d.hi(),
            d.lo()
        );
    }
}

// ----------------------------------------------------------- N2 gen_harmonic

#[test]
fn n2_gen_harmonic_df64_where_n_pow_1_minus_a_overflows() {
    let cases: [(f64, f64, (f64, f64)); 3] = [
        (
            1e4,
            -76.25,
            (1.2995047360377722e307, -3.4564554682241354e289),
        ),
        (1e4, -76.5, (1.295328955639185e308, 2.714431127064219e291)),
        (1e3, -102.3, (8.093462544814556e307, 2.712975987891481e291)),
    ];
    for (n, a, want) in cases {
        let d = specials::gen_harmonic(Df64::from(n), Df64::from(a));
        let e = rel(d, want);
        // exp(-a ln i) with |a ln i| near 709 costs about 700 ε (see `lib.rs`).
        assert!(
            e <= 5e-30,
            "Df64 gen_harmonic({n}, {a}) = ({}, {}), err {e:e}",
            d.hi(),
            d.lo()
        );
    }
    // The true value overflows: both strategies give +inf.
    assert_eq!(specials::gen_harmonic(1e5, -61.3), f64::INFINITY);
    assert_eq!(
        specials::gen_harmonic(Df64::from(1e5), Df64::from(-61.3)).hi(),
        f64::INFINITY
    );
}

// ---------------------------------------------------------------- N3 trigamma

#[test]
fn n3_trigamma_df64_subnormal_is_infinite() {
    for x in [5e-324, 1e-310, 2.2e-308, 1e-300, 7.4e-155] {
        assert_eq!(specials::trigamma(x), f64::INFINITY, "f64 trigamma({x})");
        let d = specials::trigamma(Df64::from(x));
        assert_eq!(
            d.hi(),
            f64::INFINITY,
            "Df64 trigamma({x}) = ({}, {})",
            d.hi(),
            d.lo()
        );
    }
    // Just above the overflow, the new branch keeps full accuracy.
    for (x, want) in [
        (7.5e-155, (1.7777777777777775e308, -2.7458982980522117e291)),
        (1e-154, (1e308, 4.320353973117618e291)),
    ] {
        let e = rel(specials::trigamma(Df64::from(x)), want);
        assert!(e <= 1e-30, "Df64 trigamma({x}): err {e:e}");
    }
}

// -------------------------------------------------------------- N4 digammainv

#[test]
fn n4_digammainv_df64_holds_1e_30_below_minus_1e154() {
    let cases: [(f64, (f64, f64)); 4] = [
        (-1e155, (1e-155, -2.148703788699233e-172)),
        (-1e200, (1e-200, 4.816661538840688e-217)),
        (-1e250, (1e-250, 2.49036296629014e-267)),
        (-1e290, (9.999999999999999e-291, 1.1549024816968258e-307)),
    ];
    for (y, want) in cases {
        let d = specials::digammainv(Df64::from(y));
        let e = rel(d, want);
        assert!(
            e <= 1e-30,
            "Df64 digammainv({y}) = ({}, {}), err {e:e}",
            d.hi(),
            d.lo()
        );
    }
}

// ------------------------------------------------------ N5 gen_harmonic hang

#[test]
fn n5_gen_harmonic_returns_for_infinite_and_huge_n() {
    let zeta2 = (1.6449340668482264, 3.040672350398476e-17);
    within(20, "gen_harmonic at n = inf", move || {
        // n = +inf: the limit, the same in both strategies.
        for a in [0.5, 1.0, 0.0, -3.0, f64::NEG_INFINITY] {
            assert_eq!(
                specials::gen_harmonic(f64::INFINITY, a),
                f64::INFINITY,
                "f64 (inf, {a})"
            );
            let d = specials::gen_harmonic(Df64::from(f64::INFINITY), Df64::from(a));
            assert_eq!(d.hi(), f64::INFINITY, "Df64 (inf, {a})");
        }
        let f = specials::gen_harmonic(f64::INFINITY, 2.0);
        assert!(
            (f - zeta2.0).abs() <= 2e-16 * zeta2.0,
            "f64 (inf, 2) = {f:e}"
        );
        let d = specials::gen_harmonic(Df64::from(f64::INFINITY), Df64::from(2.0));
        assert!(
            rel(d, zeta2) <= 1e-30,
            "Df64 (inf, 2) = ({}, {})",
            d.hi(),
            d.lo()
        );
        for a in [f64::INFINITY, f64::MAX] {
            assert_eq!(specials::gen_harmonic(f64::INFINITY, a), 1.0);
            assert_eq!(
                specials::gen_harmonic(Df64::from(f64::INFINITY), Df64::from(a)).hi(),
                1.0
            );
        }
    });
    within(20, "gen_harmonic at n > 2^53", move || {
        let cases: [(f64, f64, (f64, f64)); 3] = [
            (1e17, 0.5, (632455530.5733213, 1.6703069007331235e-08)),
            (1e9, 0.5, (63244.092864670165, 3.5313719126324657e-13)),
            (
                1e17,
                1.0000000001,
                (39.72116216919516, -3.4857689974869073e-15),
            ),
        ];
        for (n, a, want) in cases {
            let f = specials::gen_harmonic(n, a);
            let ef = ((f - want.0) / want.0).abs();
            assert!(
                ef <= 4e-15,
                "f64 gen_harmonic({n}, {a}) = {f:e}, err {ef:e}"
            );
            let e = rel(specials::gen_harmonic(Df64::from(n), Df64::from(a)), want);
            assert!(e <= 1e-30, "Df64 gen_harmonic({n}, {a}): err {e:e}");
        }
    });
    // The Euler-Maclaurin cutoff N = 40 + |a| made the direct sum O(|a|) (no end
    // for |a| near f64::MAX), and `-a ln i` overflowed to a NaN low word.
    within(20, "gen_harmonic at huge |a|", move || {
        for (n, a, want) in [
            (1e17, 1e100, 1.0),
            (1e4, f64::MAX, 1.0),
            (1e5, -f64::MAX, f64::INFINITY),
            (1e300, 1e300, 1.0),
        ] {
            assert_eq!(specials::gen_harmonic(n, a), want, "f64 ({n}, {a})");
            assert_eq!(
                specials::gen_harmonic(Df64::from(n), Df64::from(a)).hi(),
                want,
                "Df64 ({n}, {a})"
            );
        }
    });
}

// ------------------------------------------------------- N6/N7 stirling2

#[test]
fn n6_stirling2_truncates_in_both_strategies() {
    let p259 = 2.0_f64.powi(259); // S(260, 2) = 2^259 - 1, rounds to 2^259
    let p218 = 2.0_f64.powi(218);
    let cases = [
        (260.5, 2.0, p259),
        (260.5, 2.5, p259),
        (219.5, 2.5, p218),
        (5.1, 5.5, 1.0),
        (5.7, 5.2, 1.0),
        (1e5, 0.5, 0.0),
        (260.5, 0.5, 0.0),
        (-0.5, -0.5, 0.0),
        (-0.5, 0.0, 0.0),
        (0.5, 0.5, 1.0), // S(0, 0)
        (f64::INFINITY, 5.0, f64::INFINITY),
        (f64::INFINITY, 1.5, 1.0),
        (f64::INFINITY, 0.5, 0.0),
        (5.0, f64::INFINITY, 0.0),
        (1e17, 2.0, f64::INFINITY),
        (1e4, 261.5, f64::INFINITY),
    ];
    for (n, k, want) in cases {
        assert_eq!(specials::stirling2(n, k), want, "f64 stirling2({n}, {k})");
        assert_eq!(
            specials::stirling2(Df64::from(n), Df64::from(k)).hi(),
            want,
            "Df64 stirling2({n}, {k})"
        );
    }
    // Non-integer arguments beyond n = 219 give the values at the truncated
    // integers: f64 is Temme's approximation, Df64 is exact.
    for (n, k, want) in [
        (230.9, 3.7, (9.11459414497156e108, 4.9911420361211423e92)),
        (
            250.2,
            10.99,
            (2.7557319222983898e243, 5.218016721488181e226),
        ),
    ] {
        let f = specials::stirling2(n, k);
        assert_eq!(f, specials::stirling2(n.trunc(), k.trunc()));
        assert!(
            ((f - want.0) / want.0).abs() <= 1e-5,
            "f64 stirling2({n}, {k}) = {f:e}"
        );
        let e = rel(specials::stirling2(Df64::from(n), Df64::from(k)), want);
        assert!(e <= 1e-30, "Df64 stirling2({n}, {k}): err {e:e}");
    }
}

#[test]
fn n7_stirling2_df64_keeps_the_low_word_of_large_arguments() {
    let n = Df64::from(1e17);
    let one = Df64::from(1.0);
    let two = Df64::from(2.0);
    let e = rel(
        specials::stirling2(n, n - one),
        (5e33, 2.2212384506478592e17),
    );
    assert!(e <= 1e-30, "S(1e17, 1e17 - 1): err {e:e}");
    let e = rel(
        specials::stirling2(n, n - two),
        (1.25e67, 6.601388524395825e48),
    );
    assert!(e <= 1e-30, "S(1e17, 1e17 - 2): err {e:e}");
    let m = Df64::from(2.0_f64.powi(53));
    let e = rel(
        specials::stirling2(m + one, m - one),
        (8.227522786606029e62, 3.044795077772713e46),
    );
    assert!(e <= 1e-30, "S(2^53 + 1, 2^53 - 1): err {e:e}");
    // Huge arguments no longer run the recurrence (this did not return).
    within(20, "stirling2(MAX, 1e100)", || {
        let d = specials::stirling2(Df64::from(f64::MAX), Df64::from(1e100));
        assert!(!d.hi().is_nan() || specials::stirling2(f64::MAX, 1e100).is_nan());
    });
}

// ---------------------------------------------------------------- N8 poch

#[test]
fn n8_poch_df64_subnormal_a() {
    let cases: [(f64, f64, (f64, f64)); 4] = [
        (
            1e-310,
            170.5,
            (5.562092414559983e-05, -1.3957807319807326e-21),
        ),
        (
            -1e-310,
            170.5,
            (-5.562092414559983e-05, 1.3957807319807326e-21),
        ),
        (1e-300, 170.5, (556209.241456, -4.2924868069622174e-11)),
        (
            1e-250,
            3.25,
            (2.5492569667185293e-250, 1.4999548935599847e-266),
        ),
    ];
    for (a, m, want) in cases {
        let d = specials::poch(Df64::from(a), Df64::from(m));
        let e = rel(d, want);
        assert!(
            e <= 1e-29,
            "Df64 poch({a}, {m}) = ({}, {}), err {e:e}",
            d.hi(),
            d.lo()
        );
    }
    // a and a + m both tiny: the ordinary path, which the fix must not take.
    for (a, m) in [(5e-324, 5e-324), (1e-310, 1e-310)] {
        let d = specials::poch(Df64::from(a), Df64::from(m)).hi();
        assert!((d - 0.5).abs() < 1e-15, "Df64 poch({a}, {m}) = {d:e}");
    }
}

// ---------------------------------------------------- the review's agreement grid

/// Over fixed extreme and non-finite arguments: where f64 gives ±∞, ±0 or NaN,
/// Df64 gives the same; where f64 is finite, Df64 is finite too.
#[test]
fn strategies_agree_on_extreme_arguments() {
    let base = [
        0.0,
        5e-324,
        1e-310,
        2.2250738585072014e-308,
        1e-300,
        1e-200,
        7.4e-155,
        1e-100,
        1e-20,
        0.5,
        1.0,
        2.0,
        2.5,
        10.0,
        100.5,
        170.25,
        261.5,
        1e4,
        1e5,
        9007199254740992.0,
        1e17,
        1e100,
        1e300,
        f64::MAX,
        f64::INFINITY,
    ];
    let mut v = vec![f64::NAN, -0.0];
    for x in base {
        v.push(x);
        if x != 0.0 {
            v.push(-x);
        }
    }
    fn check(what: String, f: f64, d: f64, bad: &mut Vec<String>) {
        let ok = if f.is_nan() {
            d.is_nan()
        } else if f.is_infinite() || f == 0.0 {
            d == f
        } else {
            d.is_finite()
        };
        if !ok {
            bad.push(format!("{what}: f64 {f:e}, Df64 {d:e}"));
        }
    }
    let mut bad = Vec::new();
    let one: [Unary; 3] = [
        ("zetac", specials::zetac, specials::zetac),
        ("trigamma", specials::trigamma, specials::trigamma),
        ("digammainv", specials::digammainv, specials::digammainv),
    ];
    for (name, f, g) in one {
        for &x in &v {
            check(
                format!("{name}({x:e})"),
                f(x),
                g(Df64::from(x)).hi(),
                &mut bad,
            );
        }
    }
    let two: [Binary; 2] = [
        (
            "gen_harmonic",
            specials::gen_harmonic,
            specials::gen_harmonic,
        ),
        ("stirling2", specials::stirling2, specials::stirling2),
    ];
    let v2 = v.clone();
    let bad2 = within(60, "the agreement grid", move || {
        let mut bad = Vec::new();
        for (name, f, g) in two {
            for &x in &v2 {
                for &y in &v2 {
                    let d = g(Df64::from(x), Df64::from(y)).hi();
                    check(format!("{name}({x:e}, {y:e})"), f(x, y), d, &mut bad);
                }
            }
        }
        bad
    });
    bad.extend(bad2);
    assert!(
        bad.is_empty(),
        "{} disagreements:\n{}",
        bad.len(),
        bad.join("\n")
    );
}
