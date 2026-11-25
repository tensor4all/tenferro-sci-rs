//! The precision-adaptive algorithms evaluated in `f64`.
//!
//! The public API sends `f64` to the xsf kernels, so these tests check the
//! generic algorithms' logic (branches, recurrences, truncation) at `f64`
//! precision against the same mpmath reference points. Their double-double
//! accuracy is checked through the public API in `tests/reference.rs`.

use crate::generic;
use crate::refdata::{error, rows};

/// (function, relative tolerance, absolute floor)
const CASES: [(&str, f64, f64); 30] = [
    ("gamma", 4e-15, 0.0),
    ("lgamma", 4e-15, 0.0),
    ("rgamma", 4e-15, 0.0),
    // The f64 rounding of the tabulated zero of ψ shifts the result at the
    // zero itself by ~1e-16, so the criterion is absolute there.
    ("digamma", 4e-15, 1.0),
    ("zeta", 8e-15, 0.0),
    // exp(ln B) for arguments beyond Γ's range: error ~ |ln B| ε.
    ("beta", 5e-13, 0.0),
    ("lbeta", 4e-15, 0.0),
    // The rest of the gamma family (#8). poch, binom and gen_harmonic pay
    // |ln Γ| ε in f64 (exp of a difference of ln Γ values, or of -a ln i).
    ("poch", 1e-12, 0.0),
    // x = -3.75 is next to a zero of ln|Γ(1+x)| (x ≈ -3.74): absolute accuracy
    // eps there gives 3e-14 relative.
    ("lgam1p", 5e-14, 0.0),
    ("binom", 1e-12, 0.0),
    ("zetac", 1e-12, 0.0),
    ("trigamma", 4e-15, 0.0),
    ("digammainv", 4e-15, 0.0),
    ("gen_harmonic", 4e-13, 0.0),
    // 6e5 additions at n = 2e5: the f64 recurrence accumulates ~1e-14.
    ("stirling2", 1e-13, 0.0),
    // The elliptic family (#11), evaluated in f64 by the double-double algorithms.
    ("ellipk", 1e-13, 0.0),
    ("ellipkm1", 1e-13, 0.0),
    ("ellipe", 1e-13, 0.0),
    ("ellipkinc", 1e-13, 0.0),
    ("ellipeinc", 1e-13, 0.0),
    ("ellipj_sn", 1e-13, 0.0),
    ("ellipj_cn", 1e-13, 0.0),
    ("ellipj_dn", 1e-13, 0.0),
    ("ellipj_ph", 1e-13, 0.0),
    ("elliprc", 1e-13, 0.0),
    ("elliprf", 1e-13, 0.0),
    ("elliprd", 1e-13, 0.0),
    ("elliprj", 1e-13, 0.0),
    ("elliprg", 1e-13, 0.0),
    ("agm", 1e-13, 0.0),
];

fn eval(name: &str, a: &[f64]) -> f64 {
    match name {
        "gamma" => generic::gamma::gamma(a[0]),
        "lgamma" => generic::gamma::lgamma(a[0]),
        "rgamma" => generic::gamma::rgamma(a[0]),
        "digamma" => generic::digamma::digamma(a[0]),
        "zeta" => generic::zeta::zeta(a[0], a[1]),
        "beta" => generic::beta::beta(a[0], a[1]),
        "lbeta" => generic::beta::lbeta(a[0], a[1]),
        "poch" => generic::poch::poch(a[0], a[1]),
        "lgam1p" => generic::lgam1p::lgam1p(a[0]),
        "binom" => generic::binom::binom(a[0], a[1]),
        "zetac" => generic::zeta::zetac(a[0]),
        "trigamma" => generic::trigamma::trigamma(a[0]),
        "digammainv" => generic::digammainv::digammainv(a[0]),
        "gen_harmonic" => generic::gen_harmonic::gen_harmonic(a[0], a[1]),
        "stirling2" => match generic::stirling2::prepare(a[0], a[1]) {
            generic::stirling2::Prepared::Value(v) => v,
            generic::stirling2::Prepared::Kernel(n, k) => generic::stirling2::stirling2(n, k),
        },
        "ellipk" => generic::ellip::ellipk(a[0]),
        "ellipkm1" => generic::ellip::ellipkm1(a[0]),
        "ellipe" => generic::ellip::ellipe(a[0]),
        "ellipkinc" => generic::ellip::ellipkinc(a[0], a[1]),
        "ellipeinc" => generic::ellip::ellipeinc(a[0], a[1]),
        "ellipj_sn" => generic::ellip::ellipj(a[0], a[1]).0,
        "ellipj_cn" => generic::ellip::ellipj(a[0], a[1]).1,
        "ellipj_dn" => generic::ellip::ellipj(a[0], a[1]).2,
        "ellipj_ph" => generic::ellip::ellipj(a[0], a[1]).3,
        "elliprc" => generic::carlson::elliprc(a[0], a[1]),
        "elliprf" => generic::carlson::elliprf(a[0], a[1], a[2]),
        "elliprd" => generic::carlson::elliprd(a[0], a[1], a[2]),
        "elliprj" => generic::carlson::elliprj(a[0], a[1], a[2], a[3]),
        "elliprg" => generic::carlson::elliprg(a[0], a[1], a[2]),
        "agm" => generic::agm::agm(a[0], a[1]),
        _ => unreachable!("{name}"),
    }
}

#[test]
fn generic_algorithms_in_f64_match_mpmath() {
    for (name, tol, floor) in CASES {
        for r in rows(name).iter().filter(|r| r.is_f64()) {
            let a: Vec<f64> = r.args_as();
            let got = eval(name, &a);
            let scale = crate::refdata::jacobi_scale(
                name,
                a[0],
                a.get(1).copied().unwrap_or(0.0),
                r.reference.0,
            );
            let floor = floor.max(crate::refdata::jacobi_floor(
                scale,
                100.0,
                f64::EPSILON,
                tol,
            ));
            let e = error(got, r.reference, floor);
            assert!(
                e <= tol,
                "{name}{a:?} (line {}): got {got:e}, ref {:?}, err {e:e} > {tol:e}",
                r.line,
                r.reference
            );
        }
    }
}

#[test]
fn sinpi_is_exact_at_half_integers_and_odd_in_x() {
    for (x, s) in [
        (0.5, 1.0),
        (1.5, -1.0),
        (-0.5, -1.0),
        (-2.5, -1.0),
        (1e10 + 0.5, 1.0),
    ] {
        assert_eq!(generic::sinpi(x), s, "sinpi({x})");
    }
    assert_eq!(generic::sinpi(3.0_f64), 0.0);
}

/// GKP (6.43) for `S(n, n - j)` against the recurrence, where the recurrence is
/// still cheap: exact integers below 2^106, relative 1e-30 above.
#[cfg(feature = "xprec")]
#[test]
fn near_diagonal_matches_the_recurrence() {
    use xprec::Df64;
    for n in [70_u32, 100, 219, 500, 2000] {
        for j in [1_usize, 2, 3, 7, 20, 33, 64] {
            if j as u32 >= n {
                continue;
            }
            let nn = Df64::from(f64::from(n));
            let kk = nn - Df64::from(j as f64);
            let want = generic::stirling2::stirling2(nn, kk);
            let got = generic::stirling2::near_diagonal(nn, j);
            if want.hi().is_infinite() {
                assert_eq!(got.hi(), f64::INFINITY, "S({n}, {n} - {j})");
                continue;
            }
            let e = ((got - want) / want).abs().hi();
            assert!(e <= 1e-30, "S({n}, {n} - {j}): err {e:e}");
        }
    }
}

/// The integer-degree recurrences and the Legendre-function recurrences of `generic`, evaluated
/// in `f64`, against the mpmath rows. The last argument of a row is its scale (the largest term of
/// the recurrence), so the absolute floor near a zero is `100 (n + 1) eps scale`.
#[test]
fn generic_orthogonal_algorithms_in_f64_match_mpmath() {
    use crate::generic::{legendre as gl, orthogonal as go};
    let tol = 1e-10;
    let names = [
        "eval_jacobi_int",
        "eval_sh_jacobi_int",
        "eval_gegenbauer_int",
        "eval_chebyt_int",
        "eval_chebyu_int",
        "eval_chebyc_int",
        "eval_chebys_int",
        "eval_sh_chebyt_int",
        "eval_sh_chebyu_int",
        "eval_legendre_int",
        "eval_sh_legendre_int",
        "eval_laguerre_int",
        "eval_genlaguerre_int",
        "eval_hermite",
        "eval_hermitenorm",
        "assoc_legendre_p",
        "assoc_legendre_p_t3",
        "assoc_legendre_p_norm",
        "assoc_legendre_p_norm_t3",
        "sph_legendre_p",
        "pmv_int",
    ];
    for name in names {
        let all = rows(name);
        assert!(all.len() > 60, "{name}: {} rows", all.len());
        for r in all.iter().filter(|r| r.is_f64()) {
            let a: Vec<f64> = r.args_as();
            // `j + 2 alpha - 1` absorbs alpha of 1e-8 or less in f64 (double-double keeps it in the low word)
            if name == "eval_gegenbauer_int" && a[1].abs() < 1e-8 {
                continue;
            }
            let n = a[0] as i64;
            let got = match name {
                "eval_jacobi_int" => go::jacobi(n, a[1], a[2], a[3]),
                "eval_sh_jacobi_int" => go::sh_jacobi(n, a[1], a[2], a[3]),
                "eval_gegenbauer_int" => go::gegenbauer(n, a[1], a[2]),
                "eval_chebyt_int" => go::chebyt(n, a[1]),
                "eval_chebyu_int" => go::chebyu(n, a[1]),
                "eval_chebyc_int" => go::chebyc(n, a[1]),
                "eval_chebys_int" => go::chebys(n, a[1]),
                "eval_sh_chebyt_int" => go::sh_chebyt(n, a[1]),
                "eval_sh_chebyu_int" => go::sh_chebyu(n, a[1]),
                "eval_legendre_int" => go::legendre(n, a[1]),
                "eval_sh_legendre_int" => go::sh_legendre(n, a[1]),
                "eval_laguerre_int" => go::genlaguerre(n, 0.0, a[1]),
                "eval_genlaguerre_int" => go::genlaguerre(n, a[1], a[2]),
                "eval_hermite" => go::hermite(n, a[1]),
                "eval_hermitenorm" => go::hermitenorm(n, a[1]),
                "assoc_legendre_p" => {
                    gl::assoc_legendre_p(false, n as i32, a[1] as i32, a[2], false)
                }
                "assoc_legendre_p_t3" => {
                    gl::assoc_legendre_p(false, n as i32, a[1] as i32, a[2], true)
                }
                "assoc_legendre_p_norm" => {
                    gl::assoc_legendre_p(true, n as i32, a[1] as i32, a[2], false)
                }
                "assoc_legendre_p_norm_t3" => {
                    gl::assoc_legendre_p(true, n as i32, a[1] as i32, a[2], true)
                }
                "sph_legendre_p" => gl::sph_legendre_p(n as i32, a[1] as i32, a[2]),
                _ => gl::pmv_int(a[0], a[1], a[2]).expect("integer arguments"),
            };
            // degree and order set the size of the absolute floor
            let size = if name.contains("legendre") || name == "pmv_int" {
                a[0].abs().max(a[1].abs())
            } else {
                a[0].abs()
            };
            let scale = a[a.len() - 1];
            let floor = 100.0 * (size + 1.0) * f64::EPSILON * scale;
            let reference = r.reference.0 + r.reference.1;
            let ok = got.is_finite()
                && ((got - reference).abs() <= tol * reference.abs().max(floor / tol)
                    || (got - reference).abs() <= floor);
            // sph and normalized rows near |z| = 1 carry the 1 - z z rounding of the input
            let ok = ok
                || (name.starts_with("assoc") || name == "sph_legendre_p") && {
                    let e = error(got, r.reference, floor / tol);
                    e <= 1e-8
                };
            assert!(
                ok,
                "{name}{a:?} (line {}): got {got:e}, mpmath {:?}",
                r.line, r.reference
            );
        }
    }
}
