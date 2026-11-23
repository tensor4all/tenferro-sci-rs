//! The families re-exported from `few_special_functions`, one mpmath value
//! each. The values come from the reference tables of `few_special_functions`
//! 0.2.1 (`tests/reference/*.tsv`, mpmath 1.3.0 at 50 digits), rounded to the
//! nearest `f64`. Each family is validated in full there; these cases check
//! that every re-export resolves to the pinned implementation.

use specials::{
    bose_einstein, clausen, coulomb, dawson, debye, fermi_dirac, fresnel, marcum_q,
    parabolic_cylinder, voigt, whittaker,
};

/// Relative tolerance. Every family's measured maximum over its full table
/// is below 5e-13 except Coulomb `psi` and parabolic-cylinder `w`, and both
/// are well below this at the points used here.
const TOLERANCE: f64 = 1e-13;

fn check(name: &str, actual: f64, expected: f64) {
    let error = ((actual - expected) / expected).abs();
    assert!(
        error <= TOLERANCE,
        "{name}: expected {expected:e}, got {actual:e} (relative error {error:.1e})"
    );
}

#[test]
fn re_exported_families_match_mpmath() {
    check(
        "fermi_dirac_integral(0.5, 1)",
        fermi_dirac::fermi_dirac_integral(0.5, 1.0),
        1.3963752806665641,
    );
    check(
        "bose_einstein_integral(0.5, -1)",
        bose_einstein::bose_einstein_integral(0.5, -1.0),
        0.37969571496313553,
    );
    check(
        "Cl_2(1)",
        clausen::clausen(clausen::ClausenOrder::Two, 1.0),
        1.0139591323607684,
    );
    check(
        "debye_function(1, 1, 1)",
        debye::debye_function(1.0, 1.0, 1.0),
        0.7775046341122482,
    );
    check(
        "coulomb::f(1, 0.5, 2)",
        coulomb::f(1.0, 0.5, 2.0),
        0.6185123212891975,
    );
    check(
        "coulomb::g(1, 0.5, 2)",
        coulomb::g(1.0, 0.5, 2.0),
        1.1982980556731446,
    );
    check(
        "coulomb::psi(1, 0.5, 2)",
        coulomb::psi(1.0, 0.5, 2.0, None).re,
        -3.265366139066061,
    );
    check(
        "whittaker_m(0.5, 0.25, 2)",
        whittaker::whittaker_m(0.5, 0.25, 2.0),
        0.9948565892668343,
    );
    check(
        "whittaker_w(0.5, 0.25, 2)",
        whittaker::whittaker_w(0.5, 0.25, 2.0),
        0.534013946067451,
    );
    check(
        "marcum_q(1, 1, 2)",
        marcum_q::marcum_q(1.0, 1.0, 2.0),
        0.26901206003591,
    );
    check("dawson(1)", dawson::dawson(1.0), 0.5380795069127684);
    check("fresnel_c(2)", fresnel::fresnel_c(2.0), 0.48825340607534073);
    check("fresnel_s(2)", fresnel::fresnel_s(2.0), 0.34341567836369824);
    check("voigt(1, 0.5)", voigt::voigt(1.0, 0.5), 0.3549003328675779);
    check(
        "parabolic_cylinder::u(0.5, 1)",
        parabolic_cylinder::u(0.5, 1.0),
        0.5106437410796607,
    );
    check(
        "parabolic_cylinder::v(0.5, 1)",
        parabolic_cylinder::v(0.5, 1.0),
        1.0245040556536147,
    );
}

#[test]
fn special_error_is_re_exported() {
    let error: Result<f64, specials::SpecialError> =
        fermi_dirac::fermi_dirac_integral_checked(0.5, 1.0);
    assert_eq!(error, Ok(fermi_dirac::fermi_dirac_integral(0.5, 1.0)));
}
