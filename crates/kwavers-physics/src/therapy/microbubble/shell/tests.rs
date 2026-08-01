use super::properties::MarmottantShellProperties;
use super::state::ShellState;
use aequitas::systems::si::quantities::{DynamicViscosity, Length, SurfaceTension, Velocity};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;

#[test]
fn test_create_shell() {
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    shell.validate().unwrap();
    assert_eq!(shell.state, ShellState::Elastic);
    assert!(!shell.has_ruptured);
}

#[test]
fn test_sono_vue_shell() {
    let shell = MarmottantShellProperties::sono_vue(Length::from_base(1.25e-6)).unwrap();
    shell.validate().unwrap();
    assert_eq!(shell.radius_equilibrium.into_base(), 1.25e-6);
}

#[test]
fn test_surface_tension_buckled() {
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    let chi = shell.surface_tension(Length::from_base(0.8e-6));
    assert_eq!(chi.into_base(), 0.0);
}

#[test]
fn test_surface_tension_elastic() {
    // Marmottant 2005 references the elastic regime to R_buckling (= 0.85·R0
    // here), so χ(R) = κ_s(R²/R_buckling² − 1). The previous assertion used the
    // R_equilibrium reference (R²/R0² − 1), which is analytically incorrect: it
    // yields χ(R_buckling) = κ_s(0.85² − 1) < 0, an unphysical negative surface
    // tension. See `surface_tension` for the derivation.
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    let r = Length::from_base(1.1e-6);
    let r_buckling = shell.radius_buckling;
    let chi = shell.surface_tension(r);
    let expected = 1.0
        * ((r.into_base() * r.into_base()) / (r_buckling.into_base() * r_buckling.into_base())
            - 1.0);
    assert!(
        (chi.into_base() - expected).abs() < 1e-10,
        "expected {expected}, got {}",
        chi.into_base()
    );
    // Continuity / non-negativity: χ(R_buckling) = 0 exactly (was −0.2775 before).
    assert!(shell.surface_tension(r_buckling).into_base().abs() < 1e-12);
    // And χ ≥ 0 across the elastic regime (no negative surface tension).
    for k in 0..=20 {
        let rr = r_buckling + (shell.radius_rupture - r_buckling) * f64::from(k) / 20.0;
        assert!(
            shell.surface_tension(rr).into_base() >= -1e-12,
            "negative χ at R={}",
            rr.into_base()
        );
    }
}

#[test]
fn test_surface_tension_ruptured() {
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    let chi = shell.surface_tension(Length::from_base(2.0e-6));
    // Beyond the rupture radius the shell-bound regime collapses and
    // χ saturates at the water-air surface tension at 20 °C, sourced
    // from `core::constants::cavitation::SURFACE_TENSION_WATER` =
    // 0.0728 N/m. The prior assertion of 0.072 N/m was a literal that
    // did not match the SSOT-sourced default.
    assert_eq!(chi.into_base(), SURFACE_TENSION_WATER);
}

#[test]
fn test_state_transitions() {
    let mut shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();

    assert_eq!(shell.state, ShellState::Elastic);

    shell.update_state(Length::from_base(0.8e-6));
    assert_eq!(shell.state, ShellState::Buckled);
    assert!(shell.is_buckled());

    shell.update_state(Length::from_base(1.0e-6));
    assert_eq!(shell.state, ShellState::Elastic);
    assert!(shell.is_elastic());

    shell.update_state(Length::from_base(2.0e-6));
    assert_eq!(shell.state, ShellState::Ruptured);
    assert!(shell.is_ruptured());
    assert!(shell.has_ruptured);

    // Rupture is irreversible
    shell.update_state(Length::from_base(1.0e-6));
    assert_eq!(shell.state, ShellState::Ruptured);
    assert!(shell.has_ruptured);
}

#[test]
fn test_pressure_contribution() {
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    let p = shell.pressure_contribution(Length::from_base(1.0e-6), Velocity::from_base(10.0));
    assert!(p.into_base() > 0.0);
}

#[test]
fn test_strain_calculation() {
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    assert_eq!(shell.strain(Length::from_base(1.0e-6)).into_base(), 0.0);
    assert!((shell.strain(Length::from_base(0.9e-6)).into_base() + 0.1).abs() < 1e-10);
    assert!((shell.strain(Length::from_base(1.2e-6)).into_base() - 0.2).abs() < 1e-10);
}

#[test]
fn test_validation_invalid_buckling_ratio() {
    let result = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        1.5,
        1.6,
    );
    assert!(result.is_err());
}

#[test]
fn test_validation_invalid_rupture_ratio() {
    let result = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        0.9,
    );
    assert!(result.is_err());
}

#[test]
fn test_surface_tension_derivative() {
    // dχ/dR = 2κ_s·R/R_buckling² (R_buckling reference per Marmottant 2005),
    // consistent with the χ(R) correctness fix above.
    let shell = MarmottantShellProperties::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0e-9),
        0.85,
        1.6,
    )
    .unwrap();
    let r = Length::from_base(1.0e-6);
    let r_buckling = shell.radius_buckling;
    let dchi_dr = shell.surface_tension_derivative(r).into_base();
    let expected = 2.0 * 1.0 * r.into_base() / (r_buckling.into_base() * r_buckling.into_base());
    assert!(
        (dchi_dr - expected).abs() < 1e-10,
        "expected {expected}, got {dchi_dr}"
    );
    assert_eq!(
        shell
            .surface_tension_derivative(Length::from_base(0.8e-6))
            .into_base(),
        0.0
    );
    assert_eq!(
        shell
            .surface_tension_derivative(Length::from_base(2.0e-6))
            .into_base(),
        0.0
    );
}

#[test]
fn test_drug_delivery_shell() {
    let shell = MarmottantShellProperties::drug_delivery(Length::from_base(2.0e-6)).unwrap();
    shell.validate().unwrap();
    assert!(shell.radius_rupture.into_base() < 2.0e-6 * 1.5);
}
