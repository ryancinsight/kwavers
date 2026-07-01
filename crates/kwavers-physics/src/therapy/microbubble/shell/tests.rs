use super::properties::MarmottantShellProperties;
use super::state::ShellState;
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;

#[test]
fn test_create_shell() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    shell.validate().unwrap();
    assert_eq!(shell.state, ShellState::Elastic);
    assert!(!shell.has_ruptured);
}

#[test]
fn test_sono_vue_shell() {
    let shell = MarmottantShellProperties::sono_vue(1.25e-6).unwrap();
    shell.validate().unwrap();
    assert_eq!(shell.radius_equilibrium, 1.25e-6);
}

#[test]
fn test_surface_tension_buckled() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let chi = shell.surface_tension(0.8e-6);
    assert_eq!(chi, 0.0);
}

#[test]
fn test_surface_tension_elastic() {
    // Marmottant 2005 references the elastic regime to R_buckling (= 0.85·R0
    // here), so χ(R) = κ_s(R²/R_buckling² − 1). The previous assertion used the
    // R_equilibrium reference (R²/R0² − 1), which is analytically incorrect: it
    // yields χ(R_buckling) = κ_s(0.85² − 1) < 0, an unphysical negative surface
    // tension. See `surface_tension` for the derivation.
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let r = 1.1e-6;
    let r_buckling = shell.radius_buckling;
    let chi = shell.surface_tension(r);
    let expected = 1.0 * ((r * r) / (r_buckling * r_buckling) - 1.0);
    assert!(
        (chi - expected).abs() < 1e-10,
        "expected {expected}, got {chi}"
    );
    // Continuity / non-negativity: χ(R_buckling) = 0 exactly (was −0.2775 before).
    assert!(shell.surface_tension(r_buckling).abs() < 1e-12);
    // And χ ≥ 0 across the elastic regime (no negative surface tension).
    for k in 0..=20 {
        let rr = r_buckling + (shell.radius_rupture - r_buckling) * f64::from(k) / 20.0;
        assert!(shell.surface_tension(rr) >= -1e-12, "negative χ at R={rr}");
    }
}

#[test]
fn test_surface_tension_ruptured() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let chi = shell.surface_tension(2.0e-6);
    // Beyond the rupture radius the shell-bound regime collapses and
    // χ saturates at the water-air surface tension at 20 °C, sourced
    // from `core::constants::cavitation::SURFACE_TENSION_WATER` =
    // 0.0728 N/m. The prior assertion of 0.072 N/m was a literal that
    // did not match the SSOT-sourced default.
    assert_eq!(chi, SURFACE_TENSION_WATER);
}

#[test]
fn test_state_transitions() {
    let mut shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();

    assert_eq!(shell.state, ShellState::Elastic);

    shell.update_state(0.8e-6);
    assert_eq!(shell.state, ShellState::Buckled);
    assert!(shell.is_buckled());

    shell.update_state(1.0e-6);
    assert_eq!(shell.state, ShellState::Elastic);
    assert!(shell.is_elastic());

    shell.update_state(2.0e-6);
    assert_eq!(shell.state, ShellState::Ruptured);
    assert!(shell.is_ruptured());
    assert!(shell.has_ruptured);

    // Rupture is irreversible
    shell.update_state(1.0e-6);
    assert_eq!(shell.state, ShellState::Ruptured);
    assert!(shell.has_ruptured);
}

#[test]
fn test_pressure_contribution() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let p = shell.pressure_contribution(1.0e-6, 10.0);
    assert!(p > 0.0);
}

#[test]
fn test_strain_calculation() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    assert_eq!(shell.strain(1.0e-6), 0.0);
    assert!((shell.strain(0.9e-6) + 0.1).abs() < 1e-10);
    assert!((shell.strain(1.2e-6) - 0.2).abs() < 1e-10);
}

#[test]
fn test_validation_invalid_buckling_ratio() {
    let result = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 1.5, 1.6);
    assert!(result.is_err());
}

#[test]
fn test_validation_invalid_rupture_ratio() {
    let result = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 0.9);
    assert!(result.is_err());
}

#[test]
fn test_surface_tension_derivative() {
    // dχ/dR = 2κ_s·R/R_buckling² (R_buckling reference per Marmottant 2005),
    // consistent with the χ(R) correctness fix above.
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let r = 1.0e-6;
    let r_buckling = shell.radius_buckling;
    let dchi_dr = shell.surface_tension_derivative(r);
    let expected = 2.0 * 1.0 * r / (r_buckling * r_buckling);
    assert!(
        (dchi_dr - expected).abs() < 1e-10,
        "expected {expected}, got {dchi_dr}"
    );
    assert_eq!(shell.surface_tension_derivative(0.8e-6), 0.0);
    assert_eq!(shell.surface_tension_derivative(2.0e-6), 0.0);
}

#[test]
fn test_drug_delivery_shell() {
    let shell = MarmottantShellProperties::drug_delivery(2.0e-6).unwrap();
    shell.validate().unwrap();
    assert!(shell.radius_rupture < 2.0e-6 * 1.5);
}
