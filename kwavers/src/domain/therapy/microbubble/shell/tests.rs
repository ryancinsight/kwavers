use super::properties::MarmottantShellProperties;
use super::state::ShellState;

#[test]
fn test_create_shell() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    assert!(shell.validate().is_ok());
    assert_eq!(shell.state, ShellState::Elastic);
    assert!(!shell.has_ruptured);
}

#[test]
fn test_sono_vue_shell() {
    let shell = MarmottantShellProperties::sono_vue(1.25e-6).unwrap();
    assert!(shell.validate().is_ok());
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
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let r = 1.1e-6;
    let chi = shell.surface_tension(r);
    let expected = 1.0 * ((1.1 * 1.1) / (1.0 * 1.0) - 1.0);
    assert!((chi - expected).abs() < 1e-10);
}

#[test]
fn test_surface_tension_ruptured() {
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let chi = shell.surface_tension(2.0e-6);
    assert_eq!(chi, 0.072);
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
    let shell = MarmottantShellProperties::new(1.0e-6, 1.0, 1.0e-9, 0.85, 1.6).unwrap();
    let r = 1.0e-6;
    let dchi_dr = shell.surface_tension_derivative(r);
    let expected = 2.0 * 1.0 * r / (1.0e-6 * 1.0e-6);
    assert!((dchi_dr - expected).abs() < 1e-10);
    assert_eq!(shell.surface_tension_derivative(0.8e-6), 0.0);
    assert_eq!(shell.surface_tension_derivative(2.0e-6), 0.0);
}

#[test]
fn test_drug_delivery_shell() {
    let shell = MarmottantShellProperties::drug_delivery(2.0e-6).unwrap();
    assert!(shell.validate().is_ok());
    assert!(shell.radius_rupture < 2.0e-6 * 1.5);
}
