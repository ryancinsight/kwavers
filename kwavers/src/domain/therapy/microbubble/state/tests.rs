use super::*;

#[test]
fn test_create_sono_vue() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::sono_vue(pos).unwrap();

    assert!(state.radius > 0.0);
    assert_eq!(state.radius, state.radius_equilibrium);
    assert_eq!(state.wall_velocity, 0.0);
    assert!(state.temperature > 0.0);
    assert!(state.validate().is_ok());
}

#[test]
fn test_create_definity() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::definity(pos).unwrap();

    assert!(state.radius > 0.0);
    assert!(state.radius > 1.0e-6);
    assert!(state.validate().is_ok());
}

#[test]
fn test_drug_loaded() {
    let pos = Position3D::zero();
    let drug_conc = 100.0;
    let state = MicrobubbleState::drug_loaded(2.0, drug_conc, pos).unwrap();

    assert_eq!(state.drug_concentration, drug_conc);
    assert!(state.drug_mass() > 0.0);
    assert_eq!(state.drug_remaining_fraction(), 1.0);
    assert!(state.validate().is_ok());
}

#[test]
fn test_validation_negative_radius() {
    let pos = Position3D::zero();
    let result = MicrobubbleState::new(-1.0e-6, 1.0, 1.0, 0.0, pos);
    assert!(result.is_err());
}

#[test]
fn test_validation_negative_drug() {
    let pos = Position3D::zero();
    let result = MicrobubbleState::new(1.0e-6, 1.0, 1.0, -10.0, pos);
    assert!(result.is_err());
}

#[test]
fn test_volume_calculation() {
    let pos = Position3D::zero();
    let radius = 1.0e-6;
    let state = MicrobubbleState::new(radius, 1.0, 1.0, 0.0, pos).unwrap();

    let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    assert!((state.volume() - expected_volume).abs() < 1e-20);
}

#[test]
fn test_compression_ratio() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

    assert_eq!(state.compression_ratio(), 1.0);
    assert!(!state.is_compressed());
    assert!(!state.is_expanded());

    state.radius = 0.5e-6;
    assert_eq!(state.compression_ratio(), 0.5);
    assert!(state.is_compressed());
    assert!(!state.is_expanded());

    state.radius = 2.0e-6;
    assert_eq!(state.compression_ratio(), 2.0);
    assert!(!state.is_compressed());
    assert!(state.is_expanded());
}

#[test]
fn test_cavitation_criterion() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

    assert!(!state.is_cavitating());

    state.radius = 2.5e-6;
    assert!(state.is_cavitating());
}

#[test]
fn test_resonance_frequency() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

    let f0 = state.resonance_frequency();
    assert!(f0 > 2e6 && f0 < 5e6);
}

#[test]
fn test_energy_conservation_zero_at_equilibrium() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();

    assert!(state.kinetic_energy().abs() < 1e-20);
    assert!(state.potential_energy().abs() < 1e-15);
}

/// Potential energy is positive for both compression and expansion (Brennen 1995, §4.1).
#[test]
fn test_potential_energy_positive_for_expansion_and_compression() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(1.0e-6, 1.0, 1.0, 0.0, pos).unwrap();
    let r0 = state.radius_equilibrium;

    state.radius = r0 * 0.5;
    let e_compressed = state.potential_energy();
    assert!(
        e_compressed > 0.0,
        "E_pot must be positive for compressed bubble, got {e_compressed:.3e}"
    );

    state.radius = r0 * 2.0;
    let e_expanded = state.potential_energy();
    assert!(
        e_expanded > 0.0,
        "E_pot must be positive for expanded bubble, got {e_expanded:.3e}"
    );

    assert!(
        e_expanded > e_compressed,
        "Expanded energy ({e_expanded:.3e}) should exceed compressed ({e_compressed:.3e})"
    );
}

#[test]
fn test_position_distance() {
    let pos1 = Position3D::new(0.0, 0.0, 0.0);
    let pos2 = Position3D::new(3.0, 4.0, 0.0);

    assert_eq!(pos1.distance_to(&pos2), 5.0);
}

#[test]
fn test_velocity_magnitude() {
    let vel = Velocity3D::new(3.0, 4.0, 0.0);
    assert_eq!(vel.magnitude(), 5.0);
}
