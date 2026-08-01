use super::*;
use aequitas::systems::si::quantities::{
    DynamicViscosity, Length, MassDensity, Pressure, SurfaceTension, Velocity,
};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;

const AMBIENT_PRESSURE: f64 = ATMOSPHERIC_PRESSURE;
// SSOT: 37 °C + 273.15 = 310.15 K (exact Celsius-to-Kelvin conversion).
const BODY_TEMPERATURE: f64 = BODY_TEMPERATURE_K;

fn expected_gas_moles(radius: f64) -> f64 {
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    (AMBIENT_PRESSURE * volume) / (GAS_CONSTANT * BODY_TEMPERATURE)
}

fn assert_invalid_value(result: KwaversResult<MicrobubbleState>, parameter: &str, reason: &str) {
    match result {
        Err(KwaversError::Validation(ValidationError::InvalidValue {
            parameter: actual_parameter,
            reason: actual_reason,
            ..
        })) => {
            assert_eq!(actual_parameter, parameter);
            assert_eq!(actual_reason, reason);
        }
        Err(error) => panic!("expected validation invalid-value error, got {error:?}"),
        Ok(state) => panic!("expected validation error, got {state:?}"),
    }
}

#[test]
fn test_create_sono_vue() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::sono_vue(pos).unwrap();

    assert_eq!(state.radius.into_base().to_bits(), 1.25e-6_f64.to_bits());
    assert_eq!(state.radius, state.radius_equilibrium);
    assert_eq!(state.wall_velocity.into_base().to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        state.wall_acceleration.into_base().to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        state.temperature.into_base().to_bits(),
        BODY_TEMPERATURE.to_bits()
    );
    assert_eq!(
        state.pressure_internal.into_base().to_bits(),
        AMBIENT_PRESSURE.to_bits()
    );
    assert_eq!(
        state.pressure_liquid.into_base().to_bits(),
        AMBIENT_PRESSURE.to_bits()
    );
    assert_eq!(
        state.surface_tension.into_base().to_bits(),
        SURFACE_TENSION_WATER.to_bits()
    );
    assert_eq!(
        state.gas_moles.into_base().to_bits(),
        expected_gas_moles(1.25e-6).to_bits()
    );
    state.validate().unwrap();
}

#[test]
fn test_create_definity() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::definity(pos).unwrap();

    assert_eq!(state.radius.into_base().to_bits(), 1.5e-6_f64.to_bits());
    assert_eq!(
        state.shell_elasticity.into_base().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        state.shell_viscosity.into_base().to_bits(),
        1.2e-9_f64.to_bits()
    );
    assert_eq!(
        state.shell_radius_buckling.into_base().to_bits(),
        (1.5e-6_f64 * 0.9).to_bits()
    );
    assert_eq!(
        state.shell_radius_rupture.into_base().to_bits(),
        (1.5e-6_f64 * 1.5).to_bits()
    );
    state.validate().unwrap();
}

#[test]
fn test_drug_loaded() {
    let pos = Position3D::zero();
    let drug_conc = 100.0;
    let state = MicrobubbleState::drug_loaded(
        Length::from_base(2.0e-6),
        MassDensity::from_base(drug_conc),
        pos,
    )
    .unwrap();

    assert_eq!(state.drug_concentration.into_base(), drug_conc);
    let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * (2.0e-6_f64).powi(3);
    let expected_mass = drug_conc * expected_volume;
    assert_eq!(
        state.drug_mass().into_base().to_bits(),
        expected_mass.to_bits()
    );
    assert_eq!(
        state.drug_remaining_fraction().into_base().to_bits(),
        1.0_f64.to_bits()
    );
    state.validate().unwrap();
}

#[test]
fn test_validation_negative_radius() {
    let pos = Position3D::zero();
    let result = MicrobubbleState::new(
        Length::from_base(-1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    );
    assert_invalid_value(result, "radius_equilibrium", "must be positive");
}

#[test]
fn test_validation_negative_drug() {
    let pos = Position3D::zero();
    let result = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(-10.0),
        pos,
    );
    assert_invalid_value(result, "drug_concentration", "must be non-negative");
}

#[test]
fn test_validate_rejects_mutated_invalid_runtime_state() {
    let pos = Position3D::zero();

    let mut negative_radius = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();
    negative_radius.radius = Length::from_base(0.0);
    assert_invalid_value(
        negative_radius.validate().map(|()| negative_radius),
        "radius",
        "must be positive",
    );

    let mut negative_pressure = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();
    negative_pressure.pressure_internal = Pressure::from_base(-1.0);
    assert_invalid_value(
        negative_pressure.validate().map(|()| negative_pressure),
        "pressure_internal",
        "must be non-negative",
    );
}

#[test]
fn test_volume_calculation() {
    let pos = Position3D::zero();
    let radius = 1.0e-6;
    let state = MicrobubbleState::new(
        Length::from_base(radius),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();

    let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    assert!((state.volume().into_base() - expected_volume).abs() < 1e-20);
}

#[test]
fn test_compression_ratio() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();

    assert_eq!(state.compression_ratio().into_base(), 1.0);
    assert!(!state.is_compressed());
    assert!(!state.is_expanded());

    state.radius = Length::from_base(0.5e-6);
    assert_eq!(state.compression_ratio().into_base(), 0.5);
    assert!(state.is_compressed());
    assert!(!state.is_expanded());

    state.radius = Length::from_base(2.0e-6);
    assert_eq!(state.compression_ratio().into_base(), 2.0);
    assert!(!state.is_compressed());
    assert!(state.is_expanded());
}

#[test]
fn test_cavitation_criterion() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();

    assert!(!state.is_cavitating());

    state.radius = Length::from_base(2.5e-6);
    assert!(state.is_cavitating());
}

#[test]
fn test_resonance_frequency() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();

    let f0 = state.resonance_frequency().into_base();
    assert!(f0 > 2.0 * MHZ_TO_HZ && f0 < 5.0 * MHZ_TO_HZ);
}

#[test]
fn test_energy_conservation_zero_at_equilibrium() {
    let pos = Position3D::zero();
    let state = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();

    assert!(state.kinetic_energy().into_base().abs() < 1e-20);
    assert!(state.potential_energy().into_base().abs() < 1e-15);
}

/// Potential energy is positive for both compression and expansion (Brennen 1995, §4.1).
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_potential_energy_positive_for_expansion_and_compression() {
    let pos = Position3D::zero();
    let mut state = MicrobubbleState::new(
        Length::from_base(1.0e-6),
        SurfaceTension::from_base(1.0),
        DynamicViscosity::from_base(1.0),
        MassDensity::from_base(0.0),
        pos,
    )
    .unwrap();
    let r0 = state.radius_equilibrium.into_base();

    state.radius = Length::from_base(r0 * 0.5);
    let e_compressed = state.potential_energy().into_base();
    assert!(
        e_compressed > 0.0,
        "E_pot must be positive for compressed bubble, got {e_compressed:.3e}"
    );

    state.radius = Length::from_base(r0 * 2.0);
    let e_expanded = state.potential_energy().into_base();
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
    let pos1 = Position3D::new(
        Length::from_base(0.0),
        Length::from_base(0.0),
        Length::from_base(0.0),
    );
    let pos2 = Position3D::new(
        Length::from_base(3.0),
        Length::from_base(4.0),
        Length::from_base(0.0),
    );

    assert_eq!(pos1.distance_to(&pos2).into_base(), 5.0);
}

#[test]
fn test_velocity_magnitude() {
    let vel = Velocity3D::new(
        Velocity::from_base(3.0),
        Velocity::from_base(4.0),
        Velocity::from_base(0.0),
    );
    assert_eq!(vel.magnitude().into_base(), 5.0);
}
