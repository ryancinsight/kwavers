use super::{CouplingStrategy, FieldCoupler};
use crate::core::constants::{
    fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE},
    thermodynamic::SPECIFIC_HEAT_WATER,
};
use crate::core::error::KwaversError;
use ndarray::Array3;

/// A large-magnitude field with a small *relative* change must be reported
/// as converged, even though the absolute difference is large.
///
/// Absolute tolerance would fail to converge here (diff >> tolerance), but
/// relative tolerance correctly identifies convergence (rel_diff << tolerance).
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_convergence_relative_not_absolute() {
    let coupler = FieldCoupler {
        strategy: CouplingStrategy::Strong,
        coupling_strength: 1.0,
        max_iterations: 10,
        tolerance: 1e-6,
    };

    // Field magnitude: 1e6 Pa; absolute change: 1 Pa → relative change: 1e-6
    let prev = vec![Array3::from_elem((4, 4, 4), 1_000_000.0_f64)];
    let curr = vec![Array3::from_elem((4, 4, 4), 1_000_001.0_f64)];

    assert!(
        coupler.check_convergence(&prev, &curr),
        "relative change of 1e-6 at 1 MPa must be reported as converged"
    );
}

/// Two identical fields must always be converged regardless of magnitude.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_convergence_identical_fields() {
    let coupler = FieldCoupler::new(CouplingStrategy::Strong);
    let field = vec![Array3::from_elem((4, 4, 4), 42.0_f64)];
    assert!(coupler.check_convergence(&field, &field));
}

/// A field where the change exceeds the relative tolerance must NOT converge.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_convergence_large_relative_change_not_converged() {
    let coupler = FieldCoupler {
        strategy: CouplingStrategy::Strong,
        coupling_strength: 1.0,
        max_iterations: 10,
        tolerance: 1e-6,
    };

    let prev = vec![Array3::from_elem((4, 4, 4), 1.0_f64)];
    let curr = vec![Array3::from_elem((4, 4, 4), 1.1_f64)];

    assert!(
        !coupler.check_convergence(&prev, &curr),
        "10 % relative change must NOT be reported as converged"
    );
}

/// Verify that DENSITY_WATER_NOMINAL matches the expected water density.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[test]
fn test_density_water_nominal_is_1000() {
    assert!(
        (DENSITY_WATER_NOMINAL - 1000.0).abs() < 1.0,
        "DENSITY_WATER_NOMINAL ({DENSITY_WATER_NOMINAL}) must be ≈ 1000 kg/m³"
    );
}

/// Verify SPECIFIC_HEAT_WATER is within the published range for water at 20°C.
///
/// NIST data: c_p(water, 20°C) = 4181.8 J/(kg·K). Range [4150, 4220].
/// # Panics
/// - Panics if assertion fails.
///
#[test]
fn test_specific_heat_water_within_literature_range() {
    assert!(
        SPECIFIC_HEAT_WATER > 4150.0 && SPECIFIC_HEAT_WATER < 4220.0,
        "SPECIFIC_HEAT_WATER ({SPECIFIC_HEAT_WATER}) outside NIST range [4150, 4220] J/(kg·K)"
    );
}

/// Weak coupling must evaluate the three physical coupling edges without
/// mutating the source pressure field.
///
/// Analytical references:
/// - photoelastic modulation: `I₁ = I₀(1 + s·10⁻¹²p·dt)`
/// - optical heating: `ΔT_o = μ_a I₁ dt / (ρc_p)`
/// - acoustic absorption heating: `ΔT_a = αp²dt / (ρ²c_pc)`
/// # Panics
/// - Panics if a value differs from the analytical coupling update.
///
#[test]
fn weak_coupling_updates_targets_from_source_fields() {
    let coupler = FieldCoupler::new(CouplingStrategy::Weak);
    let pressure = 2.0e6_f64;
    let temperature = 37.0_f64;
    let light = 4.0_f64;
    let dt = 0.5_f64;
    let mut fields = vec![
        Array3::from_elem((2, 2, 2), pressure),
        Array3::from_elem((2, 2, 2), temperature),
        Array3::from_elem((2, 2, 2), light),
    ];

    coupler
        .couple_fields(&mut fields, dt)
        .expect("weak coupling fields have matching dimensions");

    let expected_light = light * (1.0 + 1.0e-12 * pressure * dt);
    let expected_optical_delta =
        10.0 * expected_light * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
    let acoustic_intensity = pressure * pressure / (DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE);
    let expected_acoustic_delta =
        0.5 * acoustic_intensity * dt / (DENSITY_WATER_NOMINAL * SPECIFIC_HEAT_WATER);
    let expected_temperature = temperature + expected_optical_delta + expected_acoustic_delta;

    assert_eq!(fields[0][[0, 0, 0]], pressure);
    assert!(
        (fields[2][[0, 0, 0]] - expected_light).abs() < 1e-12,
        "light {}, expected {}",
        fields[2][[0, 0, 0]],
        expected_light
    );
    assert!(
        (fields[1][[0, 0, 0]] - expected_temperature).abs() < 1e-12,
        "temperature {}, expected {}",
        fields[1][[0, 0, 0]],
        expected_temperature
    );
}

/// Coupling requires collocated pressure, thermal, and optical volumes.
/// # Panics
/// - Panics if mismatch rejection fails or if validation mutates any field.
///
#[test]
fn weak_coupling_rejects_mismatched_shapes_before_mutation() {
    let coupler = FieldCoupler::new(CouplingStrategy::Weak);
    let mut fields = vec![
        Array3::from_elem((2, 2, 2), 1.0),
        Array3::from_elem((2, 2, 2), 2.0),
        Array3::from_elem((1, 2, 2), 3.0),
    ];

    let err = coupler
        .couple_fields(&mut fields, 1.0)
        .expect_err("mismatched optical field shape must be rejected");

    match err {
        KwaversError::DimensionMismatch(message) => {
            assert!(message.contains("FieldCoupler edge"));
            assert!(message.contains("matching shapes"));
        }
        other => panic!("expected dimension mismatch, got {other:?}"),
    }
    assert_eq!(fields[0][[0, 0, 0]], 1.0);
    assert_eq!(fields[1][[0, 0, 0]], 2.0);
    assert_eq!(fields[2][[0, 0, 0]], 3.0);
}
