use super::{CouplingStrategy, FieldCoupler};
use crate::core::constants::{
    fundamental::DENSITY_WATER_NOMINAL, thermodynamic::SPECIFIC_HEAT_WATER,
};
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
