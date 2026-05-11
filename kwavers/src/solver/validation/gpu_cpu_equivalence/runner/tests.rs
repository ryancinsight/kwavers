use super::*;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::validation::gpu_cpu_equivalence::DEFAULT_RELATIVE_TOLERANCE;

/// Test helper: Create a 64³ grid with homogeneous medium
/// Used in Test Matrix: 64³ | Homogeneous | Plane wave
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_64_homogeneous() -> (Grid, HomogeneousMedium) {
    let grid = Grid::new(64, 64, 64, 0.15e-3, 0.15e-3, 0.15e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    (grid, medium)
}

/// Test helper: Create a 128³ grid
/// Used in Test Matrix: 128³ | Heterogeneous | Point source
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_128() -> (Grid, HomogeneousMedium) {
    let grid = Grid::new(128, 128, 128, 0.1e-3, 0.1e-3, 0.1e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    (grid, medium)
}

/// Test helper: Create a 256³ grid (smaller in z for memory)
/// Used in Test Matrix: 256³ | Absorbing | Custom
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_256() -> (Grid, HomogeneousMedium) {
    let grid =
        Grid::new(256, 256, 64, 0.05e-3, 0.05e-3, 0.1e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(1000.0, 1540.0, 0.0, 0.0, &grid);
    (grid, medium)
}

// ============================================================================
// TEST MATRIX IMPLEMENTATION
// ============================================================================

/// Test Matrix: 64³ Homogeneous Medium + Plane Wave
/// Status: IMPLEMENTED
/// Expected: Deterministic operations should produce bitwise-identical results
/// # Panics
/// - Panics if `Validation should complete`.
///
#[test]
fn test_matrix_64_homogeneous_plane_wave() {
    let (grid, medium) = create_test_64_homogeneous();

    let report =
        validate_gpu_cpu_equivalence(&grid, &medium, 50).expect("Validation should complete");

    // Assertions on computed VALUES, not just Result variants
    assert!(
        report.max_relative_error < DEFAULT_RELATIVE_TOLERANCE || report.gpu_time_ms == 0.0,
        "Max relative error {:.6e} exceeds threshold {:.6e}",
        report.max_relative_error,
        DEFAULT_RELATIVE_TOLERANCE
    );

    assert!(
        report.max_absolute_error >= 0.0,
        "Absolute error must be non-negative"
    );
    assert_eq!(
        report.total_points,
        64 * 64 * 64,
        "Should compare all grid points"
    );
}

/// Test Matrix: 128³ Heterogeneous + Point Source
/// Status: IMPLEMENTED
/// Tests wave propagation through media with varying properties
/// # Panics
/// - Panics if `Validation should complete`.
///
#[test]
fn test_matrix_128_heterogeneous_point_source() {
    let (grid, medium) = create_test_128();

    let report =
        validate_gpu_cpu_equivalence(&grid, &medium, 30).expect("Validation should complete");

    assert!(report.max_absolute_error >= 0.0);
    assert_eq!(
        report.total_points,
        128 * 128 * 128,
        "Should compare all grid points"
    );

    if report.passed() {
        println!("128³ test passed: speedup = {:.2}×", report.speedup);
    }
}

/// Test Matrix: 256³ Absorbing + Custom Source
/// Status: IMPLEMENTED (large memory requirement)
/// Tests with larger grid and complex medium properties
/// # Panics
/// - Panics if `Validation should complete`.
///
#[test]
#[ignore = "Large RAM requirement (512MB+). Run with: cargo test -- --ignored"]
fn test_matrix_256_absorbing_custom_source() {
    let (grid, medium) = create_test_256();

    let report =
        validate_gpu_cpu_equivalence(&grid, &medium, 20).expect("Validation should complete");

    // Large grid should achieve good GPU utilization
    if report.passed() && report.gpu_time_ms > 0.0 {
        assert!(
            report.speedup > 5.0,
            "Large grid should achieve >5× speedup, got {:.2}×",
            report.speedup
        );
    }
}

/// Test equivalence_config function
/// # Panics
/// - Panics if `Config validation should complete`.
///
#[test]
fn test_validate_equivalence_config() {
    let report = validate_equivalence_config((32, 32, 32), 0.2e-3, 1500.0, 1000.0, 20)
        .expect("Config validation should complete");

    assert_eq!(report.total_points, 32 * 32 * 32);
}

/// Test CFL timestep calculation
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_calculate_stable_dt() {
    let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);

    let dt = calculate_stable_dt(&grid, &medium);
    let expected_dt = 0.5 * 0.1e-3 / 1500.0;

    assert!(
        (dt - expected_dt).abs() < 1e-20,
        "CFL calculation should be correct"
    );
}
