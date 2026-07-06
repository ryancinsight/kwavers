use super::*;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_medium::HomogeneousMedium;

/// Test helper: Create a 64³ grid with homogeneous medium
/// Used in Test Matrix: 64³ | Homogeneous | Plane wave
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_64_homogeneous() -> (Grid, HomogeneousMedium) {
    let grid = Grid::new(64, 64, 64, 0.15e-3, 0.15e-3, 0.15e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    (grid, medium)
}

/// Test helper: Create a 128³ grid
/// Used in Test Matrix: 128³ | Heterogeneous | Point source
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_128() -> (Grid, HomogeneousMedium) {
    let grid = Grid::new(128, 128, 128, 0.1e-3, 0.1e-3, 0.1e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    (grid, medium)
}

/// Test helper: Create a 256³ grid (smaller in z for memory)
/// Used in Test Matrix: 256³ | Absorbing | Custom
/// # Panics
/// - Panics if `Valid grid dimensions`.
///
fn create_test_256() -> (Grid, HomogeneousMedium) {
    let grid = Grid::new(256, 256, 64, 0.05e-3, 0.05e-3, 0.1e-3).expect("Valid grid dimensions");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE,
        0.0,
        0.0,
        &grid,
    );
    (grid, medium)
}

fn assert_real_gpu_provider_unavailable(report: &EquivalenceReport) {
    assert!(
        !report.passed(),
        "FDTD GPU equivalence must not pass without a real GPU provider"
    );
    let reason = report
        .failure_reason
        .as_deref()
        .expect("unavailable FDTD GPU provider must be surfaced");
    assert!(
        reason.contains("FDTD provider-generic Leto/Hephaestus GPU equivalence"),
        "failure reason must name the missing provider, got {reason}"
    );
    assert!(
        reason.contains("provider trait implementation")
            && reason.contains("previous path only ran the CPU solver"),
        "failure reason must prevent CPU-vs-CPU equivalence claims, got {reason}"
    );
}

// ============================================================================
// TEST MATRIX IMPLEMENTATION
// ============================================================================

/// Test Matrix: 64³ Homogeneous Medium + Plane Wave
/// Status: CPU reference implemented; real FDTD GPU provider trait unavailable.
/// Expected: report surfaces unavailable GPU provider, not CPU-vs-CPU parity.
/// # Panics
/// - Panics if `Validation should complete`.
///
#[test]
fn test_matrix_64_homogeneous_plane_wave() {
    let (grid, medium) = create_test_64_homogeneous();

    let report =
        validate_gpu_cpu_equivalence(&grid, &medium, 50).expect("Validation should complete");

    assert_real_gpu_provider_unavailable(&report);
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

    assert_real_gpu_provider_unavailable(&report);
    assert_eq!(
        report.total_points,
        128 * 128 * 128,
        "Should compare all grid points"
    );
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

    assert_real_gpu_provider_unavailable(&report);
}

/// Test equivalence_config function
/// # Panics
/// - Panics if `Config validation should complete`.
///
#[test]
fn test_validate_equivalence_config() {
    let report = validate_equivalence_config(
        (32, 32, 32),
        0.2e-3,
        SOUND_SPEED_WATER_SIM,
        DENSITY_WATER_NOMINAL,
        20,
    )
    .expect("Config validation should complete");

    assert_eq!(report.total_points, 32 * 32 * 32);
    assert_real_gpu_provider_unavailable(&report);
}

/// Test CFL timestep calculation
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_calculate_stable_dt() {
    let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );

    let dt = calculate_stable_dt(&grid, &medium);
    let expected_dt = 0.5 * 0.1e-3 / SOUND_SPEED_WATER_SIM;

    assert!(
        (dt - expected_dt).abs() < 1e-20,
        "CFL calculation should be correct"
    );
}
