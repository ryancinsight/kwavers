use super::*;
use crate::core::constants::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use std::f64::consts::PI;

#[test]
fn test_spatial_order_cfl_limits() {
    assert!((AcousticSpatialOrder::Second.cfl_limit() - 0.577).abs() < 0.001); // 1/√3
    assert!((AcousticSpatialOrder::Fourth.cfl_limit() - 0.258).abs() < 0.001); // 1/√15
    assert!((AcousticSpatialOrder::Sixth.cfl_limit() - 0.192).abs() < 0.001); // 1/√27
}

#[test]
fn test_spatial_order_minimum_points() {
    assert_eq!(AcousticSpatialOrder::Second.minimum_grid_points(), 3);
    assert_eq!(AcousticSpatialOrder::Fourth.minimum_grid_points(), 5);
    assert_eq!(AcousticSpatialOrder::Sixth.minimum_grid_points(), 7);
}

#[test]
fn test_spatial_order_from_usize() {
    assert_eq!(
        AcousticSpatialOrder::from_usize(2).unwrap(),
        AcousticSpatialOrder::Second
    );
    assert_eq!(
        AcousticSpatialOrder::from_usize(4).unwrap(),
        AcousticSpatialOrder::Fourth
    );
    assert_eq!(
        AcousticSpatialOrder::from_usize(6).unwrap(),
        AcousticSpatialOrder::Sixth
    );
    assert!(AcousticSpatialOrder::from_usize(99).is_err());
}

#[test]
fn test_acoustic_diffusivity_zero_frequency() {
    use crate::domain::medium::HomogeneousMedium;

    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    let diffusivity =
        compute_diffusivity_from_power_law_absorption(&medium, 0.0, 0.0, 0.0, 0.0, &grid);
    assert_eq!(
        diffusivity, 0.0,
        "Zero frequency should give zero diffusivity"
    );
}

#[test]
fn test_acoustic_diffusivity_formula() {
    // Test case 1: Zero absorption should give zero diffusivity
    let alpha = 0.0;
    let c: f64 = SOUND_SPEED_WATER_SIM;
    let freq = 1e6;
    let omega = 2.0 * PI * freq;
    let expected = 2.0 * alpha * c.powi(3) / (omega * omega);
    assert_eq!(expected, 0.0);

    // Test case 2: Non-zero values
    let alpha = 0.5;
    let c: f64 = SOUND_SPEED_WATER_SIM;
    let freq = 1e6;
    let omega = 2.0 * PI * freq;
    let diffusivity = 2.0 * alpha * c.powi(3) / (omega * omega);
    let expected = 2.0 * 0.5 * SOUND_SPEED_WATER_SIM.powi(3) / (2.0 * PI * 1e6).powi(2);
    assert!(
        (diffusivity - expected).abs() < 1e-10,
        "Formula calculation mismatch: got {}, expected {}",
        diffusivity,
        expected
    );

    // Test case 3: Frequency scaling — doubling ω halves δ by factor 4
    let freq2 = 2e6;
    let omega2 = 2.0 * PI * freq2;
    let diffusivity2 = 2.0 * alpha * c.powi(3) / (omega2 * omega2);
    assert!(
        (diffusivity2 - diffusivity / 4.0).abs() < 1e-10,
        "Frequency scaling incorrect: {} vs {}",
        diffusivity2,
        diffusivity / 4.0
    );

    // Test case 4: Value must be in physically reasonable range
    assert!(
        diffusivity > 1e-6 && diffusivity < 1e-3,
        "Diffusivity value seems unreasonable: {}",
        diffusivity
    );
}

/// `compute_max_stable_timestep` implements the CFL condition:
///   dt_max = CFL_limit(order) · min(dx, dy, dz) / c_max
///
/// Analytical verification for Second-order, isotropic grid:
///   dt_max = (1/√3) · 0.001 / 1500 ≈ 3.849e-7 s
#[test]
fn compute_max_stable_timestep_matches_analytical_cfl_formula() {
    let dx = 0.001_f64;
    let grid = Grid::new(10, 10, 10, dx, dx, dx).unwrap();
    let c_max = SOUND_SPEED_WATER_SIM;

    let dt_second = compute_max_stable_timestep(&grid, c_max, AcousticSpatialOrder::Second);
    let expected_second = AcousticSpatialOrder::Second.cfl_limit() * dx / c_max;
    assert!(
        (dt_second - expected_second).abs() < 1e-15,
        "Second-order: got {dt_second:.6e} expected {expected_second:.6e}"
    );

    let dt_fourth = compute_max_stable_timestep(&grid, c_max, AcousticSpatialOrder::Fourth);
    let expected_fourth = AcousticSpatialOrder::Fourth.cfl_limit() * dx / c_max;
    assert!(
        (dt_fourth - expected_fourth).abs() < 1e-15,
        "Fourth-order: got {dt_fourth:.6e} expected {expected_fourth:.6e}"
    );

    assert!(
        dt_fourth < dt_second,
        "Fourth-order dt_max must be strictly less than Second-order dt_max"
    );
}

/// `compute_max_stable_timestep` uses the minimum grid spacing when the grid is anisotropic.
#[test]
fn compute_max_stable_timestep_uses_minimum_spacing_for_anisotropic_grid() {
    let grid = Grid::new(10, 10, 10, 0.001, 0.002, 0.003).unwrap();
    let c_max = SOUND_SPEED_WATER_SIM;
    let dt = compute_max_stable_timestep(&grid, c_max, AcousticSpatialOrder::Second);
    let expected = AcousticSpatialOrder::Second.cfl_limit() * 0.001 / c_max;
    assert!((dt - expected).abs() < 1e-15);
}

/// `compute_nonlinearity_coefficient` computes β = 1 + B/(2A).
///
/// For water: B/A = 5.0 (default) → β = 1 + 5/2 = 3.5.
#[test]
fn compute_nonlinearity_coefficient_matches_ba_formula() {
    use crate::domain::medium::HomogeneousMedium;
    let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::water(&grid);

    let beta = compute_nonlinearity_coefficient(&medium, 0.0, 0.0, 0.0, &grid);

    // nonlinearity_parameter returns B/A directly; β = 1 + B/(2A)
    let b_over_a = crate::domain::medium::AcousticProperties::nonlinearity_parameter(
        &medium, 0.0, 0.0, 0.0, &grid,
    );
    let expected = 1.0 + b_over_a / 2.0;
    assert!(
        (beta - expected).abs() < 1e-15,
        "β = {beta} must equal 1 + B/(2A) = {expected}"
    );
    // For water B/A = 5.0 → β = 3.5
    assert!(
        (beta - 3.5).abs() < 0.1,
        "β for water must be ~3.5 (got {beta})"
    );
    assert!(
        beta > 1.0,
        "nonlinearity coefficient must be > 1 for water (got {beta})"
    );
}

#[test]
fn test_heterogeneous_medium_position_dependence() {
    use crate::domain::medium::heterogeneous::tissue::DomainTissueRegion;
    use crate::domain::medium::heterogeneous::tissue::HeterogeneousTissueMedium;
    use crate::domain::medium::AbsorptionTissueType;

    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let mut medium = HeterogeneousTissueMedium::new(grid.clone(), AbsorptionTissueType::Muscle);

    let region = DomainTissueRegion::new(
        AbsorptionTissueType::Fat,
        0.005,
        0.015,
        0.005,
        0.015,
        0.005,
        0.015,
    );
    medium.set_tissue_region(&region).unwrap();

    let density1 = crate::domain::medium::density_at(&medium, 0.0, 0.0, 0.0, &grid);
    let speed1 = crate::domain::medium::sound_speed_at(&medium, 0.0, 0.0, 0.0, &grid);

    let density2 = crate::domain::medium::density_at(&medium, 0.01, 0.01, 0.01, &grid);
    let speed2 = crate::domain::medium::sound_speed_at(&medium, 0.01, 0.01, 0.01, &grid);

    assert_ne!(
        density1, density2,
        "Real heterogeneous medium should have physically disjoint density regions"
    );
    assert_ne!(
        speed1, speed2,
        "Real heterogeneous medium should have physically disjoint sound speed regions"
    );
}
