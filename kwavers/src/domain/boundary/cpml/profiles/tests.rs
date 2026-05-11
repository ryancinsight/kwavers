use super::*;
use crate::domain::boundary::cpml::config::CPMLConfig;
use crate::domain::grid::Grid;

/// The k-Wave wall value is `sigma_max = pml_alpha * c0 / dx`.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_cpml_sigma_max_formula() {
    let c0 = 1500.0_f64;
    let dx = 1e-3_f64;
    let pml_size = 10_usize;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;

    let grid = Grid::new(32, 32, 32, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    let expected_sigma_max = pml_alpha * (c0 / dx);
    let actual = profiles.sigma_x[0];
    assert!(
        (actual - expected_sigma_max).abs() / expected_sigma_max < 0.01,
        "sigma_max = {actual:.1} should match k-Wave formula {expected_sigma_max:.1}"
    );
}

/// Singleton axes must remain CPML-neutral for lower-dimensional embeddings.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_singleton_axis_profiles_are_neutral() {
    let c0 = 1500.0_f64;
    let dx = 1e-3_f64;
    let dt = 1e-7_f64;

    let grid = Grid::new(32, 32, 1, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(10).with_alpha(2.0);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    assert!(profiles.sigma_x.iter().any(|&v| v > 0.0));
    assert!(profiles.sigma_y.iter().any(|&v| v > 0.0));
    assert!(profiles.sigma_z.iter().all(|&v| v == 0.0));
    assert!(profiles.sigma_z_sgz.iter().all(|&v| v == 0.0));
    assert!(profiles.kappa_z.iter().all(|&v| v == 1.0));
    assert!(profiles.alpha_z.iter().all(|&v| v == 0.0));
    assert!(profiles.a_z.iter().all(|&v| v == 0.0));
    assert!(profiles.b_z.iter().all(|&v| v == 1.0));
}

/// Roden-Gedney coefficients reduce to `b = exp(-sigma dt)` and `a = b - 1`.
/// # Panics
/// - Panics if `grid`.
/// - Panics if `CPMLProfiles::new should succeed`.
///
#[test]
fn test_cpml_recursive_convolution_coefficients() {
    let c0 = 1500.0_f64;
    let dx = 1e-3_f64;
    let pml_size = 10_usize;
    let dt = 1e-7_f64;
    let pml_alpha = 2.0_f64;

    let nx = 32;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("grid");
    let config = CPMLConfig::with_thickness(pml_size).with_alpha(pml_alpha);

    let profiles =
        CPMLProfiles::new(&config, &grid, c0, dt).expect("CPMLProfiles::new should succeed");

    let sigma_max = profiles.sigma_x[0];
    let expected_b = (-sigma_max * dt).exp();
    let expected_a = expected_b - 1.0;

    assert!(
        (profiles.b_x[0] - expected_b).abs() < 1e-12,
        "b_x at PML wall: expected {expected_b:.10}, got {:.10}",
        profiles.b_x[0]
    );
    assert!(
        (profiles.a_x[0] - expected_a).abs() < 1e-12,
        "a_x at PML wall: expected {expected_a:.10}, got {:.10}",
        profiles.a_x[0]
    );

    let mid = nx / 2;
    assert!(
        (profiles.b_x[mid] - 1.0).abs() < 1e-14,
        "b_x at interior must be 1.0, got {}",
        profiles.b_x[mid]
    );
    assert!(
        profiles.a_x[mid].abs() < 1e-14,
        "a_x at interior must be 0.0, got {}",
        profiles.a_x[mid]
    );
}
