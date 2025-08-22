//! Comparison tests between FDTD and PSTD solvers
//!
//! These tests verify that both solvers produce similar results for simple cases
//! and compare against analytical solutions where available.

use kwavers::{
    Grid, HomogeneousMedium, PluginManager,
    FdtdConfig, FdtdPlugin, PstdConfig, PstdPlugin
};
use ndarray::{Array3, Array4};

/// Test 1D plane wave propagation - both solvers should give similar results
#[test]
fn test_plane_wave_propagation() {
    // Create a simple 1D-like grid
    let grid = Grid::new(128, 16, 16, 0.5e-3, 0.5e-3, 0.5e-3);

    // Homogeneous medium (water)
    let medium = HomogeneousMedium::water(&grid);

    // Initial Gaussian pulse
    let sigma = 10.0 * grid.dx;
    let center = grid.nx as f64 * grid.dx / 2.0;
    let mut initial_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));

    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        let value = ((-(x - center).powi(2) / (2.0 * sigma.powi(2))).exp()) * 1e6;
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                initial_pressure[[i, j, k]] = value;
            }
        }
    }

    // Run with FDTD
    let fdtd_result = run_fdtd_simulation(&grid, &medium, &initial_pressure);

    // Run with PSTD
    let pstd_result = run_pstd_simulation(&grid, &medium, &initial_pressure);

    // Compare results - they should be similar
    let max_diff = (&fdtd_result - &pstd_result)
        .mapv(f64::abs)
        .fold(0.0f64, |a, &b| a.max(b));
    let max_val = fdtd_result
        .mapv(f64::abs)
        .fold(0.0f64, |a, &b| a.max(b))
        .max(pstd_result.mapv(f64::abs).fold(0.0f64, |a, &b| a.max(b)));

    // Allow 5% relative difference due to numerical dispersion
    assert!(
        max_diff / max_val < 0.05,
        "FDTD and PSTD results differ by more than 5%: {:.2}%",
        100.0 * max_diff / max_val
    );
}

/// Test against analytical solution for standing wave
#[test]
fn test_standing_wave_analytical() {
    // Create a 1D grid with reflecting boundaries
    let grid = Grid::new(100, 16, 16, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::water(&grid);

    // Wavelength and frequency
    let wavelength = 20.0 * grid.dx;
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let c = 1500.0; // sound speed in water
    let omega = k * c;

    // Initial condition: sin(kx)
    let mut initial_pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        initial_pressure[[i, 8, 8]] = (k * x).sin() * 1e6;
    }

    // Time to simulate - quarter period
    let period = 2.0 * std::f64::consts::PI / omega;
    let t_end = period / 4.0;

    // Run simulations
    let fdtd_result = run_fdtd_simulation_timed(&grid, &medium, &initial_pressure, t_end);
    let pstd_result = run_pstd_simulation_timed(&grid, &medium, &initial_pressure, t_end);

    // Analytical solution at t = T/4 should be cos(kx)
    let mut analytical = Array3::zeros((grid.nx, 16, 16));
    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        analytical[[i, 8, 8]] = (k * x).cos() * 1e6;
    }

    // Check FDTD accuracy
    let fdtd_error = calculate_relative_error(&fdtd_result, &analytical);
    assert!(
        fdtd_error < 0.02,
        "FDTD error too large: {:.2}%",
        100.0 * fdtd_error
    );

    // Check PSTD accuracy - should be more accurate
    let pstd_error = calculate_relative_error(&pstd_result, &analytical);
    assert!(
        pstd_error < 0.01,
        "PSTD error too large: {:.2}%",
        100.0 * pstd_error
    );
    assert!(
        pstd_error < fdtd_error,
        "PSTD should be more accurate than FDTD"
    );
}

/// Helper function to run FDTD simulation
fn run_fdtd_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> Array3<f64> {
    run_fdtd_simulation_timed(grid, medium, initial_pressure, 50e-6)
}

fn run_fdtd_simulation_timed(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> Array3<f64> {
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: 0.95,
        subgridding: false,
        subgrid_factor: 2,
    };

    let cfl_factor = config.cfl_factor;
    let mut plugin_manager = PluginManager::new();
    plugin_manager
        .add_plugin(Box::new(FdtdPlugin::new(config, grid).unwrap()))
        .unwrap();

    // Initialize fields - must match UnifiedFieldType::COUNT
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    // Time stepping
    let c = 1500.0; // sound speed
    let dt = cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c;
    let n_steps = (t_end / dt).ceil() as usize;

    plugin_manager.initialize(grid, medium).unwrap();

    for _step in 0..n_steps {
        let t = _step as f64 * dt;
        plugin_manager
            .execute(&mut fields, grid, medium, dt, t)
            .unwrap();
    }

    fields.slice(ndarray::s![0, .., .., ..]).to_owned()
}

/// Helper function to run PSTD simulation
fn run_pstd_simulation(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
) -> Array3<f64> {
    run_pstd_simulation_timed(grid, medium, initial_pressure, 50e-6)
}

fn run_pstd_simulation_timed(
    grid: &Grid,
    medium: &HomogeneousMedium,
    initial_pressure: &Array3<f64>,
    t_end: f64,
) -> Array3<f64> {
    let config = PstdConfig {
        use_kspace_correction: true,
        correction_method: kwavers::solver::pstd::CorrectionMethod::Sinc,
        use_antialiasing: true,
        use_absorption: false,
        absorption_alpha: 0.0,
        absorption_y: 1.0,
        cfl_factor: 0.3,
        max_steps: 1000,
        dispersive_media: false,
        pml_layers: 4,
        pml_alpha: 0.0,
    };

    let cfl_factor = config.cfl_factor;
    let mut plugin_manager = PluginManager::new();
    plugin_manager
        .add_plugin(Box::new(PstdPlugin::new(config, grid).unwrap()))
        .unwrap();

    // Initialize fields - must match UnifiedFieldType::COUNT
    let mut fields = Array4::zeros((17, grid.nx, grid.ny, grid.nz));
    fields
        .slice_mut(ndarray::s![0, .., .., ..])
        .assign(initial_pressure);

    // Time stepping
    let c = 1500.0; // sound speed
    let dt = cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c;
    let n_steps = (t_end / dt).ceil() as usize;

    plugin_manager.initialize(grid, medium).unwrap();

    for _step in 0..n_steps {
        let t = _step as f64 * dt;
        plugin_manager
            .execute(&mut fields, grid, medium, dt, t)
            .unwrap();
    }

    fields.slice(ndarray::s![0, .., .., ..]).to_owned()
}

/// Calculate relative error between two fields
fn calculate_relative_error(computed: &Array3<f64>, analytical: &Array3<f64>) -> f64 {
    let diff = (computed - analytical).mapv(f64::abs);
    let max_diff = diff.fold(0.0f64, |a, &b| a.max(b));
    let max_analytical = analytical.mapv(f64::abs).fold(0.0f64, |a, &b| a.max(b));

    max_diff / max_analytical
}

/// Test numerical dispersion characteristics
#[test]
fn test_dispersion_characteristics() {
    // Create a longer 1D grid to observe dispersion
    let grid = Grid::new(256, 16, 16, 0.5e-3, 0.5e-3, 0.5e-3);
    let medium = HomogeneousMedium::water(&grid);

    // Create a narrow Gaussian pulse (high frequency content)
    let sigma = 2.0 * grid.dx;
    let center = 50.0 * grid.dx;
    let mut initial_pressure = Array3::zeros((grid.nx, 16, 16));

    for i in 0..grid.nx {
        let x = i as f64 * grid.dx;
        initial_pressure[[i, 8, 8]] = ((-(x - center).powi(2) / (2.0 * sigma.powi(2))).exp()) * 1e6;
    }

    // Run both simulations for longer time
    let t_end = 100e-6;
    let fdtd_result = run_fdtd_simulation_timed(&grid, &medium, &initial_pressure, t_end);
    let pstd_result = run_pstd_simulation_timed(&grid, &medium, &initial_pressure, t_end);

    // Find peak positions
    let fdtd_peak = find_peak_position(&fdtd_result);
    let pstd_peak = find_peak_position(&pstd_result);

    // Expected position (assuming no dispersion)
    let c = 1500.0; // sound speed
    let expected_position = center + c * t_end;
    let expected_index = (expected_position / grid.dx).round() as usize;

    // PSTD should have less dispersion error
    let fdtd_error = usize::abs_diff(fdtd_peak, expected_index);
    let pstd_error = usize::abs_diff(pstd_peak, expected_index);

    assert!(
        pstd_error <= fdtd_error,
        "PSTD should have less dispersion than FDTD. PSTD error: {}, FDTD error: {}",
        pstd_error,
        fdtd_error
    );

    // Both should be reasonably accurate
    assert!(
        fdtd_error < 5,
        "FDTD dispersion error too large: {} grid points",
        fdtd_error
    );
    assert!(
        pstd_error < 2,
        "PSTD dispersion error too large: {} grid points",
        pstd_error
    );
}

/// Find the position of the peak in a 1D field
fn find_peak_position(field: &Array3<f64>) -> usize {
    let mut max_val = 0.0;
    let mut max_idx = 0;

    for i in 0..field.shape()[0] {
        let val = field[[i, 8, 8]].abs();
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx
}