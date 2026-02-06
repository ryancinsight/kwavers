//! Integration Tests for Photoacoustic Imaging Module
//!
//! This module contains comprehensive integration tests for the photoacoustic imaging
//! simulator, covering the complete simulation pipeline from optical fluence computation
//! to image reconstruction.
//!
//! ## Test Coverage
//!
//! - **Simulator Creation**: Initialization and configuration
//! - **Optical Computation**: Fluence distribution and wavelength dependence
//! - **Acoustic Computation**: Initial pressure generation and wave propagation
//! - **Reconstruction**: Universal back-projection and time-reversal algorithms
//! - **Validation**: Analytical comparison and physical correctness
//! - **Integration**: End-to-end simulation pipeline
//!
//! ## Test Strategy
//!
//! Tests follow the Test-Driven Development (TDD) principles:
//! 1. Red: Specify expected behavior
//! 2. Green: Implement minimal correct solution
//! 3. Refactor: Improve design while maintaining correctness
//!
//! Each test validates:
//! - Mathematical correctness (against analytical solutions)
//! - Physical validity (non-negative pressure, energy conservation, etc.)
//! - Numerical stability (finite values, no NaN/Inf)
//! - API contracts (dimensions, return types)

use super::core::PhotoacousticSimulator;
use super::reconstruction;
use crate::domain::grid::Grid;
use crate::domain::imaging::photoacoustic::PhotoacousticOpticalProperties;
use crate::domain::medium::homogeneous::HomogeneousMedium;
use approx::assert_relative_eq;
use ndarray::Array3;

#[test]
fn test_photoacoustic_creation() {
    let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium);
    assert!(simulator.is_ok());
}

#[test]
fn test_fluence_computation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence = simulator.compute_fluence();
    assert!(fluence.is_ok());

    let fluence_data = fluence.unwrap();
    assert_eq!(fluence_data.dim(), (16, 16, 8));

    // Check that fluence decreases with depth (fundamental physical property)
    let surface_fluence = fluence_data[[8, 8, 0]];
    let deep_fluence = fluence_data[[8, 8, 7]];
    assert!(
        surface_fluence > deep_fluence,
        "Fluence should decrease with depth due to absorption and scattering"
    );

    // Validate physical correctness
    for &val in fluence_data.iter() {
        assert!(val >= 0.0, "Fluence must be non-negative");
        assert!(val.is_finite(), "Fluence must be finite");
    }
}

#[test]
fn test_initial_pressure_computation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence = simulator.compute_fluence().unwrap();
    let initial_pressure = simulator.compute_initial_pressure(&fluence);

    assert!(initial_pressure.is_ok());
    let pressure_data = initial_pressure.unwrap();
    assert_eq!(pressure_data.pressure.dim(), (16, 16, 8));
    assert!(pressure_data.max_pressure > 0.0);

    // Validate physical correctness
    for &val in pressure_data.pressure.iter() {
        assert!(val >= 0.0, "Pressure must be non-negative");
        assert!(val.is_finite(), "Pressure must be finite");
    }
}

#[test]
fn test_simulation() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let mut simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence = simulator.compute_fluence().unwrap();
    let initial_pressure = simulator.compute_initial_pressure(&fluence).unwrap();
    let result = simulator.simulate(&initial_pressure);

    assert!(result.is_ok());
    let sim_result = result.unwrap();

    // Ensure time sampling and pressure frames are consistent
    assert_eq!(sim_result.pressure_fields.len(), sim_result.time.len());
    assert!(
        sim_result.pressure_fields.len() >= 2,
        "Should have multiple time snapshots"
    );
    assert_eq!(sim_result.reconstructed_image.dim(), (16, 16, 8));
    assert!(sim_result.snr > 0.0);

    // Validate physical correctness of pressure fields
    for field in &sim_result.pressure_fields {
        assert_eq!(field.dim(), (16, 16, 8));
        for &val in field.iter() {
            assert!(val.is_finite(), "Pressure field values must be finite");
        }
    }

    // Validate reconstructed image
    for &val in sim_result.reconstructed_image.iter() {
        assert!(val.is_finite(), "Reconstructed image values must be finite");
    }
}

#[test]
fn test_optical_properties() {
    let blood_props = PhotoacousticOpticalProperties::blood(750.0);
    let tissue_props = PhotoacousticOpticalProperties::soft_tissue(750.0);
    let tumor_props = PhotoacousticOpticalProperties::tumor(750.0);

    // Blood should have higher absorption than soft tissue (hemoglobin absorption)
    assert!(
        blood_props.absorption_coefficient > tissue_props.absorption_coefficient,
        "Blood has higher absorption due to hemoglobin"
    );

    // Tumor should have higher absorption than normal tissue (angiogenesis)
    assert!(
        tumor_props.absorption_coefficient > tissue_props.absorption_coefficient,
        "Tumors have higher absorption due to increased vascularity"
    );

    // All properties should be physically valid
    assert!(blood_props.absorption_coefficient > 0.0);
    assert!(tissue_props.absorption_coefficient > 0.0);
    assert!(tumor_props.absorption_coefficient > 0.0);
}

#[test]
fn test_analytical_validation() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let error = simulator.validate_analytical();
    assert!(error.is_ok());
    assert!(
        error.unwrap() < 1.0,
        "Relative error should be reasonable (< 100%)"
    );
}

#[test]
fn test_universal_back_projection_algorithm() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();

    // Create synthetic pressure fields (spherical wave from point source)
    // Increase time duration to allow wave to reach detectors
    // Detectors are at ~3.2 * dx = 0.0032m
    // Travel time = 0.0032 / 1500 ≈ 2.13e-6 s
    let n_time = 40;
    let dt = 1e-7;
    let mut pressure_fields = Vec::with_capacity(n_time);
    let time_points: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();

    // Point source at center of grid
    let source_x = 8.0;
    let source_y = 8.0;
    let source_z = 4.0;

    for &time in time_points.iter() {
        let mut field = Array3::<f64>::zeros((16, 16, 8));

        // Generate spherical wave from point source
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let dx = (i as f64 - source_x) * grid.dx;
                    let dy = (j as f64 - source_y) * grid.dy;
                    let dz = (k as f64 - source_z) * grid.dz;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                    let travel_time = distance / parameters.speed_of_sound;

                    // Gaussian pulse: exp(-((t - t0)/width)^2)
                    // Width parameter for 5MHz signal ~ 2e-7s
                    let width = 2e-7;
                    let arg = (time - travel_time) / width;
                    let temporal = (-arg * arg).exp();

                    let amplitude = 1.0 / (distance.max(1e-6));
                    field[[i, j, k]] = amplitude * temporal;
                }
            }
        }
        pressure_fields.push(field);
    }

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    // Perform universal back-projection reconstruction
    let reconstructed = simulator
        .time_reversal_reconstruction(&pressure_fields, &time_points)
        .unwrap();

    // Validate reconstruction quality
    assert_eq!(reconstructed.dim(), (16, 16, 8));

    // Validate that the reconstruction produces reasonable output
    // The universal back-projection algorithm should produce a non-uniform image
    // with some regions having higher intensity than others

    let mut max_intensity = f64::NEG_INFINITY;
    let mut min_intensity = f64::INFINITY;

    for &val in reconstructed.iter() {
        max_intensity = max_intensity.max(val);
        min_intensity = min_intensity.min(val);
    }

    // The image should have some variation (not be uniform)
    assert!(
        max_intensity > min_intensity,
        "Reconstructed image should not be uniform"
    );

    // All values should be finite
    assert!(
        max_intensity.is_finite(),
        "Maximum intensity should be finite"
    );
    assert!(
        min_intensity.is_finite(),
        "Minimum intensity should be finite"
    );

    // Maximum intensity should be positive (due to back-projection weighting)
    assert!(max_intensity > 0.0, "Maximum intensity should be positive");
}

#[test]
fn test_detector_interpolation_accuracy() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();

    // Create a test field with known values
    let mut field = Array3::<f64>::zeros((8, 8, 4));

    // Set specific values at grid points
    field[[2, 2, 1]] = 1.0;
    field[[3, 2, 1]] = 2.0;
    field[[2, 3, 1]] = 3.0;
    field[[3, 3, 1]] = 4.0;
    field[[2, 2, 2]] = 5.0;
    field[[3, 2, 2]] = 6.0;
    field[[2, 3, 2]] = 7.0;
    field[[3, 3, 2]] = 8.0;

    // Test interpolation at exact grid points
    let value_2_2_1 = reconstruction::interpolate_detector_signal(&grid, &field, 2.0, 2.0, 1.0);
    assert_relative_eq!(value_2_2_1, 1.0, epsilon = 1e-10);

    let value_3_3_2 = reconstruction::interpolate_detector_signal(&grid, &field, 3.0, 3.0, 2.0);
    assert_relative_eq!(value_3_3_2, 8.0, epsilon = 1e-10);

    // Test interpolation at midpoint (2.5, 2.5, 1.5)
    // Should be average of the 8 surrounding points
    let expected_mid = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
    let value_mid = reconstruction::interpolate_detector_signal(&grid, &field, 2.5, 2.5, 1.5);
    assert_relative_eq!(value_mid, expected_mid, epsilon = 1e-10);

    // Test boundary clamping
    let value_outside =
        reconstruction::interpolate_detector_signal(&grid, &field, -1.0, -1.0, -1.0);
    assert_eq!(value_outside, field[[0, 0, 0]]);

    let value_beyond = reconstruction::interpolate_detector_signal(&grid, &field, 10.0, 10.0, 10.0);
    assert_eq!(value_beyond, field[[7, 7, 3]]);
}

#[test]
fn test_spherical_spreading_correction() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    // Create a single pressure field with constant value
    let mut pressure_fields = vec![Array3::<f64>::zeros((16, 16, 8))];
    pressure_fields[0].fill(1.0); // Constant pressure field
    let time_points = vec![0.0];

    // Perform reconstruction
    let reconstructed = simulator
        .time_reversal_reconstruction(&pressure_fields, &time_points)
        .unwrap();

    // Check that reconstruction is not uniform (due to spherical spreading correction)
    let center_value = reconstructed[[8, 8, 4]];
    let edge_value = reconstructed[[0, 0, 0]];

    // Edge should have different value due to distance weighting
    assert_ne!(
        center_value, edge_value,
        "Reconstruction should have spatial variation due to 1/r weighting"
    );

    // All values should be finite and reasonable
    for &val in reconstructed.iter() {
        assert!(val.is_finite(), "Reconstructed values must be finite");
        assert!(
            val >= 0.0,
            "Reconstructed values should be non-negative due to 1/r weighting"
        );
    }
}

#[test]
fn test_multi_wavelength_fluence() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters {
        wavelengths: vec![700.0, 750.0, 800.0],
        ..Default::default()
    };

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let fluence_fields = simulator.compute_multi_wavelength_fluence();
    assert!(fluence_fields.is_ok());

    let fields = fluence_fields.unwrap();
    assert_eq!(
        fields.len(),
        3,
        "Should compute fluence for all wavelengths"
    );

    for field in &fields {
        assert_eq!(field.dim(), (8, 8, 4));
        for &val in field.iter() {
            assert!(val >= 0.0, "Fluence must be non-negative");
            assert!(val.is_finite(), "Fluence must be finite");
        }
    }
}

#[test]
fn test_multi_wavelength_simulation() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters {
        wavelengths: vec![700.0, 800.0],
        ..Default::default()
    };

    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let results = simulator.simulate_multi_wavelength();
    assert!(results.is_ok());

    let multi_results = results.unwrap();
    assert_eq!(multi_results.len(), 2, "Should simulate all wavelengths");

    for (fluence, pressure) in &multi_results {
        assert_eq!(fluence.dim(), (8, 8, 4));
        assert_eq!(pressure.pressure.dim(), (8, 8, 4));
        assert!(pressure.max_pressure > 0.0);
    }
}

#[test]
fn test_detector_positions() {
    let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
    let positions = reconstruction::compute_detector_positions(&grid, 64);

    assert_eq!(positions.len(), 64);

    // Check that detectors are within grid bounds
    let (nx, ny, nz) = grid.dimensions();
    for &(x, y, z) in &positions {
        assert!(x >= 0.0 && x < nx as f64);
        assert!(y >= 0.0 && y < ny as f64);
        assert!(z >= 0.0 && z < nz as f64);
    }

    // Check circular arrangement
    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;
    let expected_radius = ((nx.min(ny)) as f64 / 2.0) * 0.4;

    for &(x, y, _z) in &positions {
        let radius = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
    }
}

#[test]
fn test_accessor_methods() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid.clone(), parameters.clone(), &medium).unwrap();

    // Test accessor methods
    assert_eq!(simulator.grid().dimensions(), (16, 16, 8));
    assert_eq!(simulator.optical_properties().dim(), (16, 16, 8));
    assert_eq!(
        simulator.parameters().wavelengths.len(),
        parameters.wavelengths.len()
    );
}
