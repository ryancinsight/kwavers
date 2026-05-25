//! Reconstruction algorithm, detector interpolation, and analytical validation tests.

use super::super::core::PhotoacousticSimulator;
use super::super::reconstruction;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::grid::Grid;
use crate::domain::medium::homogeneous::HomogeneousMedium;
use approx::assert_relative_eq;
use ndarray::Array3;

#[test]
fn test_analytical_validation() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();
    let simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

    let error = simulator.validate_analytical().unwrap();
    assert!(error < 1.0, "Relative error should be reasonable (< 100%)");
}

#[test]
fn test_universal_back_projection_algorithm() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
    let parameters = crate::domain::imaging::photoacoustic::PhotoacousticParameters::default();

    let n_time = 40;
    let dt = 1e-7;
    let mut pressure_fields = Vec::with_capacity(n_time);
    let time_points: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();

    let source_x = 8.0;
    let source_y = 8.0;
    let source_z = 4.0;

    for &time in time_points.iter() {
        let mut field = Array3::<f64>::zeros((16, 16, 8));
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let dx = (i as f64 - source_x) * grid.dx;
                    let dy = (j as f64 - source_y) * grid.dy;
                    let dz = (k as f64 - source_z) * grid.dz;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let travel_time = distance / parameters.speed_of_sound;
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
    let reconstructed = simulator
        .time_reversal_reconstruction(&pressure_fields, &time_points)
        .unwrap();

    assert_eq!(reconstructed.dim(), (16, 16, 8));

    let mut max_intensity = f64::NEG_INFINITY;
    let mut min_intensity = f64::INFINITY;
    for &val in reconstructed.iter() {
        max_intensity = max_intensity.max(val);
        min_intensity = min_intensity.min(val);
    }

    assert!(
        max_intensity > min_intensity,
        "Reconstructed image should not be uniform"
    );
    assert!(
        max_intensity.is_finite(),
        "Maximum intensity should be finite"
    );
    assert!(
        min_intensity.is_finite(),
        "Minimum intensity should be finite"
    );
    assert!(max_intensity > 0.0, "Maximum intensity should be positive");
}

#[test]
fn test_detector_interpolation_accuracy() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();

    let mut field = Array3::<f64>::zeros((8, 8, 4));
    field[[2, 2, 1]] = 1.0;
    field[[3, 2, 1]] = 2.0;
    field[[2, 3, 1]] = 3.0;
    field[[3, 3, 1]] = 4.0;
    field[[2, 2, 2]] = 5.0;
    field[[3, 2, 2]] = 6.0;
    field[[2, 3, 2]] = 7.0;
    field[[3, 3, 2]] = 8.0;

    let value_2_2_1 = reconstruction::interpolate_detector_signal(&grid, &field, 2.0, 2.0, 1.0);
    assert_relative_eq!(value_2_2_1, 1.0, epsilon = 1e-10);

    let value_3_3_2 = reconstruction::interpolate_detector_signal(&grid, &field, 3.0, 3.0, 2.0);
    assert_relative_eq!(value_3_3_2, 8.0, epsilon = 1e-10);

    let expected_mid = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
    let value_mid = reconstruction::interpolate_detector_signal(&grid, &field, 2.5, 2.5, 1.5);
    assert_relative_eq!(value_mid, expected_mid, epsilon = 1e-10);

    let value_outside =
        reconstruction::interpolate_detector_signal(&grid, &field, -1.0, -1.0, -1.0);
    assert_eq!(value_outside, field[[0, 0, 0]]);

    let value_beyond = reconstruction::interpolate_detector_signal(&grid, &field, 10.0, 10.0, 10.0);
    assert_eq!(value_beyond, field[[7, 7, 3]]);
}

#[test]
fn test_detector_positions() {
    let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
    let positions = reconstruction::compute_detector_positions(&grid, 64);

    assert_eq!(positions.len(), 64);

    let (nx, ny, nz) = grid.dimensions();
    for &(x, y, z) in &positions {
        assert!(x >= 0.0 && x < nx as f64);
        assert!(y >= 0.0 && y < ny as f64);
        assert!(z >= 0.0 && z < nz as f64);
    }

    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;
    let expected_radius = ((nx.min(ny)) as f64 / 2.0) * 0.4;

    for &(x, y, _z) in &positions {
        let radius = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
    }
}
