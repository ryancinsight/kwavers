//! Tests for photoacoustic image reconstruction.

use super::core::{
    compute_detector_positions, interpolate_detector_signal, time_reversal_reconstruction,
};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use approx::assert_relative_eq;
use ndarray::Array3;

#[test]
fn test_detector_position_computation() {
    let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
    let positions = compute_detector_positions(&grid, 64);

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

#[test]
fn test_trilinear_interpolation_at_grid_points() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let mut field = Array3::<f64>::zeros((8, 8, 4));

    field[[2, 2, 1]] = 1.0;
    field[[3, 2, 1]] = 2.0;
    field[[2, 3, 1]] = 3.0;
    field[[3, 3, 1]] = 4.0;

    let value_2_2_1 = interpolate_detector_signal(&grid, &field, 2.0, 2.0, 1.0);
    assert_relative_eq!(value_2_2_1, 1.0, epsilon = 1e-10);

    let value_3_3_1 = interpolate_detector_signal(&grid, &field, 3.0, 3.0, 1.0);
    assert_relative_eq!(value_3_3_1, 4.0, epsilon = 1e-10);
}

#[test]
fn test_trilinear_interpolation_midpoint() {
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

    let expected_mid = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
    let value_mid = interpolate_detector_signal(&grid, &field, 2.5, 2.5, 1.5);
    assert_relative_eq!(value_mid, expected_mid, epsilon = 1e-10);
}

#[test]
fn test_boundary_clamping() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let field = Array3::<f64>::from_elem((8, 8, 4), 1.0);

    let value_outside = interpolate_detector_signal(&grid, &field, -1.0, -1.0, -1.0);
    assert_eq!(value_outside, field[[0, 0, 0]]);

    let value_beyond = interpolate_detector_signal(&grid, &field, 10.0, 10.0, 10.0);
    assert_eq!(value_beyond, field[[7, 7, 3]]);
}

#[test]
fn test_time_reversal_reconstruction_basic() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

    let n_time = 20;
    let dt = 1e-7;
    let mut pressure_fields = Vec::with_capacity(n_time);
    let time_points: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();

    let source_x = 8.0 * grid.dx;
    let source_y = 8.0 * grid.dy;
    let source_z = 4.0 * grid.dz;
    let speed_of_sound = SOUND_SPEED_WATER_SIM;

    for &time in time_points.iter() {
        let mut field = Array3::<f64>::zeros((16, 16, 8));
        for i in 0..16 {
            for j in 0..16 {
                for k in 0..8 {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let dx = x - source_x;
                    let dy = y - source_y;
                    let dz = z - source_z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    let travel_time = distance / speed_of_sound;
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

    let reconstructed =
        time_reversal_reconstruction(&grid, &pressure_fields, &time_points, speed_of_sound, 36)
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
fn test_spherical_spreading_correction() {
    let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

    let pressure_fields = vec![Array3::<f64>::from_elem((16, 16, 8), 1.0)];
    let time_points = vec![0.0];

    let reconstructed = time_reversal_reconstruction(
        &grid,
        &pressure_fields,
        &time_points,
        SOUND_SPEED_WATER_SIM,
        36,
    )
    .unwrap();

    let center_value = reconstructed[[8, 8, 4]];
    let edge_value = reconstructed[[0, 0, 0]];
    assert_ne!(center_value, edge_value);

    for &val in reconstructed.iter() {
        assert!(val.is_finite());
        assert!(val >= 0.0);
    }
}
