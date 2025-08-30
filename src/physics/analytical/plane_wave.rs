//! Plane wave analytical solutions for validation

use super::utils::DISPERSION_CORRECTION_SECOND_ORDER;
use crate::grid::Grid;
use ndarray::Array3;
use std::f64::consts::PI;

/// Plane wave analytical solutions
#[derive(Debug, Debug))]
pub struct PlaneWaveSolution;

impl PlaneWaveSolution {
    /// Generate analytical plane wave field
    pub fn generate(
        grid: &Grid,
        frequency: f64,
        amplitude: f64,
        sound_speed: f64,
        time: f64,
        direction: (f64, f64, f64),
    ) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let wavelength = sound_speed / frequency;
        let k = 2.0 * PI / wavelength;
        let omega = 2.0 * PI * frequency;

        // Normalize direction vector
        let norm = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
        let dir = (direction.0 / norm, direction.1 / norm, direction.2 / norm);

        // Apply dispersion correction for k-space methods
        let k_corrected =
            k * (1.0 + DISPERSION_CORRECTION_SECOND_ORDER * k * k * grid.dx * grid.dx);

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k_idx as f64 * grid.dz;

                    let phase = k_corrected * (dir.0 * x + dir.1 * y + dir.2 * z) - omega * time;
                    field[[i, j, k_idx] = amplitude * phase.sin();
                }
            }
        }

        field
    }

    /// Validate plane wave propagation
    pub fn validate_propagation(
        initial_field: &Array3<f64>,
        final_field: &Array3<f64>,
        grid: &Grid,
        expected_speed: f64,
        time_elapsed: f64,
        tolerance: f64,
    ) -> bool {
        use super::utils::PhysicsTestUtils;

        let (actual_speed, correlation) = PhysicsTestUtils::detect_wave_propagation_subgrid(
            initial_field,
            final_field,
            grid,
            expected_speed,
            time_elapsed,
        );

        let speed_error = (actual_speed - expected_speed).abs() / expected_speed;
        speed_error < tolerance && correlation > 0.9
    }
}
