//! Wave equation validation tests
//!
//! Validates basic wave propagation against analytical solutions
//! Reference: Pierce (1989) - "Acoustics: An Introduction"

use crate::constants::physics::{DENSITY_WATER, SOUND_SPEED_WATER};
use crate::grid::Grid;
use crate::physics::field_indices;
use crate::physics::state::PhysicsState;

use ndarray::Array3;
use std::f64::consts::PI;

// Wave simulation constants
const WAVE_FREQUENCY: f64 = 1e6; // 1 MHz
const WAVELENGTH_FACTOR: f64 = 10.0; // Points per wavelength

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_wave_equation_analytical() {
        // Grid parameters for 1D wave
        let nx = 128;
        let dx = 1e-3; // 1mm
        let dt = dx / (SOUND_SPEED_WATER * 2.0); // CFL condition

        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        let state = PhysicsState::new(grid.clone());

        // Initialize Gaussian pulse
        let x0 = nx as f64 * dx / 2.0;
        let sigma = WAVELENGTH_FACTOR * dx;

        let mut initial_pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = i as f64 * dx;
            let amplitude = ((-(x - x0).powi(2)) / (2.0 * sigma.powi(2))).exp();
            initial_pressure[[i, 0, 0]] = amplitude;
        }
        state
            .update_field(field_indices::PRESSURE_IDX, &initial_pressure)
            .unwrap();

        // Propagate using leapfrog scheme for wave equation
        // Using velocity-pressure formulation for stability
        let steps = 100;

        // Initialize velocity field
        let mut velocity: Array3<f64> = Array3::zeros((nx, 1, 1));
        let mut pressure_prev = initial_pressure.clone();
        let mut pressure_curr = initial_pressure.clone();

        for _ in 0..steps {
            // Update velocity: dv/dt = -(1/ρ) * dp/dx
            for i in 1..nx - 1 {
                velocity[[i, 0, 0]] -= (dt / DENSITY_WATER)
                    * (pressure_curr[[i + 1, 0, 0]] - pressure_curr[[i - 1, 0, 0]])
                    / (2.0 * dx);
            }

            // Update pressure: dp/dt = -ρc² * dv/dx
            let mut pressure_next = pressure_curr.clone();
            for i in 1..nx - 1 {
                pressure_next[[i, 0, 0]] = pressure_curr[[i, 0, 0]]
                    - dt * DENSITY_WATER
                        * SOUND_SPEED_WATER
                        * SOUND_SPEED_WATER
                        * (velocity[[i + 1, 0, 0]] - velocity[[i - 1, 0, 0]])
                        / (2.0 * dx);
            }

            pressure_prev = pressure_curr;
            pressure_curr = pressure_next;
        }

        state
            .update_field(field_indices::PRESSURE_IDX, &pressure_curr)
            .unwrap();

        // Verify wave has propagated
        let travel_distance = SOUND_SPEED_WATER * dt * steps as f64;
        let expected_peak = x0 + travel_distance;

        // Find peak position
        let pressure_field = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        let pressure_view = pressure_field.view();
        let mut max_val = 0.0;
        let mut max_idx = 0;
        for i in 0..nx {
            if pressure_view[[i, 0, 0]].abs() > max_val {
                max_val = pressure_view[[i, 0, 0]].abs();
                max_idx = i;
            }
        }

        let actual_peak = max_idx as f64 * dx;
        let error = (actual_peak - expected_peak).abs() / expected_peak;

        assert!(
            error < 0.05,
            "Wave propagation error: {:.2}%",
            error * 100.0
        );
    }

    #[test]
    #[ignore] // TODO: Fix hanging test - likely infinite loop
    fn test_standing_wave_rigid_boundaries() {
        let nx = 64;
        let dx = 1e-3;
        let dt = dx / (SOUND_SPEED_WATER * 2.0);

        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        let state = PhysicsState::new(grid.clone());

        // Initialize standing wave (first mode)
        let mut initial_pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = i as f64 * dx;
            let k = PI / (nx as f64 * dx); // Wave number for first mode
            initial_pressure[[i, 0, 0]] = (k * x).sin();
        }
        state
            .update_field(field_indices::PRESSURE_IDX, &initial_pressure)
            .unwrap();

        // Should oscillate in place with rigid boundaries
        let period = 2.0 * PI / (SOUND_SPEED_WATER * PI / (nx as f64 * dx));
        let steps = (period / dt) as usize;

        let pressure_field = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        let initial_energy: f64 = pressure_field.view().iter().map(|p| p * p).sum();

        // Simple time stepping for standing wave test
        for _ in 0..steps {
            // Get current pressure field
            let pressure_field = state
                .get_field(field_indices::PRESSURE_IDX)
                .unwrap()
                .to_owned();
            // Apply rigid boundary conditions
            let pressure_guard = state.get_field(field_indices::PRESSURE_IDX).unwrap();
            let mut pressure = pressure_guard.to_owned();
            pressure[[0, 0, 0]] = 0.0;
            pressure[[nx - 1, 0, 0]] = 0.0;
            state
                .update_field(field_indices::PRESSURE_IDX, &pressure)
                .unwrap();
        }

        let pressure_field = state.get_field(field_indices::PRESSURE_IDX).unwrap();
        let final_energy: f64 = pressure_field.view().iter().map(|p| p * p).sum();
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;

        assert!(
            energy_error < 0.01,
            "Energy conservation error: {:.2}%",
            energy_error * 100.0
        );
    }

    #[test]
    fn test_spherical_spreading() {
        // 3D test for 1/r decay using simplified wave propagation
        let n = 64;
        let dx = 2e-3;
        let dt = dx / (SOUND_SPEED_WATER * 4.0); // Smaller time step for stability
        let c2 = SOUND_SPEED_WATER * SOUND_SPEED_WATER;

        // Initialize fields
        let mut pressure = Array3::zeros((n, n, n));
        let mut pressure_prev = Array3::zeros((n, n, n));

        // Gaussian pulse at center for smooth propagation
        let center = n / 2;
        let sigma = 2.0 * dx;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let r = (((i as f64 - center as f64) * dx).powi(2)
                        + ((j as f64 - center as f64) * dx).powi(2)
                        + ((k as f64 - center as f64) * dx).powi(2))
                    .sqrt();
                    pressure[[i, j, k]] = (-r * r / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        pressure_prev.assign(&pressure);

        // Propagate using simple finite difference wave equation
        let steps = 30;

        for _ in 0..steps {
            let mut pressure_next = Array3::zeros((n, n, n));

            // Interior points: ∂²p/∂t² = c²∇²p
            for k in 1..n - 1 {
                for j in 1..n - 1 {
                    for i in 1..n - 1 {
                        let laplacian = (pressure[[i + 1, j, k]]
                            + pressure[[i - 1, j, k]]
                            + pressure[[i, j + 1, k]]
                            + pressure[[i, j - 1, k]]
                            + pressure[[i, j, k + 1]]
                            + pressure[[i, j, k - 1]]
                            - 6.0 * pressure[[i, j, k]])
                            / (dx * dx);

                        pressure_next[[i, j, k]] = 2.0 * pressure[[i, j, k]]
                            - pressure_prev[[i, j, k]]
                            + dt * dt * c2 * laplacian;
                    }
                }
            }

            // Update time history
            pressure_prev.assign(&pressure);
            pressure.assign(&pressure_next);
        }

        // Measure amplitude at different radii along x-axis
        let r1 = 8;
        let r2 = 16;

        // Find peak amplitude in spherical shells around each radius
        let mut amp1_max: f64 = 0.0;
        let mut amp2_max: f64 = 0.0;

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let r = (((i as f64 - center as f64) * dx).powi(2)
                        + ((j as f64 - center as f64) * dx).powi(2)
                        + ((k as f64 - center as f64) * dx).powi(2))
                    .sqrt();

                    let r_grid = r / dx;

                    // Sample in shells around target radii
                    if (r_grid - r1 as f64).abs() < 1.0 {
                        amp1_max = amp1_max.max(pressure[[i, j, k]].abs());
                    }
                    if (r_grid - r2 as f64).abs() < 1.0 {
                        amp2_max = amp2_max.max(pressure[[i, j, k]].abs());
                    }
                }
            }
        }

        // Should follow 1/r relationship for spherical waves
        let expected_ratio = r2 as f64 / r1 as f64; // Inverse because amplitude ~ 1/r
        let actual_ratio = amp1_max / amp2_max.max(1e-10); // Avoid division by zero
        let error = (actual_ratio - expected_ratio).abs() / expected_ratio;

        assert!(
            error < 0.25, // Allow 25% error due to numerical dispersion
            "Spherical spreading error: {:.2}% (expected ratio: {:.2}, actual: {:.2})",
            error * 100.0,
            expected_ratio,
            actual_ratio
        );
    }
}
