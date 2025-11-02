//! Wave equation validation tests
//!
//! Validates basic wave propagation against analytical solutions
//! Reference: Pierce (1989) - "Acoustics: An Introduction"

use crate::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
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
        // CORRECTED: Proper 1D wave equation implementation
        // ∂²p/∂t² = c² ∂²p/∂x²

        let nx = 128;
        let dx = 1e-3; // 1mm spacing
        let dt = dx / SOUND_SPEED_WATER; // CFL = 1.0 for 1D wave equation
        let c2 = SOUND_SPEED_WATER * SOUND_SPEED_WATER;

        let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
        let state = PhysicsState::new(grid.clone());

        // Initialize Gaussian pulse at center
        let x0 = nx as f64 * dx / 2.0;
        let sigma = WAVELENGTH_FACTOR * dx;

        let mut pressure_curr = Array3::zeros((nx, 1, 1));
        let mut pressure_prev = Array3::zeros((nx, 1, 1));

        for i in 0..nx {
            let x = i as f64 * dx;
            let amplitude = ((-(x - x0).powi(2)) / (2.0 * sigma.powi(2))).exp();
            pressure_curr[[i, 0, 0]] = amplitude;
            pressure_prev[[i, 0, 0]] = amplitude; // Initial condition
        }

        state
            .update_field(field_indices::PRESSURE_IDX, &pressure_curr)
            .unwrap();

        // Propagate using standard leapfrog scheme for wave equation
        let steps = 50;

        for _ in 0..steps {
            let mut pressure_next = Array3::zeros((nx, 1, 1));

            // Interior points: ∂²p/∂t² = c² ∂²p/∂x²
            for i in 1..nx - 1 {
                let d2p_dx2 = (pressure_curr[[i + 1, 0, 0]] - 2.0 * pressure_curr[[i, 0, 0]]
                             + pressure_curr[[i - 1, 0, 0]]) / (dx * dx);

                pressure_next[[i, 0, 0]] = 2.0 * pressure_curr[[i, 0, 0]]
                                          - pressure_prev[[i, 0, 0]]
                                          + dt * dt * c2 * d2p_dx2;
            }

            // Boundary conditions: zero pressure (free ends)
            pressure_next[[0, 0, 0]] = 0.0;
            pressure_next[[nx - 1, 0, 0]] = 0.0;

            pressure_prev.assign(&pressure_curr);
            pressure_curr.assign(&pressure_next);
        }

        state
            .update_field(field_indices::PRESSURE_IDX, &pressure_curr)
            .unwrap();

        // Verify wave propagation physics
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
            error < 0.10, // Allow 10% error due to numerical dispersion
            "Wave propagation error: {:.2}% (expected: {:.3}, actual: {:.3})",
            error * 100.0, expected_peak, actual_peak
        );
    }

    #[test]
    fn test_standing_wave_rigid_boundaries() {
        // CORRECTED: Proper standing wave physics with rigid boundaries
        // Rigid boundaries: zero normal velocity (dp/dx = 0)

        let nx = 64;
        let dx = 1e-3;
        let dt = dx / SOUND_SPEED_WATER; // CFL = 1.0 for stability
        let c2 = SOUND_SPEED_WATER * SOUND_SPEED_WATER;

        let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
        let state = PhysicsState::new(grid.clone());

        // Initialize standing wave (cosine mode for rigid boundaries)
        let mut pressure_curr = Array3::zeros((nx, 1, 1));
        let mut pressure_prev = Array3::zeros((nx, 1, 1));

        // Standing wave: p(x,t) = A cos(kx) cos(ωt)
        // First mode: k = π/L, ω = c k
        let k = PI / ((nx - 1) as f64 * dx); // Wave number
        let amplitude = 1.0;

        for i in 0..nx {
            let x = i as f64 * dx;
            let initial_p = amplitude * (k * x).cos();
            pressure_curr[[i, 0, 0]] = initial_p;
            pressure_prev[[i, 0, 0]] = initial_p; // Start at t=0
        }

        state
            .update_field(field_indices::PRESSURE_IDX, &pressure_curr)
            .unwrap();

        let initial_energy: f64 = pressure_curr.iter().map(|p| p * p).sum();

        // Propagate for one full period
        let omega = SOUND_SPEED_WATER * k;
        let period = 2.0 * PI / omega;
        let steps = (period / dt).round() as usize;

        for _ in 0..steps {
            let mut pressure_next = Array3::zeros((nx, 1, 1));

            // Interior points: ∂²p/∂t² = c² ∂²p/∂x²
            for i in 1..nx - 1 {
                let d2p_dx2 = (pressure_curr[[i + 1, 0, 0]] - 2.0 * pressure_curr[[i, 0, 0]]
                             + pressure_curr[[i - 1, 0, 0]]) / (dx * dx);

                pressure_next[[i, 0, 0]] = 2.0 * pressure_curr[[i, 0, 0]]
                                          - pressure_prev[[i, 0, 0]]
                                          + dt * dt * c2 * d2p_dx2;
            }

            // Rigid boundary conditions: dp/dx = 0 (zero normal velocity)
            // This means: p[1] = p[0] and p[nx-2] = p[nx-1]
            pressure_next[[0, 0, 0]] = pressure_next[[1, 0, 0]];
            pressure_next[[nx - 1, 0, 0]] = pressure_next[[nx - 2, 0, 0]];

            pressure_prev.assign(&pressure_curr);
            pressure_curr.assign(&pressure_next);
        }

        let final_energy: f64 = pressure_curr.iter().map(|p| p * p).sum();
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;

        assert!(
            energy_error < 0.05, // Allow 5% energy loss due to numerical errors
            "Energy conservation error: {:.3}% (should be <5% for proper standing wave)",
            energy_error * 100.0
        );
    }

    #[test]
    fn test_spherical_spreading() {
        // 3D test for 1/r decay using point source
        let n = 48;
        let dx = 1e-3;
        let dt = dx / (SOUND_SPEED_WATER * 2.0); // CFL = 0.5 for stability
        let c2 = SOUND_SPEED_WATER * SOUND_SPEED_WATER;

        // Initialize fields
        let mut pressure: Array3<f64> = Array3::zeros((n, n, n));
        let mut pressure_prev: Array3<f64> = Array3::zeros((n, n, n));

        // Point source at center
        let center = n / 2;

        // Store maximum amplitudes at different radii
        let mut max_at_r1: f64 = 0.0;
        let mut max_at_r2: f64 = 0.0;
        let r1 = 8.0 * dx; // First measurement radius
        let r2 = 16.0 * dx; // Second measurement radius

        // Propagate with continuous source
        let steps = 100;
        let source_freq = 1e6; // 1 MHz
        let omega = 2.0 * std::f64::consts::PI * source_freq;

        for step in 0..steps {
            let mut pressure_next = Array3::zeros((n, n, n));
            let time = step as f64 * dt;

            // Apply sinusoidal point source at center
            let source_amplitude = if step < 20 {
                (omega * time).sin() * 0.1
            } else {
                0.0 // Turn off source after initial pulse
            };

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

                        // Add source at center
                        if i == center && j == center && k == center {
                            pressure_next[[i, j, k]] += source_amplitude;
                        }
                    }
                }
            }

            // Update time history
            pressure_prev.assign(&pressure);
            pressure.assign(&pressure_next);

            // Measure amplitudes after wave has propagated
            if step > 40 {
                for k in center - 2..=center + 2 {
                    for j in center - 2..=center + 2 {
                        for i in 0..n {
                            let r = ((i as f64 - center as f64) * dx).abs();

                            // Check at measurement radii
                            if (r - r1).abs() < dx {
                                max_at_r1 = max_at_r1.max(pressure[[i, j, k]].abs());
                            }
                            if (r - r2).abs() < dx {
                                max_at_r2 = max_at_r2.max(pressure[[i, j, k]].abs());
                            }
                        }
                    }
                }
            }
        }

        // Should follow 1/r relationship for spherical waves
        let expected_ratio = r2 / r1; // Amplitude ratio should be inverse of radius ratio
        let actual_ratio = max_at_r1 / max_at_r2.max(1e-10);
        let error = (actual_ratio - expected_ratio).abs() / expected_ratio;

        assert!(
            error < 0.30, // Allow 30% error due to numerical effects
            "Spherical spreading error: {:.2}% (expected ratio: {:.2}, actual: {:.2})",
            error * 100.0,
            expected_ratio,
            actual_ratio
        );
    }
}
