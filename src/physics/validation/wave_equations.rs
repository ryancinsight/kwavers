//! Wave equation validation tests
//!
//! Validates basic wave propagation against analytical solutions
//! Reference: Pierce (1989) - "Acoustics: An Introduction"

use crate::grid::Grid;
use crate::physics::state::PhysicsState;
use crate::physics::field_indices;
use crate::solver::time_integration::{RungeKutta4, TimeStepper};
use crate::solver::time_integration::time_stepper::RK4Config;
use ndarray::{s, Array3};
use std::f64::consts::PI;

// Physical constants
const SOUND_SPEED_WATER: f64 = 1500.0; // m/s
const DENSITY_WATER: f64 = 1000.0; // kg/m³
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
        let mut state = PhysicsState::new(grid.clone());

        // Initialize Gaussian pulse
        let x0 = nx as f64 * dx / 2.0;
        let sigma = WAVELENGTH_FACTOR * dx;

        let mut initial_pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = i as f64 * dx;
            let amplitude = ((-(x - x0).powi(2)) / (2.0 * sigma.powi(2))).exp();
            initial_pressure[[i, 0, 0]] = amplitude;
        }
        state.update_field(field_indices::PRESSURE_IDX, &initial_pressure).unwrap();

        // Propagate using RK4
        let config = RK4Config::default();
        let mut solver = RungeKutta4::new(config);
        let steps = 100;

        for _ in 0..steps {
            solver.step(&mut state, &grid);
        }

        // Verify wave has propagated
        let travel_distance = SOUND_SPEED_WATER * dt * steps as f64;
        let expected_peak = x0 + travel_distance;

        // Find peak position
        let pressure_field = state.pressure().unwrap();
        let mut max_val = 0.0;
        let mut max_idx = 0;
        for i in 0..nx {
            if pressure_field[[i, 0, 0]].abs() > max_val {
                max_val = pressure_field[[i, 0, 0]].abs();
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
    fn test_standing_wave_rigid_boundaries() {
        let nx = 64;
        let dx = 1e-3;
        let dt = dx / (SOUND_SPEED_WATER * 2.0);

        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        let mut state = PhysicsState::new(grid.clone());

        // Initialize standing wave (first mode)
        let mut initial_pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = i as f64 * dx;
            let k = PI / (nx as f64 * dx); // Wave number for first mode
            initial_pressure[[i, 0, 0]] = (k * x).sin();
        }
        state.update_field(field_indices::PRESSURE_IDX, &initial_pressure).unwrap();

        // Should oscillate in place with rigid boundaries
        let period = 2.0 * PI / (SOUND_SPEED_WATER * PI / (nx as f64 * dx));
        let steps = (period / dt) as usize;

        let pressure_field = state.pressure().unwrap();
        let initial_energy: f64 = pressure_field.iter().map(|p| p * p).sum();

        let config = RK4Config::default();
        let mut solver = RungeKutta4::new(config);
        for _ in 0..steps {
            solver.step(&mut state, &grid);
            // Apply rigid boundary conditions
            let mut pressure = state.pressure().unwrap().to_owned();
            pressure[[0, 0, 0]] = 0.0;
            pressure[[nx - 1, 0, 0]] = 0.0;
            state.update_field(field_indices::PRESSURE_IDX, &pressure).unwrap();
        }

        let pressure_field = state.pressure().unwrap();
        let final_energy: f64 = pressure_field.iter().map(|p| p * p).sum();
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;

        assert!(
            energy_error < 0.01,
            "Energy conservation error: {:.2}%",
            energy_error * 100.0
        );
    }

    #[test]
    fn test_spherical_spreading() {
        // 3D test for 1/r decay
        let n = 32;
        let dx = 5e-3;
        let dt = dx / (SOUND_SPEED_WATER * 2.0);

        let grid = Grid::new(n, n, n, dx, dx, dx);
        let mut state = PhysicsState::new(grid.clone());

        // Point source at center
        let center = n / 2;
        let mut initial_pressure = Array3::zeros((n, n, n));
        initial_pressure[[center, center, center]] = 1.0;
        state.update_field(field_indices::PRESSURE_IDX, &initial_pressure).unwrap();

        let config = RK4Config::default();
        let mut solver = RungeKutta4::new(config);
        let steps = 20;

        for _ in 0..steps {
            solver.step(&mut state, &grid);
        }

        // Measure amplitude at different radii
        let r1 = 5;
        let r2 = 10;

        let pressure_field = state.pressure().unwrap();
        let amp1 = pressure_field[[center + r1, center, center]].abs();
        let amp2 = pressure_field[[center + r2, center, center]].abs();

        // Should follow 1/r relationship
        let expected_ratio = r1 as f64 / r2 as f64;
        let actual_ratio = amp1 / amp2;
        let error = (actual_ratio - expected_ratio).abs() / expected_ratio;

        assert!(
            error < 0.15,
            "Spherical spreading error: {:.2}%",
            error * 100.0
        );
    }
}
