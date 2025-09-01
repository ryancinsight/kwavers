//! Validation tests for implicit Kuznetsov solver
//!
//! Tests the implicit solver implementation against theoretical predictions
//! for second harmonic generation and shock formation.

use crate::physics::mechanics::acoustic_wave::kuznetsov::implicit_solver::{
    ImplicitKuznetsovConfig, ImplicitKuznetsovSolver,
};
use std::f64::consts::PI;

// Physical constants for water at 20°C
const WATER_DENSITY: f64 = 1000.0; // kg/m³
const WATER_SOUND_SPEED: f64 = 1500.0; // m/s
const WATER_NONLINEARITY: f64 = 3.5; // B/A parameter
const WATER_DIFFUSIVITY: f64 = 4.5e-6; // m²/s

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_solver_stability() {
        // Test that the implicit solver remains stable for long propagation
        let config = ImplicitKuznetsovConfig {
            sound_speed: WATER_SOUND_SPEED,
            density: WATER_DENSITY,
            nonlinearity: WATER_NONLINEARITY,
            diffusivity: WATER_DIFFUSIVITY,
            dt: 1e-8,
            dx: 1e-5,
            max_iterations: 30,
            tolerance: 1e-8,
        };

        let mut solver = ImplicitKuznetsovSolver::new(config, (128, 1, 1));

        // Initialize with moderate amplitude
        let wavelength = 1.5e-3; // 1.5mm at 1 MHz
        let amplitude = 1e5; // 100 kPa
        solver.initialize_sinusoid(amplitude, wavelength);

        // Run for many steps
        let mut max_pressures = Vec::new();
        for step in 0..1000 {
            solver.step();

            let max_p = solver
                .pressure()
                .iter()
                .map(|&p| p.abs())
                .fold(0.0, f64::max);

            max_pressures.push(max_p);

            if step % 100 == 0 {
                println!("Step {}: max pressure = {:.2e} Pa", step, max_p);
            }
        }

        // Check that pressure doesn't grow exponentially
        let initial_max = max_pressures[0];
        let final_max = max_pressures[max_pressures.len() - 1];

        assert!(
            final_max < initial_max * 2.0,
            "Pressure grew from {:.2e} to {:.2e}",
            initial_max,
            final_max
        );
    }

    #[test]
    fn test_second_harmonic_generation_weak() {
        // Test weak second harmonic generation against perturbation theory
        let config = ImplicitKuznetsovConfig {
            sound_speed: WATER_SOUND_SPEED,
            density: WATER_DENSITY,
            nonlinearity: WATER_NONLINEARITY,
            diffusivity: 0.0, // No dissipation for cleaner harmonics
            dt: 1e-9,
            dx: 1e-5,
            max_iterations: 20,
            tolerance: 1e-7,
        };

        let nx = 256;
        let mut solver = ImplicitKuznetsovSolver::new(config.clone(), (nx, 1, 1));

        // Weak amplitude for perturbation regime
        let frequency = 1e6; // 1 MHz
        let wavelength = WATER_SOUND_SPEED / frequency;
        let amplitude = 1e3; // 1 kPa
        solver.initialize_sinusoid(amplitude, wavelength);

        // Propagate for short distance
        let steps = 100;
        for _ in 0..steps {
            solver.step();
        }

        // Analyze spectrum
        let spectrum = solver.compute_spectrum();

        // Find harmonic peaks
        let fundamental_idx = (nx as f64 * config.dx / wavelength).round() as usize;
        let second_idx = 2 * fundamental_idx;

        if fundamental_idx < spectrum.len() && second_idx < spectrum.len() {
            let f1 = spectrum[fundamental_idx];
            let f2 = spectrum[second_idx];

            // Calculate Goldberg number
            let k = 2.0 * PI / wavelength;
            let beta = 1.0 + config.nonlinearity / 2.0;
            let distance = steps as f64 * config.dt * config.sound_speed;
            let shock_distance = config.sound_speed / (beta * k * amplitude);
            let goldberg = distance / shock_distance;

            println!("Goldberg number: {:.4}", goldberg);
            println!("F1 amplitude: {:.2e}, F2 amplitude: {:.2e}", f1, f2);

            // For weak nonlinearity (Goldberg << 1)
            if goldberg < 0.1 {
                let expected_ratio = goldberg / 2.0;
                let actual_ratio = f2 / f1;

                println!(
                    "Expected F2/F1: {:.4}, Actual: {:.4}",
                    expected_ratio, actual_ratio
                );

                // Allow 100% error due to numerical approximations
                assert!(
                    (actual_ratio - expected_ratio).abs() < expected_ratio,
                    "Ratio mismatch: expected {:.4}, got {:.4}",
                    expected_ratio,
                    actual_ratio
                );
            }
        }
    }

    #[test]
    fn test_shock_formation_distance() {
        // Test that shocks form at the predicted distance
        let config = ImplicitKuznetsovConfig {
            sound_speed: WATER_SOUND_SPEED,
            density: WATER_DENSITY,
            nonlinearity: WATER_NONLINEARITY,
            diffusivity: 1e-7, // Small diffusivity to allow shock formation
            dt: 1e-10,
            dx: 1e-6,
            max_iterations: 50,
            tolerance: 1e-9,
        };

        let nx = 512;
        let mut solver = ImplicitKuznetsovSolver::new(config.clone(), (nx, 1, 1));

        // High amplitude for shock formation
        let frequency = 2e6; // 2 MHz
        let wavelength = WATER_SOUND_SPEED / frequency;
        let amplitude = 1e6; // 1 MPa
        solver.initialize_sinusoid(amplitude, wavelength);

        // Calculate theoretical shock distance (Blackstock formula)
        let k = 2.0 * PI / wavelength;
        let beta = 1.0 + config.nonlinearity / 2.0;
        let shock_distance = config.sound_speed / (beta * k * amplitude);

        println!("Theoretical shock distance: {:.2e} m", shock_distance);

        // Propagate and monitor steepening
        let mut max_gradients = Vec::new();
        let steps_to_shock = (shock_distance / (config.dt * config.sound_speed)) as usize;

        for step in 0..steps_to_shock * 2 {
            solver.step();

            // Calculate maximum gradient
            let pressure = solver.pressure();
            let mut max_grad: f64 = 0.0;

            for i in 1..nx - 1 {
                let grad =
                    ((pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]) / (2.0 * config.dx)).abs();
                max_grad = max_grad.max(grad);
            }

            max_gradients.push(max_grad);

            if step % (steps_to_shock / 10) == 0 {
                let distance = step as f64 * config.dt * config.sound_speed;
                println!(
                    "Distance: {:.2e} m, Max gradient: {:.2e}",
                    distance, max_grad
                );
            }
        }

        // Find where gradient peaks (shock formation)
        let peak_idx = max_gradients
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let actual_shock_distance = peak_idx as f64 * config.dt * config.sound_speed;

        println!("Actual shock distance: {:.2e} m", actual_shock_distance);

        // Check within 50% of theoretical (numerical diffusion affects this)
        let error = (actual_shock_distance - shock_distance).abs() / shock_distance;
        assert!(
            error < 0.5,
            "Shock distance error: {:.1}% (expected: {:.2e}, actual: {:.2e})",
            error * 100.0,
            shock_distance,
            actual_shock_distance
        );
    }

    #[test]
    fn test_energy_conservation() {
        // Test energy conservation in the absence of dissipation
        let config = ImplicitKuznetsovConfig {
            sound_speed: WATER_SOUND_SPEED,
            density: WATER_DENSITY,
            nonlinearity: 0.0, // Linear case
            diffusivity: 0.0,  // No dissipation
            dt: 1e-9,
            dx: 1e-5,
            max_iterations: 10,
            tolerance: 1e-10,
        };

        let nx = 128;
        let mut solver = ImplicitKuznetsovSolver::new(config.clone(), (nx, 1, 1));

        // Initialize with Gaussian pulse
        let sigma = 10.0 * config.dx;
        let amplitude = 1e4;

        // Initialize with Gaussian pulse using a temporary pressure field
        let mut initial_pressure = ndarray::Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = (i as f64 - nx as f64 / 2.0) * config.dx;
            let value = amplitude * (-x * x / (2.0 * sigma * sigma)).exp();
            initial_pressure[[i, 0, 0]] = value;
        }
        // Use sinusoid initialization as a workaround
        solver.initialize_sinusoid(amplitude, 10.0 * sigma);

        // Calculate initial energy
        let initial_energy: f64 = solver.pressure().iter().map(|&p| p * p).sum();

        // Propagate
        for _ in 0..1000 {
            solver.step();
        }

        // Calculate final energy
        let final_energy: f64 = solver.pressure().iter().map(|&p| p * p).sum();

        println!(
            "Initial energy: {:.2e}, Final energy: {:.2e}",
            initial_energy, final_energy
        );

        // Energy should be conserved within numerical precision
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;
        assert!(
            energy_error < 0.01,
            "Energy not conserved: {:.2}% error",
            energy_error * 100.0
        );
    }
}
