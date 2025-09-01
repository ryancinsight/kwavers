//! Nonlinear acoustics validation tests
//!
//! Reference: Hamilton & Blackstock (1998) - "Nonlinear Acoustics"

use crate::grid::Grid;
use crate::physics::mechanics::acoustic_wave::kuznetsov::config::{
    AcousticEquationMode, KuznetsovConfig,
};
use crate::physics::mechanics::acoustic_wave::kuznetsov::solver::KuznetsovWave;
use crate::physics::traits::AcousticWaveModel;
use ndarray::{Array3, Array4};
use std::f64::consts::PI;

// Nonlinearity parameters
const BETA_WATER: f64 = 3.5; // Nonlinearity parameter B/A for water
const ATTENUATION_WATER: f64 = 2.17e-3; // dB/cm/MHz^2
const SHOCK_DISTANCE_FACTOR: f64 = 1.0; // Normalized shock distance

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // TODO: Fix nonlinear term formulation
    fn test_kuznetsov_second_harmonic() {
        // Test second harmonic generation in nonlinear propagation
        let nx = 256;
        let dx = 1e-4;
        let frequency: f64 = 1e6; // 1 MHz

        use crate::physics::mechanics::acoustic_wave::kuznetsov::AcousticEquationMode;
        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::FullKuznetsov,
            cfl_factor: 0.5,
            nonlinearity_coefficient: BETA_WATER,
            acoustic_diffusivity: ATTENUATION_WATER * 1500.0_f64.powi(3)
                / (2.0 * std::f64::consts::PI * std::f64::consts::PI * frequency.powi(2)),
            use_k_space_correction: false,
            k_space_correction_order: 2,
            spatial_order: 4,
            adaptive_time_stepping: false,
            max_pressure: 1e9,
            shock_capturing: true,
            history_levels: 3,
            nonlinearity_scaling: 1.0,
            diffusivity: 1.0,
        };

        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        let mut solver =
            KuznetsovWave::new(config, &grid).expect("Failed to create Kuznetsov solver");

        // Initialize sinusoidal wave
        let wavelength = 1500.0 / frequency;
        let k = 2.0 * PI / wavelength;
        let amplitude = 1e6; // 1 MPa

        // Calculate time step
        let dt = 0.5 * dx / 1500.0; // CFL condition

        // Initialize fields array (4D: [field_type, x, y, z])
        let mut fields = Array4::zeros((1, nx, 1, 1)); // Single field for pressure
        let mut prev_pressure = Array3::zeros((nx, 1, 1));

        // Initialize pressure field with sinusoidal wave
        for i in 0..nx {
            let x = i as f64 * dx;
            fields[[0, i, 0, 0]] = amplitude * (k * x).sin();
            // Initialize prev_pressure for proper time stepping (assuming wave traveling at c)
            prev_pressure[[i, 0, 0]] = amplitude * (k * (x - 1500.0 * dt)).sin();
        }

        // Create a null source and medium for testing
        use crate::medium::HomogeneousMedium;
        use crate::source::NullSource;
        let source = NullSource::new();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut t = 0.0;

        // Propagate to develop harmonics
        let steps = 100;
        for step in 0..steps {
            // Store current pressure as previous for next step
            for i in 0..nx {
                prev_pressure[[i, 0, 0]] = fields[[0, i, 0, 0]];
            }

            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t);
            t += dt;

            // Check for NaN
            if step % 20 == 0 || step == steps - 1 {
                let max_p = fields.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
                let has_nan = fields.iter().any(|p| p.is_nan());
                println!(
                    "Step {}: max pressure = {:.2e}, has NaN = {}",
                    step, max_p, has_nan
                );
                if has_nan {
                    panic!("NaN detected at step {}", step);
                }
            }
        }

        // Extract pressure for analysis
        let mut pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            pressure[[i, 0, 0]] = fields[[0, i, 0, 0]];
        }

        // FFT to extract harmonics
        use rustfft::{num_complex::Complex, FftPlanner};
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nx);

        let mut spectrum: Vec<Complex<f64>> =
            pressure.iter().map(|&p| Complex::new(p, 0.0)).collect();

        fft.process(&mut spectrum);

        // Find fundamental and second harmonic peaks
        let fundamental_idx = (frequency * nx as f64 * dx / 1500.0) as usize;
        let second_harmonic_idx = 2 * fundamental_idx;

        let fundamental_amp = spectrum[fundamental_idx].norm();
        let second_harmonic_amp = spectrum[second_harmonic_idx].norm();

        println!("Fundamental amplitude: {:.2e}", fundamental_amp);
        println!("Second harmonic amplitude: {:.2e}", second_harmonic_amp);
        println!(
            "Fundamental index: {}, Second harmonic index: {}",
            fundamental_idx, second_harmonic_idx
        );

        // Theoretical ratio from Blackstock
        let propagation_distance = steps as f64 * dx;
        let shock_distance = 1500.0 / (BETA_WATER * k * amplitude);
        let sigma = propagation_distance / shock_distance;

        let expected_ratio = sigma / 2.0; // Linear approximation for small sigma
        let actual_ratio = if fundamental_amp > 1e-10 {
            second_harmonic_amp / fundamental_amp
        } else {
            println!(
                "Warning: fundamental amplitude too small: {:.2e}",
                fundamental_amp
            );
            0.0
        };

        println!(
            "Expected ratio: {:.4}, Actual ratio: {:.4}",
            expected_ratio, actual_ratio
        );

        if expected_ratio > 0.0 {
            let error = (actual_ratio - expected_ratio).abs() / expected_ratio;
            assert!(
                error < 0.2,
                "Second harmonic generation error: {:.2}% (expected: {:.4}, actual: {:.4})",
                error * 100.0,
                expected_ratio,
                actual_ratio
            );
        } else {
            panic!("Invalid expected ratio: {:.4}", expected_ratio);
        }
    }

    #[test]
    #[ignore] // TODO: Fix nonlinear term formulation
    fn test_shock_formation_distance() {
        // Validate shock formation distance for plane wave
        let nx = 512;
        let dx = 1e-4;
        let frequency = 2e6;
        let amplitude = 2e6; // 2 MPa

        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::FullKuznetsov,
            cfl_factor: 0.5,
            nonlinearity_coefficient: BETA_WATER,
            acoustic_diffusivity: 0.0, // No attenuation for cleaner shock
            use_k_space_correction: false,
            k_space_correction_order: 2,
            spatial_order: 4,
            adaptive_time_stepping: false,
            max_pressure: 1e9,
            shock_capturing: true,
            history_levels: 3,
            nonlinearity_scaling: 1.0,
            diffusivity: 0.0,
        };

        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        let mut solver =
            KuznetsovWave::new(config, &grid).expect("Failed to create Kuznetsov solver");

        // Initialize sine wave
        let wavelength = 1500.0 / frequency;
        let k = 2.0 * PI / wavelength;

        // Initialize fields array (4D: [field_type, x, y, z])
        let mut fields = Array4::zeros((1, nx, 1, 1));
        let prev_pressure = Array3::zeros((nx, 1, 1));

        for i in 0..nx / 4 {
            let x = i as f64 * dx;
            fields[[0, i, 0, 0]] = amplitude * (k * x).sin();
        }

        // Create a null source and medium for testing
        use crate::medium::HomogeneousMedium;
        use crate::source::NullSource;
        let source = NullSource::new();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);

        // Calculate time step
        let dt = 0.5 * dx / 1500.0; // CFL condition
        let mut t = 0.0;

        // Theoretical shock distance (Blackstock Eq. 4.23)
        let shock_distance = 1500.0 / (BETA_WATER * k * amplitude);
        let steps_to_shock = (shock_distance / dx) as usize;

        // Propagate to near shock formation
        for _ in 0..(steps_to_shock as f64 * 0.9) as usize {
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t);
            t += dt;
        }

        // Extract pressure for analysis
        let mut pressure = Array3::zeros((nx, 1, 1));
        for i in 0..nx {
            pressure[[i, 0, 0]] = fields[[0, i, 0, 0]];
        }

        // Check for steepening (increased gradient)
        let mut max_gradient: f64 = 0.0;
        for i in 1..nx - 1 {
            let gradient = (pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]).abs() / (2.0 * dx);
            max_gradient = max_gradient.max(gradient);
        }

        // Gradient should be significantly increased near shock
        let initial_gradient = amplitude * k;
        let steepening_factor = max_gradient / initial_gradient;

        assert!(
            steepening_factor > 5.0,
            "Insufficient shock steepening: {:.2}x",
            steepening_factor
        );
    }
}
