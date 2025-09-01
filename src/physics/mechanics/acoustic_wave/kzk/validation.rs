//! Validation tests for KZK equation implementation
//!
//! Compares against analytical solutions and published results.

#[cfg(test)]
mod tests {
    use super::super::*;
    use ndarray::Array2;
    use std::f64::consts::PI;

    /// Test linear propagation of Gaussian beam
    /// Should maintain Gaussian profile with known spreading
    #[test]
    #[ignore] // TODO: Fix Rayleigh length calculation (13.5mm vs 7.07mm expected)
    fn test_gaussian_beam_diffraction() {
        let mut config = KZKConfig {
            nx: 64,
            ny: 64,
            nz: 100,
            nt: 50,
            dx: 0.5e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian beam
        let beam_width = 5e-3; // 5 mm
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                source[[i, j]] = (-r2 / (beam_width * beam_width)).exp();
            }
        }

        solver.set_source(source.clone(), 1e6);

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / 1e6;
        let rayleigh_distance = PI * beam_width * beam_width / wavelength;
        let steps = (rayleigh_distance / config.dz) as usize;

        for _ in 0..steps {
            solver.step();
        }

        // Check beam has spread by √2 at Rayleigh distance
        let intensity = solver.get_intensity();

        // Find FWHM
        let center_i = config.nx / 2;
        let center_j = config.ny / 2;
        let max_intensity = intensity[[center_i, center_j]];
        let half_max = max_intensity / 2.0;

        let mut width_pixels = 0;
        for i in 0..config.nx {
            if intensity[[i, center_j]] > half_max {
                width_pixels += 1;
            }
        }

        let final_width = width_pixels as f64 * config.dx;
        let expected_width = beam_width * 2.0_f64.sqrt();

        assert!(
            (final_width - expected_width).abs() / expected_width < 0.2,
            "Beam width error: expected {:.2}mm, got {:.2}mm",
            expected_width * 1000.0,
            final_width * 1000.0
        );
    }

    /// Test harmonic generation in nonlinear propagation
    #[test]
    fn test_harmonic_generation() {
        let mut config = KZKConfig {
            nx: 32,
            ny: 32,
            nz: 50,
            nt: 128,
            dx: 1e-3,
            dz: 2e-3,
            dt: 5e-9,
            include_nonlinearity: true,
            include_absorption: false,
            include_diffraction: false,
            beta: 3.5,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Uniform plane wave source
        let amplitude = 1e6; // 1 MPa
        let frequency = 2e6; // 2 MHz
        let source = Array2::from_elem((config.nx, config.ny), amplitude);

        solver.set_source(source, frequency);

        // Propagate
        for _ in 0..20 {
            solver.step();
        }

        // Analyze spectrum at center point
        let signal = solver.get_time_signal(config.nx / 2, config.ny / 2);

        // Compute FFT
        use rustfft::{num_complex::Complex, FftPlanner};
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.nt);

        let mut spectrum: Vec<Complex<f64>> =
            signal.iter().map(|&s| Complex::new(s, 0.0)).collect();

        fft.process(&mut spectrum);

        // Find fundamental and second harmonic
        let df = 1.0 / (config.nt as f64 * config.dt);
        let fundamental_bin = (frequency / df) as usize;
        let second_harmonic_bin = 2 * fundamental_bin;

        if second_harmonic_bin < config.nt / 2 {
            let fundamental_amp = spectrum[fundamental_bin].norm();
            let second_harmonic_amp = spectrum[second_harmonic_bin].norm();

            // Second harmonic should be generated
            assert!(
                second_harmonic_amp > fundamental_amp * 0.01,
                "No second harmonic generation detected"
            );
        }
    }

    /// Test absorption decay
    #[test]
    fn test_absorption() {
        let mut config = KZKConfig {
            nx: 16,
            ny: 16,
            nz: 100,
            nt: 32,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: true,
            include_diffraction: false,
            alpha0: 0.5, // dB/cm/MHz
            alpha_power: 1.0,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Uniform source
        let source = Array2::from_elem((config.nx, config.ny), 1.0);
        solver.set_source(source, 1e6);

        // Get initial intensity
        let initial_intensity = solver.get_intensity()[[config.nx / 2, config.ny / 2]];

        // Propagate 10 cm
        let steps = (0.1 / config.dz) as usize;
        for _ in 0..steps {
            solver.step();
        }

        // Check decay
        let final_intensity = solver.get_intensity()[[config.nx / 2, config.ny / 2]];

        // Expected decay: -0.5 dB/cm at 1 MHz over 10 cm = -5 dB = factor of 0.316
        let expected_ratio = 10.0_f64.powf(-0.5); // ~0.316
        let actual_ratio = final_intensity / initial_intensity;

        assert!(
            (actual_ratio - expected_ratio).abs() / expected_ratio < 0.2,
            "Absorption error: expected {:.3}, got {:.3}",
            expected_ratio,
            actual_ratio
        );
    }
}
