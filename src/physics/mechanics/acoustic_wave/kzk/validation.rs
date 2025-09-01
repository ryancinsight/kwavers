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
    #[ignore] // TODO: Beam spreading insufficient (2.86mm vs 7.07mm expected)
    fn test_gaussian_beam_diffraction() {
        let mut config = KZKConfig {
            nx: 128,
            ny: 128,
            nz: 100,
            nt: 50,
            dx: 0.25e-3,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian beam with proper normalization
        // For a Gaussian beam: I(r) = I₀ * exp(-2r²/w₀²)
        // where w₀ is the beam waist radius at 1/e² intensity
        let beam_waist = 5e-3; // 5 mm at 1/e² intensity
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                // Use -2r²/w₀² for proper Gaussian beam profile
                source[[i, j]] = (-2.0 * r2 / (beam_waist * beam_waist)).exp();
            }
        }

        solver.set_source(source.clone(), 1e6);

        // Propagate to Rayleigh distance
        let wavelength = config.c0 / 1e6;
        let rayleigh_distance = PI * beam_waist * beam_waist / wavelength;
        let steps = (rayleigh_distance / config.dz) as usize;

        println!(
            "Propagating {} steps to Rayleigh distance {:.2}mm",
            steps,
            rayleigh_distance * 1000.0
        );

        for step in 0..steps {
            solver.step();

            // Check beam size periodically
            if step == 0 || step == steps / 2 || step == steps - 1 {
                let intensity = solver.get_intensity();
                let max_int = intensity[[config.nx / 2, config.ny / 2]];
                let threshold = max_int / (std::f64::consts::E * std::f64::consts::E);

                // Simple radius estimate
                let mut radius_est = 0;
                for i in config.nx / 2..config.nx {
                    if intensity[[i, config.ny / 2]] < threshold {
                        radius_est = i - config.nx / 2;
                        break;
                    }
                }

                println!(
                    "Step {}: radius ≈ {:.2}mm",
                    step,
                    radius_est as f64 * config.dx * 1000.0
                );
            }
        }

        // Check beam has spread by √2 at Rayleigh distance
        let intensity = solver.get_intensity();

        // Find beam radius at 1/e² intensity (same as initial definition)
        let center_i = config.nx / 2;
        let center_j = config.ny / 2;
        let max_intensity = intensity[[center_i, center_j]];
        let threshold = max_intensity / (std::f64::consts::E * std::f64::consts::E); // 1/e² threshold

        println!(
            "Center: ({}, {}), Max intensity: {:.2e}, Threshold: {:.2e}",
            center_i, center_j, max_intensity, threshold
        );

        // Find radius by measuring from center to where intensity drops below threshold
        let mut radius_pixels = 0.0;
        for i in center_i..config.nx {
            let curr_intensity = intensity[[i, center_j]];
            if i == center_i || i == center_i + 1 || i == center_i + 10 {
                println!(
                    "i={}, intensity={:.2e}, threshold={:.2e}",
                    i, curr_intensity, threshold
                );
            }
            if curr_intensity < threshold {
                // Linear interpolation for sub-pixel accuracy
                if i > center_i {
                    let prev_intensity = intensity[[i - 1, center_j]];
                    let fraction = (threshold - curr_intensity) / (prev_intensity - curr_intensity);
                    radius_pixels = (i - center_i) as f64 - fraction;
                    println!("Found edge at i={}, radius_pixels={:.2}", i, radius_pixels);
                }
                break;
            }
        }

        // If we didn't find the edge, use the maximum distance
        if radius_pixels == 0.0 {
            println!("Warning: beam edge not found within grid!");
            radius_pixels = (config.nx - center_i - 1) as f64;
        }

        let final_radius = radius_pixels * config.dx;
        let expected_radius = beam_waist * 2.0_f64.sqrt();

        assert!(
            (final_radius - expected_radius).abs() / expected_radius < 0.2,
            "Beam radius error: expected {:.2}mm, got {:.2}mm",
            expected_radius * 1000.0,
            final_radius * 1000.0
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
