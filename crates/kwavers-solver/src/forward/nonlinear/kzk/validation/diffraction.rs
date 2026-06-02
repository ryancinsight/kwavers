//! Diffraction validation tests for KZK equation implementation.

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;
    use crate::forward::nonlinear::kzk::constants::*;
    use ndarray::Array2;
    use std::f64::consts::PI;

    /// Test linear propagation of Gaussian beam (COMPREHENSIVE - Tier 3)
    /// Should maintain Gaussian profile with known spreading
    ///
    /// This test propagates to full Rayleigh distance with default grid.
    /// Execution time: >60s, classified as Tier 3 comprehensive validation.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>60s execution time)"]
    fn test_gaussian_beam_diffraction() {
        let config = KZKConfig {
            nx: DEFAULT_GRID_SIZE,
            ny: DEFAULT_GRID_SIZE,
            nz: 100,
            nt: 50,
            dx: DEFAULT_DX,
            dz: DEFAULT_DZ,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        let beam_waist = DEFAULT_BEAM_WAIST;
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                source[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        solver.set_source(source.clone(), DEFAULT_FREQUENCY);

        let wavelength = config.c0 / DEFAULT_FREQUENCY;
        let rayleigh_distance = PI * beam_waist * beam_waist / wavelength;
        let steps = (rayleigh_distance / config.dz) as usize;

        println!(
            "Propagating {} steps to Rayleigh distance {:.2}mm",
            steps,
            rayleigh_distance * 1000.0
        );

        for step in 0..steps {
            solver.step();

            if step == 0 || step == steps / 2 || step == steps - 1 {
                let intensity = solver.get_intensity();
                let max_int = intensity[[config.nx / 2, config.ny / 2]];
                let threshold = max_int / (std::f64::consts::E * std::f64::consts::E);

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

        let intensity = solver.get_intensity();
        let center_i = config.nx / 2;
        let center_j = config.ny / 2;
        let max_intensity = intensity[[center_i, center_j]];
        let threshold = max_intensity / (std::f64::consts::E * std::f64::consts::E);

        println!(
            "Center: ({}, {}), Max intensity: {:.2e}, Threshold: {:.2e}",
            center_i, center_j, max_intensity, threshold
        );

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
                if i > center_i {
                    let prev_intensity = intensity[[i - 1, center_j]];
                    let fraction = (threshold - curr_intensity) / (prev_intensity - curr_intensity);
                    radius_pixels = (i - center_i) as f64 - fraction;
                    println!("Found edge at i={}, radius_pixels={:.2}", i, radius_pixels);
                }
                break;
            }
        }

        if radius_pixels == 0.0 {
            println!("Warning: beam edge not found within grid!");
            radius_pixels = (config.nx - center_i - 1) as f64;
        }

        let final_radius = radius_pixels * config.dx;
        let expected_radius = beam_waist * 2.0_f64.sqrt();

        assert!(
            (final_radius - expected_radius).abs() / expected_radius < 0.35,
            "Beam radius error: expected {:.2}mm, got {:.2}mm (within 35% tolerance for numerical diffusion)",
            expected_radius * 1000.0,
            final_radius * 1000.0
        );
    }

    /// Test linear propagation of Gaussian beam (FAST - Tier 1)
    /// Fast version with reduced grid and fewer steps for CI/CD.
    /// Execution time: <2s, classified as Tier 1 fast validation.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_gaussian_beam_diffraction_fast() {
        let config = KZKConfig {
            nx: 32,
            ny: 32,
            nz: 20,
            nt: 16,
            dx: 2e-4,
            dz: 1e-3,
            dt: 1e-8,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        let beam_waist = 1e-3;
        let mut source = Array2::zeros((config.nx, config.ny));

        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x * x + y * y;
                source[[i, j]] = (-r2 / (beam_waist * beam_waist)).exp();
            }
        }

        solver.set_source(source, MHZ_TO_HZ);

        for _ in 0..3 {
            solver.step();
        }

        let intensity = solver.get_intensity();
        let center = intensity[[config.nx / 2, config.ny / 2]];
        assert!(center > 0.0, "intensity must be positive at beam center");

        // Gaussian beam property: peak at center, monotone decay radially.
        // After only 3 propagation steps on a 32×32 grid the diffraction spreading
        // is far less than the beam waist, so the on-axis maximum must exceed the
        // intensity at the domain corner.
        let corner = intensity[[0, 0]];
        assert!(
            center > corner,
            "beam center intensity ({center:.3e}) must exceed corner ({corner:.3e}): \
             Gaussian beam must be peaked on-axis"
        );
    }
}
