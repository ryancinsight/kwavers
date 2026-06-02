//! Validation tests for Kuznetsov equation solver
//!
//! These tests ensure the solver correctly implements the physics of the
//! Kuznetsov equation by comparing against analytical solutions and
//! established benchmarks.

#[cfg(test)]
mod tests {
    use super::super::{AcousticEquationMode, KuznetsovConfig, KuznetsovWave};
    use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
    use kwavers_core::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use kwavers_domain::grid::Grid;
    use kwavers_domain::medium::HomogeneousMedium;
    use kwavers_domain::source::PointSource;
    use kwavers_physics::traits::AcousticWaveModel;
    use ndarray::Array4;

    /// Test linear wave propagation (nonlinearity = 0, diffusivity = 0)
    /// Should match standard linear acoustic wave equation
    ///
    /// Fast version with reduced grid (16³) and 20 steps for CI/CD (<1s execution).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_linear_propagation() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let dt = 1e-7;

        // Create linear configuration
        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::Linear,
            nonlinearity_coefficient: 0.0,
            acoustic_diffusivity: 0.0,
            ..Default::default()
        };

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // Point source at center
        use kwavers_domain::signal::{Signal, SineWave};
        use std::sync::Arc;
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
        let position = grid.indices_to_coordinates(grid.nx / 2, grid.ny / 2, grid.nz / 2);
        let source = PointSource::new(position, signal);

        // Initialize pressure field
        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let prev_pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // Propagate for fewer time steps for fast validation
        let n_steps = 20;
        for step in 0..n_steps {
            let t = step as f64 * dt;
            solver
                .update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t)
                .unwrap();
        }

        // Check that wave has propagated outward
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        let center_val = pressure[[grid.nx / 2, grid.ny / 2, grid.nz / 2]].abs();
        let edge_val = pressure[[0, grid.ny / 2, grid.nz / 2]].abs();

        // In linear propagation, energy should spread from center
        assert!(center_val > 0.0, "Center should have non-zero pressure");
        assert!(edge_val < center_val, "Wave should decay with distance");
    }

    /// Test that homogeneous media warning is not triggered
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_homogeneous_no_warning() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let config = KuznetsovConfig::default();
        let mut solver = KuznetsovWave::new(config, &grid).unwrap();

        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);
        use kwavers_domain::signal::{Signal, SineWave};
        use std::sync::Arc;
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(MHZ_TO_HZ, 1.0, 0.0));
        let position = grid.indices_to_coordinates(16, 16, 16);
        let source = PointSource::new(position, signal);

        let mut fields = Array4::zeros((1, 32, 32, 32));
        let prev = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // This should not produce a warning for homogeneous media
        solver
            .update_wave(&mut fields, &prev, &source, &grid, &medium, 1e-7, 0.0)
            .unwrap();
    }

    /// Test energy conservation in linear regime (COMPREHENSIVE - Tier 3)
    ///
    /// This test runs 200 iterations on a 64³ grid for thorough validation.
    /// Execution time: >30s, classified as Tier 3 comprehensive validation.
    /// Use `cargo test -- --ignored` for full validation suite.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>30s execution time)"]
    fn test_energy_conservation_linear() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
        let dt = 5e-8; // Small timestep for stability

        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::Linear,
            nonlinearity_coefficient: 0.0,
            acoustic_diffusivity: 0.0,
            ..Default::default()
        };

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // No source (zero amplitude)
        use kwavers_domain::source::NullSource;
        let source = NullSource::new();

        // Initialize with Gaussian pulse
        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);

        // Create initial Gaussian pulse
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        let sigma = 5.0; // grid points
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let r2 = ((i as f64 - center.0 as f64).powi(2)
                        + (j as f64 - center.1 as f64).powi(2)
                        + (k as f64 - center.2 as f64).powi(2))
                        / (sigma * sigma);
                    pressure[[i, j, k]] = (-r2).exp();
                }
            }
        }

        // Compute initial energy
        let initial_energy: f64 = pressure.iter().map(|&p| p * p).sum();

        // Propagate for many steps
        let prev_pressure = pressure.to_owned();
        for step in 0..200 {
            let t = step as f64 * dt;
            solver
                .update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t)
                .unwrap();
        }

        // Compute final energy
        let final_pressure = fields.index_axis(ndarray::Axis(0), 0);
        let final_energy: f64 = final_pressure.iter().map(|&p| p * p).sum();

        // A pressure-only Gaussian with zero initial velocity is NOT a traveling
        // wave — it decomposes into two counter-propagating shells.  As the shells
        // propagate, acoustic energy transfers between the pressure and velocity
        // fields.  For a pure-pressure initial condition, Σ p² (the pressure-
        // squared integral measured here) stabilises at ≈ 50 % of its initial
        // value once the transient phase is complete; the other 50 % resides in
        // the velocity field, which this solver does not expose externally.
        //
        // Σ p² is therefore NOT an acoustic energy invariant for this class of
        // initial conditions.  The correct invariant is
        //   E_total = ∫ (p²/(ρc²) + ρu²) dV,
        // which requires access to the velocity field.
        //
        // Acceptance bounds [0.30, 0.75] × initial:
        //   - Lower: well below 0.5 indicates spurious numerical dissipation.
        //   - Upper: above 0.75 indicates numerical instability (energy growth).
        assert!(
            final_energy >= 0.30 * initial_energy,
            "p² integral vanished: solver is numerically dissipating energy; \
             final/initial = {:.3}",
            final_energy / initial_energy
        );
        assert!(
            final_energy <= 0.75 * initial_energy,
            "p² integral too large: traveling-wave energy partition not reached \
             or solver is amplifying; final/initial = {:.3}",
            final_energy / initial_energy
        );
    }

    /// Test energy conservation in linear regime (FAST - Tier 1)
    ///
    /// Fast version with reduced grid (16³) and fewer steps (20) for CI/CD.
    /// Execution time: <1s, classified as Tier 1 fast validation.
    ///
    /// Note: This is a smoke test to verify basic solver functionality.
    /// Comprehensive energy conservation validation is in the ignored test.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_energy_conservation_linear_fast() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
        let dt = 5e-8; // Small timestep for stability

        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::Linear,
            nonlinearity_coefficient: 0.0,
            acoustic_diffusivity: 0.0,
            ..Default::default()
        };

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // No source (zero amplitude)
        use kwavers_domain::source::NullSource;
        let source = NullSource::new();

        // Initialize with Gaussian pulse
        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let mut pressure = fields.index_axis_mut(ndarray::Axis(0), 0);

        // Create initial Gaussian pulse
        let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
        let sigma = 2.0; // grid points (smaller for smaller grid)
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let r2 = ((i as f64 - center.0 as f64).powi(2)
                        + (j as f64 - center.1 as f64).powi(2)
                        + (k as f64 - center.2 as f64).powi(2))
                        / (sigma * sigma);
                    pressure[[i, j, k]] = (-r2).exp();
                }
            }
        }

        // Compute initial energy
        let initial_energy: f64 = pressure.iter().map(|&p| p * p).sum();

        // Propagate for fewer steps for fast validation
        let prev_pressure = pressure.to_owned();
        for step in 0..20 {
            let t = step as f64 * dt;
            solver
                .update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t)
                .unwrap();
        }

        // Compute final energy
        let final_pressure = fields.index_axis(ndarray::Axis(0), 0);
        let final_energy: f64 = final_pressure.iter().map(|&p| p * p).sum();

        // Verify solver ran without panicking and energy is in reasonable range
        // For fast test with reduced grid, we just check energy didn't explode or vanish
        assert!(final_energy > 0.0, "Energy should be positive");
        assert!(
            final_energy < 100.0 * initial_energy,
            "Energy shouldn't explode"
        );
    }

    /// Test that nonlinear effects produce expected harmonic generation (COMPREHENSIVE - Tier 3)
    ///
    /// This test runs 1000 iterations on a 128×64×64 grid for thorough validation.
    /// Execution time: >60s, classified as Tier 3 comprehensive validation.
    /// Use `cargo test -- --ignored` for full validation suite.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>60s execution time)"]
    fn test_nonlinear_harmonic_generation() {
        let grid = Grid::new(128, 64, 64, 1e-4, 1e-3, 1e-3).unwrap();
        let dt = 1e-8;

        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::Westervelt, // Nonlinear, no diffusion
            nonlinearity_coefficient: 5.2,                   // Water B/A parameter
            acoustic_diffusivity: 0.0,
            ..Default::default()
        };

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // Sinusoidal source
        let frequency = MHZ_TO_HZ; // 1 MHz
        use kwavers_domain::signal::{Signal, SineWave};
        use std::sync::Arc;
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, MPA_TO_PA, 0.0));
        let position = grid.indices_to_coordinates(10, grid.ny / 2, grid.nz / 2);
        let source = PointSource::new(position, signal);

        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let prev_pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // Propagate to allow nonlinear effects to develop
        let n_steps = 1000;
        for step in 0..n_steps {
            let t = step as f64 * dt;
            solver
                .update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t)
                .unwrap();
        }

        // Analyze spectrum at a propagated distance
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        let probe_value = pressure[[grid.nx * 3 / 4, grid.ny / 2, grid.nz / 2]];

        // Nonlinear propagation produces energy at harmonics
        // **Test scope**: Validates non-zero pressure propagation (functional test)
        // Full spectral validation requires FFT harmonic analysis (see integration tests)
        assert!(
            probe_value.abs() > 0.0,
            "Should have non-zero pressure at probe point"
        );
    }

    /// Test that nonlinear effects produce expected harmonic generation (FAST - Tier 1)
    ///
    /// Fast version with reduced grid (32×16×16) and fewer steps (50) for CI/CD.
    /// Execution time: <2s, classified as Tier 1 fast validation.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_nonlinear_harmonic_generation_fast() {
        let grid = Grid::new(32, 16, 16, 1e-4, 1e-3, 1e-3).unwrap();
        let dt = 1e-8;

        let config = KuznetsovConfig {
            equation_mode: AcousticEquationMode::Westervelt, // Nonlinear, no diffusion
            nonlinearity_coefficient: 5.2,                   // Water B/A parameter
            acoustic_diffusivity: 0.0,
            ..Default::default()
        };

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // Sinusoidal source
        let frequency = MHZ_TO_HZ; // 1 MHz
        use kwavers_domain::signal::{Signal, SineWave};
        use std::sync::Arc;
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, MPA_TO_PA, 0.0));
        let position = grid.indices_to_coordinates(5, grid.ny / 2, grid.nz / 2);
        let source = PointSource::new(position, signal);

        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let prev_pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // Propagate fewer steps for fast validation
        let n_steps = 50;
        for step in 0..n_steps {
            let t = step as f64 * dt;
            solver
                .update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t)
                .unwrap();
        }

        // Analyze spectrum at a propagated distance
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        let probe_value = pressure[[grid.nx * 3 / 4, grid.ny / 2, grid.nz / 2]];

        // Nonlinear propagation produces energy at harmonics
        // **Test scope**: Validates non-zero pressure propagation (functional fast test)
        // Full spectral validation requires FFT harmonic analysis (see integration tests)
        assert!(
            probe_value.abs() > 0.0,
            "Should have non-zero pressure at probe point"
        );
    }
}
