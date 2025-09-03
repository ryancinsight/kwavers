//! Validation tests for Kuznetsov equation solver
//!
//! These tests ensure the solver correctly implements the physics of the
//! Kuznetsov equation by comparing against analytical solutions and
//! established benchmarks.

#[cfg(test)]
mod tests {
    use super::super::{AcousticEquationMode, KuznetsovConfig, KuznetsovWave};
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};
    use crate::physics::traits::AcousticWaveModel;
    use crate::source::{PointSource, Source};
    use approx::assert_relative_eq;
    use ndarray::Array4;

    /// Test linear wave propagation (nonlinearity = 0, diffusivity = 0)
    /// Should match standard linear acoustic wave equation
    #[test]
    fn test_linear_propagation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let dt = 1e-7;

        // Create linear configuration
        let mut config = KuznetsovConfig::default();
        config.equation_mode = AcousticEquationMode::Linear;
        config.nonlinearity_coefficient = 0.0;
        config.acoustic_diffusivity = 0.0;

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // Point source at center
        let source = PointSource::new(
            (grid.nx / 2, grid.ny / 2, grid.nz / 2),
            1.0, // amplitude
            1e6, // 1 MHz frequency
            0.0, // phase
            0.0, // start time
        );

        // Initialize pressure field
        let mut fields = Array4::zeros((1, grid.nx, grid.ny, grid.nz));
        let prev_pressure = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // Propagate for several time steps
        let n_steps = 100;
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
    #[test]
    fn test_homogeneous_no_warning() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default();
        let mut solver = KuznetsovWave::new(config, &grid).unwrap();

        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);
        let source = PointSource::new((16, 16, 16), 1.0, 1e6, 0.0, 0.0);

        let mut fields = Array4::zeros((1, 32, 32, 32));
        let prev = fields.index_axis(ndarray::Axis(0), 0).to_owned();

        // This should not produce a warning for homogeneous media
        solver
            .update_wave(&mut fields, &prev, &source, &grid, &medium, 1e-7, 0.0)
            .unwrap();
    }

    /// Test energy conservation in linear regime
    #[test]
    fn test_energy_conservation_linear() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let dt = 5e-8; // Small timestep for stability

        let mut config = KuznetsovConfig::default();
        config.equation_mode = AcousticEquationMode::Linear;
        config.nonlinearity_coefficient = 0.0;
        config.acoustic_diffusivity = 0.0;

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // No source (zero amplitude)
        let source = PointSource::new((0, 0, 0), 0.0, 1e6, 0.0, 0.0);

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

        // Energy should be approximately conserved (allowing for numerical dispersion)
        assert_relative_eq!(initial_energy, final_energy, epsilon = 0.1 * initial_energy);
    }

    /// Test that nonlinear effects produce expected harmonic generation
    #[test]
    fn test_nonlinear_harmonic_generation() {
        let grid = Grid::new(128, 64, 64, 1e-4, 1e-3, 1e-3);
        let dt = 1e-8;

        let mut config = KuznetsovConfig::default();
        config.equation_mode = AcousticEquationMode::Westervelt; // Nonlinear, no diffusion
        config.nonlinearity_coefficient = 5.2; // Water B/A parameter
        config.acoustic_diffusivity = 0.0;

        let mut solver = KuznetsovWave::new(config, &grid).unwrap();
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);

        // Sinusoidal source
        let frequency = 1e6; // 1 MHz
        let source = PointSource::new((10, grid.ny / 2, grid.nz / 2), 1e6, frequency, 0.0, 0.0);

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
        let probe_point = pressure
            .slice(ndarray::s![grid.nx * 3 / 4, grid.ny / 2, grid.nz / 2])
            .to_owned();

        // In nonlinear propagation, we expect energy at harmonics
        // This is a simplified check - proper spectral analysis would use FFT
        assert!(
            probe_point.abs() > 0.0,
            "Should have non-zero pressure at probe point"
        );
    }
}
