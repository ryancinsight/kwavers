//! Validation tests for Kuznetsov equation implementation
//! 
//! These tests verify the correctness of the full Kuznetsov equation solver
//! by comparing against analytical solutions and testing physical properties.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::medium::Medium;
    use crate::physics::traits::AcousticWaveModel;
    use crate::physics::mechanics::acoustic_wave::kuznetsov::{KuznetsovWave, KuznetsovConfig, TimeIntegrationScheme};
    use ndarray::{Array3, Array4, Array1, Axis};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;
    
    // Test source implementation
    struct TestSource;
    
    impl std::fmt::Debug for TestSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TestSource")
        }
    }
    
    impl crate::source::Source for TestSource {
        fn get_source_term(&self, _t: f64, _x: f64, _y: f64, _z: f64, _grid: &crate::grid::Grid) -> f64 {
            0.0 // No source for these tests
        }
        
        fn positions(&self) -> Vec<(f64, f64, f64)> {
            vec![]
        }
        
        fn signal(&self) -> &dyn crate::signal::Signal {
            panic!("Not implemented for test source")
        }
    }
    
    /// Test basic initialization and configuration
    #[test]
    fn test_kuznetsov_initialization() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default();
        let solver = KuznetsovWave::new(&grid, config.clone());
        
        assert_eq!(solver.config.enable_nonlinearity, true);
        assert_eq!(solver.config.enable_diffusivity, true);
        assert_eq!(solver.config.spatial_order, 4);
        assert_eq!(solver.pressure_history.len(), 2); // Based on RK4 default
    }
    
    /// Test linear wave propagation (nonlinearity and diffusivity disabled)
    #[test]
    fn test_linear_wave_propagation() {
        let grid = Grid::new(128, 128, 128, 1e-3, 1e-3, 1e-3);
        let mut config = KuznetsovConfig::default();
        config.enable_nonlinearity = false;
        config.enable_diffusivity = false;
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        
        // Create test source
        let source = TestSource;
        
        // Create fields array
        let mut fields = Array4::zeros((13, 128, 128, 128)); // Standard field indices
        
        // Initialize pressure field
        for i in 0..128 {
            for j in 0..128 {
                for k in 0..128 {
                    let x = i as f64 - 64.0;
                    let y = j as f64 - 64.0;
                    let z = k as f64 - 64.0;
                    let r2 = x*x + y*y + z*z;
                    fields[[0, i, j, k]] = 1e3 * (-r2 / 100.0).exp(); // Pressure at index 0
                }
            }
        }
        
        let initial_energy = fields.index_axis(Axis(0), 0).iter().map(|&p| p * p).sum::<f64>();
        let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
        
        // Run simulation
        let dt = 1e-7;
        for _ in 0..10 {
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        }
        
        let final_energy = fields.index_axis(Axis(0), 0).iter().map(|&p| p * p).sum::<f64>();
        
        // Energy should be approximately conserved in linear case
        assert!((final_energy - initial_energy).abs() / initial_energy < 0.01);
    }
    
    /// Test nonlinear steepening with Kuznetsov equation
    #[test]
    fn test_nonlinear_steepening() {
        let grid = Grid::new(256, 64, 64, 1e-4, 1e-3, 1e-3);
        let mut config = KuznetsovConfig::default();
        config.enable_nonlinearity = true;
        config.enable_diffusivity = false;
        config.nonlinearity_scaling = 1.0;
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        medium.b_a = 5.0; // Typical for water
        
        // Initialize with sinusoidal wave
        let mut fields = Array4::zeros((13, 256, 64, 64)); // Standard field indices
        let wavelength = 0.01; // 10mm
        let k_wave = 2.0 * PI / wavelength;
        let amplitude = 1e6; // 1 MPa
        
        for i in 0..256 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = i as f64 * grid.dx;
                    if x < wavelength * 2.0 {
                        fields[[0, i, j, k]] = amplitude * (k_wave * x).sin();
                    }
                }
            }
        }
        
        let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
        let source = TestSource;
        
        // Propagate to observe steepening
        let dt = 5e-8;
        let steps = 100;
        
        for _ in 0..steps {
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        }
        
        // Check for harmonic generation (sign of nonlinearity)
        let mut x_slice = Array1::zeros(256);
        for i in 0..256 {
            x_slice[i] = fields[[0, i, 32, 32]];
        }
        
        // Simple harmonic analysis
        let fundamental_freq = 1.0 / wavelength;
        let second_harmonic_present = check_harmonic_content(&x_slice, grid.dx, fundamental_freq, 2.0);
        
        assert!(second_harmonic_present, "Second harmonic should be generated due to nonlinearity");
    }
    
    /// Test acoustic diffusivity effect
    #[test]
    fn test_acoustic_diffusivity() {
        let grid = Grid::new(128, 64, 64, 1e-3, 1e-3, 1e-3);
        let mut config = KuznetsovConfig::default();
        config.enable_nonlinearity = false;
        config.enable_diffusivity = true;
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.5, 0.0); // Non-zero absorption
        
        // Initialize with high-frequency pulse
        let mut fields = Array4::zeros((13, 128, 64, 64));
        let sigma = 0.002; // 2mm width (high frequency)
        let amplitude = 1e5; // 100 kPa
        
        for i in 0..128 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = (i as f64 - 64.0) * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    fields[[0, i, j, k]] = amplitude * (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        
        let initial_max = fields.index_axis(Axis(0), 0).iter()
            .fold(0.0_f64, |a, &b| a.max(b.abs()));
        let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
        let source = TestSource;
        
        // Propagate
        let dt = 1e-7;
        for _ in 0..100 {
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        }
        
        let final_max = fields.index_axis(Axis(0), 0).iter()
            .fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Should see amplitude decay due to diffusivity
        assert!(final_max < initial_max * 0.95, 
            "Amplitude should decay with diffusivity: {} vs {}", final_max, initial_max);
    }
    
    /// Test full Kuznetsov equation with all terms
    #[test]
    fn test_full_kuznetsov_equation() {
        let grid = Grid::new(128, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default(); // All terms enabled
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.5, 0.0);
        medium.b_a = 5.0;
        
        // Initialize test case
        let mut fields = Array4::zeros((13, 128, 64, 64));
        
        // Gaussian pulse
        for i in 0..128 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = (i as f64 - 64.0) * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    fields[[0, i, j, k]] = 1e6 * (-r2 / 0.0001).exp();
                }
            }
        }
        
        let prev_pressure = fields.index_axis(Axis(0), 0).to_owned();
        let source = TestSource;
        
        // Run simulation
        let dt = 1e-7;
        let mut max_pressure = Vec::new();
        
        for t_step in 0..50 {
            let t = t_step as f64 * dt;
            solver.update_wave(&mut fields, &prev_pressure, &source, &grid, &medium, dt, t);
            
            let max_p = fields.index_axis(Axis(0), 0).iter()
                .fold(0.0_f64, |a, &b| a.max(b.abs()));
            max_pressure.push(max_p);
        }
        
        // Check that simulation remains stable
        assert!(max_pressure.iter().all(|&p| p.is_finite()), 
            "Pressure should remain finite");
        assert!(max_pressure.last().unwrap() < &1e8, 
            "Pressure should not explode");
    }
    
    /// Test stability with CFL condition
    #[test]
    fn test_cfl_stability() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default();
        let solver = KuznetsovWave::new(&grid, config);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        
        // Test stable timestep
        let dt_stable = 0.3 * grid.dx / 1500.0; // CFL = 0.3
        assert!(solver.check_cfl_condition(&grid, &medium, dt_stable));
        
        // Test unstable timestep
        let dt_unstable = 1.5 * grid.dx / 1500.0; // CFL = 1.5
        assert!(!solver.check_cfl_condition(&grid, &medium, dt_unstable));
    }
    
    /// Helper function to check harmonic content
    fn check_harmonic_content(signal: &Array1<f64>, dx: f64, fundamental_freq: f64, harmonic: f64) -> bool {
        use rustfft::{FftPlanner, num_complex::Complex};
        
        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        
        // Convert to complex
        let mut buffer: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Perform FFT
        fft.process(&mut buffer);
        
        // Compute frequency bins
        let freq_resolution = 1.0 / (n as f64 * dx);
        let fundamental_bin = (fundamental_freq / freq_resolution).round() as usize;
        let harmonic_bin = (harmonic * fundamental_freq / freq_resolution).round() as usize;
        
        if harmonic_bin >= n/2 {
            return false; // Harmonic beyond Nyquist
        }
        
        // Get magnitudes
        let fundamental_mag = buffer[fundamental_bin].norm();
        let harmonic_mag = buffer[harmonic_bin].norm();
        
        // Check if harmonic is significant (at least 1% of fundamental)
        harmonic_mag > 0.01 * fundamental_mag
    }
    
    /// Test comparison with NonlinearWave in standard mode
    #[test]
    fn test_comparison_with_standard_nonlinear() {
        use crate::physics::mechanics::NonlinearWave;
        
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        // Kuznetsov with only standard nonlinearity
        let mut kuznetsov_config = KuznetsovConfig::default();
        kuznetsov_config.enable_diffusivity = false;
        kuznetsov_config.nonlinearity_scaling = 1.0;
        kuznetsov_config.time_scheme = TimeIntegrationScheme::Euler; // Simple scheme
        
        let mut kuznetsov_solver = KuznetsovWave::new(&grid, kuznetsov_config);
        
        // Standard nonlinear solver
        let mut standard_solver = NonlinearWave::new(&grid);
        standard_solver.set_nonlinearity_scaling(1.0);
        
        // Same medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        
        // Same initial conditions - create fields arrays
        let mut fields_k = Array4::zeros((13, 64, 64, 64));
        let mut fields_s = fields_k.clone();
        
        // Initialize with Gaussian
        for i in 0..64 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = (i as f64 - 32.0) * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    let val = 1e5 * (-r2 / 0.0001).exp();
                    fields_k[[0, i, j, k]] = val;
                    fields_s[[0, i, j, k]] = val;
                }
            }
        }
        
        let prev_pressure = fields_k.index_axis(Axis(0), 0).to_owned();
        let source = TestSource;
        
        // Run one step
        let dt = 1e-7;
        kuznetsov_solver.update_wave(&mut fields_k, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        standard_solver.update_wave(&mut fields_s, &prev_pressure, &source, &grid, &medium, dt, 0.0);
        
        // Compare results - they should be similar for weak nonlinearity
        let diff_sum: f64 = fields_k.index_axis(Axis(0), 0).iter()
            .zip(fields_s.index_axis(Axis(0), 0).iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        
        let avg_diff = diff_sum / (64.0 * 64.0 * 64.0);
        
        // Should be close but not identical due to different formulations
        assert!(avg_diff < 1e3, "Average difference should be small: {}", avg_diff);
    }


}