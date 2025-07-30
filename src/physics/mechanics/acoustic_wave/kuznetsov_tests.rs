//! Validation tests for Kuznetsov equation implementation
//! 
//! These tests verify the correctness of the full Kuznetsov equation solver
//! by comparing against analytical solutions and testing physical properties.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::physics::traits::AcousticWaveModel;
    use ndarray::{Array3, Array4};
    use approx::assert_relative_eq;
    use std::f64::consts::PI;
    
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
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0);
        
        // Initialize with Gaussian pulse
        let mut pressure = Array3::zeros((128, 128, 128));
        let mut velocity = Array4::zeros((3, 128, 128, 128));
        let source = Array3::zeros((128, 128, 128));
        
        let sigma = 0.01; // 10mm width
        let amplitude = 1e6; // 1 MPa
        
        for i in 0..128 {
            for j in 0..128 {
                for k in 0..128 {
                    let x = (i as f64 - 64.0) * grid.dx;
                    let y = (j as f64 - 64.0) * grid.dy;
                    let z = (k as f64 - 64.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    pressure[[i, j, k]] = amplitude * (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        
        let initial_energy: f64 = pressure.iter().map(|&p| p * p).sum();
        
        // Propagate for a few steps
        let dt = 1e-7;
        for _ in 0..10 {
            solver.update_wave(&mut pressure, &mut velocity, &source, &grid, &medium, dt, 0.0).unwrap();
        }
        
        let final_energy: f64 = pressure.iter().map(|&p| p * p).sum();
        
        // Energy should be approximately conserved in linear case without absorption
        assert_relative_eq!(final_energy, initial_energy, epsilon = 0.01 * initial_energy);
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
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0);
        medium.b_a = 5.0; // Typical for water
        
        // Initialize with sinusoidal wave
        let mut pressure = Array3::zeros((256, 64, 64));
        let mut velocity = Array4::zeros((3, 256, 64, 64));
        let source = Array3::zeros((256, 64, 64));
        
        let wavelength = 0.01; // 10mm
        let k_wave = 2.0 * PI / wavelength;
        let amplitude = 1e6; // 1 MPa
        
        for i in 0..256 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = i as f64 * grid.dx;
                    if x < wavelength * 2.0 {
                        pressure[[i, j, k]] = amplitude * (k_wave * x).sin();
                    }
                }
            }
        }
        
        // Store initial waveform
        let initial_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Propagate to observe steepening
        let dt = 5e-8;
        let steps = 100;
        
        for _ in 0..steps {
            solver.update_wave(&mut pressure, &mut velocity, &source, &grid, &medium, dt, 0.0).unwrap();
        }
        
        // Check for harmonic generation (sign of nonlinearity)
        // Compute FFT along x-axis at center
        let mut x_slice = Array1::zeros(256);
        for i in 0..256 {
            x_slice[i] = pressure[[i, 32, 32]];
        }
        
        // Simple harmonic analysis
        let fundamental_freq = 1.0 / wavelength;
        let second_harmonic_present = check_harmonic_content(&x_slice, grid.dx, fundamental_freq, 2.0);
        
        assert!(second_harmonic_present, "Second harmonic should be generated due to nonlinearity");
    }
    
    /// Test acoustic diffusivity (third-order time derivative)
    #[test]
    fn test_acoustic_diffusivity() {
        let grid = Grid::new(128, 64, 64, 1e-4, 1e-3, 1e-3);
        let mut config = KuznetsovConfig::default();
        config.enable_nonlinearity = false;
        config.enable_diffusivity = true;
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5); // Non-zero absorption
        
        // Initialize with high-frequency pulse
        let mut pressure = Array3::zeros((128, 64, 64));
        let mut velocity = Array4::zeros((3, 128, 64, 64));
        let source = Array3::zeros((128, 64, 64));
        
        let sigma = 0.001; // 1mm width (high frequency)
        let amplitude = 1e6;
        
        for i in 0..128 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = (i as f64 - 64.0) * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    pressure[[i, j, k]] = amplitude * (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        
        let initial_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Propagate with diffusivity
        let dt = 1e-8;
        let steps = 50;
        
        // Need to initialize history for third-order derivatives
        for _ in 0..3 {
            solver.update_history(&pressure);
        }
        
        for _ in 0..steps {
            solver.update_wave(&mut pressure, &mut velocity, &source, &grid, &medium, dt, 0.0).unwrap();
        }
        
        let final_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Diffusivity should cause amplitude decay
        assert!(final_max < initial_max * 0.95, "Diffusivity should cause amplitude decay");
    }
    
    /// Test full Kuznetsov equation with all terms
    #[test]
    fn test_full_kuznetsov_equation() {
        let grid = Grid::new(128, 64, 64, 2e-4, 2e-4, 2e-4);
        let config = KuznetsovConfig::default(); // All terms enabled
        
        let mut solver = KuznetsovWave::new(&grid, config);
        let mut medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5);
        medium.b_a = 5.0;
        
        // Initialize test case
        let mut pressure = Array3::zeros((128, 64, 64));
        let mut velocity = Array4::zeros((3, 128, 64, 64));
        let source = Array3::zeros((128, 64, 64));
        
        // Create a focused beam
        for i in 0..128 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = i as f64 * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    
                    // Gaussian beam profile
                    let r2 = y*y + z*z;
                    let beam_width = 0.005; // 5mm
                    let envelope = (-r2 / (beam_width * beam_width)).exp();
                    
                    // Sinusoidal carrier
                    if x < 0.02 { // 20mm extent
                        let k_wave = 2.0 * PI / 0.0015; // 1.5mm wavelength
                        pressure[[i, j, k]] = 2e6 * envelope * (k_wave * x).sin();
                    }
                }
            }
        }
        
        // Propagate
        let dt = 5e-8;
        let steps = 100;
        
        for step in 0..steps {
            let t = step as f64 * dt;
            solver.update_wave(&mut pressure, &mut velocity, &source, &grid, &medium, dt, t).unwrap();
        }
        
        // Verify solution remains stable and physical
        let has_nan = pressure.iter().any(|&p| p.is_nan());
        let has_inf = pressure.iter().any(|&p| p.is_infinite());
        assert!(!has_nan, "Solution should not contain NaN");
        assert!(!has_inf, "Solution should not contain infinity");
        
        let max_pressure = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_pressure < 1e8, "Pressure should remain bounded");
    }
    
    /// Test stability with CFL condition
    #[test]
    fn test_cfl_stability() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = KuznetsovConfig::default();
        let solver = KuznetsovWave::new(&grid, config);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0);
        
        // Test stable timestep
        let dt_stable = 0.3 * grid.dx / 1500.0; // CFL = 0.3
        assert!(solver.check_cfl_condition(&grid, &medium, dt_stable));
        
        // Test unstable timestep
        let dt_unstable = 1.5 * grid.dx / 1500.0; // CFL = 1.5
        assert!(!solver.check_cfl_condition(&grid, &medium, dt_unstable));
    }
    
    /// Helper function to check for harmonic content
    fn check_harmonic_content(signal: &Array1<f64>, dx: f64, fundamental: f64, harmonic: f64) -> bool {
        use crate::fft::fft_1d;
        
        let n = signal.len();
        let fft_result = fft_1d(signal).unwrap();
        
        // Find frequency bins
        let df = 1.0 / (n as f64 * dx);
        let fundamental_bin = (fundamental / df).round() as usize;
        let harmonic_bin = (harmonic * fundamental / df).round() as usize;
        
        if fundamental_bin < n/2 && harmonic_bin < n/2 {
            let fundamental_mag = fft_result[fundamental_bin].norm();
            let harmonic_mag = fft_result[harmonic_bin].norm();
            
            // Check if harmonic is at least 1% of fundamental
            harmonic_mag > 0.01 * fundamental_mag
        } else {
            false
        }
    }
    
    use ndarray::Array1;
    
    /// Test comparison with NonlinearWave in standard mode
    #[test]
    fn test_comparison_with_standard_nonlinear() {
        use crate::physics::mechanics::NonlinearWave;
        
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        // Setup Kuznetsov solver without diffusivity for fair comparison
        let mut kuznetsov_config = KuznetsovConfig::default();
        kuznetsov_config.enable_diffusivity = false;
        kuznetsov_config.time_scheme = TimeIntegrationScheme::Euler; // Simple scheme
        let mut kuznetsov_solver = KuznetsovWave::new(&grid, kuznetsov_config);
        
        // Setup standard NonlinearWave
        let mut standard_solver = NonlinearWave::new(&grid);
        standard_solver.set_nonlinearity_scaling(1.0);
        
        // Same medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0);
        
        // Same initial conditions
        let mut pressure_k = Array3::zeros((64, 64, 64));
        let mut pressure_s = Array3::zeros((64, 64, 64));
        let mut velocity_k = Array4::zeros((3, 64, 64, 64));
        let mut velocity_s = Array4::zeros((3, 64, 64, 64));
        
        // Initialize both with same Gaussian
        for i in 0..64 {
            for j in 0..64 {
                for k in 0..64 {
                    let x = (i as f64 - 32.0) * grid.dx;
                    let y = (j as f64 - 32.0) * grid.dy;
                    let z = (k as f64 - 32.0) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    let value = 1e5 * (-r2 / 0.0001).exp();
                    pressure_k[[i, j, k]] = value;
                    pressure_s[[i, j, k]] = value;
                }
            }
        }
        
        // Propagate both
        let source = Array3::zeros((64, 64, 64));
        let dt = 1e-7;
        
        for _ in 0..5 {
            kuznetsov_solver.update_wave(&mut pressure_k, &mut velocity_k, &source, &grid, &medium, dt, 0.0).unwrap();
            standard_solver.update_wave(&mut pressure_s, &mut velocity_s, &source, &grid, &medium, dt, 0.0).unwrap();
        }
        
        // Results should be similar for small amplitude, short propagation
        let max_diff = pressure_k.iter()
            .zip(pressure_s.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, |acc, x| acc.max(x));
        
        let max_pressure = pressure_k.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Allow for some numerical differences
        assert!(max_diff < 0.1 * max_pressure, 
            "Kuznetsov and standard nonlinear should give similar results for weak nonlinearity");
    }
}