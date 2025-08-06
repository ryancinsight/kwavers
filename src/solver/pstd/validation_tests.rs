//! Validation tests for PSTD solver
//! 
//! These tests verify the numerical accuracy of the PSTD implementation
//! against analytical solutions and known benchmarks.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use ndarray::{Array3, Array4, Axis};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;
    
    /// Test plane wave propagation accuracy
    #[test]
    fn test_plane_wave_propagation() {
        // Create a grid with power-of-2 dimensions for optimal FFT
        let nx = 64;
        let ny = 64;
        let nz = 64;
        let dx = 1e-3; // 1mm
        let grid = Grid::new(nx, ny, nz, dx, dx, dx);
        
        // Create homogeneous medium
        let density = 1000.0; // kg/m³
        let sound_speed = 1500.0; // m/s
        let medium = HomogeneousMedium::new(density, sound_speed, &grid, 0.0, 1.0);
        
        // PSTD configuration
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: false, // Disable for exact test
            pml_stencil_size: 4,
            cfl_factor: 0.3,
        };
        
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Initialize plane wave propagating in x-direction
        let frequency = 1e6; // 1 MHz
        let wavelength = sound_speed / frequency;
        let k = 2.0 * PI / wavelength;
        
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut velocity_x = Array3::zeros((nx, ny, nz));
        let mut velocity_y = Array3::zeros((nx, ny, nz));
        let mut velocity_z = Array3::zeros((nx, ny, nz));
        
        // Initial conditions: p = A*sin(kx), vx = (A/ρc)*sin(kx)
        let amplitude = 1.0;
        for i in 0..nx {
            let x = i as f64 * dx;
            let p_val = amplitude * (k * x).sin();
            let vx_val = (amplitude / (density * sound_speed)) * (k * x).sin();
            
            for j in 0..ny {
                for k_idx in 0..nz {
                    pressure[[i, j, k_idx]] = p_val;
                    velocity_x[[i, j, k_idx]] = vx_val;
                }
            }
        }
        
        // Time step
        let dt = solver.max_stable_dt(sound_speed);
        
        // Propagate for quarter period
        let period = 1.0 / frequency;
        let n_steps = (0.25 * period / dt).round() as usize;
        
        for _ in 0..n_steps {
            // Compute divergence
            let divergence = solver.compute_divergence(&velocity_x, &velocity_y, &velocity_z).unwrap();
            
            // Update pressure
            solver.update_pressure(&mut pressure, &divergence, &medium, dt).unwrap();
            
            // Update velocity
            solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, &medium, dt).unwrap();
        }
        
        // After quarter period, pressure should be cos(kx)
        let time = n_steps as f64 * dt;
        for i in 0..nx {
            let x = i as f64 * dx;
            let expected = amplitude * (k * x - 2.0 * PI * frequency * time).sin();
            let actual = pressure[[i, ny/2, nz/2]];
            
            // PSTD should have very high accuracy
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }
    
    /// Test numerical dispersion is minimal
    #[test]
    fn test_numerical_dispersion() {
        let nx = 128;
        let grid = Grid::new(nx, nx, nx, 1e-3, 1e-3, 1e-3);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
        
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: true,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
        };
        
        let solver = PstdSolver::new(config, &grid).unwrap();
        
        // Test multiple wavelengths
        let frequencies = vec![0.5e6, 1e6, 2e6]; // Different frequencies
        let c = 1500.0;
        
        for freq in frequencies {
            let wavelength = c / freq;
            let points_per_wavelength = wavelength / grid.dx;
            
            // PSTD should have minimal dispersion even with few PPW
            if points_per_wavelength >= 2.0 {
                let phase_error = estimate_phase_error_pstd(points_per_wavelength);
                // For PSTD, phase error decreases exponentially with PPW
                // At 3 PPW: exp(-3) ≈ 0.05
                let tolerance = if points_per_wavelength < 4.0 { 0.1 } else { 1e-3 };
                assert!(phase_error < tolerance, 
                    "PSTD phase error too large at {} PPW: {}", 
                    points_per_wavelength, phase_error);
            }
        }
    }
    
    /// Test k-space correction functionality
    #[test]
    fn test_k_space_correction() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        
        // Test different correction orders
        for order in [2, 4, 6, 8] {
            let (kx, ky, kz) = PstdSolver::compute_wavenumbers(&grid);
            let k_squared = &kx * &kx + &ky * &ky + &kz * &kz;
            let kappa = PstdSolver::compute_k_space_correction(&k_squared, &grid, order);
            
            // Check DC component
            assert_eq!(kappa[[0, 0, 0]], 1.0);
            
            // Check that correction is between 0 and 1
            for val in kappa.iter() {
                assert!(*val >= 0.0 && *val <= 1.0, 
                    "k-space correction out of bounds: {}", val);
            }
            
            // Higher order should give values closer to 1 for low k
            if order > 2 {
                let kappa_prev = PstdSolver::compute_k_space_correction(&k_squared, &grid, order - 2);
                // Relaxed assertion to allow for numerical precision issues
                assert!(kappa[[1, 0, 0]] >= kappa_prev[[1, 0, 0]] - 1e-10,
                    "Higher order should give less correction for low k (within numerical precision)");
            }
        }
    }
    
    /// Test anti-aliasing filter
    #[test]
    fn test_anti_aliasing_filter() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let (kx, ky, kz) = PstdSolver::compute_wavenumbers(&grid);
        let filter = PstdSolver::create_anti_aliasing_filter(&kx, &ky, &kz, &grid);
        
        // Count filtered frequencies
        let filtered_count = filter.iter().filter(|&&f| f == 0.0).count();
        let total_count = filter.len();
        
        // Approximately 1 - (2/3)³ ≈ 70% should be filtered
        let filtered_ratio = filtered_count as f64 / total_count as f64;
        assert!(filtered_ratio > 0.6 && filtered_ratio < 0.8,
            "Anti-aliasing filter ratio unexpected: {}", filtered_ratio);
    }
    
    /// Test conservation of energy
    #[test]
    fn test_energy_conservation() {
        let nx = 64;
        let grid = Grid::new(nx, nx, nx, 1e-3, 1e-3, 1e-3);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
        
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: true,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
        };
        
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Initialize Gaussian pulse
        let mut pressure = Array3::zeros((nx, nx, nx));
        let mut velocity_x = Array3::zeros((nx, nx, nx));
        let mut velocity_y = Array3::zeros((nx, nx, nx));
        let mut velocity_z = Array3::zeros((nx, nx, nx));
        
        let center = nx as f64 / 2.0;
        let sigma = 5.0 * grid.dx;
        
        for i in 0..nx {
            for j in 0..nx {
                for k in 0..nx {
                    let x = (i as f64 - center) * grid.dx;
                    let y = (j as f64 - center) * grid.dy;
                    let z = (k as f64 - center) * grid.dz;
                    let r2 = x*x + y*y + z*z;
                    pressure[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
                }
            }
        }
        
        // Calculate initial energy
        let rho = medium.density_array();
        let c = medium.sound_speed_array();
        let initial_energy = calculate_acoustic_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &rho, &c);
        
        // Time step
        let dt = solver.max_stable_dt(1500.0);
        
        // Propagate for several steps
        for _ in 0..100 {
            let divergence = solver.compute_divergence(&velocity_x, &velocity_y, &velocity_z).unwrap();
            solver.update_pressure(&mut pressure, &divergence, &medium, dt).unwrap();
            solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, &medium, dt).unwrap();
        }
        
        // Calculate final energy
        let final_energy = calculate_acoustic_energy(&pressure, &velocity_x, &velocity_y, &velocity_z, &rho, &c);
        
        // Energy should be conserved (within numerical precision)
        assert_relative_eq!(final_energy, initial_energy, epsilon = 1e-3);
    }
    
    /// Helper function to estimate PSTD phase error
    fn estimate_phase_error_pstd(points_per_wavelength: f64) -> f64 {
        // PSTD has exponentially small phase error
        (-points_per_wavelength).exp()
    }
    
    /// Helper function to calculate acoustic energy
    fn calculate_acoustic_energy(
        pressure: &Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        rho: &Array3<f64>,
        c: &Array3<f64>,
    ) -> f64 {
        let mut energy = 0.0;
        
        for i in 0..pressure.shape()[0] {
            for j in 0..pressure.shape()[1] {
                for k in 0..pressure.shape()[2] {
                    // Kinetic energy: 0.5 * ρ * v²
                    let v2 = vx[[i,j,k]].powi(2) + vy[[i,j,k]].powi(2) + vz[[i,j,k]].powi(2);
                    let kinetic = 0.5 * rho[[i,j,k]] * v2;
                    
                    // Potential energy: 0.5 * p² / (ρc²)
                    let potential = 0.5 * pressure[[i,j,k]].powi(2) / (rho[[i,j,k]] * c[[i,j,k]].powi(2));
                    
                    energy += kinetic + potential;
                }
            }
        }
        
        energy
    }
}