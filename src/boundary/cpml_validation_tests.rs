//! Validation tests for Convolutional PML implementation
//! 
//! These tests verify the performance and correctness of the C-PML
//! boundary conditions, especially for grazing angle absorption.

#[cfg(test)]
mod tests {
    use crate::boundary::{CPMLBoundary, CPMLConfig, Boundary};
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::solver::cpml_integration::CPMLSolver;
    use ndarray::{Array3, Array4};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;
    
    /// Test plane wave absorption at various angles
    #[test]
    fn test_plane_wave_absorption() {
        let grid = Grid::new(128, 128, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Create a Gaussian pulse
        let cx = grid.nx / 2;
        let cy = grid.ny / 2;
        let cz = grid.nz / 2;
        let width = 5.0 * grid.dx;
        let mut field = create_gaussian_pulse(&grid, cx, cy, cz, width);
        
        // Apply CPML multiple times
        let initial_max = field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        
        for _ in 0..10 {
            cpml.apply_acoustic(&mut field, &grid, 0).unwrap();
        }
        
        let final_max = field.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        
        println!("Initial max: {:.2e}, Final max: {:.2e}", initial_max, final_max);
        
        // Check that the field hasn't grown (basic stability check)
        assert!(final_max <= initial_max * 1.1, 
            "Field amplitude grew too much: initial={:.2e}, final={:.2e}", 
            initial_max, final_max);
        
        // Check that absorption occurred in the boundary regions
        let thickness = cpml.config.thickness;
        let mut boundary_sum = 0.0;
        let mut interior_sum = 0.0;
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let val = field[[i, j, k]].abs();
                    if i < thickness || i >= grid.nx - thickness ||
                       j < thickness || j >= grid.ny - thickness ||
                       k < thickness || k >= grid.nz - thickness {
                        boundary_sum += val;
                    } else {
                        interior_sum += val;
                    }
                }
            }
        }
        
        // Boundary should have lower average amplitude due to absorption
        let boundary_cells = (grid.nx * grid.ny * grid.nz) - 
                           ((grid.nx - 2*thickness) * (grid.ny - 2*thickness) * (grid.nz - 2*thickness));
        let interior_cells = (grid.nx - 2*thickness) * (grid.ny - 2*thickness) * (grid.nz - 2*thickness);
        
        let boundary_avg = boundary_sum / boundary_cells as f64;
        let interior_avg = interior_sum / interior_cells as f64;
        
        println!("Boundary avg: {:.2e}, Interior avg: {:.2e}", boundary_avg, interior_avg);
        
        // The boundary average should be less than interior (absorption effect)
        assert!(boundary_avg < interior_avg * 0.95, 
            "Boundary absorption not effective: boundary_avg={:.2e}, interior_avg={:.2e}", 
            boundary_avg, interior_avg);
    }
    
    /// Test grazing angle performance
    #[test]
    fn test_grazing_angle_performance() {
        let grid = Grid::new(150, 150, 150, 1e-3, 1e-3, 1e-3);
        
        // Standard config
        let standard_config = CPMLConfig::default();
        let mut standard_cpml = CPMLBoundary::new(standard_config, &grid).unwrap();
        
        // Grazing angle optimized config
        let grazing_config = CPMLConfig::for_grazing_angles();
        let mut grazing_cpml = CPMLBoundary::new(grazing_config, &grid).unwrap();
        
        // Create near-grazing wave (85°)
        let mut field_standard = create_plane_wave(&grid, 85.0, 1e6);
        let mut field_grazing = field_standard.clone();
        let initial_energy = compute_field_energy(&field_standard);
        
        // Apply boundaries
        for _ in 0..30 {
            standard_cpml.apply_acoustic(&mut field_standard, &grid, 0).unwrap();
            grazing_cpml.apply_acoustic(&mut field_grazing, &grid, 0).unwrap();
        }
        
        let standard_reflection = compute_field_energy(&field_standard) / initial_energy;
        let grazing_reflection = compute_field_energy(&field_grazing) / initial_energy;
        
        println!("Standard C-PML at 85°: {:.2e}", standard_reflection);
        println!("Grazing-optimized C-PML at 85°: {:.2e}", grazing_reflection);
        println!("Improvement factor: {:.2}x", standard_reflection / grazing_reflection);
        
        // Grazing-optimized should perform better (at least 2x improvement)
        assert!(grazing_reflection < 0.5 * standard_reflection,
            "Grazing-optimized C-PML should perform better at grazing angles");
    }
    
    /// Test memory variable update consistency
    #[test]
    fn test_memory_variable_consistency() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Create test gradient
        let mut gradient = Array3::ones((64, 64, 64));
        
        // Update memory variables multiple times
        for _ in 0..10 {
            cpml.update_acoustic_memory(&gradient, 0).unwrap();
        }
        
        // Memory variables should converge to steady state
        let psi = &cpml.psi_acoustic;
        let psi_x = psi.index_axis(ndarray::Axis(0), 0);
        
        // Check that memory variables are bounded
        for val in psi_x.iter() {
            assert!(val.is_finite(), "Memory variable contains non-finite value");
            assert!(val.abs() < 100.0, "Memory variable diverging");
        }
    }
    
    /// Test C-PML solver integration
    #[test]
    fn test_cpml_solver_integration() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let mut solver = CPMLSolver::new(config, &grid).unwrap();
        
        // Create medium
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 0.0);
        
        // Initialize fields
        let mut pressure = create_gaussian_pulse(&grid, 16, 16, 16, 5.0);
        let mut velocity = Array4::zeros((3, 32, 32, 32));
        
        let initial_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Run simulation with C-PML
        let dt = 1e-7;
        for step in 0..30 {
            solver.update_acoustic_field(&mut pressure, &mut velocity, &grid, &medium, dt, step).unwrap();
        }
        
        let final_max = pressure.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        
        // Field should be attenuated but stable
        assert!(final_max < initial_max, "Field should be attenuated");
        assert!(final_max > 0.0, "Field should not completely vanish");
        assert!(pressure.iter().all(|&p| p.is_finite()), "Field contains non-finite values");
    }
    
    /// Test reflection coefficient estimation
    #[test]
    fn test_reflection_estimation() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::for_grazing_angles();
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Test reflection estimates
        let r_0 = cpml.estimate_reflection(0.0);
        let r_45 = cpml.estimate_reflection(45.0);
        let r_60 = cpml.estimate_reflection(60.0);
        let r_80 = cpml.estimate_reflection(80.0);
        let r_89 = cpml.estimate_reflection(89.0);
        
        // Verify monotonic increase with angle
        assert!(r_45 > r_0, "Reflection should increase with angle");
        assert!(r_60 > r_45, "Reflection should increase with angle");
        assert!(r_80 > r_60, "Reflection should increase with angle");
        assert!(r_89 > r_80, "Reflection should increase with angle");
        
        // All should still be small
        assert!(r_89 < 1e-5, "Even at 89°, reflection should be small");
    }
    
    /// Test profile smoothness and continuity
    #[test]
    fn test_profile_continuity() {
        let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig {
            thickness: 20,
            polynomial_order: 3.0,
            ..Default::default()
        };
        
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Check sigma profile continuity
        let mut max_diff = 0.0f64;
        for i in 1..100 {
            let diff = (cpml.sigma_x[i] - cpml.sigma_x[i-1]).abs();
            max_diff = max_diff.max(diff);
            if i < 25 || i > 75 {
                println!("i={}: sigma[{}]={:.6}, sigma[{}]={:.6}, diff={:.6}", 
                         i, i-1, cpml.sigma_x[i-1], i, cpml.sigma_x[i], diff);
            }
        }
        
        // For polynomial grading with order 3, the maximum difference should be proportional to sigma_max
        let sigma_max = cpml.sigma_x[0].max(cpml.sigma_x[99]);
        let threshold = sigma_max * 0.2; // Allow up to 20% of max sigma as difference
        assert!(max_diff < threshold, 
                "Sigma profile has discontinuity: max_diff={:.3} > threshold={:.3}", 
                max_diff, threshold);
        
        // Check that profiles go to zero at interface
        assert_eq!(cpml.sigma_x[20], 0.0, "Sigma should be zero at PML interface");
        assert_eq!(cpml.sigma_x[79], 0.0, "Sigma should be zero at PML interface");
        
        // Check kappa starts at 1
        assert_eq!(cpml.kappa_x[20], 1.0, "Kappa should be 1 at PML interface");
        assert_eq!(cpml.kappa_x[79], 1.0, "Kappa should be 1 at PML interface");
    }
    
    /// Test dispersive media support
    #[test]
    fn test_dispersive_media_support() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Initially no dispersive support
        assert!(cpml.psi_dispersive.is_none());
        
        // Enable dispersive support
        cpml.enable_dispersive_support();
        
        // Should now have dispersive memory variables
        assert!(cpml.psi_dispersive.is_some());
        
        if let Some(ref psi_disp) = cpml.psi_dispersive {
            let expected_dim: (usize, usize, usize, usize) = (3, 64, 64, 64);
            assert_eq!(psi_disp.dim(), expected_dim);
        }
    }
    
    /// Test frequency domain application
    #[test]
    fn test_frequency_domain_cpml() {
        use rustfft::num_complex::Complex;
        
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        let config = CPMLConfig::default();
        let mut cpml = CPMLBoundary::new(config, &grid).unwrap();
        
        // Create test field in frequency domain
        let mut field_freq = Array3::from_elem((64, 64, 64), Complex::new(1.0, 0.0));
        
        // Apply C-PML in frequency domain
        cpml.apply_acoustic_freq(&mut field_freq, &grid, 0).unwrap();
        
        // Check that field is modified in PML regions
        // At boundaries, field should be attenuated
        assert!(field_freq[[0, 32, 32]].norm() < 1.0, 
            "Field should be attenuated at boundary");
        
        // In center, field should be mostly unchanged
        assert_relative_eq!(field_freq[[32, 32, 32]].norm(), 1.0, epsilon = 0.01);
    }
    
    // Helper functions
    
    /// Create a plane wave at given angle
    fn create_plane_wave(grid: &Grid, angle_degrees: f64, frequency: f64) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let angle_rad = angle_degrees * PI / 180.0;
        let k = 2.0 * PI * frequency / 1500.0; // Wave number (assuming c=1500 m/s)
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k_idx in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    
                    // Plane wave propagating at angle from x-axis
                    let phase = k * (x * angle_rad.cos() + y * angle_rad.sin());
                    field[[i, j, k_idx]] = phase.cos();
                }
            }
        }
        
        field
    }
    
    /// Create a Gaussian pulse
    fn create_gaussian_pulse(
        grid: &Grid, 
        cx: usize, 
        cy: usize, 
        cz: usize, 
        width: f64
    ) -> Array3<f64> {
        let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let dx = (i as f64 - cx as f64) * grid.dx;
                    let dy = (j as f64 - cy as f64) * grid.dy;
                    let dz = (k as f64 - cz as f64) * grid.dz;
                    
                    let r2 = dx*dx + dy*dy + dz*dz;
                    field[[i, j, k]] = (-r2 / (width * width)).exp();
                }
            }
        }
        
        field
    }
    
    /// Compute total field energy
    fn compute_field_energy(field: &Array3<f64>) -> f64 {
        field.iter().map(|&v| v * v).sum()
    }
    
    /// Compute energy in interior region (excluding PML)
    fn compute_interior_energy(field: &Array3<f64>, thickness: usize) -> f64 {
        let (nx, ny, nz) = field.dim();
        let mut energy = 0.0;
        
        for i in thickness..nx-thickness {
            for j in thickness..ny-thickness {
                for k in thickness..nz-thickness {
                    energy += field[[i, j, k]].powi(2);
                }
            }
        }
        
        energy
    }
    
    /// Compute simple Laplacian for wave propagation
    fn compute_laplacian(field: &Array3<f64>, grid: &Grid) -> Array3<f64> {
        let (nx, ny, nz) = field.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));
        
        // Simple 2nd order central difference
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let d2dx2 = (field[[i+1, j, k]] - 2.0*field[[i, j, k]] + field[[i-1, j, k]]) / (grid.dx * grid.dx);
                    let d2dy2 = (field[[i, j+1, k]] - 2.0*field[[i, j, k]] + field[[i, j-1, k]]) / (grid.dy * grid.dy);
                    let d2dz2 = (field[[i, j, k+1]] - 2.0*field[[i, j, k]] + field[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    laplacian[[i, j, k]] = d2dx2 + d2dy2 + d2dz2;
                }
            }
        }
        
        laplacian
    }
}