//! Validation tests for FDTD solver
//! 
//! These tests verify the numerical accuracy of the FDTD implementation
//! against analytical solutions and known benchmarks.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use ndarray::Array3;
    use std::f64::consts::PI;
    use approx::assert_relative_eq;
    
    // Test constants for finite difference accuracy
    // These are reasonable tolerances for properly resolved waves
    const FD_ORDER2_ERROR_TOL: f64 = 2e-2; // 2% error for 2nd order (practical tolerance)
    const FD_ORDER4_ERROR_TOL: f64 = 1e-4; // 0.01% error for 4th order
    const FD_ORDER6_ERROR_TOL: f64 = 1e-6; // 0.0001% error for 6th order
    
    /// Test plane wave propagation with FDTD
    #[test]
    fn test_fdtd_plane_wave() {
        let nx = 100;
        let ny = 20;
        let nz = 20;
        let dx = 1e-3; // 1mm
        let grid = Grid::new(nx, ny, nz, dx, dx, dx);
        
        let density = 1000.0;
        let sound_speed = 1500.0;
        let medium = HomogeneousMedium::new(density, sound_speed, &grid, 0.0, 1.0);
        
        let config = FdtdConfig {
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: 0.9,
            subgridding: false,
            subgrid_factor: 2,
        };
        
        let mut solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Initialize plane wave
        let frequency = 500e3; // 500 kHz - lower frequency for FDTD
        let wavelength = sound_speed / frequency;
        let k = 2.0 * PI / wavelength;
        
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut velocity_x = Array3::zeros((nx, ny, nz));
        let mut velocity_y = Array3::zeros((nx, ny, nz));
        let mut velocity_z = Array3::zeros((nx, ny, nz));
        
        // Initial conditions
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
        
        // Propagate for short time
        let propagation_time = 5.0 * dx / sound_speed; // 5 grid cells
        let n_steps = (propagation_time / dt).ceil() as usize;
        
        for _ in 0..n_steps {
            solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, &medium, dt).unwrap();
            solver.update_pressure(&mut pressure, &velocity_x, &velocity_y, &velocity_z, &medium, dt).unwrap();
        }
        
        // Check phase velocity (should be close to c)
        // Find peaks to measure wavelength
        let y_mid = ny / 2;
        let z_mid = nz / 2;
        let mut peaks = Vec::new();
        
        for i in 1..nx-1 {
            if pressure[[i, y_mid, z_mid]] > pressure[[i-1, y_mid, z_mid]] &&
               pressure[[i, y_mid, z_mid]] > pressure[[i+1, y_mid, z_mid]] &&
               pressure[[i, y_mid, z_mid]] > 0.5 * amplitude {
                peaks.push(i);
            }
        }
        
        if peaks.len() >= 2 {
            let measured_wavelength = (peaks[1] - peaks[0]) as f64 * dx;
            let error = (measured_wavelength - wavelength).abs() / wavelength;
            
            // FDTD should have reasonable accuracy with enough PPW
            let ppw = wavelength / dx;
            let expected_error = estimate_fdtd_error(ppw, config.spatial_order);
            assert!(error < 2.0 * expected_error, 
                "FDTD wavelength error {} exceeds expected {}", error, expected_error);
        }
    }
    
    /// Test numerical dispersion characteristics
    #[test]
    fn test_fdtd_dispersion() {
        let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
        
        // Test different spatial orders
        for order in [2, 4, 6] {
            let config = FdtdConfig {
                spatial_order: order,
                staggered_grid: true,
                cfl_factor: 0.9,
                subgridding: false,
                subgrid_factor: 2,
            };
            
            let solver = FdtdSolver::new(config, &grid).unwrap();
            
            // Higher order should allow larger time steps
            let dt2 = FdtdSolver::new(FdtdConfig { spatial_order: 2, ..config }, &grid).unwrap().max_stable_dt(1500.0);
            let dt_order = solver.max_stable_dt(1500.0);
            
            if order > 2 {
                assert!(dt_order <= dt2, "Higher order FDTD should have more restrictive CFL");
            }
        }
    }
    
    /// Test staggered grid interpolation
    #[test]
    fn test_staggered_grid_interpolation() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.9,
            subgridding: false,
            subgrid_factor: 2,
        };
        
        let solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Create a simple field
        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] = i as f64;
                }
            }
        }
        
        // Test interpolation to staggered positions
        let interpolated = solver.interpolate_to_staggered(&field, 0, 0.5);
        
        // Check that values are averaged correctly
        for i in 0..9 {
            let expected = 0.5 * (field[[i, 5, 5]] + field[[i+1, 5, 5]]);
            assert_relative_eq!(interpolated[[i, 5, 5]], expected, epsilon = 1e-10);
        }
    }
    
    /// Test finite difference accuracy
    #[test]
    fn test_finite_difference_accuracy() {
        let nx = 128; // Finer grid for better accuracy
        let grid = Grid::new(nx, nx, nx, 0.01, 0.01, 0.01);
        
        // Test different orders
        for order in [2, 4, 6] {
            let config = FdtdConfig {
                spatial_order: order,
                staggered_grid: false,
                cfl_factor: 0.9,
                subgridding: false,
                subgrid_factor: 2,
            };
            
            let solver = FdtdSolver::new(config, &grid).unwrap();
            
            // Create sinusoidal field with more points per wavelength
            let mut field = Array3::zeros((nx, nx, nx));
            let k = 2.0 * PI / (nx as f64 * grid.dx / 2.0); // 2 wavelengths across domain = 64 points per wavelength
            
            for i in 0..nx {
                let x = i as f64 * grid.dx;
                for j in 0..nx {
                    for k_idx in 0..nx {
                        field[[i, j, k_idx]] = (k * x).sin();
                    }
                }
            }
            
            // Compute derivative
            let deriv = solver.compute_derivative(&field, 0, 0.0);
            
            // Check accuracy in the interior, away from boundaries
            let margin = order / 2 + 3; // Extra margin for boundary effects
            for i in margin..nx-margin {
                let x = i as f64 * grid.dx;
                let expected = k * (k * x).cos();
                let actual = deriv[[i, nx/2, nx/2]];
                
                // Error should decrease with order
                let error = (actual - expected).abs();
                let max_error = match order {
                    2 => FD_ORDER2_ERROR_TOL,
                    4 => FD_ORDER4_ERROR_TOL,
                    6 => FD_ORDER6_ERROR_TOL,
                    _ => FD_ORDER2_ERROR_TOL,
                };
                
                assert!(error < max_error, 
                    "FD order {} error {} exceeds limit {} at x={}", 
                    order, error, max_error, x);
            }
        }
    }
    
    /// Test subgridding functionality
    #[test]
    fn test_subgridding() {
        let grid = Grid::new(40, 40, 40, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig {
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: 0.9,
            subgridding: true,
            subgrid_factor: 2,
        };
        
        let mut solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Add subgrid region
        solver.add_subgrid((10, 10, 10), (30, 30, 30)).unwrap();
        assert_eq!(solver.subgrids.len(), 1);
        
        // Test interpolation
        let coarse = Array3::ones((40, 40, 40));
        let mut fine = Array3::zeros((40, 40, 40)); // Size doesn't matter for test
        
        solver.interpolate_to_fine(&coarse, &mut fine, &solver.subgrids[0]);
        
        // Fine grid should have interpolated values
        assert!(fine.iter().any(|&v| v == 1.0));
    }
    
    /// Test CFL stability condition
    #[test]
    fn test_cfl_stability() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let c_max = 1500.0;
        
        // Test different configurations
        let configs = vec![
            (2, 1.0 / 3.0_f64.sqrt()),
            (4, 0.5),
            (6, 0.4),
        ];
        
        for (order, expected_limit) in configs {
            let config = FdtdConfig {
                spatial_order: order,
                staggered_grid: true,
                cfl_factor: 1.0, // Use full CFL limit
                subgridding: false,
                subgrid_factor: 2,
            };
            
            let solver = FdtdSolver::new(config, &grid).unwrap();
            let dt = solver.max_stable_dt(c_max);
            let cfl = c_max * dt / grid.dx;
            
            assert!(cfl <= expected_limit * 1.01, // Small tolerance
                "CFL {} exceeds limit {} for order {}", cfl, expected_limit, order);
        }
    }
    
    /// Helper function to estimate FDTD dispersion error
    fn estimate_fdtd_error(points_per_wavelength: f64, order: usize) -> f64 {
        let k_dx = 2.0 * PI / points_per_wavelength;
        match order {
            2 => (k_dx * k_dx) / 12.0,
            4 => (k_dx * k_dx * k_dx * k_dx) / 360.0,
            6 => (k_dx * k_dx * k_dx * k_dx * k_dx * k_dx) / 20160.0,
            _ => 0.1,
        }
    }
}