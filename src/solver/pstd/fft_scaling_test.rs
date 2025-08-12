//! Tests for FFT scaling consistency in PSTD solver
//!
//! This module validates that FFT normalization is correctly and consistently
//! applied across all PSTD operations to ensure proper wave physics.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use ndarray::{Array3, Array4};
    use num_complex::Complex;
    use std::f64::consts::PI;
    
    /// Test that FFT round-trip preserves field values with correct scaling
    #[test]
    fn test_fft_round_trip_scaling() {
        let nx = 32;
        let ny = 32;
        let nz = 32;
        let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        
        // Create a test field with known values
        let mut field = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 / nx as f64;
                    let y = j as f64 / ny as f64;
                    let z = k as f64 / nz as f64;
                    field[[i, j, k]] = (2.0 * PI * x).sin() * (2.0 * PI * y).cos() * (2.0 * PI * z).sin();
                }
            }
        }
        
        // Store original for comparison
        let original = field.clone();
        
        // Forward FFT
        let mut fields_4d = Array4::zeros((1, nx, ny, nz));
        fields_4d.index_axis_mut(ndarray::Axis(0), 0).assign(&field);
        let field_hat = crate::utils::fft_3d(&fields_4d, 0, &grid);
        
        // Inverse FFT
        let reconstructed = crate::utils::ifft_3d(&field_hat, &grid);
        
        // Check that values are preserved (within numerical tolerance)
        let max_error = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        
        assert!(max_error < 1e-10, "FFT round-trip error too large: {}", max_error);
    }
    
    /// Test that pressure update scaling is consistent
    #[test]
    fn test_pressure_update_scaling() {
        let nx = 16;
        let ny = 16;
        let nz = 16;
        let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
        
        let config = PstdConfig {
            k_space_correction: false,  // Disable for simpler testing
            k_space_order: 2,
            anti_aliasing: false,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
            use_leapfrog: false,  // Use Euler for simplicity
        };
        
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Create a simple divergence field
        let mut divergence = Array3::zeros((nx, ny, nz));
        let amplitude = 1.0;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = i as f64 / nx as f64 - 0.5;
                    let y = j as f64 / ny as f64 - 0.5;
                    let z = k as f64 / nz as f64 - 0.5;
                    let r = (x*x + y*y + z*z).sqrt();
                    divergence[[i, j, k]] = amplitude * (-r * r).exp();
                }
            }
        }
        
        // Apply pressure update
        let mut pressure = Array3::zeros((nx, ny, nz));
        let dt = 1e-6;
        solver.update_pressure(&mut pressure, &divergence, &medium, dt).unwrap();
        
        // Check that pressure has reasonable values (not scaled incorrectly)
        let max_pressure = pressure.iter().map(|p| p.abs()).fold(0.0, f64::max);
        let expected_order = amplitude * dt * medium.sound_speed(0.0, 0.0, 0.0, &grid).powi(2) * medium.density(0.0, 0.0, 0.0, &grid);
        
        // Pressure should be on the same order of magnitude as expected
        assert!(max_pressure > expected_order * 0.1 && max_pressure < expected_order * 10.0,
                "Pressure scaling incorrect: max={}, expected order={}", max_pressure, expected_order);
    }
    
    /// Test that velocity update scaling is consistent
    #[test]
    fn test_velocity_update_scaling() {
        let nx = 16;
        let ny = 16;
        let nz = 16;
        let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
        
        let config = PstdConfig {
            k_space_correction: false,
            k_space_order: 2,
            anti_aliasing: false,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
            use_leapfrog: false,
        };
        
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Create a simple pressure field
        let mut pressure = Array3::zeros((nx, ny, nz));
        let amplitude = 1e3;  // 1 kPa
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = 2.0 * PI * i as f64 / nx as f64;
                    pressure[[i, j, k]] = amplitude * x.sin();
                }
            }
        }
        
        // Apply velocity update
        let mut vx = Array3::zeros((nx, ny, nz));
        let mut vy = Array3::zeros((nx, ny, nz));
        let mut vz = Array3::zeros((nx, ny, nz));
        let dt = 1e-6;
        
        solver.update_velocity(&mut vx, &mut vy, &mut vz, &pressure, &medium, dt).unwrap();
        
        // Check that velocity gradients are reasonable
        let max_vx = vx.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let density = medium.density(0.0, 0.0, 0.0, &grid);
        let expected_order = amplitude * dt / (density * grid.dx);  // dp/dx ~ amplitude * k, k ~ 2π/L
        
        // Velocity should be on the same order of magnitude as expected
        assert!(max_vx > expected_order * 0.1 && max_vx < expected_order * 10.0,
                "Velocity scaling incorrect: max={}, expected order={}", max_vx, expected_order);
    }
    
    /// Test conservation of energy with correct FFT scaling
    #[test]
    fn test_energy_conservation() {
        let nx = 32;
        let ny = 32;
        let nz = 32;
        let grid = Grid::new(nx, ny, nz, 1e-3, 1e-3, 1e-3);
        let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.0, 1.0);
        
        let config = PstdConfig {
            k_space_correction: true,
            k_space_order: 4,
            anti_aliasing: true,
            pml_stencil_size: 4,
            cfl_factor: 0.3,
            use_leapfrog: true,
        };
        
        let mut solver = PstdSolver::new(config, &grid).unwrap();
        
        // Initialize with a Gaussian pulse
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut vx = Array3::zeros((nx, ny, nz));
        let mut vy = Array3::zeros((nx, ny, nz));
        let mut vz = Array3::zeros((nx, ny, nz));
        
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cz = nz as f64 / 2.0;
        let sigma = 3.0;
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dx = (i as f64 - cx) * grid.dx;
                    let dy = (j as f64 - cy) * grid.dy;
                    let dz = (k as f64 - cz) * grid.dz;
                    let r2 = dx*dx + dy*dy + dz*dz;
                    pressure[[i, j, k]] = 1e3 * (-r2 / (2.0 * sigma * sigma * grid.dx * grid.dx)).exp();
                }
            }
        }
        
        // Compute initial energy
        let rho = medium.density(0.0, 0.0, 0.0, &grid);
        let c = medium.sound_speed(0.0, 0.0, 0.0, &grid);
        let initial_energy = compute_total_energy(&pressure, &vx, &vy, &vz, rho, c, &grid);
        
        // Time step
        let dt = 0.5 * config.cfl_factor * grid.dx.min(grid.dy).min(grid.dz) / c;
        
        // Evolve for several time steps
        for _ in 0..10 {
            // Compute divergence
            let divergence = solver.compute_divergence(&vx, &vy, &vz).unwrap();
            
            // Update pressure
            solver.update_pressure(&mut pressure, &divergence, &medium, dt).unwrap();
            
            // Update velocity
            solver.update_velocity(&mut vx, &mut vy, &mut vz, &pressure, &medium, dt).unwrap();
        }
        
        // Compute final energy
        let final_energy = compute_total_energy(&pressure, &vx, &vy, &vz, rho, c, &grid);
        
        // Energy should be approximately conserved (allowing for numerical dissipation)
        let energy_change = ((final_energy - initial_energy) / initial_energy).abs();
        assert!(energy_change < 0.01, "Energy not conserved: change = {}%", energy_change * 100.0);
    }
    
    fn compute_total_energy(
        pressure: &Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        rho: f64,
        c: f64,
        grid: &Grid,
    ) -> f64 {
        let mut energy = 0.0;
        let dv = grid.dx * grid.dy * grid.dz;
        
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Kinetic energy: 0.5 * rho * v^2
                    let v2 = vx[[i,j,k]].powi(2) + vy[[i,j,k]].powi(2) + vz[[i,j,k]].powi(2);
                    let kinetic = 0.5 * rho * v2;
                    
                    // Potential energy: p^2 / (2 * rho * c^2)
                    let potential = pressure[[i,j,k]].powi(2) / (2.0 * rho * c * c);
                    
                    energy += (kinetic + potential) * dv;
                }
            }
        }
        
        energy
    }
}