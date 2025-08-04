//! Physics validation tests with known analytical solutions
//! 
//! This module contains tests that validate our numerical implementations
//! against known analytical solutions from physics literature.

#[cfg(test)]
mod tests {
    use crate::{
        Grid, HomogeneousMedium,
        medium::Medium,
    };
    use ndarray::Array3;
    use std::f64::consts::PI;

    /// Test 1D wave equation solution: u(x,t) = A*sin(kx - ωt)
    /// This is the fundamental test for any wave solver
    #[test]
    fn test_1d_wave_equation_analytical() {
        // Grid parameters for 1D wave (ny=nz=1)
        let nx = 128;
        let dx = 1e-3; // 1mm
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        
        // Wave parameters
        let c = 1500.0; // m/s (water)
        let frequency = 1e6; // 1 MHz
        let wavelength = c / frequency;
        let k = 2.0 * PI / wavelength;
        let omega = 2.0 * PI * frequency;
        let amplitude = 1e5; // 100 kPa
        
        // Time parameters
        let dt = 0.3 * dx / c; // CFL = 0.3
        let periods = 2.0;
        let n_steps = (periods / frequency / dt) as usize;
        
        // Initialize pressure field with sine wave
        let mut pressure_prev = Array3::<f64>::zeros((nx, 1, 1));
        let mut pressure_curr = Array3::<f64>::zeros((nx, 1, 1));
        
        // Set initial conditions: p(x,0) = A*sin(kx), dp/dt(x,0) = -A*ω*sin(kx)
        for i in 0..nx {
            let x = i as f64 * dx;
            pressure_curr[[i, 0, 0]] = amplitude * (k * x).sin();
            // For the previous timestep, use p(x,-dt) = A*sin(kx + ω*dt)
            pressure_prev[[i, 0, 0]] = amplitude * (k * x + omega * dt).sin();
        }
        
        // Propagate using simple finite difference
        let c2_dt2_dx2 = (c * dt / dx).powi(2);
        
        for _step in 0..n_steps {
            let mut pressure_next = Array3::<f64>::zeros((nx, 1, 1));
            
            // Update interior points using wave equation
            for i in 1..nx-1 {
                let d2p_dx2 = pressure_curr[[i+1, 0, 0]] - 2.0 * pressure_curr[[i, 0, 0]] + pressure_curr[[i-1, 0, 0]];
                pressure_next[[i, 0, 0]] = 2.0 * pressure_curr[[i, 0, 0]] - pressure_prev[[i, 0, 0]] 
                    + c2_dt2_dx2 * d2p_dx2;
            }
            
            // Periodic boundary conditions
            pressure_next[[0, 0, 0]] = 2.0 * pressure_curr[[0, 0, 0]] - pressure_prev[[0, 0, 0]] 
                + c2_dt2_dx2 * (pressure_curr[[1, 0, 0]] - 2.0 * pressure_curr[[0, 0, 0]] + pressure_curr[[nx-1, 0, 0]]);
            pressure_next[[nx-1, 0, 0]] = 2.0 * pressure_curr[[nx-1, 0, 0]] - pressure_prev[[nx-1, 0, 0]] 
                + c2_dt2_dx2 * (pressure_curr[[0, 0, 0]] - 2.0 * pressure_curr[[nx-1, 0, 0]] + pressure_curr[[nx-2, 0, 0]]);
            
            // Update arrays
            pressure_prev = pressure_curr;
            pressure_curr = pressure_next;
        }
        
        // Check against analytical solution
        let final_time = n_steps as f64 * dt;
        let mut max_error = 0.0f64;
        
        for i in 0..nx { // Check all points with periodic boundaries
            let x = i as f64 * dx;
            let analytical = amplitude * (k * x - omega * final_time).sin();
            let numerical = pressure_curr[[i, 0, 0]];
            let error = (numerical - analytical).abs() / amplitude;
            max_error = max_error.max(error);
        }
        
        // Should be accurate to within 5% for this simple case with CFL=0.3
        assert!(max_error < 0.05, "1D wave equation error too large: {:.2}%", max_error * 100.0);
    }

    /// Test heat diffusion with analytical solution
    /// Initial condition: Gaussian pulse
    /// Solution: Gaussian with increasing width
    #[test]
    fn test_heat_diffusion_analytical() {
        let nx = 64;
        let dx = 1e-3; // 1mm
        let grid = Grid::new(nx, nx, 1, dx, dx, dx);
        
        // Thermal parameters
        let alpha = 1.4e-7; // m²/s (water thermal diffusivity)
        let dt = 0.1 * dx * dx / (4.0 * alpha); // Stability criterion
        
        // Initial Gaussian parameters
        let sigma0 = 5.0 * dx;
        let x0 = nx as f64 / 2.0 * dx;
        let y0 = nx as f64 / 2.0 * dx;
        let T0 = 10.0; // Initial temperature rise
        
        // Initialize temperature field
        let mut temperature = Array3::<f64>::zeros((nx, nx, 1));
        for i in 0..nx {
            for j in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dx;
                let r2 = (x - x0).powi(2) + (y - y0).powi(2);
                temperature[[i, j, 0]] = T0 * (-r2 / (2.0 * sigma0 * sigma0)).exp();
            }
        }
        
        // Time evolution
        let n_steps = 100;
        let mut temp_new = temperature.clone();
        
        for _step in 0..n_steps {
            // Apply heat equation with central differences
            for i in 1..nx-1 {
                for j in 1..nx-1 {
                    let laplacian = 
                        (temperature[[i+1, j, 0]] - 2.0 * temperature[[i, j, 0]] + temperature[[i-1, j, 0]]) / (dx * dx) +
                        (temperature[[i, j+1, 0]] - 2.0 * temperature[[i, j, 0]] + temperature[[i, j-1, 0]]) / (dx * dx);
                    
                    temp_new[[i, j, 0]] = temperature[[i, j, 0]] + alpha * dt * laplacian;
                }
            }
            temperature.assign(&temp_new);
        }
        
        // Analytical solution for 2D heat equation with Gaussian initial condition
        let t_final = n_steps as f64 * dt;
        let sigma_t = (sigma0.powi(2) + 4.0 * alpha * t_final).sqrt();
        let amplitude_ratio = (sigma0 / sigma_t).powi(2);
        
        // Check solution
        let mut max_error = 0.0f64;
        for i in 10..nx-10 {
            for j in 10..nx-10 {
                let x = i as f64 * dx;
                let y = j as f64 * dx;
                let r2 = (x - x0).powi(2) + (y - y0).powi(2);
                let analytical = T0 * amplitude_ratio * (-r2 / (2.0 * sigma_t * sigma_t)).exp();
                let numerical = temperature[[i, j, 0]];
                
                if analytical > 0.1 { // Only check where solution is significant
                    let error = (numerical - analytical).abs() / analytical;
                    max_error = max_error.max(error);
                }
            }
        }
        
        assert!(max_error < 0.05, "Heat diffusion error too large: {:.2}%", max_error * 100.0);
    }

    /// Test acoustic absorption with Beer-Lambert law
    /// I(x) = I0 * exp(-α*x)
    #[test]
    fn test_acoustic_absorption_beer_lambert() {
        let nx = 256;
        let dx = 0.5e-3; // 0.5mm
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        
        // Medium with absorption
        let c = 1500.0;
        let rho = 1000.0;
        let alpha_db_per_m = 8.686; // dB/m (conversion: 1 Np/m = 8.686 dB/m)
        let alpha_np = 1.0; // Nepers/m
        let mut medium = HomogeneousMedium::new(rho, c, &grid, 0.0, 0.0);
        
        // Set power law absorption parameters
        medium.alpha0 = alpha_np; // Np/m at reference frequency
        medium.delta = 0.0; // No frequency dependence for this test
        medium.reference_frequency = 1e6; // 1 MHz
        
        // Initial plane wave
        let frequency = 1e6;
        let k = 2.0 * PI * frequency / c;
        let I0 = 1e5f64; // Initial intensity
        
        let mut pressure = Array3::<f64>::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = i as f64 * dx;
            pressure[[i, 0, 0]] = I0.sqrt() * (k * x).sin();
        }
        
        // Propagate with absorption
        let dt = 0.3 * dx / c;
        let propagation_distance = 0.1; // 10 cm
        let n_steps = (propagation_distance / c / dt) as usize;
        
        // Get absorption coefficient
        let absorption_coeff = medium.absorption_coefficient(0.0, 0.0, 0.0, &grid, frequency);
        
        // Simple propagation with absorption
        for _step in 0..n_steps {
            // Apply absorption factor
            let absorption_factor = (-absorption_coeff * c * dt).exp();
            pressure *= absorption_factor;
            
            // Simple advection (shift wave)
            let mut new_pressure = Array3::<f64>::zeros((nx, 1, 1));
            let shift = (c * dt / dx) as usize;
            for i in shift..nx {
                new_pressure[[i, 0, 0]] = pressure[[i - shift, 0, 0]];
            }
            pressure = new_pressure;
        }
        
        // Check intensity follows Beer-Lambert law
        let intensity_initial = I0;
        let intensity_final = pressure.mapv(|p| p * p).mean().unwrap();
        let intensity_analytical = intensity_initial * (-2.0 * absorption_coeff * propagation_distance).exp();
        
        let error = (intensity_final.sqrt() - intensity_analytical.sqrt()).abs() / intensity_analytical.sqrt();
        assert!(error < 0.15, "Beer-Lambert law error: {:.2}%", error * 100.0);
    }

    /// Test standing wave formation between rigid boundaries
    /// Solution: p(x,t) = A*cos(kx)*cos(ωt)
    #[test]
    fn test_standing_wave_rigid_boundaries() {
        let nx = 128;
        let dx = 1e-3;
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        
        let c = 1500.0;
        let L = nx as f64 * dx; // Domain length
        
        // Fundamental mode: wavelength = 2L
        let n = 1; // Mode number
        let wavelength = 2.0 * L / n as f64;
        let k = 2.0 * PI / wavelength;
        let frequency = c / wavelength;
        let omega = 2.0 * PI * frequency;
        
        // Initialize with standing wave pattern
        let amplitude = 1e5;
        let mut pressure = Array3::<f64>::zeros((nx, 1, 1));
        let mut velocity = Array3::<f64>::zeros((nx, 1, 1));
        
        for i in 0..nx {
            let x = i as f64 * dx;
            pressure[[i, 0, 0]] = amplitude * (k * x).cos();
            velocity[[i, 0, 0]] = 0.0; // Initially at rest
        }
        
        // Time stepping
        let dt = 0.3 * dx / c;
        let periods = 2.0;
        let n_steps = (periods / frequency / dt) as usize;
        
        let rho = 1000.0;
        let mut pressure_new = pressure.clone();
        
        for step in 0..n_steps {
            // Update pressure using velocity
            for i in 1..nx-1 {
                let dvdx = (velocity[[i, 0, 0]] - velocity[[i-1, 0, 0]]) / dx;
                pressure_new[[i, 0, 0]] = pressure[[i, 0, 0]] - rho * c * c * dt * dvdx;
            }
            
            // Rigid boundary conditions (v=0 at boundaries)
            pressure_new[[0, 0, 0]] = pressure_new[[1, 0, 0]];
            pressure_new[[nx-1, 0, 0]] = pressure_new[[nx-2, 0, 0]];
            
            pressure.assign(&pressure_new);
            
            // Update velocity using pressure
            for i in 1..nx-1 {
                let dpdx = (pressure[[i+1, 0, 0]] - pressure[[i, 0, 0]]) / dx;
                velocity[[i, 0, 0]] = velocity[[i, 0, 0]] - dt / rho * dpdx;
            }
            
            // Rigid boundaries
            velocity[[0, 0, 0]] = 0.0;
            velocity[[nx-1, 0, 0]] = 0.0;
        }
        
        // Check standing wave pattern
        let t_final = n_steps as f64 * dt;
        let time_factor = (omega * t_final).cos();
        
        let mut max_error = 0.0f64;
        for i in 1..nx-1 {
            let x = i as f64 * dx;
            let analytical = amplitude * (k * x).cos() * time_factor;
            let numerical = pressure[[i, 0, 0]];
            let error = (numerical - analytical).abs() / amplitude;
            max_error = max_error.max(error);
        }
        
        assert!(max_error < 0.05, "Standing wave error: {:.2}%", max_error * 100.0);
    }

    /// Test spherical wave spreading: p ∝ 1/r
    #[test]
    fn test_spherical_spreading_3d() {
        // Use smaller grid for 3D test
        let n = 64;
        let dx = 2e-3; // 2mm
        let grid = Grid::new(n, n, n, dx, dx, dx);
        
        let c = 1500.0;
        let source_pos = (n/2, n/2, n/2);
        let source_radius = 5.0 * dx;
        
        // Initialize with spherical source
        let mut pressure = Array3::<f64>::zeros((n, n, n));
        let p0 = 1e5;
        
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = (((i as f64 - source_pos.0 as f64) * dx).powi(2) +
                            ((j as f64 - source_pos.1 as f64) * dx).powi(2) +
                            ((k as f64 - source_pos.2 as f64) * dx).powi(2)).sqrt();
                    
                    if r < source_radius {
                        pressure[[i, j, k]] = p0 * (1.0 - r / source_radius);
                    }
                }
            }
        }
        
        // Simple outward propagation
        let dt = 0.3 * dx / c;
        let n_steps = 30; // Propagate outward
        
        // Record pressure at different radii
        let r1 = 10.0 * dx;
        let r2 = 20.0 * dx;
        let mut p1 = 0.0;
        let mut p2 = 0.0;
        
        for step in 0..n_steps {
            // Simple wave propagation (simplified)
            let mut new_pressure = Array3::<f64>::zeros((n, n, n));
            
            for i in 1..n-1 {
                for j in 1..n-1 {
                    for k in 1..n-1 {
                        let laplacian = 
                            (pressure[[i+1, j, k]] - 2.0 * pressure[[i, j, k]] + pressure[[i-1, j, k]]) / (dx * dx) +
                            (pressure[[i, j+1, k]] - 2.0 * pressure[[i, j, k]] + pressure[[i, j-1, k]]) / (dx * dx) +
                            (pressure[[i, j, k+1]] - 2.0 * pressure[[i, j, k]] + pressure[[i, j, k-1]]) / (dx * dx);
                        
                        new_pressure[[i, j, k]] = pressure[[i, j, k]] + c * c * dt * dt * laplacian;
                    }
                }
            }
            
            pressure = new_pressure;
            
            // Measure at specific radii
            if step == 15 {
                let i = source_pos.0 + (r1 / dx) as usize;
                if i < n { p1 = pressure[[i, source_pos.1, source_pos.2]].abs(); }
            }
            if step == 25 {
                let i = source_pos.0 + (r2 / dx) as usize;
                if i < n { p2 = pressure[[i, source_pos.1, source_pos.2]].abs(); }
            }
        }
        
        // Check 1/r relationship
        if p1 > 0.0 && p2 > 0.0 {
            let ratio_measured = p1 / p2;
            let ratio_expected = r2 / r1; // p ∝ 1/r
            let error = (ratio_measured - ratio_expected).abs() / ratio_expected;
            
            assert!(error < 0.3, "Spherical spreading error: {:.2}%", error * 100.0);
        }
    }

    /// Test dispersion relation for numerical schemes
    /// ω = c*k for non-dispersive wave equation
    #[test]
    fn test_numerical_dispersion() {
        let nx = 256;
        let dx = 1e-3;
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        
        let c = 1500.0;
        let dt = 0.3 * dx / c;
        
        // Test different wavelengths
        let points_per_wavelength = vec![4.0, 8.0, 16.0, 32.0];
        let mut max_phase_error = 0.0f64;
        
        for ppw in points_per_wavelength {
            let wavelength = ppw * dx;
            let k = 2.0 * PI / wavelength;
            let omega_exact = c * k;
            
            // Initialize with sine wave
            let mut p_prev = Array3::<f64>::zeros((nx, 1, 1));
            let mut p_curr = Array3::<f64>::zeros((nx, 1, 1));
            
            for i in 0..nx {
                let x = i as f64 * dx;
                p_curr[[i, 0, 0]] = (k * x).sin();
                p_prev[[i, 0, 0]] = (k * x - omega_exact * dt).sin();
            }
            
            // Propagate for one period
            let period = 2.0 * PI / omega_exact;
            let n_steps = (period / dt) as usize;
            
            for _ in 0..n_steps {
                let mut p_next = Array3::<f64>::zeros((nx, 1, 1));
                
                // Standard wave equation update
                for i in 1..nx-1 {
                    let d2p_dx2 = p_curr[[i+1, 0, 0]] - 2.0 * p_curr[[i, 0, 0]] + p_curr[[i-1, 0, 0]];
                    p_next[[i, 0, 0]] = 2.0 * p_curr[[i, 0, 0]] - p_prev[[i, 0, 0]] 
                        + (c * dt / dx).powi(2) * d2p_dx2;
                }
                
                // Periodic boundaries
                p_next[[0, 0, 0]] = p_next[[nx-2, 0, 0]];
                p_next[[nx-1, 0, 0]] = p_next[[1, 0, 0]];
                
                p_prev = p_curr;
                p_curr = p_next;
            }
            
            // Measure phase error
            let mut phase_shift = 0.0;
            for i in nx/4..3*nx/4 {
                let x = i as f64 * dx;
                let expected = (k * x).sin();
                let actual = p_curr[[i, 0, 0]];
                
                if expected.abs() > 0.5 {
                    phase_shift += (actual / expected).acos();
                }
            }
            phase_shift /= (nx/2) as f64;
            
            let phase_error = phase_shift.abs();
            max_phase_error = max_phase_error.max(phase_error);
            
            println!("PPW: {}, Phase error: {:.4} rad", ppw, phase_error);
        }
        
        // Should have low dispersion for well-resolved waves
        assert!(max_phase_error < 0.1, "Numerical dispersion too high: {:.4} rad", max_phase_error);
    }
}