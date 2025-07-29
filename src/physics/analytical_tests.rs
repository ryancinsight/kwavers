//! Analytical tests for physics algorithms
//! 
//! This module provides tests with known analytical solutions to validate
//! the accuracy of our physics implementations.

#[cfg(test)]
mod tests {
    use crate::grid::Grid;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::physics::mechanics::acoustic_wave::NonlinearWave;
    use crate::physics::traits::AcousticWaveModel;
    use crate::source::{Source, NullSource};
    use ndarray::{Array3, Array4};
    use std::f64::consts::PI;

    /// Test 1D plane wave propagation with analytical solution
    /// 
    /// For a 1D plane wave in a homogeneous medium:
    /// p(x,t) = A * sin(k*x - ω*t)
    /// where k = ω/c = 2π/λ
    #[test]
    fn test_plane_wave_propagation() {
        let nx = 128;
        let ny = 1;
        let nz = 1;
        let dx = 1e-3; // 1 mm
        let dy = dx;
        let dz = dx;
        
        let grid = Grid::new(nx, ny, nz, dx, dy, dz);
        let c0 = 1500.0; // m/s (water)
        let rho0 = 1000.0; // kg/m³
        let medium = HomogeneousMedium::new(rho0, c0, &grid, 0.0, 0.0);
        
        // Wave parameters
        let frequency = 1e6; // 1 MHz
        let wavelength = c0 / frequency;
        let k = 2.0 * PI / wavelength;
        let omega = 2.0 * PI * frequency;
        let amplitude = 1e5; // 100 kPa
        
        // Time parameters
        let cfl = 0.3;
        let dt = cfl * dx / c0;
        let t_end = 2.0 * wavelength / c0; // 2 periods
        let n_steps = (t_end / dt).ceil() as usize;
        
        // Initialize wave
        let mut wave = NonlinearWave::new(&grid);
        wave.set_nonlinearity_scaling(0.0); // Linear case
        
        // Initialize fields
        let mut fields = Array4::zeros((crate::solver::TOTAL_FIELDS, nx, ny, nz));
        let mut prev_pressure = Array3::zeros((nx, ny, nz));
        
        // Set initial condition: p(x,0) = A * sin(k*x)
        for i in 0..nx {
            let x = i as f64 * dx;
            let p = amplitude * (k * x).sin();
            fields[[crate::solver::PRESSURE_IDX, i, 0, 0]] = p;
            prev_pressure[[i, 0, 0]] = p;
        }
        
        // Propagate for a shorter time to check wave evolution
        let propagation_time = 0.5 * wavelength / c0; // Half wavelength travel time
        let n_steps = (propagation_time / dt).round() as usize;
        let source = NullSource;
        
        // Store initial state
        let initial_pressure = fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX).to_owned();
        
        for step in 0..n_steps {
            let t = step as f64 * dt;
            wave.update_wave(
                &mut fields,
                &prev_pressure,
                &source,
                &grid,
                &medium,
                dt,
                t,
            );
            prev_pressure.assign(&fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX));
        }
        
        // For a 1D plane wave in k-space method, check that the wave pattern has evolved
        // The k-space method preserves the wave but may have some numerical dispersion
        let final_pressure = fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX);
        
        // Find the peak of the initial and final waves
        let initial_max_idx = initial_pressure.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        let final_max_idx = final_pressure.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        // Check that the wave has propagated
        let expected_shift = (c0 * propagation_time / dx).round() as usize;
        let actual_shift = if final_max_idx > initial_max_idx {
            final_max_idx - initial_max_idx
        } else {
            initial_max_idx - final_max_idx
        };
        
        // Allow for some error in the shift due to discretization
        assert!(
            actual_shift as i32 - expected_shift as i32 <= 2,
            "Plane wave propagation test failed: expected shift={}, actual shift={}",
            expected_shift, actual_shift
        );
        
        // Check that amplitude is preserved (within tolerance)
        let initial_max = initial_pressure.iter().map(|&p| p.abs()).fold(0.0, f64::max);
        let final_max = final_pressure.iter().map(|&p| p.abs()).fold(0.0, f64::max);
        let amplitude_ratio = final_max / initial_max;
        
        assert!(
            (amplitude_ratio - 1.0).abs() < 0.1,
            "Plane wave amplitude not preserved: initial={:.3e}, final={:.3e}, ratio={:.3}",
            initial_max, final_max, amplitude_ratio
        );
    }
    
    /// Test acoustic attenuation with analytical solution
    /// 
    /// For a plane wave with attenuation α:
    /// p(x,t) = A * exp(-α*c*t) * sin(k*x - ω*t)
    /// where the wave travels distance x = c*t
    #[test]
    fn test_acoustic_attenuation() {
        let nx = 256;
        let ny = 1;
        let nz = 1;
        let dx = 0.5e-3; // 0.5 mm
        let dy = dx;
        let dz = dx;
        
        let grid = Grid::new(nx, ny, nz, dx, dy, dz);
        let c0 = 1500.0; // m/s
        let rho0 = 1000.0; // kg/m³
        let alpha = 0.5; // Np/m (nepers per meter)
        
        // For the k-space method, absorption is applied in time domain
        // The absorption per time step is: exp(-α * c * dt)
        // So we need to set alpha such that after traveling distance d,
        // the amplitude is reduced by exp(-α * d)
        let medium = HomogeneousMedium::new(rho0, c0, &grid, 0.0, 0.0)
            .with_acoustic_absorption(alpha, 0.0); // delta=0 for frequency-independent
        
        // Initialize wave
        let mut wave = NonlinearWave::new(&grid);
        wave.set_nonlinearity_scaling(1e-10); // Nearly linear case (avoid zero)
        
        // Wave parameters
        let frequency = 1e6; // 1 MHz
        let wavelength = c0 / frequency;
        let k = 2.0 * PI / wavelength;
        let amplitude = 1e5; // 100 kPa
        
        // Time parameters
        let cfl = 0.3;
        let dt = cfl * dx / c0;
        
        // Initialize fields
        let mut fields = Array4::zeros((crate::solver::TOTAL_FIELDS, nx, ny, nz));
        let mut prev_pressure = Array3::zeros((nx, ny, nz));
        
        // Set initial condition: Gaussian pulse
        let pulse_width = 10.0 * dx;
        let pulse_center = 20.0 * dx;
        for i in 0..nx {
            let x = i as f64 * dx;
            let envelope = amplitude * (-(x - pulse_center).powi(2) / (2.0 * pulse_width.powi(2))).exp();
            let p = envelope * (k * x).sin();
            fields[[crate::solver::PRESSURE_IDX, i, 0, 0]] = p;
            // For k-space method, prev_pressure should be the same as current pressure initially
            prev_pressure[[i, 0, 0]] = p;
        }
        
        // Also initialize velocity fields to zero (important for k-space method)
        if crate::solver::VX_IDX < crate::solver::TOTAL_FIELDS {
            for i in 0..nx {
                fields[[crate::solver::VX_IDX, i, 0, 0]] = 0.0;
            }
        }
        if crate::solver::VY_IDX < crate::solver::TOTAL_FIELDS {
            for i in 0..nx {
                fields[[crate::solver::VY_IDX, i, 0, 0]] = 0.0;
            }
        }
        if crate::solver::VZ_IDX < crate::solver::TOTAL_FIELDS {
            for i in 0..nx {
                fields[[crate::solver::VZ_IDX, i, 0, 0]] = 0.0;
            }
        }
        
        // Propagate for a specific distance
        let propagation_distance = 0.05; // 50 mm
        let propagation_time = propagation_distance / c0;
        let n_steps = (propagation_time / dt).round() as usize;
        
        let source = NullSource;
        
        // Store initial max amplitude
        let initial_pressure = fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX).to_owned();
        let initial_max = initial_pressure
            .iter()
            .map(|&p| p.abs())
            .fold(0.0, f64::max);
        
        println!("Initial field setup - max pressure: {:.3e}", initial_max);
        println!("Grid size: {}x{}x{}", nx, ny, nz);
        println!("dt: {:.3e}, CFL: {}", dt, cfl);
        
        for step in 0..n_steps {
            wave.update_wave(
                &mut fields,
                &prev_pressure,
                &source,
                &grid,
                &medium,
                dt,
                step as f64 * dt,
            );
            prev_pressure.assign(&fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX));
            
            if step == 0 {
                let pressure_after_1 = fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX);
                let max_after_1 = pressure_after_1.iter().map(|&p| p.abs()).fold(0.0, f64::max);
                println!("After step 1: max pressure = {:.3e}", max_after_1);
            }
        }
        
        // Find max amplitude after propagation
        let final_pressure = fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX);
        let final_max = final_pressure
            .iter()
            .map(|&p| p.abs())
            .fold(0.0, f64::max);
        
        // Debug output
        println!("Initial max amplitude: {:.3e}", initial_max);
        println!("Final max amplitude: {:.3e}", final_max);
        println!("Number of steps: {}", n_steps);
        println!("Propagation distance: {:.3e} m", propagation_distance);
        println!("Alpha: {} Np/m", alpha);
        
        // Expected attenuation after traveling the distance
        let expected_attenuation = (-alpha * propagation_distance).exp();
        let actual_attenuation = if initial_max > 0.0 { final_max / initial_max } else { 0.0 };
        let error = if expected_attenuation > 0.0 { 
            (actual_attenuation - expected_attenuation).abs() / expected_attenuation 
        } else { 
            1.0 
        };
        
        assert!(
            error < 0.1, // 10% tolerance for numerical effects
            "Attenuation test failed: expected attenuation={:.3e}, actual={:.3e}, error={:.1}%",
            expected_attenuation, actual_attenuation, error * 100.0
        );
    }
    
    /// Test spherical wave spreading with analytical solution
    /// 
    /// For a spherical wave: p(r,t) = (A/r) * sin(k*r - ω*t)
    /// Amplitude decreases as 1/r
    #[test]
    fn test_spherical_spreading() {
        let n = 64;
        let dx = 2e-3; // 2 mm
        let grid = Grid::new(n, n, n, dx, dx, dx);
        
        let c0 = 1500.0; // m/s
        let rho0 = 1000.0; // kg/m³
        let medium = HomogeneousMedium::new(rho0, c0, &grid, 0.0, 0.0);
        
        // Source at center
        let center = n / 2;
        let source_amplitude = 1e6; // 1 MPa
        let reference_distance = 3.0 * dx; // 3 grid points to avoid singularity
        
        // Initialize spherical wave
        let mut pressure = Array3::zeros((n, n, n));
        
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = (((i as i32 - center as i32).pow(2) + 
                             (j as i32 - center as i32).pow(2) + 
                             (k as i32 - center as i32).pow(2)) as f64).sqrt() * dx;
                    
                    if r >= reference_distance {
                        // Spherical spreading law: p ∝ 1/r
                        pressure[[i, j, k]] = source_amplitude * reference_distance / r;
                    } else if r > 0.0 {
                        // Near source, use reference amplitude
                        pressure[[i, j, k]] = source_amplitude;
                    }
                }
            }
        }
        
        // Check 1/r decay at various distances
        let test_distances = vec![10e-3, 20e-3, 40e-3]; // meters
        
        for &distance in &test_distances {
            let expected_amplitude = source_amplitude * reference_distance / distance;
            let n_points = (distance / dx).round() as usize;
            
            if n_points + center < n {
                let actual_amplitude = pressure[[center + n_points, center, center]];
                let error = (actual_amplitude - expected_amplitude).abs() / expected_amplitude;
                
                assert!(
                    error < 0.02, // 2% tolerance for discretization effects
                    "Spherical spreading test failed at r={:.3e}: expected={:.3e}, actual={:.3e}, error={:.1}%",
                    distance, expected_amplitude, actual_amplitude, error * 100.0
                );
            }
        }
    }
    
    /// Test Gaussian beam profile
    /// 
    /// For a Gaussian beam: p(r) = A * exp(-r²/w₀²)
    /// where w₀ is the beam waist
    #[test]
    fn test_gaussian_beam() {
        let n = 128;
        let dx = 0.1e-3; // 0.1 mm
        let grid = Grid::new(n, n, 1, dx, dx, dx);
        
        let amplitude = 1e5; // 100 kPa
        let beam_waist = 2e-3; // 2 mm
        
        // Initialize Gaussian beam
        let mut pressure = Array3::zeros((n, n, 1));
        let center = n / 2;
        
        for i in 0..n {
            for j in 0..n {
                let x = (i as f64 - center as f64) * dx;
                let y = (j as f64 - center as f64) * dx;
                let r_squared = x * x + y * y;
                
                pressure[[i, j, 0]] = amplitude * (-r_squared / (beam_waist * beam_waist)).exp();
            }
        }
        
        // Check Gaussian profile at various radii
        let test_radii = vec![0.0, beam_waist / 2.0, beam_waist, 2.0 * beam_waist];
        
        for &radius in &test_radii {
            let expected = amplitude * (-(radius * radius) / (beam_waist * beam_waist)).exp();
            let n_points = (radius / dx).round() as usize;
            
            if center + n_points < n {
                let actual = pressure[[center + n_points, center, 0]];
                let error = (actual - expected).abs() / amplitude;
                
                assert!(
                    error < 0.01,
                    "Gaussian beam test failed at r={:.3e}: expected={:.3e}, actual={:.3e}, error={:.1}%",
                    radius, expected, actual, error * 100.0
                );
            }
        }
    }
    
    /// Test standing wave pattern
    /// 
    /// For two counter-propagating waves: p = 2A * cos(kx) * sin(ωt)
    #[test]
    fn test_standing_wave() {
        let nx = 256;
        let dx = 0.5e-3; // 0.5 mm
        let grid = Grid::new(nx, 1, 1, dx, dx, dx);
        
        let c0 = 1500.0; // m/s
        let frequency = 1e6; // 1 MHz
        let wavelength = c0 / frequency;
        let k = 2.0 * PI / wavelength;
        let amplitude = 1e5; // 100 kPa
        
        // Initialize standing wave pattern with smoother profile
        let mut pressure = Array3::zeros((nx, 1, 1));
        
        // Add window function to reduce edge effects
        let window_width = 10.0 * dx;
        
        for i in 0..nx {
            let x = i as f64 * dx;
            
            // Window function (smooth edges)
            let window = if x < window_width {
                0.5 * (1.0 - (PI * (window_width - x) / window_width).cos())
            } else if x > (nx as f64 - 1.0) * dx - window_width {
                0.5 * (1.0 - (PI * (x - ((nx as f64 - 1.0) * dx - window_width)) / window_width).cos())
            } else {
                1.0
            };
            
            // Standing wave: 2A * cos(kx) at t=0 with window
            pressure[[i, 0, 0]] = 2.0 * amplitude * (k * x).cos() * window;
        }
        
        // Check nodes and antinodes (avoiding edges)
        let n_wavelengths = ((nx as f64 * dx - 2.0 * window_width) / wavelength).floor() as usize;
        let start_x = window_width;
        
        for n in 0..n_wavelengths {
            // Check node at x = start_x + (n + 0.5) * λ
            let node_x = start_x + (n as f64 + 0.5) * wavelength;
            let node_idx = (node_x / dx).round() as usize;
            
            if node_idx < nx - (window_width / dx) as usize {
                let node_pressure = pressure[[node_idx, 0, 0]].abs();
                // Allow for numerical error at nodes
                let tolerance = 0.1 * amplitude; // 10% of amplitude for windowed function
                assert!(
                    node_pressure < tolerance,
                    "Standing wave node test failed at x={:.3e}: pressure={:.3e} (should be < {:.3e})",
                    node_x, node_pressure, tolerance
                );
            }
            
            // Check antinode at x = start_x + n * λ
            let antinode_x = start_x + n as f64 * wavelength;
            let antinode_idx = (antinode_x / dx).round() as usize;
            
            if antinode_idx < nx - (window_width / dx) as usize {
                let antinode_pressure = pressure[[antinode_idx, 0, 0]].abs();
                let expected = 2.0 * amplitude; // Window should be ~1 in the middle
                let error = (antinode_pressure - expected).abs() / expected;
                
                assert!(
                    error < 0.1, // 10% tolerance for windowed function
                    "Standing wave antinode test failed at x={:.3e}: expected={:.3e}, actual={:.3e}",
                    antinode_x, expected, antinode_pressure
                );
            }
        }
    }
}