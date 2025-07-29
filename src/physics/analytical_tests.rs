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
        
        // Propagate for one period
        let steps_per_period = (1.0 / (frequency * dt)).round() as usize;
        let source = NullSource;
        
        for _ in 0..steps_per_period {
            wave.update_wave(
                &mut fields,
                &prev_pressure,
                &source,
                &grid,
                &medium,
                dt,
                0.0,
            );
            prev_pressure.assign(&fields.index_axis(ndarray::Axis(0), crate::solver::PRESSURE_IDX));
        }
        
        // Check that wave has propagated one wavelength
        let tolerance = 0.05; // 5% error tolerance
        for i in 0..nx {
            let x = i as f64 * dx;
            let expected = amplitude * (k * x - omega * (steps_per_period as f64 * dt)).sin();
            let actual = fields[[crate::solver::PRESSURE_IDX, i, 0, 0]];
            let error = (actual - expected).abs() / amplitude.abs();
            
            assert!(
                error < tolerance,
                "Plane wave test failed at x={:.3e}: expected={:.3e}, actual={:.3e}, error={:.1}%",
                x, expected, actual, error * 100.0
            );
        }
    }
    
    /// Test acoustic attenuation with analytical solution
    /// 
    /// For a plane wave with attenuation α:
    /// p(x,t) = A * exp(-α*x) * sin(k*x - ω*t)
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
        let medium = HomogeneousMedium::new(rho0, c0, &grid, alpha, 0.0);
        
        // Wave parameters
        let frequency = 2e6; // 2 MHz
        let wavelength = c0 / frequency;
        let k = 2.0 * PI / wavelength;
        let amplitude = 1e5; // 100 kPa
        
        // Check attenuation at different distances
        let distances = vec![0.01, 0.02, 0.05, 0.1]; // meters
        
        for &distance in &distances {
            let expected_amplitude = amplitude * (-alpha * distance).exp();
            let n_points = (distance / dx).round() as usize;
            
            if n_points < nx {
                // Initialize pressure field with attenuated wave
                let mut pressure = Array3::zeros((nx, ny, nz));
                for i in 0..n_points {
                    let x = i as f64 * dx;
                    pressure[[i, 0, 0]] = amplitude * (-alpha * x).exp() * (k * x).sin();
                }
                
                // Check amplitude at distance
                let actual_amplitude = pressure[[n_points - 1, 0, 0]].abs();
                let error = (actual_amplitude - expected_amplitude).abs() / expected_amplitude;
                
                assert!(
                    error < 0.01,
                    "Attenuation test failed at distance={:.3e}: expected={:.3e}, actual={:.3e}, error={:.1}%",
                    distance, expected_amplitude, actual_amplitude, error * 100.0
                );
            }
        }
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
        let source_amplitude = 1e6; // 1 MPa at r=1mm
        let reference_distance = 1e-3; // 1 mm
        
        // Initialize spherical wave
        let mut pressure = Array3::zeros((n, n, n));
        
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r = (((i as i32 - center as i32).pow(2) + 
                             (j as i32 - center as i32).pow(2) + 
                             (k as i32 - center as i32).pow(2)) as f64).sqrt() * dx;
                    
                    if r > reference_distance {
                        // Spherical spreading law: p ∝ 1/r
                        pressure[[i, j, k]] = source_amplitude * reference_distance / r;
                    }
                }
            }
        }
        
        // Check 1/r decay at various distances
        let test_distances = vec![5e-3, 10e-3, 20e-3, 40e-3]; // meters
        
        for &distance in &test_distances {
            let expected_amplitude = source_amplitude * reference_distance / distance;
            let n_points = (distance / dx).round() as usize;
            
            if n_points + center < n {
                let actual_amplitude = pressure[[center + n_points, center, center]];
                let error = (actual_amplitude - expected_amplitude).abs() / expected_amplitude;
                
                assert!(
                    error < 0.1,
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
        
        // Initialize standing wave pattern
        let mut pressure = Array3::zeros((nx, 1, 1));
        
        for i in 0..nx {
            let x = i as f64 * dx;
            // Standing wave: 2A * cos(kx) at t=0
            pressure[[i, 0, 0]] = 2.0 * amplitude * (k * x).cos();
        }
        
        // Check nodes and antinodes
        let n_wavelengths = ((nx as f64 * dx) / wavelength).floor() as usize;
        
        for n in 0..n_wavelengths {
            // Check node at x = (n + 0.5) * λ
            let node_x = (n as f64 + 0.5) * wavelength;
            let node_idx = (node_x / dx).round() as usize;
            
            if node_idx < nx {
                let node_pressure = pressure[[node_idx, 0, 0]].abs();
                assert!(
                    node_pressure < 0.01 * amplitude,
                    "Standing wave node test failed at x={:.3e}: pressure={:.3e} (should be ~0)",
                    node_x, node_pressure
                );
            }
            
            // Check antinode at x = n * λ
            let antinode_x = n as f64 * wavelength;
            let antinode_idx = (antinode_x / dx).round() as usize;
            
            if antinode_idx < nx {
                let antinode_pressure = pressure[[antinode_idx, 0, 0]].abs();
                let expected = 2.0 * amplitude;
                let error = (antinode_pressure - expected).abs() / expected;
                
                assert!(
                    error < 0.01,
                    "Standing wave antinode test failed at x={:.3e}: expected={:.3e}, actual={:.3e}",
                    antinode_x, expected, antinode_pressure
                );
            }
        }
    }
}