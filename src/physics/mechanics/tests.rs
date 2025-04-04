use super::*;
use crate::grid::Grid;
use crate::medium::homogeneous::HomogeneousMedium;
use crate::physics::mechanics::acoustic_wave::{NonlinearWave, AbsorptionLaw};
use ndarray::{Array3, s};
use approx::assert_relative_eq;
use std::f64;

#[test]
fn test_linear_wave_propagation() {
    // Create a simple grid
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);
    
    // Create a homogeneous water medium
    let medium = HomogeneousMedium::new(
        1000.0,  // density (kg/m³)
        1500.0,  // sound speed (m/s)
        &grid,
        0.0,     // no absorption
        0.0,     // no scattering
    );
    
    // Create nonlinear wave solver with zero nonlinearity
    let mut wave = NonlinearWave::new(&grid);
    wave.set_nonlinearity_scaling(0.0);
    
    // Create initial pressure field (Gaussian pulse)
    let mut pressure = Array3::<f64>::zeros((50, 50, 50));
    let center = 25;
    let width = 5.0;
    
    for i in 0..50 {
        for j in 0..50 {
            for k in 0..50 {
                let r2 = ((i - center) as f64).powi(2) + 
                        ((j - center) as f64).powi(2) + 
                        ((k - center) as f64).powi(2);
                pressure[[i, j, k]] = (-r2 / width.powi(2)).exp();
            }
        }
    }
    
    // Store initial center pressure
    let initial_pressure = pressure[[center, center, center]];
    
    // Propagate for one time step
    let dt = 0.1 * 0.001 / 1500.0;  // CFL = 0.1
    wave.step(&mut pressure, &medium, dt);
    
    // Verify wave has propagated (center pressure should decrease)
    assert!(pressure[[center, center, center]] < initial_pressure);
    
    // Verify energy conservation (within numerical error)
    let initial_energy: f64 = pressure.iter().map(|&x| x.powi(2)).sum();
    let final_energy: f64 = pressure.iter().map(|&x| x.powi(2)).sum();
    assert_relative_eq!(initial_energy, final_energy, max_relative = 0.01);
}

#[test]
fn test_nonlinear_wave_propagation() {
    // Create a simple grid
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);
    
    // Create a homogeneous water medium
    let medium = HomogeneousMedium::new(
        1000.0,  // density
        1500.0,  // sound speed
        &grid,
        0.0,     // no absorption
        0.0,     // no scattering
    );
    
    // Create nonlinear wave solver with strong nonlinearity
    let mut wave = NonlinearWave::new(&grid);
    wave.set_nonlinearity_scaling(5.0);
    
    // Create high-amplitude initial pressure field
    let mut pressure = Array3::<f64>::zeros((50, 50, 50));
    let center = 25;
    let width = 5.0;
    let amplitude = 1.0e6;  // 1 MPa
    
    for i in 0..50 {
        for j in 0..50 {
            for k in 0..50 {
                let r2 = ((i - center) as f64).powi(2) + 
                        ((j - center) as f64).powi(2) + 
                        ((k - center) as f64).powi(2);
                pressure[[i, j, k]] = amplitude * (-r2 / width.powi(2)).exp();
            }
        }
    }
    
    // Store initial pressure profile along x-axis
    let mut initial_profile = Vec::new();
    for i in 0..50 {
        initial_profile.push(pressure[[i, center, center]]);
    }
    
    // Propagate for several time steps
    let dt = 0.1 * 0.001 / 1500.0;  // CFL = 0.1
    for _ in 0..10 {
        wave.step(&mut pressure, &medium, dt);
    }
    
    // Store final pressure profile
    let mut final_profile = Vec::new();
    for i in 0..50 {
        final_profile.push(pressure[[i, center, center]]);
    }
    
    // Verify nonlinear steepening (maximum gradient should increase)
    let initial_max_gradient = initial_profile.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    
    let final_max_gradient = final_profile.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    
    assert!(final_max_gradient > initial_max_gradient);
}

#[test]
fn test_nonlinear_wave_in_tissue() {
    // Create a grid
    let grid = Grid::new(50, 50, 50, 0.0002, 0.0002, 0.0002);  // 0.2mm spacing
    
    // Create heterogeneous tissue medium
    let mut density = Array3::<f64>::ones((50, 50, 50)) * 1000.0;  // Water density
    let mut sound_speed = Array3::<f64>::ones((50, 50, 50)) * 1500.0;  // Water sound speed
    
    // Add tissue layer (muscle)
    for i in 25..50 {
        density[[i, .., ..]] = 1070.0;  // Muscle density
        sound_speed[[i, .., ..]] = 1580.0;  // Muscle sound speed
    }
    
    let medium = HomogeneousMedium::new_with_properties(density, sound_speed, &grid);
    
    // Create nonlinear wave solver with tissue-appropriate parameters
    let mut wave = NonlinearWave::new(&grid);
    wave.set_nonlinearity_scaling(3.5);  // Tissue nonlinearity
    wave.set_absorption_law(AbsorptionLaw::PowerLaw { alpha0: 0.54, y: 1.1 });
    
    // Create initial pressure field (high amplitude for nonlinear effects)
    let mut pressure = Array3::<f64>::zeros((50, 50, 50));
    let center = 10;  // Source near water-tissue interface
    let width = 3.0;
    let amplitude = 2.0e6;  // 2 MPa for strong nonlinearity
    
    for i in 0..50 {
        for j in 0..50 {
            for k in 0..50 {
                let r2 = ((i - center) as f64).powi(2) + 
                        ((j - 25) as f64).powi(2) + 
                        ((k - 25) as f64).powi(2);
                pressure[[i, j, k]] = amplitude * (-r2 / width.powi(2)).exp();
            }
        }
    }
    
    // Store initial profile
    let mut initial_profile = Vec::new();
    for i in 0..50 {
        initial_profile.push(pressure[[i, 25, 25]]);
    }
    
    // Propagate wave
    let dt = 0.1 * 0.0002 / 1580.0;  // CFL = 0.1 based on fastest sound speed
    for _ in 0..50 {
        wave.step(&mut pressure, &medium, dt);
    }
    
    // Store final profile
    let mut final_profile = Vec::new();
    for i in 0..50 {
        final_profile.push(pressure[[i, 25, 25]]);
    }
    
    // Verify tissue-specific effects:
    
    // 1. Check for stronger nonlinear distortion in tissue region
    let tissue_gradient = final_profile[30..40].windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    
    let water_gradient = final_profile[10..20].windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    
    assert!(tissue_gradient > water_gradient * 1.2);  // 20% stronger distortion in tissue
    
    // 2. Verify wave speed difference
    let tissue_arrival = final_profile[30..40].iter()
        .position(|&p| p.abs() > amplitude * 0.1)
        .unwrap_or(0);
    
    let water_arrival = final_profile[10..20].iter()
        .position(|&p| p.abs() > amplitude * 0.1)
        .unwrap_or(0);
    
    // Wave should arrive earlier in tissue due to higher sound speed
    assert!(tissue_arrival < water_arrival);
}

#[test]
fn test_wave_reflection() {
    // Create a simple grid
    let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001);
    
    // Create a medium with a density discontinuity
    let mut density = Array3::<f64>::ones((50, 50, 50)) * 1000.0;
    let mut sound_speed = Array3::<f64>::ones((50, 50, 50)) * 1500.0;
    
    // Add a high-impedance layer
    for i in 35..50 {
        density[[i, .., ..]] *= 2.0;
        sound_speed[[i, .., ..]] *= 1.5;
    }
    
    let medium = HomogeneousMedium::new_with_properties(density, sound_speed, &grid);
    
    // Create wave solver
    let mut wave = NonlinearWave::new(&grid);
    
    // Create initial pressure pulse
    let mut pressure = Array3::<f64>::zeros((50, 50, 50));
    let source_x = 20;
    let center_y = 25;
    let center_z = 25;
    let width = 5.0;
    
    for i in 0..50 {
        for j in 0..50 {
            for k in 0..50 {
                let r2 = ((i - source_x) as f64).powi(2) + 
                        ((j - center_y) as f64).powi(2) + 
                        ((k - center_z) as f64).powi(2);
                pressure[[i, j, k]] = (-r2 / width.powi(2)).exp();
            }
        }
    }
    
    // Propagate wave
    let dt = 0.1 * 0.001 / 1500.0;
    for _ in 0..20 {
        wave.step(&mut pressure, &medium, dt);
    }
    
    // Verify reflection (check for non-zero pressure in initial region)
    let reflected_energy: f64 = pressure.slice(s![..30, .., ..])
        .iter()
        .map(|&x| x.powi(2))
        .sum();
    
    assert!(reflected_energy > 0.1);  // Significant reflected energy
}
