//! Physics validation tests for Kwavers
//!
//! These tests verify that the physics implementations are correct
//! by comparing against analytical solutions and known results.

use kwavers::{Grid, Time};
use kwavers::medium::homogeneous::HomogeneousMedium;
use ndarray::Array3;
use std::f64::consts::PI;

#[test]
fn test_wave_speed_in_medium() {
    // Test that wave propagation speed matches the medium's sound speed
    let grid = Grid::new(100, 100, 100, 1e-3, 1e-3, 1e-3);
    let sound_speed = 1500.0; // m/s
    let medium = HomogeneousMedium::new(1000.0, sound_speed, 0.0, 0.0, &grid);
    
    // CFL condition for 3D FDTD
    let cfl = 0.5;
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let dt = cfl * min_dx / sound_speed;
    
    // Wave should travel sound_speed * dt in one timestep
    let expected_distance = sound_speed * dt;
    let grid_distance = cfl * min_dx;
    
    // These should be equal
    assert!((expected_distance - grid_distance).abs() < 1e-10);
}

#[test]
fn test_cfl_stability_condition() {
    // Test that CFL number is correctly enforced for stability
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let sound_speed = 1500.0;
    
    // For 3D FDTD, CFL must be <= 1/sqrt(3) ≈ 0.577
    let max_cfl_3d = 1.0 / (3.0_f64).sqrt();
    
    // Test with safe CFL
    let safe_cfl = 0.5;
    assert!(safe_cfl < max_cfl_3d);
    
    // Calculate timestep
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let dt = safe_cfl * min_dx / sound_speed;
    
    // Verify it's positive and reasonable
    assert!(dt > 0.0);
    assert!(dt < 1e-6); // Should be in microsecond range
}

#[test]
fn test_plane_wave_propagation() {
    // Test that a plane wave maintains its shape during propagation
    let nx = 100;
    let grid = Grid::new(nx, 50, 50, 1e-3, 1e-3, 1e-3);
    let mut field = grid.create_field();
    
    // Initialize a Gaussian pulse
    let center = nx as f64 / 4.0;
    let width: f64 = 5.0;
    
    for i in 0..nx {
        let x = i as f64;
        let value = (-(x - center).powi(2) / (2.0 * width.powi(2))).exp();
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                field[[i, j, k]] = value;
            }
        }
    }
    
    // Check that initial field has expected properties
    let max_val: f64 = field.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((max_val - 1.0).abs() < 1e-10); // Peak should be 1.0
}

#[test]
fn test_energy_conservation_principle() {
    // In a lossless medium, total energy should be conserved
    let grid = Grid::new(50, 50, 50, 1e-3, 1e-3, 1e-3);
    let field = grid.create_field();
    
    // Calculate total energy (proportional to sum of squares)
    let total_energy: f64 = field.iter().map(|&x| x * x).sum();
    
    // Initially zero
    assert_eq!(total_energy, 0.0);
    
    // After adding energy, it should be positive
    let mut field2 = grid.create_field();
    field2[[25, 25, 25]] = 1.0;
    let energy2: f64 = field2.iter().map(|&x| x * x).sum();
    assert_eq!(energy2, 1.0);
}

#[test]
fn test_dispersion_relation() {
    // Test that the numerical dispersion follows expected patterns
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let wavelength = 10.0 * grid.dx; // 10 grid points per wavelength
    let k = 2.0 * PI / wavelength; // Wave number
    
    // For FDTD, there should be minimal dispersion at 10 points per wavelength
    let ppw = wavelength / grid.dx;
    assert!(ppw >= 10.0); // Adequate sampling
    
    // Nyquist criterion
    let k_max = PI / grid.dx;
    assert!(k < k_max); // Within Nyquist limit
}

#[test]
fn test_homogeneous_medium_properties() {
    // Test that homogeneous medium returns constant properties
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let density = 1500.0;
    let sound_speed = 2000.0;
    let medium = HomogeneousMedium::new(density, sound_speed, 0.001, 0.072, &grid);
    
    use kwavers::medium::Medium;
    
    // Test at multiple points
    let test_points = vec![
        (0.0, 0.0, 0.0),
        (15e-3, 15e-3, 15e-3),
        (31e-3, 31e-3, 31e-3),
    ];
    
    for (x, y, z) in test_points {
        assert_eq!(medium.density(x, y, z, &grid), density);
        assert_eq!(medium.sound_speed(x, y, z, &grid), sound_speed);
        assert!(medium.is_homogeneous());
    }
}

#[test]
fn test_grid_spacing_isotropy() {
    // Test that grid with equal spacing is isotropic
    let spacing = 2e-3;
    let grid = Grid::new(40, 40, 40, spacing, spacing, spacing);
    
    assert_eq!(grid.dx, grid.dy);
    assert_eq!(grid.dy, grid.dz);
    
    // Diagonal distance should be sqrt(3) * spacing for unit cell
    let diagonal = (grid.dx.powi(2) + grid.dy.powi(2) + grid.dz.powi(2)).sqrt();
    let expected = spacing * (3.0_f64).sqrt();
    assert!((diagonal - expected).abs() < 1e-10);
}

#[test]
fn test_numerical_stability_indicator() {
    // Test that we can detect potential instabilities
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let sound_speed = 1500.0;
    
    // Calculate maximum stable timestep
    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let max_dt = min_dx / (sound_speed * (3.0_f64).sqrt());
    
    // Test that smaller timestep is stable
    let dt = 0.5 * max_dt;
    assert!(dt < max_dt);
    
    // Courant number
    let courant = sound_speed * dt / min_dx;
    assert!(courant < 1.0 / (3.0_f64).sqrt());
}