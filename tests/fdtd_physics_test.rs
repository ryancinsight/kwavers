//! FDTD Physics Validation Tests
//! 
//! Critical tests to ensure the FDTD solver produces physically correct results.

use kwavers::{
    Grid, 
    solver::fdtd::{FdtdConfig, FdtdSolver},
    medium::homogeneous::HomogeneousMedium,
    source::PointSource,
};
use ndarray::Array3;
use std::f64::consts::PI;

/// Test that a wave propagates at the correct speed
#[test]
fn test_wave_propagation_speed() {
    // Create a small grid for fast testing
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let config = FdtdConfig::default();
    let solver = FdtdSolver::new(config, &grid).unwrap();
    
    // Create medium with known sound speed
    let sound_speed = 1500.0; // m/s (water)
    let medium = HomogeneousMedium::new(1000.0, sound_speed, 0.0, 0.0, &grid);
    
    // Initialize pressure field
    let mut pressure = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_x = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_y = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_z = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    
    // Add initial pulse at center
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    pressure[[center.0, center.1, center.2]] = 1.0;
    
    // Propagate for a known time
    let dt = 0.5 * grid.dx / sound_speed; // CFL = 0.5
    let num_steps = 10;
    
    for _ in 0..num_steps {
        solver.step(
            &mut pressure.view_mut(),
            &mut velocity_x.view_mut(),
            &mut velocity_y.view_mut(),
            &mut velocity_z.view_mut(),
            &medium,
            dt,
        ).unwrap();
    }
    
    // Check that wave has propagated the expected distance
    let expected_distance = sound_speed * dt * num_steps as f64;
    let grid_distance = expected_distance / grid.dx;
    
    // Wave should have moved approximately grid_distance cells from center
    let expected_radius = grid_distance as usize;
    
    // Check that pressure is non-zero at expected radius
    if expected_radius < grid.nx / 2 {
        let test_point = (center.0 + expected_radius, center.1, center.2);
        assert!(
            pressure[[test_point.0, test_point.1, test_point.2]].abs() > 1e-6,
            "Wave should have reached test point"
        );
    }
}

/// Test energy conservation in a closed system
#[test]
fn test_energy_conservation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let config = FdtdConfig::default();
    let solver = FdtdSolver::new(config, &grid).unwrap();
    
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    
    let mut pressure = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_x = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_y = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_z = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    
    // Add initial energy
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    pressure[[center.0, center.1, center.2]] = 1.0;
    
    // Calculate initial energy
    let initial_energy: f64 = pressure.iter().map(|p| p * p).sum();
    
    // Propagate
    let dt = 0.3 * grid.dx / 1500.0; // Conservative CFL
    for _ in 0..5 {
        solver.step(
            &mut pressure.view_mut(),
            &mut velocity_x.view_mut(),
            &mut velocity_y.view_mut(),
            &mut velocity_z.view_mut(),
            &medium,
            dt,
        ).unwrap();
    }
    
    // Calculate final energy (pressure + kinetic)
    let pressure_energy: f64 = pressure.iter().map(|p| p * p).sum();
    let kinetic_energy: f64 = velocity_x.iter().map(|v| v * v).sum()
        + velocity_y.iter().map(|v| v * v).sum()
        + velocity_z.iter().map(|v| v * v).sum();
    let final_energy = pressure_energy + kinetic_energy;
    
    // Energy should be approximately conserved (within numerical error)
    let energy_ratio = final_energy / initial_energy;
    assert!(
        (energy_ratio - 1.0).abs() < 0.1,
        "Energy not conserved: ratio = {}", energy_ratio
    );
}

/// Test that CFL condition is enforced
#[test]
fn test_cfl_stability() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let config = FdtdConfig::default();
    let solver = FdtdSolver::new(config, &grid).unwrap();
    
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    
    // Calculate maximum stable time step
    let max_dt = 0.5 * grid.dx / 1500.0; // CFL = 0.5 for 3D
    
    // This should be stable
    let mut pressure = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_x = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_y = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    let mut velocity_z = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));
    
    pressure[[16, 16, 16]] = 1.0;
    
    // Should complete without NaN or infinity
    for _ in 0..10 {
        solver.step(
            &mut pressure.view_mut(),
            &mut velocity_x.view_mut(),
            &mut velocity_y.view_mut(),
            &mut velocity_z.view_mut(),
            &medium,
            max_dt,
        ).unwrap();
    }
    
    // Check no NaN or infinity
    assert!(pressure.iter().all(|p| p.is_finite()));
    assert!(velocity_x.iter().all(|v| v.is_finite()));
}