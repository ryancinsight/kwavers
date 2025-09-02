//! Integration tests for complete acoustic simulation workflows
//!
//! These tests validate end-to-end functionality including:
//! - Wave propagation accuracy
//! - Boundary condition effectiveness  
//! - Source-receiver coupling
//! - Energy conservation

use kwavers::{
    boundary::BoundaryType, configuration::Configuration, grid::Grid,
    medium::homogeneous::HomogeneousMedium, physics::constants_physics::*,
    solver::fdtd::FDTDSolver, source::point::PointSource,
};
use ndarray::Array3;

/// Validate that a point source creates spherical waves
#[test]
fn test_spherical_wave_propagation() {
    // Create small test grid
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);

    // Water medium
    let medium = HomogeneousMedium::new(
        DENSITY_WATER,
        SPEED_OF_SOUND_WATER,
        0.01, // Optical absorption [1/m]
        0.1,  // Optical scattering [1/m]
        &grid,
    );

    // Point source at center
    let mut source = PointSource::new(
        [32.0, 32.0, 32.0], // Center position
        1e6,                // 1 MHz
        1e5,                // 100 kPa amplitude
    );

    // FDTD solver with PML boundaries
    let mut solver = FDTDSolver::new(&grid, &medium);
    solver.set_boundary(BoundaryType::PML);

    // Initialize fields
    let mut pressure = Array3::zeros((64, 64, 64));

    // Run for 100 timesteps
    let dt = solver.calculate_stable_timestep(&grid, &medium);
    for step in 0..100 {
        let t = step as f64 * dt;
        source.apply(&mut pressure, &grid, t);
        solver.step(&mut pressure, &grid, &medium, dt);
    }

    // Verify spherical symmetry
    let center = [32, 32, 32];
    let radius = 10;

    let mut values_at_radius = Vec::new();
    for i in 0..64 {
        for j in 0..64 {
            for k in 0..64 {
                let r = ((i as i32 - center[0]).pow(2)
                    + (j as i32 - center[1]).pow(2)
                    + (k as i32 - center[2]).pow(2)) as f64;
                let r = r.sqrt();

                if (r - radius as f64).abs() < 1.0 {
                    values_at_radius.push(pressure[[i, j, k]]);
                }
            }
        }
    }

    // Check that values at same radius are similar (spherical symmetry)
    let mean: f64 = values_at_radius.iter().sum::<f64>() / values_at_radius.len() as f64;
    let variance: f64 = values_at_radius
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / values_at_radius.len() as f64;
    let std_dev = variance.sqrt();

    // Relative standard deviation should be small for spherical wave
    let relative_std = std_dev / mean.abs().max(1e-10);
    assert!(
        relative_std < 0.1,
        "Wave is not spherically symmetric: relative std = {:.3}",
        relative_std
    );
}

/// Test energy conservation in lossless medium
#[test]
fn test_energy_conservation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(
        DENSITY_WATER,
        SPEED_OF_SOUND_WATER,
        0.01, // Optical absorption [1/m]
        0.1,  // Optical scattering [1/m]
        &grid,
    );

    let mut solver = FDTDSolver::new(&grid, &medium);
    solver.set_boundary(BoundaryType::Periodic); // Periodic for energy conservation

    // Initialize with Gaussian pulse
    let mut pressure = Array3::zeros((32, 32, 32));
    let center = [16.0, 16.0, 16.0];
    let sigma = 3.0;

    for i in 0..32 {
        for j in 0..32 {
            for k in 0..32 {
                let r2 = (i as f64 - center[0]).powi(2)
                    + (j as f64 - center[1]).powi(2)
                    + (k as f64 - center[2]).powi(2);
                pressure[[i, j, k]] = 1e5 * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    // Calculate initial energy
    let initial_energy: f64 = pressure.iter().map(|p| p * p).sum();

    // Propagate for 50 timesteps
    let dt = solver.calculate_stable_timestep(&grid, &medium);
    for _ in 0..50 {
        solver.step(&mut pressure, &grid, &medium, dt);
    }

    // Calculate final energy
    let final_energy: f64 = pressure.iter().map(|p| p * p).sum();

    // Energy should be conserved (within numerical precision)
    let energy_ratio = final_energy / initial_energy;
    assert!(
        (energy_ratio - 1.0).abs() < 0.01,
        "Energy not conserved: ratio = {:.3}",
        energy_ratio
    );
}

/// Test that PML boundaries absorb outgoing waves
#[test]
fn test_pml_absorption() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(
        DENSITY_WATER,
        SPEED_OF_SOUND_WATER,
        0.01, // Optical absorption [1/m]
        0.1,  // Optical scattering [1/m]
        &grid,
    );

    let mut solver = FDTDSolver::new(&grid, &medium);
    solver.set_boundary(BoundaryType::PML);
    solver.set_pml_thickness(10);

    // Create pulse near boundary
    let mut pressure = Array3::zeros((64, 64, 64));
    pressure[[10, 32, 32]] = 1e5; // Near x-min boundary

    // Propagate until wave should hit boundary
    let dt = solver.calculate_stable_timestep(&grid, &medium);
    for _ in 0..100 {
        solver.step(&mut pressure, &grid, &medium, dt);
    }

    // Check that energy in PML region is minimal
    let mut pml_energy = 0.0;
    let mut interior_energy = 0.0;

    for i in 0..64 {
        for j in 0..64 {
            for k in 0..64 {
                let p2 = pressure[[i, j, k]].powi(2);
                if i < 10 || i >= 54 || j < 10 || j >= 54 || k < 10 || k >= 54 {
                    pml_energy += p2;
                } else {
                    interior_energy += p2;
                }
            }
        }
    }

    // Most energy should have been absorbed
    let total_energy = pml_energy + interior_energy;
    assert!(
        pml_energy / total_energy.max(1e-10) < 0.01,
        "PML did not absorb wave: PML energy fraction = {:.3}",
        pml_energy / total_energy
    );
}

/// Test focusing with a phased array
#[test]
fn test_phased_array_focusing() {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(
        DENSITY_WATER,
        SPEED_OF_SOUND_WATER,
        0.01, // Optical absorption [1/m]
        0.1,  // Optical scattering [1/m]
        &grid,
    );

    // Create linear array of sources
    let num_elements = 8;
    let spacing = 2e-3; // 2mm spacing
    let focal_point = [32.0, 32.0, 48.0]; // Focus 16mm away

    let mut sources = Vec::new();
    for i in 0..num_elements {
        let x = 32.0 + (i as f64 - num_elements as f64 / 2.0) * spacing / grid.dx;
        let position = [x, 32.0, 16.0];

        // Calculate delay for focusing
        let distance = ((position[0] - focal_point[0]).powi(2)
            + (position[1] - focal_point[1]).powi(2)
            + (position[2] - focal_point[2]).powi(2))
        .sqrt();
        let delay = distance / SPEED_OF_SOUND_WATER;

        let mut source = PointSource::new(position, 1e6, 1e5);
        source.set_delay(delay);
        sources.push(source);
    }

    // Run simulation
    let mut solver = FDTDSolver::new(&grid, &medium);
    let mut pressure = Array3::zeros((64, 64, 64));

    let dt = solver.calculate_stable_timestep(&grid, &medium);
    for step in 0..200 {
        let t = step as f64 * dt;
        for source in &mut sources {
            source.apply(&mut pressure, &grid, t);
        }
        solver.step(&mut pressure, &grid, &medium, dt);
    }

    // Check that maximum pressure is near focal point
    let mut max_pressure = 0.0;
    let mut max_position = [0, 0, 0];

    for i in 0..64 {
        for j in 0..64 {
            for k in 0..64 {
                if pressure[[i, j, k]].abs() > max_pressure {
                    max_pressure = pressure[[i, j, k]].abs();
                    max_position = [i, j, k];
                }
            }
        }
    }

    // Maximum should be within 3 grid points of focal point
    let distance = ((max_position[0] as f64 - focal_point[0]).powi(2)
        + (max_position[1] as f64 - focal_point[1]).powi(2)
        + (max_position[2] as f64 - focal_point[2]).powi(2))
    .sqrt();

    assert!(
        distance < 3.0,
        "Focus not at expected location: distance = {:.1} grid points",
        distance
    );
}
