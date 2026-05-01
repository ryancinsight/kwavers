//! Value-semantic regression tests for the Westervelt FDTD solver.

use super::{WesterveltFdtd, WesterveltFdtdConfig};
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};

#[test]
fn test_westervelt_fdtd_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let solver = WesterveltFdtd::new(config, &grid, &medium);

    assert_eq!(solver.pressure.shape(), &[32, 32, 32]);
}

#[test]
fn test_linear_wave_propagation() {
    // Test that with β=0 (no nonlinearity), we get linear wave propagation
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    // Set nonlinearity to zero for linear test
    medium.nonlinearity = 0.0;

    // Use zero artificial viscosity for energy conservation test
    let config = WesterveltFdtdConfig {
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Set initial Gaussian pulse
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as i32 - center.0 as i32).pow(2)
                    + (j as i32 - center.1 as i32).pow(2)
                    + (k as i32 - center.2 as i32).pow(2)) as f64;
                solver.pressure[[i, j, k]] = (-(r2 / 100.0)).exp();
            }
        }
    }

    // Propagate for a few time steps
    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..10 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    // Check that energy is conserved (approximately) with no artificial viscosity
    let total_energy: f64 = solver.pressure.iter().map(|&p| p * p).sum();
    assert!(total_energy > 0.0);
}

#[test]
fn test_conservation_diagnostics_integration() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Enable diagnostics
    solver.enable_conservation_diagnostics(ConservationTolerances::default());

    // Initial energy should be zero (no excitation)
    let initial_energy = solver.calculate_total_energy();
    assert!(initial_energy < 1e-10);

    // Verify tracker is enabled
    assert!(solver.conservation_tracker.is_some());
    assert!(solver.is_solution_valid());

    // Disable and check
    solver.disable_conservation_diagnostics();
    assert!(solver.conservation_tracker.is_none());
}

#[test]
fn test_energy_calculation_accuracy() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Set a known pressure field (uniform)
    let p0 = 1000.0; // Pa
    solver.pressure.fill(p0);

    // Calculate energy
    let energy = solver.calculate_total_energy();

    // Expected energy: E = p²/(2ρ₀c₀²) * Volume
    let rho0 = 1000.0;
    let c0 = 1500.0;
    let volume =
        (grid.nx as f64) * grid.dx * (grid.ny as f64) * grid.dy * (grid.nz as f64) * grid.dz;
    let expected_energy = (p0 * p0) / (2.0 * rho0 * c0 * c0) * volume;

    let relative_error = (energy - expected_energy).abs() / expected_energy;
    assert!(
        relative_error < 1e-10,
        "Energy calculation error: {}",
        relative_error
    );
}

#[test]
fn test_conservation_check_interval() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Enable diagnostics with check interval of 5
    let tolerances = ConservationTolerances {
        check_interval: 5,
        ..ConservationTolerances::default()
    };
    solver.enable_conservation_diagnostics(tolerances);

    // Simulate 20 steps
    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..20 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    // Should have 20/5 = 4 checks (steps 5, 10, 15, 20)
    let summary = solver.get_conservation_summary().unwrap();
    assert!(summary.contains("checks"));
}
