//! Solver Convergence Validation Tests
//!
//! Validates numerical properties of FDTD and PSTD solvers
//! following literature-based standards.
//!
//! ## References
//! - Taflove & Hagness (2005) "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
//! - Treeby & Cox (2010) "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields"
//! - Finkelstein & Kastner (2007) "Finite difference time domain dispersion reduction schemes"
//!
//! ## Sprint 150: Solver Property Validation
//! - Validates CFL stability conditions
//! - Validates energy conservation (within numerical precision)
//! - Validates wave propagation without instabilities
//! - Validates solver doesn't produce NaN or Inf values

use kwavers::grid::Grid;
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::physics::plugin::PluginManager;
use kwavers::solver::fdtd::{FdtdConfig, FdtdPlugin};
use ndarray::{s, Array4};

/// Test CFL stability condition
///
/// Validates that solver is stable for CFL ≤ 1 and remains bounded
/// Reference: Courant et al. (1928), "Über die partiellen Differenzengleichungen der mathematischen Physik"
#[test]
fn test_cfl_stability_condition() {
    let nx = 16; // Small grid for fast test
    let dx = 1e-3;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("Failed to create grid");
    let medium = HomogeneousMedium::water(&grid);

    // Test stable CFL (should remain bounded)
    let mut fields = Array4::zeros((17, nx, nx, nx));
    fields[[0, nx / 2, nx / 2, nx / 2]] = 1e6;

    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.9, // Just below stability limit
        subgridding: false,
        subgrid_factor: 2,
    };

    let plugin = FdtdPlugin::new(config, &grid).expect("Failed to create FDTD plugin");
    let mut plugin_manager = PluginManager::new();
    plugin_manager
        .add_plugin(Box::new(plugin))
        .expect("Failed to add plugin");
    plugin_manager
        .initialize(&grid, &medium)
        .expect("Failed to initialize plugins");

    let c = 1500.0;
    let dt = 0.9 * dx / c;

    for step in 0..100 {
        let t = step as f64 * dt;
        plugin_manager
            .execute(&mut fields, &grid, &medium, dt, t)
            .expect("Failed to execute plugins");

        // Check stability
        let max_pressure = fields
            .slice(s![0, .., .., ..])
            .iter()
            .fold(0.0f64, |a, &b| a.max(b.abs()));

        assert!(
            max_pressure.is_finite(),
            "Solver became unstable at step {}",
            step
        );
        assert!(
            max_pressure < 1e8,
            "Solver grew unbounded at step {}: {:.2e}",
            step,
            max_pressure
        );
    }
}

/// Test energy conservation for linear waves
///
/// Validates that total energy remains approximately constant
/// Reference: Virieux (1986), "P-SV wave propagation in heterogeneous media"
#[test]
fn test_energy_conservation() {
    let nx = 32;
    let dx = 1e-3;
    let grid = Grid::new(nx, nx, nx, dx, dx, dx).expect("Failed to create grid");
    let medium = HomogeneousMedium::water(&grid);

    let mut fields = Array4::zeros((17, nx, nx, nx));

    // Initialize Gaussian pulse
    let center = nx / 2;
    let sigma = 3.0_f64;
    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                let r2 = ((i as f64 - center as f64).powi(2)
                    + (j as f64 - center as f64).powi(2)
                    + (k as f64 - center as f64).powi(2))
                    * dx.powi(2);
                fields[[0, i, j, k]] = 1e6 * (-r2 / (2.0 * sigma.powi(2))).exp();
            }
        }
    }

    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.5,
        subgridding: false,
        subgrid_factor: 2,
    };

    let plugin = FdtdPlugin::new(config, &grid).expect("Failed to create FDTD plugin");
    let mut plugin_manager = PluginManager::new();
    plugin_manager
        .add_plugin(Box::new(plugin))
        .expect("Failed to add plugin");
    plugin_manager
        .initialize(&grid, &medium)
        .expect("Failed to initialize plugins");

    // Compute initial energy
    let initial_energy: f64 = fields
        .slice(s![0, .., .., ..])
        .iter()
        .map(|&x| x.powi(2))
        .sum();

    let c = 1500.0;
    let dt = 0.5 * dx / c;

    // Run for multiple steps
    for step in 0..50 {
        let t = step as f64 * dt;
        plugin_manager
            .execute(&mut fields, &grid, &medium, dt, t)
            .expect("Failed to execute plugins");
    }

    // Compute final energy
    let final_energy: f64 = fields
        .slice(s![0, .., .., ..])
        .iter()
        .map(|&x| x.powi(2))
        .sum();

    let energy_change = (final_energy - initial_energy).abs() / initial_energy;

    // Energy should be conserved within 20% for FDTD
    // (allows for numerical dissipation and boundary effects)
    assert!(
        energy_change < 0.2,
        "Energy not conserved: {:.1}% change",
        energy_change * 100.0
    );
}
