//! Energy conservation, absorption decay, and diagnostics check-interval tests.

use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};
use crate::solver::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};

#[test]
fn test_energy_calculation_accuracy() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let p0 = 1000.0; // Pa
    solver.pressure.fill(p0);

    let energy = solver.calculate_total_energy();

    // E = p²/(2ρ₀c₀²) * Volume
    let rho0 = DENSITY_WATER_NOMINAL;
    let c0 = SOUND_SPEED_WATER_SIM;
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
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let tolerances = ConservationTolerances {
        check_interval: 5,
        ..ConservationTolerances::default()
    };
    solver.enable_conservation_diagnostics(tolerances);

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..20 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    let summary = solver.get_conservation_summary().unwrap();
    assert!(summary.contains("checks"));
}

/// **Theorem (absorption sign, Stokes-Kirchhoff):**
/// With the multiplicative per-step absorption `p *= exp(−α·c·Δt)`, the L2
/// energy norm `‖p‖₂ = sqrt(∑p²)` must be strictly smaller after N steps than
/// before, for any α > 0.
///
/// Verification: α = 5 Np/m, c = 1500 m/s, Δt ≈ 3.85×10⁻⁷ s.
/// Per-step decay factor ≈ exp(−5·1500·3.85e-7) ≈ 0.9971.
/// After 30 steps: 0.9971³⁰ ≈ 0.918 → ~8% L2 energy reduction.
#[test]
fn absorption_causes_amplitude_decay_not_growth() {
    let n = 24usize;
    let grid = Grid::new(n, n, n, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    medium
        .set_acoustic_properties(5.0, 1.0, medium.nonlinearity_coefficient())
        .unwrap();

    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: true,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let cx = n / 2;
    let cy = n / 2;
    let cz = n / 2;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as i32 - cx as i32).pow(2)
                    + (j as i32 - cy as i32).pow(2)
                    + (k as i32 - cz as i32).pow(2)) as f64;
                let val = 1.0e4 * (-(r2 / 9.0)).exp();
                solver.pressure[[i, j, k]] = val;
                solver.pressure_prev[[i, j, k]] = val;
            }
        }
    }
    let initial_max = solver.pressure.iter().cloned().fold(0.0f64, f64::max);

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for step in 0..30 {
        solver
            .update(&medium, &grid, &[], step as f64 * dt, dt)
            .unwrap();
    }

    let final_max = solver
        .pressure
        .iter()
        .cloned()
        .map(f64::abs)
        .fold(0.0f64, f64::max);

    assert!(
        final_max < initial_max,
        "absorption must reduce peak amplitude: final_max={final_max:.4e} initial_max={initial_max:.4e}"
    );
}
