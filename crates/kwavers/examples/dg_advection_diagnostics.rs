//! DG scalar-advection diagnostics for solver-discrepancy audits.
//!
//! The current `DGSolver` core advances the scalar periodic advection equation
//! ```text
//!   u_t + c u_x = 0
//! ```
//! with nodal DG, Lax-Friedrichs flux, and SSP-RK3. This is not yet the same
//! first-order acoustic pressure/velocity system used by FDTD and PSTD. The
//! correct comparison gate is therefore DG versus the analytical periodic
//! advection solution, while `pstd_fdtd_comparison.rs` covers acoustic
//! FDTD/PSTD pressure-field discrepancies.
//!
//! ## Theorem
//!
//! On a periodic domain of length `L`, the exact solution for initial condition
//! `u(x,0)=sin(2*pi*x/L)` is `u(x,t)=sin(2*pi*(x-c*t)/L)`. A conservative DG
//! semi-discretization must preserve the quadrature-weighted global mass, and
//! phase/amplitude discrepancies against this shifted wave quantify the scalar
//! DG operator before it is promoted into an acoustic-system comparison.
//!
//! ## References
//!
//! - Hesthaven & Warburton (2008). *Nodal Discontinuous Galerkin Methods*.
//! - Cockburn & Shu (2001). *J. Sci. Comput.* 16(3):173-261.
//! - Shu & Osher (1988). *J. Comput. Phys.* 77(2):439-471.
//! - Pierce (1989). *Acoustics: An Introduction to Its Physical Principles*.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers_solver::forward::pstd::dg::{DGConfig, DGSolver};
use leto::{Array1, Array3};
use std::f64::consts::PI;
use std::io::Write;
use std::sync::Arc;

#[path = "dg_advection_diagnostics/error_measures.rs"]
mod error_measures;
#[path = "dg_advection_diagnostics/reference_solutions.rs"]
mod reference_solutions;
use error_measures::{
    acoustic_energy, amplitude, left_going_invariant_error, phase_error, relative_l2, weighted_mass,
};
use reference_solutions::{
    exact_bidirectional_acoustic, exact_shifted_characteristic, exact_shifted_coefficients,
    initialize_right_going_characteristic, initialize_sine_coefficients, physical_coordinate,
    pressure_from_characteristic, pressure_velocity_from_characteristics, reflect_coefficients,
    velocity_from_characteristic,
};

const ELEMENTS: usize = 12;
const POLYNOMIAL_ORDER: usize = 2;
const SOUND_SPEED: f64 = 1.0;
const DENSITY: f64 = 1.0;
const DT: f64 = 0.01;
const STEPS: usize = 20;

fn main() -> KwaversResult<()> {
    let diagnostics = run_dg_advection_diagnostic()?;
    let acoustic = run_acoustic_characteristic_diagnostic()?;
    let bidirectional = run_bidirectional_acoustic_diagnostic()?;

    let _ = writeln!(std::io::stdout().lock(), "DG scalar-advection diagnostic");
    let _ = writeln!(
        std::io::stdout().lock(),
        "elements: {ELEMENTS}, polynomial_order: {POLYNOMIAL_ORDER}, steps: {STEPS}"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "equation: u_t + c u_x = 0, periodic, c = {SOUND_SPEED}"
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "relative_l2",
        diagnostics.relative_l2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "mass_error",
        diagnostics.mass_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "phase_error_rad",
        diagnostics.phase_error_rad
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "amplitude_ratio",
        diagnostics.amplitude_ratio
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "DG right-going acoustic characteristic diagnostic"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "state: w+ = p + rho*c*u, w- = p - rho*c*u = 0"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "pressure_relative_l2",
        acoustic.pressure_relative_l2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "velocity_relative_l2",
        acoustic.velocity_relative_l2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "left_invariant_error",
        acoustic.left_invariant_error
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "energy_ratio",
        acoustic.energy_ratio
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(
        std::io::stdout().lock(),
        "DG bidirectional acoustic characteristic diagnostic"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "state: w+ = sin(kx), w- = sin(kx), so p = sin(kx), u = 0 at t = 0"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "pressure_relative_l2",
        bidirectional.pressure_relative_l2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "velocity_relative_l2",
        bidirectional.velocity_relative_l2
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "{:<24} {:>16.6e}",
        "energy_ratio",
        bidirectional.energy_ratio
    );
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(),
        "interpretation: DG is checked against exact scalar advection; acoustic FDTD/PSTD field metrics are reported by pstd_fdtd_comparison.rs"
    );

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct DgDiagnostic {
    relative_l2: f64,
    mass_error: f64,
    phase_error_rad: f64,
    amplitude_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct AcousticCharacteristicDiagnostic {
    pressure_relative_l2: f64,
    velocity_relative_l2: f64,
    left_invariant_error: f64,
    energy_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct BidirectionalAcousticDiagnostic {
    pressure_relative_l2: f64,
    velocity_relative_l2: f64,
    energy_ratio: f64,
}

fn run_dg_advection_diagnostic() -> KwaversResult<DgDiagnostic> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let length = 2.0 * ELEMENTS as f64;
    let wavenumber = 2.0 * PI / length;
    let final_time = STEPS as f64 * DT;

    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let mut solver = DGSolver::new(config, grid)?;
    solver.initialize_modal_coefficients(ELEMENTS, 1);
    initialize_sine_coefficients(
        solver.modal_coefficients_mut().expect("coefficients"),
        &xi_nodes,
        wavenumber,
    );

    let initial = solver.modal_coefficients().expect("coefficients").clone();
    let mut ignored_grid_field = Array3::zeros((ELEMENTS * n_nodes, 1, 1));
    for _ in 0..STEPS {
        solver.solve_step(&mut ignored_grid_field, DT)?;
    }

    let final_coeffs = solver.modal_coefficients().expect("coefficients");
    let exact = exact_shifted_coefficients(&xi_nodes, wavenumber, SOUND_SPEED * final_time);

    Ok(DgDiagnostic {
        relative_l2: relative_l2(final_coeffs, &exact, &weights),
        mass_error: (weighted_mass(final_coeffs, &weights) - weighted_mass(&initial, &weights))
            .abs(),
        phase_error_rad: phase_error(final_coeffs, &weights, &xi_nodes, wavenumber, final_time),
        amplitude_ratio: amplitude(final_coeffs, &weights, &xi_nodes, wavenumber)
            / amplitude(&initial, &weights, &xi_nodes, wavenumber).max(f64::EPSILON),
    })
}

fn run_acoustic_characteristic_diagnostic() -> KwaversResult<AcousticCharacteristicDiagnostic> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let length = 2.0 * ELEMENTS as f64;
    let wavenumber = 2.0 * PI / length;
    let final_time = STEPS as f64 * DT;

    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let mut solver = DGSolver::new(config, grid)?;
    solver.initialize_modal_coefficients(ELEMENTS, 1);
    initialize_right_going_characteristic(
        solver.modal_coefficients_mut().expect("coefficients"),
        &xi_nodes,
        wavenumber,
    );

    let mut ignored_grid_field = Array3::zeros((ELEMENTS * n_nodes, 1, 1));
    for _ in 0..STEPS {
        solver.solve_step(&mut ignored_grid_field, DT)?;
    }

    let characteristic = solver.modal_coefficients().expect("coefficients");
    let exact_characteristic =
        exact_shifted_characteristic(&xi_nodes, wavenumber, SOUND_SPEED * final_time);
    let pressure = pressure_from_characteristic(characteristic);
    let exact_pressure = pressure_from_characteristic(&exact_characteristic);
    let velocity = velocity_from_characteristic(characteristic);
    let exact_velocity = velocity_from_characteristic(&exact_characteristic);

    Ok(AcousticCharacteristicDiagnostic {
        pressure_relative_l2: relative_l2(&pressure, &exact_pressure, &weights),
        velocity_relative_l2: relative_l2(&velocity, &exact_velocity, &weights),
        left_invariant_error: left_going_invariant_error(&pressure, &velocity),
        energy_ratio: acoustic_energy(&pressure, &velocity, &weights)
            / acoustic_energy(&exact_pressure, &exact_velocity, &weights).max(f64::EPSILON),
    })
}

fn run_bidirectional_acoustic_diagnostic() -> KwaversResult<BidirectionalAcousticDiagnostic> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let length = 2.0 * ELEMENTS as f64;
    let wavenumber = 2.0 * PI / length;
    let final_time = STEPS as f64 * DT;

    let w_plus = evolve_characteristic(&xi_nodes, |x| (wavenumber * x).sin())?;
    let reflected_minus = evolve_characteristic(&xi_nodes, |x| (wavenumber * (length - x)).sin())?;
    let w_minus = reflect_coefficients(&reflected_minus);
    let (pressure, velocity) = pressure_velocity_from_characteristics(&w_plus, &w_minus);
    let (exact_pressure, exact_velocity) =
        exact_bidirectional_acoustic(&xi_nodes, wavenumber, SOUND_SPEED * final_time);

    Ok(BidirectionalAcousticDiagnostic {
        pressure_relative_l2: relative_l2(&pressure, &exact_pressure, &weights),
        velocity_relative_l2: relative_l2(&velocity, &exact_velocity, &weights),
        energy_ratio: acoustic_energy(&pressure, &velocity, &weights)
            / acoustic_energy(&exact_pressure, &exact_velocity, &weights).max(f64::EPSILON),
    })
}

fn evolve_characteristic(
    xi_nodes: &Array1<f64>,
    initial: impl Fn(f64) -> f64,
) -> KwaversResult<Array3<f64>> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let mut solver = DGSolver::new(config, grid)?;
    solver.initialize_modal_coefficients(ELEMENTS, 1);
    {
        let coeffs = solver.modal_coefficients_mut().expect("coefficients");
        for elem in 0..ELEMENTS {
            for node in 0..xi_nodes.len() {
                coeffs[[elem, node, 0]] = initial(physical_coordinate(elem, xi_nodes[node]));
            }
        }
    }

    let mut ignored_grid_field = Array3::zeros((ELEMENTS * n_nodes, 1, 1));
    for _ in 0..STEPS {
        solver.solve_step(&mut ignored_grid_field, DT)?;
    }
    Ok(solver.modal_coefficients().expect("coefficients").clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shift_preserves_mass_for_periodic_sine() {
        let n_nodes = POLYNOMIAL_ORDER + 1;
        let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes).unwrap();
        let length = 2.0 * ELEMENTS as f64;
        let k = 2.0 * PI / length;
        let shifted = exact_shifted_coefficients(&xi_nodes, k, 0.37);

        assert!(weighted_mass(&shifted, &weights).abs() < 1.0e-12);
    }

    #[test]
    fn dg_advection_diagnostic_remains_conservative_and_bounded() {
        let diagnostics = run_dg_advection_diagnostic().unwrap();

        assert!(diagnostics.mass_error < 1.0e-12);
        assert!(diagnostics.relative_l2 < 2.0e-2);
        assert!(diagnostics.phase_error_rad < 2.0e-2);
        assert!((diagnostics.amplitude_ratio - 1.0).abs() < 2.0e-2);
    }

    #[test]
    fn right_going_acoustic_characteristic_matches_exact_pressure_and_velocity() {
        let diagnostics = run_acoustic_characteristic_diagnostic().unwrap();

        assert!(diagnostics.pressure_relative_l2 < 2.0e-2);
        assert!(diagnostics.velocity_relative_l2 < 2.0e-2);
        assert!(diagnostics.left_invariant_error < 1.0e-12);
        assert!((diagnostics.energy_ratio - 1.0).abs() < 2.0e-2);
    }

    #[test]
    fn bidirectional_acoustic_characteristics_match_standing_wave_solution() {
        let diagnostics = run_bidirectional_acoustic_diagnostic().unwrap();

        assert!(diagnostics.pressure_relative_l2 < 2.0e-2);
        assert!(diagnostics.velocity_relative_l2 < 2.0e-2);
        assert!((diagnostics.energy_ratio - 1.0).abs() < 2.0e-2);
    }
}
