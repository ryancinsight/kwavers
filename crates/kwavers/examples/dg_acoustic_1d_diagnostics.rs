//! Native 1-D acoustic DG diagnostics for solver-discrepancy audits.
//!
//! The shared fixture lives in `dg_acoustic_common.rs`. This executable prints
//! the numerical comparison table; `dg_acoustic_comparison_plot.rs` writes the
//! plotted pressure/error artifact from the same fixture.

#[path = "dg_common/dg_acoustic_common.rs"]
mod dg_acoustic_common;

use dg_acoustic_common::{
    print_common_solver_matrix, print_solver_matrix, run_embedded_gaussian_series,
    run_native_acoustic_diagnostic, ELEMENTS, POLYNOMIAL_ORDER, STEPS,
};
use kwavers_core::error::KwaversResult;

fn main() -> KwaversResult<()> {
    let diagnostic = run_native_acoustic_diagnostic()?;
    let series = run_embedded_gaussian_series()?;

    println!("DG native 1-D acoustic diagnostic");
    println!("elements: {ELEMENTS}, polynomial_order: {POLYNOMIAL_ORDER}, steps: {STEPS}");
    println!("system: p_t + rho*c^2*u_x = 0, u_t + p_x/rho = 0");
    println!();
    println!(
        "{:<36} {:>16.6e}",
        "pressure_relative_l2", diagnostic.pressure_relative_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "velocity_relative_l2", diagnostic.velocity_relative_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "pressure_characteristic_l2", diagnostic.pressure_characteristic_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "velocity_characteristic_l2", diagnostic.velocity_characteristic_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "pressure_mass_error", diagnostic.pressure_mass_error
    );
    println!(
        "{:<36} {:>16.6e}",
        "velocity_mass_error", diagnostic.velocity_mass_error
    );
    println!("{:<36} {:>16.6e}", "energy_ratio", diagnostic.energy_ratio);
    println!();
    println!("embedded 1-D Gaussian pressure matrix");
    println!("fixture: localized p0 Gaussian, zero initial velocity, homogeneous lossless medium");
    print_solver_matrix(&series.matrix);
    println!();
    println!("common p4-quadrature Gaussian pressure matrix");
    print_common_solver_matrix(&series.common_matrix);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg_acoustic_common::{
        embedded_grid, gaussian_embedded_source, run_embedded_gaussian_solver_matrix, EMBEDDED_NY,
        EMBEDDED_NZ,
    };
    use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
    use std::f64::consts::PI;

    #[test]
    fn exact_standing_wave_phase_contract_is_finite() {
        let n_nodes = POLYNOMIAL_ORDER + 1;
        let (xi_nodes, _) = gauss_lobatto_quadrature(n_nodes).unwrap();
        let k = 2.0 * PI / (2.0 * ELEMENTS as f64);

        for elem in 0..ELEMENTS {
            for node in 0..xi_nodes.len() {
                let x = 2.0 * elem as f64 + xi_nodes[node] + 1.0;
                assert!((k * x).sin().is_finite());
            }
        }
    }

    #[test]
    fn native_acoustic_matches_exact_and_characteristic_references() {
        let diagnostic = run_native_acoustic_diagnostic().unwrap();

        assert!(diagnostic.pressure_mass_error < 1.0e-12);
        assert!(diagnostic.velocity_mass_error < 1.0e-12);
        assert!(diagnostic.pressure_relative_l2 < 2.0e-2);
        assert!(diagnostic.velocity_relative_l2 < 2.0e-2);
        assert!(diagnostic.pressure_characteristic_l2 < 2.0e-2);
        assert!(diagnostic.velocity_characteristic_l2 < 2.0e-2);
        assert!((diagnostic.energy_ratio - 1.0).abs() < 2.0e-2);
    }

    #[test]
    fn gaussian_embedded_source_is_centered_and_leapfrog_compatible() {
        let grid = embedded_grid().unwrap();
        let source = gaussian_embedded_source(&grid);
        let pressure = source.p0.as_ref().expect("pressure IVP");
        let (ux, uy, uz) = source.u0.as_ref().expect("velocity half-step IVP");
        let center_i = ELEMENTS;

        assert!(pressure.iter().all(|value| value.is_finite()));
        assert!(ux.iter().all(|value| value.is_finite()));
        assert_eq!(pressure[(center_i, EMBEDDED_NY / 2, EMBEDDED_NZ / 2)], 1.0);
        assert!(uy.iter().all(|value| *value == 0.0));
        assert!(uz.iter().all(|value| *value == 0.0));
        assert!(ux[(center_i - 2, EMBEDDED_NY / 2, EMBEDDED_NZ / 2)] > 0.0);
        assert!(ux[(center_i + 2, EMBEDDED_NY / 2, EMBEDDED_NZ / 2)] < 0.0);
    }

    #[test]
    fn embedded_gaussian_matrix_reports_bounded_solver_errors() {
        let matrix = run_embedded_gaussian_solver_matrix().unwrap();

        assert!(matrix.dg_pressure_mass_error < 1.0e-12);
        assert!(matrix.dg_exact_l2 < 2.0e-2);
        assert!(matrix.fdtd_exact_l2 < 2.0e-1);
        assert!(matrix.kspace_exact_l2 < 2.0e-1);
        assert!(matrix.pstd_exact_l2 < 2.0e-1);
        assert!(matrix.fdtd_pstd_l2.is_finite());
        assert!(matrix.kspace_pstd_l2.is_finite());
        assert!(DENSITY > 0.0 && DT > 0.0);
    }

    #[test]
    fn common_sampling_matrix_uses_one_physical_grid_for_all_solvers() {
        let series = dg_acoustic_common::run_embedded_gaussian_series().unwrap();
        let common_len = series.common_pressure_lines[0].samples.len();

        assert_eq!(series.common_pressure_lines.len(), 5);
        assert_eq!(series.common_error_lines.len(), 4);
        assert!(series
            .common_pressure_lines
            .iter()
            .all(|line| line.samples.len() == common_len));
        assert!(series.common_matrix.dg_exact_l2 < 2.0e-2);
        assert!(series.common_matrix.kspace_pstd_l2.is_finite());
        assert!(series.common_matrix.dg_pstd_l2.is_finite());
    }
}
