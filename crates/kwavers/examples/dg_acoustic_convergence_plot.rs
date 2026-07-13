//! DG acoustic p-refinement convergence plot for the Gaussian comparison case.
//!
//! Outputs:
//! - `target/dg_acoustic_comparison/dg_order_convergence.png`
//! - `target/dg_acoustic_comparison/dg_order_convergence.csv`
//!
//! The plot uses one common p4 GLL quadrature grid for every order. The CSV
//! also preserves the per-order nodal metric to show when node-set changes
//! alias the comparison.

#[path = "dg_common/dg_acoustic_common.rs"]
mod dg_acoustic_common;

use anyhow::Result;
use dg_acoustic_common::{
    exact_gaussian_pressure, gaussian_profile, physical_coordinate, relative_l2, weighted_mass,
    DENSITY, DT, ELEMENTS, SOUND_SPEED, STEPS,
};
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::dg::dg_solver::acoustic::AcousticDg1DWorkspace;
use kwavers_solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers_solver::forward::pstd::dg::{DGConfig, DGSolver};
use leto::{Array1, Array3};
use plotters::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const ORDERS: [usize; 4] = [1, 2, 3, 4];
const COMMON_QUADRATURE_ORDER: usize = ORDERS[ORDERS.len() - 1];
const OUT_DIR: &str = "target/dg_acoustic_comparison";
const PNG_NAME: &str = "dg_order_convergence.png";
const CSV_NAME: &str = "dg_order_convergence.csv";

#[derive(Debug, Clone, Copy)]
struct DgConvergenceRow {
    polynomial_order: usize,
    degrees_of_freedom: usize,
    nodal_pressure_relative_l2: f64,
    common_pressure_relative_l2: f64,
    pressure_mass_error: f64,
}

fn main() -> Result<()> {
    let rows = run_convergence_sweep()?;
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;
    let png_path = out_dir.join(PNG_NAME);
    let csv_path = out_dir.join(CSV_NAME);

    write_plot(&png_path, &rows)?;
    write_csv(&csv_path, &rows)?;

    println!("DG acoustic Gaussian p-refinement convergence");
    println!("png: {}", png_path.display());
    println!("csv: {}", csv_path.display());
    println!(
        "{:<18} {:>8} {:>18} {:>18} {:>18}",
        "polynomial_order", "dofs", "nodal_l2", "common_l2", "mass_error"
    );
    for row in &rows {
        println!(
            "{:<18} {:>8} {:>18.6e} {:>18.6e} {:>18.6e}",
            row.polynomial_order,
            row.degrees_of_freedom,
            row.nodal_pressure_relative_l2,
            row.common_pressure_relative_l2,
            row.pressure_mass_error
        );
    }
    Ok(())
}

fn run_convergence_sweep() -> Result<Vec<DgConvergenceRow>> {
    ORDERS.iter().copied().map(run_dg_order).collect()
}

fn run_dg_order(polynomial_order: usize) -> Result<DgConvergenceRow> {
    let n_nodes = polynomial_order + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let mut pressure = Array3::zeros((ELEMENTS, n_nodes, 1));
    let mut velocity = Array3::zeros((ELEMENTS, n_nodes, 1));
    for elem in 0..ELEMENTS {
        for node in 0..n_nodes {
            pressure[[elem, node, 0]] = gaussian_profile(physical_coordinate(elem, xi_nodes[node]));
        }
    }
    let initial_mass = weighted_mass(&pressure, &weights);

    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, grid)?;
    let mut workspace = AcousticDg1DWorkspace::new({
        let s = pressure.shape();
        (s[0], s[1], s[2])
    });
    for _ in 0..STEPS {
        solver.step_acoustic_1d_ssp_rk3(
            &mut pressure,
            &mut velocity,
            DENSITY,
            DT,
            &mut workspace,
        )?;
    }

    let exact = exact_dg_pressure(&xi_nodes, STEPS as f64 * DT);
    let common_pressure_relative_l2 =
        common_quadrature_l2(&pressure, &xi_nodes, STEPS as f64 * DT)?;
    Ok(DgConvergenceRow {
        polynomial_order,
        degrees_of_freedom: ELEMENTS * n_nodes,
        nodal_pressure_relative_l2: relative_l2(&pressure, &exact, &weights),
        common_pressure_relative_l2,
        pressure_mass_error: (weighted_mass(&pressure, &weights) - initial_mass).abs(),
    })
}

fn exact_dg_pressure(xi_nodes: &Array1<f64>, time: f64) -> Array3<f64> {
    let mut pressure = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            pressure[[elem, node, 0]] =
                exact_gaussian_pressure(physical_coordinate(elem, xi_nodes[node]), time);
        }
    }
    pressure
}

fn common_quadrature_l2(
    pressure: &Array3<f64>,
    solution_nodes: &Array1<f64>,
    time: f64,
) -> Result<f64> {
    let (reference_nodes, reference_weights) =
        gauss_lobatto_quadrature(COMMON_QUADRATURE_ORDER + 1)?;
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for elem in 0..ELEMENTS {
        for reference_node in 0..reference_nodes.len() {
            let xi = reference_nodes[reference_node];
            let actual = interpolate_lagrange(pressure, elem, solution_nodes, xi);
            let expected = exact_gaussian_pressure(physical_coordinate(elem, xi), time);
            let diff = actual - expected;
            diff_sq += reference_weights[reference_node] * diff * diff;
            expected_sq += reference_weights[reference_node] * expected * expected;
        }
    }
    Ok(diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON))
}

fn interpolate_lagrange(
    pressure: &Array3<f64>,
    elem: usize,
    solution_nodes: &Array1<f64>,
    xi: f64,
) -> f64 {
    for node in 0..solution_nodes.len() {
        if (xi - solution_nodes[node]).abs() <= 1.0e-14 {
            return pressure[[elem, node, 0]];
        }
    }

    let mut value = 0.0;
    for node in 0..solution_nodes.len() {
        let mut basis = 1.0;
        for other in 0..solution_nodes.len() {
            if other != node {
                basis *=
                    (xi - solution_nodes[other]) / (solution_nodes[node] - solution_nodes[other]);
            }
        }
        value += pressure[[elem, node, 0]] * basis;
    }
    value
}

fn write_plot(path: &Path, rows: &[DgConvergenceRow]) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x_min, x_max, y_min, y_max) = plot_bounds(rows);
    let mut chart = ChartBuilder::on(&root)
        .caption("DG acoustic Gaussian p-refinement", ("sans-serif", 28))
        .margin(28)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("polynomial order")
        .y_desc("log10(relative L2 pressure error)")
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            rows.iter().map(|row| {
                (
                    row.polynomial_order as f64,
                    row.common_pressure_relative_l2.log10(),
                )
            }),
            ShapeStyle::from(&BLUE).stroke_width(3),
        ))?
        .label("common p4 quadrature")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], BLUE));
    chart
        .draw_series(LineSeries::new(
            rows.iter().map(|row| {
                (
                    row.polynomial_order as f64,
                    row.nodal_pressure_relative_l2.log10(),
                )
            }),
            ShapeStyle::from(&RED).stroke_width(2),
        ))?
        .label("per-order nodal quadrature")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], RED));
    chart.draw_series(rows.iter().map(|row| {
        Circle::new(
            (
                row.polynomial_order as f64,
                row.common_pressure_relative_l2.log10(),
            ),
            5,
            BLUE.filled(),
        )
    }))?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;

    root.present()?;
    Ok(())
}

fn plot_bounds(rows: &[DgConvergenceRow]) -> (f64, f64, f64, f64) {
    let order_min = rows
        .iter()
        .map(|row| row.polynomial_order)
        .min()
        .unwrap_or(1) as f64;
    let order_max = rows
        .iter()
        .map(|row| row.polynomial_order)
        .max()
        .unwrap_or(1) as f64;
    let min_log = rows
        .iter()
        .flat_map(|row| {
            [
                row.nodal_pressure_relative_l2.log10(),
                row.common_pressure_relative_l2.log10(),
            ]
        })
        .fold(f64::INFINITY, f64::min);
    let max_log = rows
        .iter()
        .flat_map(|row| {
            [
                row.nodal_pressure_relative_l2.log10(),
                row.common_pressure_relative_l2.log10(),
            ]
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let pad = ((max_log - min_log) * 0.1).max(0.2);
    (
        order_min - 0.25,
        order_max + 0.25,
        min_log - pad,
        max_log + pad,
    )
}

fn write_csv(path: &Path, rows: &[DgConvergenceRow]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "polynomial_order,degrees_of_freedom,nodal_pressure_relative_l2,common_pressure_relative_l2,pressure_mass_error"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{},{:.12e},{:.12e},{:.12e}",
            row.polynomial_order,
            row.degrees_of_freedom,
            row.nodal_pressure_relative_l2,
            row.common_pressure_relative_l2,
            row.pressure_mass_error
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dg_order_sweep_improves_gaussian_pressure_error() {
        let rows = run_convergence_sweep().unwrap();

        assert_eq!(rows.len(), ORDERS.len());
        assert!(rows
            .iter()
            .all(|row| row.nodal_pressure_relative_l2.is_finite()));
        assert!(rows
            .iter()
            .all(|row| row.common_pressure_relative_l2.is_finite()));
        assert!(rows.iter().all(|row| row.pressure_mass_error < 1.0e-12));
        assert!(rows[1].common_pressure_relative_l2 < rows[0].common_pressure_relative_l2);
        assert!(rows[3].common_pressure_relative_l2 < rows[1].common_pressure_relative_l2);
        assert!(rows[1].nodal_pressure_relative_l2 > rows[0].nodal_pressure_relative_l2);
    }

    #[test]
    fn convergence_plot_and_csv_are_nonempty_artifacts() {
        let rows = run_convergence_sweep().unwrap();
        let out_dir = PathBuf::from(OUT_DIR);
        fs::create_dir_all(&out_dir).unwrap();
        let png_path = out_dir.join("test_dg_order_convergence.png");
        let csv_path = out_dir.join("test_dg_order_convergence.csv");

        write_plot(&png_path, &rows).unwrap();
        write_csv(&csv_path, &rows).unwrap();

        assert!(fs::metadata(&png_path).unwrap().len() > 10_000);
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(csv.contains("common_pressure_relative_l2"));
        assert!(csv.contains(",60,"));
    }
}
