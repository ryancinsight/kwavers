//! Acoustic timestep-refinement sweep for the DG/FDTD/PSTD Gaussian fixture.
//!
//! Outputs:
//! - `target/dg_acoustic_comparison/timestep_sweep.png`
//! - `target/dg_acoustic_comparison/timestep_sweep.csv`
//!
//! Each solver advances to the same final time. DG is resampled onto the native
//! uniform FDTD/PSTD grid by averaging duplicate element-interface traces.

#[path = "dg_common/dg_acoustic_common.rs"]
mod dg_acoustic_common;

use anyhow::{anyhow, Result};
use dg_acoustic_common::{
    domain_length, exact_gaussian_pressure, gaussian_derivative, gaussian_profile,
    physical_coordinate, weighted_mass, DENSITY, DT, ELEMENTS, EMBEDDED_NY, EMBEDDED_NZ,
    POLYNOMIAL_ORDER, SOUND_SPEED, STEPS,
};
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::{GridSource, SourceMode};
use kwavers::solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers::solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod};
use kwavers::solver::forward::pstd::dg::dg_solver::acoustic::AcousticDg1DWorkspace;
use kwavers::solver::forward::pstd::dg::quadrature::gauss_lobatto_quadrature;
use kwavers::solver::forward::pstd::dg::{DGConfig, DGSolver};
use kwavers::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers::solver::interface::solver::Solver;
use ndarray::{Array1, Array3};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const STEP_COUNTS: [usize; 3] = [20, 40, 80];
const OUT_DIR: &str = "target/dg_acoustic_comparison";
const PNG_NAME: &str = "timestep_sweep.png";
const CSV_NAME: &str = "timestep_sweep.csv";

#[derive(Debug, Clone, Copy)]
struct TimestepRow {
    steps: usize,
    dt: f64,
    dg_exact_l2: f64,
    fdtd_exact_l2: f64,
    kspace_exact_l2: f64,
    pstd_exact_l2: f64,
    fdtd_pstd_l2: f64,
    kspace_pstd_l2: f64,
    dg_pstd_l2: f64,
    dg_pressure_mass_error: f64,
}

fn main() -> Result<()> {
    let rows = run_timestep_sweep()?;
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;
    let png_path = out_dir.join(PNG_NAME);
    let csv_path = out_dir.join(CSV_NAME);

    write_plot(&png_path, &rows)?;
    write_csv(&csv_path, &rows)?;

    println!("DG/FDTD/PSTD acoustic Gaussian timestep sweep");
    println!("png: {}", png_path.display());
    println!("csv: {}", csv_path.display());
    println!(
        "{:<8} {:>10} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "steps", "dt", "DG", "FDTD", "FDTD+k", "PSTD", "DG-PSTD"
    );
    for row in &rows {
        println!(
            "{:<8} {:>10.4e} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e}",
            row.steps,
            row.dt,
            row.dg_exact_l2,
            row.fdtd_exact_l2,
            row.kspace_exact_l2,
            row.pstd_exact_l2,
            row.dg_pstd_l2
        );
    }
    Ok(())
}

fn run_timestep_sweep() -> Result<Vec<TimestepRow>> {
    STEP_COUNTS.iter().copied().map(run_timestep_case).collect()
}

fn run_timestep_case(steps: usize) -> Result<TimestepRow> {
    let final_time = DT * STEPS as f64;
    let dt = final_time / steps as f64;
    let (dg_line, dg_mass_error) = run_dg_uniform(dt, steps)?;
    let fdtd_line = run_fdtd_uniform(dt, steps, KSpaceCorrectionMode::None)?;
    let kspace_line = run_fdtd_uniform(dt, steps, KSpaceCorrectionMode::Spectral)?;
    let pstd_line = run_pstd_uniform(dt, steps)?;
    let exact_line = exact_uniform_line(final_time);

    Ok(TimestepRow {
        steps,
        dt,
        dg_exact_l2: relative_l2(&dg_line, &exact_line),
        fdtd_exact_l2: relative_l2(&fdtd_line, &exact_line),
        kspace_exact_l2: relative_l2(&kspace_line, &exact_line),
        pstd_exact_l2: relative_l2(&pstd_line, &exact_line),
        fdtd_pstd_l2: relative_l2(&fdtd_line, &pstd_line),
        kspace_pstd_l2: relative_l2(&kspace_line, &pstd_line),
        dg_pstd_l2: relative_l2(&dg_line, &pstd_line),
        dg_pressure_mass_error: dg_mass_error,
    })
}

fn run_dg_uniform(dt: f64, steps: usize) -> Result<(Array1<f64>, f64)> {
    let n_nodes = POLYNOMIAL_ORDER + 1;
    let (xi_nodes, weights) = gauss_lobatto_quadrature(n_nodes)?;
    let mut pressure = Array3::zeros((ELEMENTS, n_nodes, 1));
    let mut velocity = Array3::zeros((ELEMENTS, n_nodes, 1));
    for elem in 0..ELEMENTS {
        for node in 0..n_nodes {
            pressure[(elem, node, 0)] = gaussian_profile(physical_coordinate(elem, xi_nodes[node]));
        }
    }
    let initial_mass = weighted_mass(&pressure, &weights);
    let grid = Arc::new(Grid::new(ELEMENTS * n_nodes, 1, 1, 1.0, 1.0, 1.0)?);
    let config = DGConfig {
        polynomial_order: POLYNOMIAL_ORDER,
        sound_speed: SOUND_SPEED,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, grid)?;
    let mut workspace = AcousticDg1DWorkspace::new(pressure.dim());
    for _ in 0..steps {
        solver.step_acoustic_1d_ssp_rk3(
            &mut pressure,
            &mut velocity,
            DENSITY,
            dt,
            &mut workspace,
        )?;
    }
    let mass_error = (weighted_mass(&pressure, &weights) - initial_mass).abs();
    Ok((resample_dg_to_uniform(&pressure, &xi_nodes)?, mass_error))
}

fn run_fdtd_uniform(
    dt: f64,
    steps: usize,
    kspace_correction: KSpaceCorrectionMode,
) -> Result<Array1<f64>> {
    let grid = embedded_grid()?;
    let medium = HomogeneousMedium::new(DENSITY, SOUND_SPEED, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        cfl_factor: dt * SOUND_SPEED,
        kspace_correction,
        dt,
        nt: steps,
        ..FdtdConfig::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, gaussian_source(&grid, dt))?;
    for _ in 0..steps {
        solver.step_forward()?;
    }
    Ok(center_line(Solver::pressure_field(&solver)))
}

fn run_pstd_uniform(dt: f64, steps: usize) -> Result<Array1<f64>> {
    let grid = embedded_grid()?;
    let medium = HomogeneousMedium::new(DENSITY, SOUND_SPEED, 0.0, 0.0, &grid);
    let config = PSTDConfig {
        dt,
        nt: steps,
        boundary: BoundaryConfig::None,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid.clone(), &medium, gaussian_source(&grid, dt))?;
    for _ in 0..steps {
        solver.step_forward()?;
    }
    Ok(center_line(Solver::pressure_field(&solver)))
}

fn embedded_grid() -> Result<Grid> {
    Ok(Grid::new(
        2 * ELEMENTS,
        EMBEDDED_NY,
        EMBEDDED_NZ,
        1.0,
        1.0,
        1.0,
    )?)
}

fn gaussian_source(grid: &Grid, dt: f64) -> GridSource {
    let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut ux = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let uy = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let uz = Array3::zeros((grid.nx, grid.ny, grid.nz));
    for i in 0..grid.nx {
        let x = i as f64;
        let ux_x = x + 0.5;
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                pressure[(i, j, k)] = gaussian_profile(x);
                ux[(i, j, k)] = 0.5 * dt / DENSITY * gaussian_derivative(ux_x);
            }
        }
    }
    GridSource {
        p0: Some(pressure),
        u0: Some((ux, uy, uz)),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    }
}

fn resample_dg_to_uniform(pressure: &Array3<f64>, xi_nodes: &Array1<f64>) -> Result<Array1<f64>> {
    let n = 2 * ELEMENTS;
    let domain = domain_length();
    let mut output = Array1::zeros(n);
    for i in 0..n {
        let x = i as f64;
        let mut sum = 0.0;
        let mut count = 0usize;
        for elem in 0..ELEMENTS {
            for node in 0..xi_nodes.len() {
                let dg_x = physical_coordinate(elem, xi_nodes[node]);
                if periodic_distance(dg_x, x, domain) <= 1.0e-12 {
                    sum += pressure[(elem, node, 0)];
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Err(anyhow!("DG trace does not contain uniform coordinate {x}"));
        }
        output[i] = sum / count as f64;
    }
    Ok(output)
}

fn center_line(field: &Array3<f64>) -> Array1<f64> {
    let (_, ny, nz) = field.dim();
    Array1::from_shape_fn(field.dim().0, |i| field[(i, ny / 2, nz / 2)])
}

fn exact_uniform_line(final_time: f64) -> Array1<f64> {
    Array1::from_shape_fn(2 * ELEMENTS, |i| {
        exact_gaussian_pressure(i as f64, final_time)
    })
}

fn periodic_distance(a: f64, b: f64, domain: f64) -> f64 {
    let distance = (a - b).abs();
    distance.min((domain - distance).abs())
}

fn relative_l2(actual: &Array1<f64>, expected: &Array1<f64>) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let diff = actual - expected;
        diff_sq += diff * diff;
        expected_sq += expected * expected;
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

fn write_plot(path: &Path, rows: &[TimestepRow]) -> Result<()> {
    let root = BitMapBackend::new(path, (1100, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x_min, x_max, y_min, y_max) = plot_bounds(rows);
    let mut chart = ChartBuilder::on(&root)
        .caption("Acoustic Gaussian timestep sweep", ("sans-serif", 28))
        .margin(28)
        .x_label_area_size(54)
        .y_label_area_size(72)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("log10(dt)")
        .y_desc("log10(relative L2 pressure error)")
        .axis_desc_style(("sans-serif", 18))
        .draw()?;
    draw_error_series(&mut chart, rows, "DG", BLUE, |row| row.dg_exact_l2)?;
    draw_error_series(&mut chart, rows, "FDTD", RED, |row| row.fdtd_exact_l2)?;
    draw_error_series(&mut chart, rows, "FDTD+k-space", GREEN, |row| {
        row.kspace_exact_l2
    })?;
    draw_error_series(&mut chart, rows, "PSTD", MAGENTA, |row| row.pstd_exact_l2)?;
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 16))
        .draw()?;
    root.present()?;
    Ok(())
}

fn draw_error_series<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    rows: &[TimestepRow],
    name: &'static str,
    color: RGBColor,
    value: impl Fn(&TimestepRow) -> f64,
) -> Result<()>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    chart
        .draw_series(LineSeries::new(
            rows.iter().map(|row| (row.dt.log10(), value(row).log10())),
            ShapeStyle::from(&color).stroke_width(3),
        ))?
        .label(name)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 28, y)], color));
    chart.draw_series(
        rows.iter()
            .map(|row| Circle::new((row.dt.log10(), value(row).log10()), 4, color.filled())),
    )?;
    Ok(())
}

fn plot_bounds(rows: &[TimestepRow]) -> (f64, f64, f64, f64) {
    let x_min = rows
        .iter()
        .map(|row| row.dt.log10())
        .fold(f64::INFINITY, f64::min);
    let x_max = rows
        .iter()
        .map(|row| row.dt.log10())
        .fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for row in rows {
        for value in [
            row.dg_exact_l2,
            row.fdtd_exact_l2,
            row.kspace_exact_l2,
            row.pstd_exact_l2,
        ] {
            let log_value = value.log10();
            y_min = y_min.min(log_value);
            y_max = y_max.max(log_value);
        }
    }
    let y_pad = ((y_max - y_min) * 0.1).max(0.2);
    (x_min - 0.05, x_max + 0.05, y_min - y_pad, y_max + y_pad)
}

fn write_csv(path: &Path, rows: &[TimestepRow]) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "steps,dt,dg_exact_l2,fdtd_exact_l2,kspace_exact_l2,pstd_exact_l2,fdtd_pstd_l2,kspace_pstd_l2,dg_pstd_l2,dg_pressure_mass_error"
    )?;
    for row in rows {
        writeln!(
            file,
            "{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.steps,
            row.dt,
            row.dg_exact_l2,
            row.fdtd_exact_l2,
            row.kspace_exact_l2,
            row.pstd_exact_l2,
            row.fdtd_pstd_l2,
            row.kspace_pstd_l2,
            row.dg_pstd_l2,
            row.dg_pressure_mass_error
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_sweep_reports_finite_uniform_grid_metrics() {
        let rows = run_timestep_sweep().unwrap();

        assert_eq!(rows.len(), STEP_COUNTS.len());
        assert_eq!(rows[0].steps, 20);
        assert_eq!(rows[1].steps, 40);
        assert_eq!(rows[2].steps, 80);
        assert!((rows[1].dt * 2.0 - rows[0].dt).abs() < 1.0e-15);
        assert!((rows[2].dt * 2.0 - rows[1].dt).abs() < 1.0e-15);
        for row in &rows {
            assert!(row.dg_exact_l2.is_finite());
            assert!(row.fdtd_exact_l2.is_finite());
            assert!(row.kspace_exact_l2.is_finite());
            assert!(row.pstd_exact_l2.is_finite());
            assert!(row.fdtd_pstd_l2.is_finite());
            assert!(row.kspace_pstd_l2.is_finite());
            assert!(row.dg_pstd_l2.is_finite());
            assert!(row.dg_pressure_mass_error < 1.0e-12);
        }
    }

    #[test]
    fn timestep_plot_and_csv_are_nonempty_artifacts() {
        let rows = run_timestep_sweep().unwrap();
        let out_dir = PathBuf::from(OUT_DIR);
        fs::create_dir_all(&out_dir).unwrap();
        let png_path = out_dir.join("test_timestep_sweep.png");
        let csv_path = out_dir.join("test_timestep_sweep.csv");

        write_plot(&png_path, &rows).unwrap();
        write_csv(&csv_path, &rows).unwrap();

        assert!(fs::metadata(&png_path).unwrap().len() > 10_000);
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(csv.contains("kspace_exact_l2"));
        assert!(csv.contains("80,"));
    }
}
