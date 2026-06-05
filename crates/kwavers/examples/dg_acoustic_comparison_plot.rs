//! Plot the shared DG/FDTD/PSTD acoustic comparison fixture.
//!
//! Outputs:
//! - `target/dg_acoustic_comparison/gaussian_pressure.png`
//! - `target/dg_acoustic_comparison/gaussian_pressure.csv`
//!
//! The artifact preserves native solver-grid traces and adds common p4
//! quadrature traces sampled at identical physical coordinates. It also adds
//! a uniform-grid DG trace on the native FDTD/PSTD grid by averaging DG traces
//! at shared element interfaces.

#[path = "dg_common/dg_acoustic_common.rs"]
mod dg_acoustic_common;

use anyhow::{anyhow, Result};
use dg_acoustic_common::{
    print_common_solver_matrix, print_solver_matrix, run_embedded_gaussian_series,
    EmbeddedGaussianSeries,
};
use plotters::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const OUT_DIR: &str = "target/dg_acoustic_comparison";
const PNG_NAME: &str = "gaussian_pressure.png";
const CSV_NAME: &str = "gaussian_pressure.csv";

fn main() -> Result<()> {
    let series = run_embedded_gaussian_series()?;
    let uniform = uniform_resampling(&series)?;
    let out_dir = PathBuf::from(OUT_DIR);
    fs::create_dir_all(&out_dir)?;
    let png_path = out_dir.join(PNG_NAME);
    let csv_path = out_dir.join(CSV_NAME);

    write_plot(&png_path, &series, &uniform)?;
    write_csv(&csv_path, &series, &uniform)?;

    println!("DG/FDTD/PSTD acoustic Gaussian comparison plot");
    println!("png: {}", png_path.display());
    println!("csv: {}", csv_path.display());
    print_solver_matrix(&series.matrix);
    println!();
    print_common_solver_matrix(&series.common_matrix);
    println!();
    print_uniform_solver_matrix(&uniform.matrix);
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct UniformGaussianMatrix {
    dg_exact_l2: f64,
    fdtd_exact_l2: f64,
    kspace_exact_l2: f64,
    pstd_exact_l2: f64,
    fdtd_pstd_l2: f64,
    kspace_pstd_l2: f64,
    dg_fdtd_l2: f64,
    dg_pstd_l2: f64,
}

#[derive(Debug, Clone)]
struct UniformGaussianSeries {
    matrix: UniformGaussianMatrix,
    pressure_lines: Vec<dg_acoustic_common::NamedLine>,
    error_lines: Vec<dg_acoustic_common::NamedLine>,
}

fn write_plot(
    path: &Path,
    series: &EmbeddedGaussianSeries,
    uniform: &UniformGaussianSeries,
) -> Result<()> {
    let root = BitMapBackend::new(path, (1500, 1200)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((3, 2));
    draw_lines(
        &panels[0],
        "Native-grid pressure at final time",
        "pressure",
        &series.pressure_lines,
    )?;
    draw_lines(
        &panels[1],
        "Common-grid pressure at final time",
        "pressure",
        &series.common_pressure_lines,
    )?;
    draw_lines(
        &panels[2],
        "Native-grid absolute error",
        "|p - p_exact|",
        &series.error_lines,
    )?;
    draw_lines(
        &panels[3],
        "Common-grid absolute error",
        "|p - p_exact|",
        &series.common_error_lines,
    )?;
    draw_lines(
        &panels[4],
        "Uniform-grid pressure at final time",
        "pressure",
        &uniform.pressure_lines,
    )?;
    draw_lines(
        &panels[5],
        "Uniform-grid absolute error",
        "|p - p_exact|",
        &uniform.error_lines,
    )?;
    root.present()?;
    Ok(())
}

fn draw_lines(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    title: &str,
    y_label: &str,
    lines: &[dg_acoustic_common::NamedLine],
) -> Result<()> {
    let (x_min, x_max, y_min, y_max) = line_bounds(lines);
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 24))
        .margin(18)
        .x_label_area_size(40)
        .y_label_area_size(58)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("x [grid cells]")
        .y_desc(y_label)
        .axis_desc_style(("sans-serif", 16))
        .draw()?;

    let colors = [BLACK, BLUE, RED, GREEN, MAGENTA, CYAN];
    for (idx, line) in lines.iter().enumerate() {
        let color = colors[idx % colors.len()];
        chart
            .draw_series(LineSeries::new(
                line.samples.iter().copied(),
                ShapeStyle::from(&color).stroke_width(2),
            ))?
            .label(line.name)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 28, y)],
                    ShapeStyle::from(&color).stroke_width(2),
                )
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 14))
        .draw()?;
    Ok(())
}

fn line_bounds(lines: &[dg_acoustic_common::NamedLine]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for line in lines {
        for &(x, y) in &line.samples {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    let y_pad = ((y_max - y_min) * 0.08).max(1.0e-12);
    (x_min, x_max, y_min - y_pad, y_max + y_pad)
}

fn write_csv(
    path: &Path,
    series: &EmbeddedGaussianSeries,
    uniform: &UniformGaussianSeries,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "kind,series,x,value")?;
    for line in &series.pressure_lines {
        write_line_rows(&mut file, "pressure", line)?;
    }
    for line in &series.error_lines {
        write_line_rows(&mut file, "absolute_error", line)?;
    }
    for line in &series.common_pressure_lines {
        write_line_rows(&mut file, "common_pressure", line)?;
    }
    for line in &series.common_error_lines {
        write_line_rows(&mut file, "common_absolute_error", line)?;
    }
    for line in &uniform.pressure_lines {
        write_line_rows(&mut file, "uniform_pressure", line)?;
    }
    for line in &uniform.error_lines {
        write_line_rows(&mut file, "uniform_absolute_error", line)?;
    }
    Ok(())
}

fn write_line_rows(
    file: &mut File,
    kind: &str,
    line: &dg_acoustic_common::NamedLine,
) -> Result<()> {
    for &(x, value) in &line.samples {
        writeln!(file, "{kind},{},{x:.12e},{value:.12e}", line.name)?;
    }
    Ok(())
}

fn uniform_resampling(series: &EmbeddedGaussianSeries) -> Result<UniformGaussianSeries> {
    let exact = find_line(&series.pressure_lines, "exact")?;
    let dg = find_line(&series.pressure_lines, "DG")?;
    let fdtd = find_line(&series.pressure_lines, "FDTD")?;
    let kspace = find_line(&series.pressure_lines, "FDTD+k-space")?;
    let pstd = find_line(&series.pressure_lines, "PSTD")?;
    let dg_uniform = resample_dg_trace_to_uniform_grid(dg, exact)?;

    let exact_values = values(exact);
    let dg_values = values(&dg_uniform);
    let fdtd_values = values(fdtd);
    let kspace_values = values(kspace);
    let pstd_values = values(pstd);
    let matrix = UniformGaussianMatrix {
        dg_exact_l2: relative_l2_slice(&dg_values, &exact_values),
        fdtd_exact_l2: relative_l2_slice(&fdtd_values, &exact_values),
        kspace_exact_l2: relative_l2_slice(&kspace_values, &exact_values),
        pstd_exact_l2: relative_l2_slice(&pstd_values, &exact_values),
        fdtd_pstd_l2: relative_l2_slice(&fdtd_values, &pstd_values),
        kspace_pstd_l2: relative_l2_slice(&kspace_values, &pstd_values),
        dg_fdtd_l2: relative_l2_slice(&dg_values, &fdtd_values),
        dg_pstd_l2: relative_l2_slice(&dg_values, &pstd_values),
    };
    let pressure_lines = vec![
        exact.clone(),
        dg_uniform,
        fdtd.clone(),
        kspace.clone(),
        pstd.clone(),
    ];
    let error_lines = vec![
        error_line("DG error", &pressure_lines[1], exact),
        error_line("FDTD error", fdtd, exact),
        error_line("FDTD+k-space error", kspace, exact),
        error_line("PSTD error", pstd, exact),
    ];
    Ok(UniformGaussianSeries {
        matrix,
        pressure_lines,
        error_lines,
    })
}

fn find_line<'a>(
    lines: &'a [dg_acoustic_common::NamedLine],
    name: &str,
) -> Result<&'a dg_acoustic_common::NamedLine> {
    lines
        .iter()
        .find(|line| line.name == name)
        .ok_or_else(|| anyhow!("missing pressure line {name}"))
}

fn resample_dg_trace_to_uniform_grid(
    dg: &dg_acoustic_common::NamedLine,
    exact_uniform: &dg_acoustic_common::NamedLine,
) -> Result<dg_acoustic_common::NamedLine> {
    let domain = dg_acoustic_common::domain_length();
    let mut samples = Vec::with_capacity(exact_uniform.samples.len());
    for &(x, _) in &exact_uniform.samples {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &(dg_x, dg_value) in &dg.samples {
            let distance = periodic_distance(dg_x, x, domain);
            if distance <= 1.0e-12 {
                sum += dg_value;
                count += 1;
            }
        }
        if count == 0 {
            return Err(anyhow!("DG trace does not contain uniform coordinate {x}"));
        }
        samples.push((x, sum / count as f64));
    }
    Ok(dg_acoustic_common::NamedLine {
        name: "DG",
        samples,
    })
}

fn periodic_distance(a: f64, b: f64, domain: f64) -> f64 {
    let distance = (a - b).abs();
    distance.min((domain - distance).abs())
}

fn values(line: &dg_acoustic_common::NamedLine) -> Vec<f64> {
    line.samples.iter().map(|&(_, value)| value).collect()
}

fn relative_l2_slice(actual: &[f64], expected: &[f64]) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let diff = actual - expected;
        diff_sq += diff * diff;
        expected_sq += expected * expected;
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

fn error_line(
    name: &'static str,
    actual: &dg_acoustic_common::NamedLine,
    expected: &dg_acoustic_common::NamedLine,
) -> dg_acoustic_common::NamedLine {
    dg_acoustic_common::NamedLine {
        name,
        samples: actual
            .samples
            .iter()
            .zip(expected.samples.iter())
            .map(|(&(x, actual), &(_, expected))| (x, (actual - expected).abs()))
            .collect(),
    }
}

fn print_uniform_solver_matrix(matrix: &UniformGaussianMatrix) {
    println!(
        "{:<36} {:>16.6e}",
        "uniform DG vs exact", matrix.dg_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "uniform FDTD vs exact", matrix.fdtd_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "uniform FDTD+k-space vs exact", matrix.kspace_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "uniform PSTD vs exact", matrix.pstd_exact_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "uniform FDTD vs PSTD", matrix.fdtd_pstd_l2
    );
    println!(
        "{:<36} {:>16.6e}",
        "uniform FDTD+k-space vs PSTD", matrix.kspace_pstd_l2
    );
    println!("{:<36} {:>16.6e}", "uniform DG vs FDTD", matrix.dg_fdtd_l2);
    println!("{:<36} {:>16.6e}", "uniform DG vs PSTD", matrix.dg_pstd_l2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_data_contains_all_solver_lines() {
        let series = run_embedded_gaussian_series().unwrap();
        let uniform = uniform_resampling(&series).unwrap();
        let names: Vec<_> = series.pressure_lines.iter().map(|line| line.name).collect();

        assert_eq!(names, vec!["exact", "DG", "FDTD", "FDTD+k-space", "PSTD"]);
        assert_eq!(series.error_lines.len(), 4);
        assert_eq!(series.common_pressure_lines.len(), 5);
        assert_eq!(series.common_error_lines.len(), 4);
        assert_eq!(uniform.pressure_lines.len(), 5);
        assert_eq!(uniform.error_lines.len(), 4);
        assert!(series.matrix.dg_exact_l2 < 2.0e-2);
        assert!(series.common_matrix.dg_exact_l2 < 2.0e-2);
        assert!(uniform.matrix.dg_exact_l2 < 2.0e-2);
        assert!(series.common_matrix.kspace_pstd_l2.is_finite());
    }

    #[test]
    fn uniform_resampling_averages_dg_interface_traces() {
        let series = run_embedded_gaussian_series().unwrap();
        let exact = find_line(&series.pressure_lines, "exact").unwrap();
        let dg = find_line(&series.pressure_lines, "DG").unwrap();
        let uniform = uniform_resampling(&series).unwrap();
        let uniform_dg = find_line(&uniform.pressure_lines, "DG").unwrap();
        let interface_x = 2.0;
        let raw_values: Vec<_> = dg
            .samples
            .iter()
            .filter(|&&(x, _)| (x - interface_x).abs() <= 1.0e-12)
            .map(|&(_, value)| value)
            .collect();
        let averaged = raw_values.iter().sum::<f64>() / raw_values.len() as f64;
        let resampled = uniform_dg
            .samples
            .iter()
            .find(|&&(x, _)| (x - interface_x).abs() <= 1.0e-12)
            .map(|&(_, value)| value)
            .unwrap();

        assert_eq!(exact.samples.len(), uniform_dg.samples.len());
        assert_eq!(raw_values.len(), 2);
        assert!((resampled - averaged).abs() <= 1.0e-14);
    }

    #[test]
    fn plot_and_csv_writers_create_nonempty_artifacts() {
        let series = run_embedded_gaussian_series().unwrap();
        let uniform = uniform_resampling(&series).unwrap();
        let out_dir = PathBuf::from(OUT_DIR);
        fs::create_dir_all(&out_dir).unwrap();
        let png_path = out_dir.join("test_gaussian_pressure.png");
        let csv_path = out_dir.join("test_gaussian_pressure.csv");

        write_plot(&png_path, &series, &uniform).unwrap();
        write_csv(&csv_path, &series, &uniform).unwrap();

        assert!(fs::metadata(&png_path).unwrap().len() > 10_000);
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(csv.contains("pressure,DG"));
        assert!(csv.contains("absolute_error,FDTD+k-space error"));
        assert!(csv.contains("common_pressure,DG"));
        assert!(csv.contains("common_absolute_error,PSTD error"));
        assert!(csv.contains("uniform_pressure,DG"));
        assert!(csv.contains("uniform_absolute_error,PSTD error"));
    }
}
