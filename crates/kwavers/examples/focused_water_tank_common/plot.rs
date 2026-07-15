use super::physics;
use super::{ProfileSet, WaterTankOutput};
use anyhow::Result;
use leto::Array2;
use plotters::coord::types::RangedCoordf64;
use plotters::coord::Shift;
use plotters::prelude::*;
use std::path::Path;

const COLORS: [RGBColor; 6] = [BLUE, RED, MAGENTA, CYAN, GREEN, BLACK];

pub fn write_plot(path: &Path, output: &WaterTankOutput) -> Result<()> {
    let root = BitMapBackend::new(path, (1900, 950)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 4));

    draw_map(
        &areas[0],
        "FDTD+CPML gated peak",
        &output.solver_fields[0].normalized_peak,
    )?;
    draw_map(
        &areas[1],
        "PSTD+CPML gated peak",
        &output.solver_fields[1].normalized_peak,
    )?;
    draw_map(
        &areas[2],
        "DG-2D gated peak",
        &output.solver_fields[2].normalized_peak,
    )?;
    draw_map(
        &areas[3],
        "DG-3D gated peak",
        &output.solver_fields[3].normalized_peak,
    )?;
    draw_map(
        &areas[4],
        "Analytical focused array",
        &output.solver_fields[4].normalized_peak,
    )?;
    draw_error_map(
        &areas[5],
        "|FDTD - PSTD| normalized map error",
        &output.solver_fields[0].normalized_peak,
        &output.solver_fields[1].normalized_peak,
    )?;
    draw_profiles(
        &areas[6],
        "Axial profiles at focus x",
        &output.profiles,
        true,
    )?;
    draw_profiles(
        &areas[7],
        "Lateral profiles at focus y",
        &output.profiles,
        false,
    )?;

    root.present()?;
    Ok(())
}

fn draw_map(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    map: &Array2<f64>,
) -> Result<()> {
    let x_max = physics::NX as f64 * physics::DX * 1.0e3;
    let y_max = physics::NY as f64 * physics::DX * 1.0e3;
    let mut chart = ChartBuilder::on(area)
        .margin(12)
        .caption(title, ("sans-serif", 20))
        .x_label_area_size(36)
        .y_label_area_size(44)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("x [mm]")
        .y_desc("y [mm]")
        .disable_mesh()
        .draw()?;

    draw_heat_cells(&mut chart, map, false)?;
    draw_source_and_focus(&mut chart)?;
    Ok(())
}

fn draw_error_map(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    lhs: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<()> {
    let error = Array2::from_shape_fn((physics::NX, physics::NY), |[i, j]| {
        (lhs[[i, j]] - rhs[[i, j]]).abs()
    });
    let x_max = physics::NX as f64 * physics::DX * 1.0e3;
    let y_max = physics::NY as f64 * physics::DX * 1.0e3;
    let mut chart = ChartBuilder::on(area)
        .margin(12)
        .caption(title, ("sans-serif", 20))
        .x_label_area_size(36)
        .y_label_area_size(44)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("x [mm]")
        .y_desc("y [mm]")
        .disable_mesh()
        .draw()?;

    draw_heat_cells(&mut chart, &error, true)?;
    draw_source_and_focus(&mut chart)?;
    Ok(())
}

fn draw_heat_cells<DB: DrawingBackend>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    map: &Array2<f64>,
    error_palette: bool,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let cell = physics::DX * 1.0e3;
    chart.draw_series((0..physics::NX).flat_map(|i| {
        (0..physics::NY).map(move |j| {
            let value = map[[i, j]].clamp(0.0, 1.0);
            let color = if error_palette {
                HSLColor(25.0 / 360.0, 0.85, 0.98 - 0.55 * value)
            } else {
                HSLColor(
                    230.0 / 360.0 - 230.0 / 360.0 * value,
                    0.88,
                    0.92 - 0.48 * value,
                )
            };
            let x = i as f64 * cell;
            let y = j as f64 * cell;
            Rectangle::new([(x, y), (x + cell, y + cell)], color.filled())
        })
    }))?;
    Ok(())
}

fn draw_source_and_focus<DB: DrawingBackend>(
    chart: &mut ChartContext<'_, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let elements = physics::elements();
    chart.draw_series(elements.iter().map(|element| {
        Circle::new(
            (
                element.x as f64 * physics::DX * 1.0e3,
                element.y as f64 * physics::DX * 1.0e3,
            ),
            2,
            BLACK.filled(),
        )
    }))?;
    chart.draw_series(std::iter::once(Cross::new(physics::focus_mm(), 8, BLACK)))?;
    Ok(())
}

fn draw_profiles(
    area: &DrawingArea<BitMapBackend<'_>, Shift>,
    title: &str,
    profiles: &[ProfileSet],
    axial: bool,
) -> Result<()> {
    let x_bounds = if axial {
        (
            -(physics::FOCUS_Y as f64) * physics::DX * 1.0e3,
            (physics::NY - physics::FOCUS_Y) as f64 * physics::DX * 1.0e3,
        )
    } else {
        (
            -(physics::FOCUS_X as f64) * physics::DX * 1.0e3,
            (physics::NX - physics::FOCUS_X) as f64 * physics::DX * 1.0e3,
        )
    };
    let mut chart = ChartBuilder::on(area)
        .margin(12)
        .caption(title, ("sans-serif", 20))
        .x_label_area_size(38)
        .y_label_area_size(48)
        .build_cartesian_2d(x_bounds.0..x_bounds.1, 0.0..1.05)?;

    chart
        .configure_mesh()
        .x_desc("offset from focus [mm]")
        .y_desc("normalized peak")
        .draw()?;

    for (idx, profile) in profiles.iter().enumerate() {
        let color = COLORS[idx % COLORS.len()];
        let samples = if axial {
            &profile.axial
        } else {
            &profile.lateral
        };
        if samples.is_empty() {
            continue;
        }
        chart
            .draw_series(LineSeries::new(
                samples.iter().copied(),
                color.stroke_width(2),
            ))?
            .label(profile.name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 18, y)], color.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.82))
        .border_style(BLACK)
        .draw()?;
    Ok(())
}
