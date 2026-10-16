//! Figure output for the linear-array study. Plots are regenerated from the run's arrays.

use anyhow::{anyhow, Result};
use eunomia::convert::IntegerTarget;
use eunomia::FloatElement;
use leto::Array2;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::{register_font, FontStyle};
use std::path::{Path, PathBuf};

/// Register Arial under the `sans-serif` family for the text-capable bitmap backend.
///
/// The ab_glyph backend resolves text by registered font bytes, not by system name,
/// so the font file is read from the Windows font directory and must exist.
///
/// # Errors
///
/// Returns an error if `%SystemRoot%\Fontsrial.ttf` is missing or invalid.
pub fn init_fonts() -> Result<()> {
    let root = std::env::var_os("SystemRoot")
        .ok_or_else(|| anyhow!("SystemRoot is not set; cannot locate a font"))?;
    let path: PathBuf = PathBuf::from(root).join("Fonts").join("arial.ttf");
    let bytes =
        std::fs::read(&path).map_err(|error| anyhow!("reading {}: {error}", path.display()))?;
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    register_font("sans-serif", FontStyle::Normal, leaked)
        .map_err(|_| anyhow!("{} is not a valid OpenType font", path.display()))?;
    Ok(())
}

/// Five-stop perceptual colour ramp (viridis) for heat maps, `t` in `[0, 1]`.
fn viridis(t: f64) -> RGBColor {
    const STOPS: [(f64, (f64, f64, f64)); 5] = [
        (0.0, (68.0, 1.0, 84.0)),
        (0.25, (59.0, 82.0, 139.0)),
        (0.5, (33.0, 145.0, 140.0)),
        (0.75, (94.0, 201.0, 98.0)),
        (1.0, (253.0, 231.0, 37.0)),
    ];
    let t = t.clamp(0.0, 1.0);
    for window in STOPS.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: f64, b: f64| u8::from_truncated((a + f * (b - a)).round());
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = STOPS[STOPS.len() - 1];
    RGBColor(
        u8::from_truncated(c.0),
        u8::from_truncated(c.1),
        u8::from_truncated(c.2),
    )
}

/// Heat map of `data[i, k]` (lateral × depth) in dB re `reference`, floored at `floor_db`.
fn draw_db_map<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title: &str,
    data: &Array2<f64>,
    dx_mm: f64,
    reference: f64,
    floor_db: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let [nx, nz] = data.shape();
    let x_max = f64::from_count(nx) * dx_mm;
    let z_max = f64::from_count(nz) * dx_mm;
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..x_max, 0.0..z_max)?;
    chart
        .configure_mesh()
        .x_desc("lateral x (mm)")
        .y_desc("depth z (mm)")
        .draw()?;
    let cells = (0..nx).flat_map(|i| (0..nz).map(move |k| (i, k)));
    chart.draw_series(cells.map(|(i, k)| {
        let db = 20.0 * (data[[i, k]].max(f64::MIN_POSITIVE) / reference).log10();
        let t = (db - floor_db) / -floor_db;
        let x0 = f64::from_count(i) * dx_mm;
        let z0 = f64::from_count(k) * dx_mm;
        Rectangle::new([(x0, z0), (x0 + dx_mm, z0 + dx_mm)], viridis(t).filled())
    }))?;
    Ok(())
}

/// Write a row of dB heat maps (one per panel) sharing a reference amplitude.
pub fn write_db_maps(
    path: &Path,
    panels: &[(&str, &Array2<f64>)],
    dx_mm: f64,
    reference: f64,
    floor_db: f64,
) -> Result<()> {
    let root =
        BitMapBackend::new(path, (520 * u32::try_from(panels.len())?, 560)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, panels.len()));
    for (area, (title, data)) in areas.iter().zip(panels) {
        draw_db_map(area, title, data, dx_mm, reference, floor_db)?;
    }
    root.present()?;
    Ok(())
}

/// Line plot of named series against a shared abscissa.
pub fn write_profiles(
    path: &Path,
    title: &str,
    x_label: &str,
    y_label: &str,
    xs: &[f64],
    series: &[(&str, RGBColor, Vec<f64>)],
) -> Result<()> {
    let root = BitMapBackend::new(path, (900, 560)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x_min, x_max) = (
        xs.first().copied().unwrap_or(0.0),
        xs.last().copied().unwrap_or(1.0),
    );
    let y_max = series
        .iter()
        .flat_map(|(_, _, ys)| ys.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max)
        .max(f64::MIN_POSITIVE);
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18))
        .margin(12)
        .x_label_area_size(45)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, 0.0..y_max * 1.05)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    for (name, colour, ys) in series {
        chart
            .draw_series(LineSeries::new(
                xs.iter().copied().zip(ys.iter().copied()),
                colour.stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], colour.stroke_width(2))
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}
