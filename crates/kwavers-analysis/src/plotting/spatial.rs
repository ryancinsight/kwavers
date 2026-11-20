use iris::color::{NamedColorMap, Rgba};
use iris_plot_html::{ColorScale, MarkerStyle, Plot};
use leto::Array3;

use super::{output::save, Grid};

/// Export positions as an interactive rotatable point cloud.
pub fn plot_positions(positions: &[(f64, f64, f64)], title: &str, filename: &str) {
    let points: Vec<_> = positions.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let plot = Rgba::new([1.0, 0.0, 0.0, 1.0])
        .map_err(anyhow::Error::from)
        .and_then(|color| {
            Ok(Plot::scatter(
                title,
                ["X", "Y", "Z"],
                &points,
                MarkerStyle::Solid { color, radius: 3 },
            )?)
        });
    save(filename, plot);
}

/// Export sampled pressure values at their spatial grid coordinates.
pub fn plot_pressure_field_3d(pressure: &Array3<f64>, grid: &Grid, title: &str, filename: &str) {
    let [nx, ny, nz] = pressure.shape();
    // Keep the existing spatial sampling of the exported cloud.
    let step = (nx / 20).max(1);
    let samples = nx.div_ceil(step) * ny.div_ceil(step) * nz.div_ceil(step);
    let mut points = Vec::with_capacity(samples);
    let mut values = Vec::with_capacity(samples);
    for i in (0..nx).step_by(step) {
        for j in (0..ny).step_by(step) {
            for k in (0..nz).step_by(step) {
                let (x, y, z) = grid.coords(i, j, k);
                points.push([x, y, z]);
                values.push(pressure[[i, j, k]]);
            }
        }
    }

    let plot = Plot::scatter(
        title,
        ["X (m)", "Y (m)", "Z (m)"],
        &points,
        MarkerStyle::Scaled {
            values: &values,
            scale: ColorScale {
                label: "Pressure (Pa)",
                map: NamedColorMap::Viridis,
            },
            radius: 2,
        },
    );
    save(filename, plot.map_err(Into::into));
}
