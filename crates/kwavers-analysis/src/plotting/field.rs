use iris::{color::NamedColorMap, view::ScalarFieldView};
use iris_plot_html::{ColorScale, Plot};
use leto::Array3;

use super::{output::save, Grid};

/// Export an XY pressure slice at the requested Z index.
///
/// # Panics
/// Panics when the index lies outside the field's Z axis.
pub fn plot_pressure_field_2d(
    pressure: &Array3<f64>,
    grid: &Grid,
    z_slice: usize,
    title: &str,
    filename: &str,
) {
    let slice = pressure
        .index_axis::<2>(2, z_slice)
        .expect("z_slice in range");
    save_slice(
        grid,
        slice.shape(),
        |index| slice[index],
        ColorScale {
            label: "Pressure (Pa)",
            map: NamedColorMap::Viridis,
        },
        title,
        filename,
    );
}

/// Export the cell-wise difference of two XY pressure slices.
///
/// # Panics
/// Panics when the index lies outside either Z axis or when the second slice
/// cannot cover the XY extent of the first slice.
pub fn plot_field_comparison(
    field1: &Array3<f64>,
    field2: &Array3<f64>,
    grid: &Grid,
    z_slice: usize,
    title: &str,
    filename: &str,
) {
    let lhs = field1
        .index_axis::<2>(2, z_slice)
        .expect("z_slice in range");
    let rhs = field2
        .index_axis::<2>(2, z_slice)
        .expect("z_slice in range");
    save_slice(
        grid,
        lhs.shape(),
        |index| lhs[index] - rhs[index],
        ColorScale {
            label: "Pressure difference (Pa)",
            map: NamedColorMap::CoolWarm,
        },
        title,
        filename,
    );
}

fn save_slice(
    grid: &Grid,
    [nx, ny]: [usize; 2],
    mut value: impl FnMut([usize; 2]) -> f64,
    scale: ColorScale<'_>,
    title: &str,
    filename: &str,
) {
    let x: Vec<_> = (0..nx).map(|i| grid.x(i)).collect();
    let y: Vec<_> = (0..ny).map(|j| grid.y(j)).collect();
    // Iris heatmaps index [y, x]. Fill the output in that order directly.
    let mut values = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        values.extend((0..nx).map(|i| value([i, j])));
    }
    let plot = ScalarFieldView::new(&values, [ny, nx])
        .map_err(anyhow::Error::from)
        .and_then(|field| {
            Ok(Plot::heatmap(
                title,
                ["X (m)", "Y (m)"],
                &x,
                &y,
                field,
                scale,
            )?)
        });
    save(filename, plot);
}
