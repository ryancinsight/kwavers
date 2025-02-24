// plotting/mod.rs
use crate::grid::Grid;
use crate::recorder::Recorder;
use crate::source::Source;
use crate::time::Time;
use log::{debug, info};
use ndarray::{Array2, Array3, Axis};
use plotly::{
    common::{ColorBar, Mode, Title},
    HeatMap, Layout, Plot, Scatter, Scatter3D,
};
use std::fs::File;
use std::io::Write;

const PRESSURE_IDX: usize = 0;
const LIGHT_IDX: usize = 1;
const TEMPERATURE_IDX: usize = 2;
const BUBBLE_RADIUS_IDX: usize = 3;

pub fn plot_positions(positions: &[(f64, f64, f64)], title: &str, filename: &str) {
    info!("Generating 3D scatter plot: {}", filename);
    let x: Vec<f64> = positions.iter().map(|&(x, _, _)| x).collect();
    let y: Vec<f64> = positions.iter().map(|&(_, y, _)| y).collect();
    let z: Vec<f64> = positions.iter().map(|&(_, _, z)| z).collect();

    let trace = Scatter3D::new(x, y, z)
        .mode(Mode::Markers)
        .marker(plotly::common::Marker::new().size(5).color("red"));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::from(title))
            .width(800)
            .height(600),
    );

    debug!("Saving plot to {}.html", filename);
    let html = plot.to_html();
    File::create(format!("{}.html", filename))
        .and_then(|mut file| file.write_all(html.as_bytes()))
        .expect("Failed to write plot HTML");
}

pub fn plot_2d_slice(
    field: &Array3<f64>,
    grid: &Grid,
    slice_axis: usize,
    slice_idx: usize,
    title: &str,
    filename: &str,
) {
    info!("Generating 2D heatmap: {}", filename);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    let (x, y, z) = match slice_axis {
        0 => {
            let x = grid.y_coordinates().to_vec();
            let y = grid.z_coordinates().to_vec();
            let z = field
                .slice(ndarray::s![slice_idx, .., ..])
                .to_shape((ny, nz))
                .unwrap()
                .to_owned();
            (x, y, z)
        }
        1 => {
            let x = grid.x_coordinates().to_vec();
            let y = grid.z_coordinates().to_vec();
            let z = field
                .slice(ndarray::s![.., slice_idx, ..])
                .to_shape((nx, nz))
                .unwrap()
                .to_owned();
            (x, y, z)
        }
        2 => {
            let x = grid.x_coordinates().to_vec();
            let y = grid.y_coordinates().to_vec();
            let z = field
                .slice(ndarray::s![.., .., slice_idx])
                .to_shape((nx, ny))
                .unwrap()
                .to_owned();
            (x, y, z)
        }
        _ => panic!("Invalid slice axis; must be 0 (x), 1 (y), or 2 (z)"),
    };

    let z_vec: Vec<Vec<f64>> = z.outer_iter().map(|row| row.to_vec()).collect();
    let trace = HeatMap::new(x, y, z_vec).color_bar(ColorBar::new().title(Title::from(title)));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::from(title))
            .width(800)
            .height(600),
    );

    debug!("Saving heatmap to {}.html", filename);
    let html = plot.to_html();
    File::create(format!("{}.html", filename))
        .and_then(|mut file| file.write_all(html.as_bytes()))
        .expect("Failed to write heatmap HTML");
}

pub fn plot_time_series(
    data: &Array2<f64>,
    time: &Time,
    sensor_idx: usize,
    title: &str,
    filename: &str,
) {
    info!(
        "Generating time series plot for sensor {}: {}",
        sensor_idx, filename
    );
    assert!(sensor_idx < data.nrows(), "Sensor index out of bounds");

    let t = time.time_vector().to_vec();
    let values = data.row(sensor_idx).to_vec();

    let trace = Scatter::new(t, values)
        .mode(Mode::Lines)
        .name(format!("Sensor {}", sensor_idx + 1).as_str());
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
        Layout::new()
            .title(Title::from(title))
            .width(800)
            .height(600),
    );

    debug!("Saving time series to {}.html", filename);
    let html = plot.to_html();
    File::create(format!("{}.html", filename))
        .and_then(|mut file| file.write_all(html.as_bytes()))
        .expect("Failed to write time series HTML");
}

pub fn plot_simulation_outputs(recorder: &Recorder, grid: &Grid, time: &Time, source: &dyn Source) {
    plot_positions(&source.positions(), "Source Positions", "source_positions");
    plot_positions(
        &recorder
            .sensor()
            .positions()
            .iter()
            .map(|&(ix, iy, iz)| {
                (
                    ix as f64 * grid.dx,
                    iy as f64 * grid.dy,
                    iz as f64 * grid.dz,
                )
            })
            .collect::<Vec<_>>(),
        "Sensor Positions",
        "sensor_positions",
    );

    if let Some((_, fields)) = recorder.fields_snapshots.last() {
        let pressure = fields.index_axis(Axis(0), PRESSURE_IDX).to_owned();
        let light = fields.index_axis(Axis(0), LIGHT_IDX).to_owned();
        let temperature = fields.index_axis(Axis(0), TEMPERATURE_IDX).to_owned();
        let cavitation = fields.index_axis(Axis(0), BUBBLE_RADIUS_IDX).to_owned();

        plot_2d_slice(
            &pressure,
            grid,
            2,
            grid.nz / 2,
            "Pressure (Pa)",
            "pressure_slice",
        );
        plot_2d_slice(
            &light,
            grid,
            2,
            grid.nz / 2,
            "Light Fluence Rate (W/m²)",
            "light_slice",
        );
        plot_2d_slice(
            &temperature,
            grid,
            2,
            grid.nz / 2,
            "Temperature (K)",
            "temperature_slice",
        );
        plot_2d_slice(
            &cavitation,
            grid,
            2,
            grid.nz / 2,
            "Bubble Radius (m)",
            "cavitation_slice",
        );
    }

    if let Some(pressure_data) = recorder.pressure_data() {
        plot_time_series(
            &pressure_data,
            time,
            0,
            "Pressure Time Series (Sensor 0)",
            "pressure_time_series",
        );
    }
    if let Some(light_data) = recorder.light_data() {
        plot_time_series(
            &light_data,
            time,
            0,
            "Light Fluence Time Series (Sensor 0)",
            "light_time_series",
        );
    }
}
