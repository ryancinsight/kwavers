//! # Plotting Module
//!
//! This module provides visualization capabilities for Kwavers simulation results.
//! It supports 2D/3D plotting of acoustic fields, pressure distributions, and temporal evolution.

use ndarray::Array3;

// Note: Field indices imported from physics::field_indices for SSOT

#[cfg(feature = "plotting")]
use plotly::{HeatMap, Layout, Plot, Scatter, Surface};

#[cfg(feature = "plotting")]
mod plotting_impl {
    use crate::grid::Grid;
    use crate::physics::field_indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
    use crate::recorder::Recorder;
    use crate::time::Time;
    use log::info;
    use ndarray::{Array2, Array3, Axis};
    use plotly::{
        common::{ColorBar, Mode, Title},
        HeatMap, Layout, Plot, Scatter, Scatter3D, Surface,
    };
    use std::fs::File;
    use std::io::Write;

    pub fn plot_positions(positions: &[(f64, f64, f64)], title: &str, filename: &str) {
        info!("Generating 3D scatter plot: {}", filename);
        // Create 3D scatter plot
        let x: Vec<f64> = positions.iter().map(|&(x, _, _)| x).collect();
        let y: Vec<f64> = positions.iter().map(|&(_, y, _)| y).collect();
        let z: Vec<f64> = positions.iter().map(|&(_, _, z)| z).collect();

        let trace = Scatter3D::new(x, y, z)
            .mode(Mode::Markers)
            .marker(plotly::common::Marker::new().size(5).color("red"));
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new().title(Title::new(title));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }

    pub fn plot_pressure_field_2d(
        pressure: &Array3<f64>,
        grid: &Grid,
        z_slice: usize,
        title: &str,
        filename: &str,
    ) {
        info!("Generating 2D pressure field plot: {}", filename);

        let slice = pressure.slice(ndarray::s![.., .., z_slice]);
        let (nx, ny) = slice.dim();

        let x: Vec<f64> = (0..nx).map(|i| i as f64 * grid.dx).collect();
        let y: Vec<f64> = (0..ny).map(|j| j as f64 * grid.dy).collect();
        let z: Vec<Vec<f64>> = (0..ny)
            .map(|j| (0..nx).map(|i| slice[[i, j]]).collect())
            .collect();

        let trace =
            HeatMap::new(x, y, z).color_bar(ColorBar::new().title(Title::new("Pressure (Pa)")));
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(Title::new(title))
            .x_axis(plotly::layout::Axis::new().title(Title::new("X (m)")))
            .y_axis(plotly::layout::Axis::new().title(Title::new("Y (m)")));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }

    pub fn plot_pressure_field_3d(
        pressure: &Array3<f64>,
        grid: &Grid,
        title: &str,
        filename: &str,
    ) {
        info!("Generating 3D pressure field plot: {}", filename);

        let (nx, ny, nz) = pressure.dim();
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        let mut values = Vec::new();

        // Sample data for visualization (every nth point to avoid overcrowding)
        let step = std::cmp::max(1, nx / 20);
        for i in (0..nx).step_by(step) {
            for j in (0..ny).step_by(step) {
                for k in (0..nz).step_by(step) {
                    x.push(i as f64 * grid.dx);
                    y.push(j as f64 * grid.dy);
                    z.push(k as f64 * grid.dz);
                    values.push(pressure[[i, j, k]]);
                }
            }
        }

        let trace = Scatter3D::new(x, y, z).mode(Mode::Markers).marker(
            plotly::common::Marker::new()
                .size(3)
                .color_bar(ColorBar::new().title(Title::new("Pressure (Pa)"))),
        );
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new().title(Title::new(title));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }

    pub fn plot_temporal_evolution(
        data: &[f64],
        time_points: &[f64],
        title: &str,
        y_label: &str,
        filename: &str,
    ) {
        info!("Generating temporal evolution plot: {}", filename);

        let trace = Scatter::new(time_points.to_vec(), data.to_vec())
            .mode(Mode::Lines)
            .name("Evolution");
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(Title::new(title))
            .x_axis(plotly::layout::Axis::new().title(Title::new("Time (s)")))
            .y_axis(plotly::layout::Axis::new().title(Title::new(y_label)));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }

    pub fn plot_field_comparison(
        field1: &Array3<f64>,
        field2: &Array3<f64>,
        grid: &Grid,
        z_slice: usize,
        title: &str,
        filename: &str,
    ) {
        info!("Generating field comparison plot: {}", filename);

        let slice1 = field1.slice(ndarray::s![.., .., z_slice]);
        let slice2 = field2.slice(ndarray::s![.., .., z_slice]);
        let (nx, ny) = slice1.dim();

        // Calculate difference
        let diff: Array2<f64> = &slice1 - &slice2;

        let x: Vec<f64> = (0..nx).map(|i| i as f64 * grid.dx).collect();
        let y: Vec<f64> = (0..ny).map(|j| j as f64 * grid.dy).collect();
        let z: Vec<Vec<f64>> = (0..ny)
            .map(|j| (0..nx).map(|i| diff[[i, j]]).collect())
            .collect();

        let trace =
            HeatMap::new(x, y, z).color_bar(ColorBar::new().title(Title::new("Difference")));
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(Title::new(title))
            .x_axis(plotly::layout::Axis::new().title(Title::new("X (m)")))
            .y_axis(plotly::layout::Axis::new().title(Title::new("Y (m)")));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }

    pub fn save_data_csv(data: &Array3<f64>, filename: &str) -> Result<(), std::io::Error> {
        info!("Saving data to CSV: {}", filename);
        let mut file = File::create(filename)?;
        let (nx, ny, nz) = data.dim();

        // Write header
        writeln!(file, "x,y,z,value")?;

        // Write data
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    writeln!(file, "{},{},{},{}", i, j, k, data[[i, j, k]])?;
                }
            }
        }

        info!("Data saved to {}", filename);
        Ok(())
    }

    pub fn plot_recorder_data(_recorder: &Recorder, filename: &str) {
        info!("Generating recorder data plot: {}", filename);

        // For now, just create a simple plot with dummy data
        let time_points: Vec<f64> = (0..100).map(|i| i as f64 * 1e-6).collect(); // Dummy time points
        let pressure_data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect(); // Dummy pressure data

        let trace = Scatter::new(time_points, pressure_data)
            .mode(Mode::Lines)
            .name("Pressure");
        let mut plot = Plot::new();
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(Title::new("Recorded Pressure Evolution"))
            .x_axis(plotly::layout::Axis::new().title(Title::new("Time (s)")))
            .y_axis(plotly::layout::Axis::new().title(Title::new("Pressure (Pa)")));
        plot.set_layout(layout);

        plot.write_html(filename);
        info!("Plot saved to {}", filename);
    }
}

// Re-export functions when plotly feature is enabled
#[cfg(feature = "plotting")]
pub use plotting_impl::*;

// Provide fallback implementations when plotly feature is disabled
#[cfg(not(feature = "plotting"))]
pub fn plot_positions(_positions: &[(f64, f64, f64)], _title: &str, _filename: &str) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

#[cfg(not(feature = "plotting"))]
pub fn plot_pressure_field_2d(
    _pressure: &Array3<f64>,
    _grid: &Grid,
    _z_slice: usize,
    _title: &str,
    _filename: &str,
) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

#[cfg(not(feature = "plotting"))]
pub fn plot_pressure_field_3d(
    _pressure: &Array3<f64>,
    _grid: &Grid,
    _title: &str,
    _filename: &str,
) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

#[cfg(not(feature = "plotting"))]
pub fn plot_temporal_evolution(
    _data: &[f64],
    _time_points: &[f64],
    _title: &str,
    _y_label: &str,
    _filename: &str,
) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

#[cfg(not(feature = "plotting"))]
pub fn plot_field_comparison(
    _field1: &Array3<f64>,
    _field2: &Array3<f64>,
    _grid: &Grid,
    _z_slice: usize,
    _title: &str,
    _filename: &str,
) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

#[cfg(not(feature = "plotting"))]
pub fn save_data_csv(_data: &Array3<f64>, _filename: &str) -> Result<(), std::io::Error> {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
    Ok(())
}

#[cfg(not(feature = "plotting"))]
pub fn plot_recorder_data(_recorder: &crate::recorder::Recorder, _filename: &str) {
    log::warn!("Plotting functionality not available - compile with 'plotly' feature");
}

// Re-export types for compatibility
pub use crate::grid::Grid;
pub use crate::recorder::Recorder;
