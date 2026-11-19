use super::{Grid, Recorder};
use leto::Array3;

fn unavailable() {
    log::warn!("HTML plotting is unavailable. Enable the 'plotting' feature.");
}

pub fn plot_positions(_positions: &[(f64, f64, f64)], _title: &str, _filename: &str) {
    unavailable();
}

pub fn plot_pressure_field_2d(
    _pressure: &Array3<f64>,
    _grid: &Grid,
    _z_slice: usize,
    _title: &str,
    _filename: &str,
) {
    unavailable();
}

pub fn plot_pressure_field_3d(
    _pressure: &Array3<f64>,
    _grid: &Grid,
    _title: &str,
    _filename: &str,
) {
    unavailable();
}

pub fn plot_temporal_evolution(
    _data: &[f64],
    _time_points: &[f64],
    _title: &str,
    _y_label: &str,
    _filename: &str,
) {
    unavailable();
}

pub fn plot_field_comparison(
    _field1: &Array3<f64>,
    _field2: &Array3<f64>,
    _grid: &Grid,
    _z_slice: usize,
    _title: &str,
    _filename: &str,
) {
    unavailable();
}

pub fn plot_recorder_data(_recorder: &Recorder, _filename: &str) {
    unavailable();
}
