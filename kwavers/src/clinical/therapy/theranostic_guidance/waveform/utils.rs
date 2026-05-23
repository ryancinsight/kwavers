//! Index, waveform, and energy utilities for the 2-D acoustic simulation.

use super::super::geometry::Point2;

pub(super) const fn linear(ix: usize, iy: usize, ny: usize) -> usize {
    ix * ny + iy
}

pub(super) fn point_to_cell(point: Point2, nx: usize, ny: usize, spacing_m: f64) -> usize {
    let cx = 0.5 * (nx - 1) as f64;
    let cy = 0.5 * (ny - 1) as f64;
    let ix = (point.x_m / spacing_m + cx)
        .round()
        .clamp(2.0, (nx - 3) as f64) as usize;
    let iy = (point.y_m / spacing_m + cy)
        .round()
        .clamp(2.0, (ny - 3) as f64) as usize;
    linear(ix, iy, ny)
}

pub(super) fn ricker(frequency_hz: f64, tau_s: f64) -> f64 {
    let x = std::f64::consts::PI * frequency_hz * (tau_s - 2.0 / frequency_hz);
    (1.0 - 2.0 * x * x) * (-x * x).exp()
}

pub(super) fn energy(values: &[f32]) -> f64 {
    values.iter().map(|value| (*value as f64).powi(2)).sum()
}
