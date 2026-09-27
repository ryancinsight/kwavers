//! Index, waveform, and energy utilities for the 2-D acoustic simulation.

pub(super) const fn linear(ix: usize, iy: usize, ny: usize) -> usize {
    ix * ny + iy
}

pub(super) fn ricker(frequency_hz: f64, tau_s: f64) -> f64 {
    let x = std::f64::consts::PI * frequency_hz * (tau_s - 2.0 / frequency_hz);
    (1.0 - 2.0 * x * x) * (-x * x).exp()
}

pub(super) fn energy(values: &[f32]) -> f64 {
    values.iter().map(|value| (*value as f64).powi(2)).sum()
}
