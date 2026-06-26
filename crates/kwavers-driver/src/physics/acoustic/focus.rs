//! Relative transmit-delay profile + nearest-step quantisation + quantisation-error metric.
//!
//! This submodule carries the timing-half of phased-array focus synthesis: the per-element
//! relative transmit delays that put all wavefronts in phase at the focus, the nearest-step
//! quantisation onto the hardware timing grid, and the worst-case quantisation error.
//!
//! Each kernel is pure-math (`f64`-in/`f64`-out, no state, no cross-slice dep) so it feeds
//! straight into the slice facade's named `pub use` re-export chain.

/// Relative transmit delays (s) for a focused linear array.
///
/// Elements lie on `x = 0`, centred at broadside. The focus is at depth `focal_m` and azimuth
/// `steer_deg`, so its lateral coordinate is `focal_m * tan(steer)`. The farthest element fires
/// first (`0` delay); nearer elements are delayed so all wavefronts reach the focus together.
#[must_use]
pub fn focused_delay_profile_s(
    n: usize,
    pitch_m: f64,
    focal_m: f64,
    steer_deg: f64,
    speed_m_s: f64,
) -> Vec<f64> {
    if n == 0 || pitch_m <= 0.0 || focal_m <= 0.0 || speed_m_s <= 0.0 {
        return Vec::new();
    }
    // A steering angle of ±90° is non-physical (end-fire): tan(±π/2) = ±∞.
    // Return zero delays rather than propagating NaN through the caller.
    if steer_deg.abs() >= 90.0 {
        return vec![0.0; n];
    }
    let center = (n as f64 - 1.0) / 2.0;
    let focus_x = focal_m * steer_deg.to_radians().tan();
    let distances: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i as f64 - center) * pitch_m;
            ((x - focus_x).powi(2) + focal_m.powi(2)).sqrt()
        })
        .collect();
    let farthest = distances.iter().copied().fold(0.0, f64::max);
    distances
        .into_iter()
        .map(|d| (farthest - d) / speed_m_s)
        .collect()
}

/// Quantise delays to a hardware timing step (s), preserving non-negative delays.
#[must_use]
pub fn quantize_delays_s(delays_s: &[f64], step_s: f64) -> Vec<f64> {
    if step_s <= 0.0 {
        return delays_s.to_vec();
    }
    delays_s
        .iter()
        .map(|&d| (d / step_s).round().max(0.0) * step_s)
        .collect()
}

/// Maximum absolute delay error (s) after quantisation.
#[must_use]
pub fn max_delay_quantization_error_s(delays_s: &[f64], quantized_s: &[f64]) -> f64 {
    delays_s
        .iter()
        .zip(quantized_s)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
