//! Medium and speed helpers for the 2-D acoustic waveform simulation.

use ndarray::Array2;

use super::super::config::TheranosticInverseConfig;
use super::super::medium::PreparedTheranosticSlice;
use crate::core::constants::acoustic_parameters::BONE_SOUND_SPEED;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

pub(super) fn lesion_speed(
    prepared: &PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
    lesion: &Array2<f64>,
) -> Array2<f64> {
    Array2::from_shape_fn(prepared.sound_speed_m_s.dim(), |idx| {
        if prepared.body_mask[idx] {
            (prepared.sound_speed_m_s[idx] + config.lesion_delta_c_m_s * lesion[idx])
                .clamp(SOUND_SPEED_WATER_SIM, BONE_SOUND_SPEED)
        } else {
            prepared.sound_speed_m_s[idx]
        }
    })
}

pub(super) fn speed_bounds(baseline_speed: &Array2<f64>, true_speed: &Array2<f64>) -> (f64, f64) {
    let (cmin, cmax) = baseline_speed
        .iter()
        .chain(true_speed.iter())
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold((f64::INFINITY, 0.0_f64), |(cmin, cmax), value| {
            (cmin.min(value), cmax.max(value))
        });
    let cmin = if cmin.is_finite() { cmin.max(1.0) } else { 1.0 };
    (cmin, cmax.max(1.0))
}

pub(super) fn reference_speed(
    prepared: &PreparedTheranosticSlice,
    baseline_speed: &Array2<f64>,
) -> f64 {
    let (sum, count) = baseline_speed
        .indexed_iter()
        .filter(|(idx, value)| prepared.body_mask[*idx] && value.is_finite() && **value > 0.0)
        .fold((0.0, 0usize), |(sum, count), (_, value)| {
            (sum + *value, count + 1)
        });
    if count == 0 {
        let (sum, count) = baseline_speed
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .fold((0.0, 0usize), |(sum, count), value| {
                (sum + value, count + 1)
            });
        return (sum / count.max(1) as f64).max(1.0);
    }
    (sum / count as f64).max(1.0)
}
