//! Shared helpers for solver dispatch.
//!
//! Contains utilities used across multiple per-solver dispatch modules:
//! recording mode parsing, data trimming, and common utilities.

use crate::domain::sensor::recorder::config::RecordingMode;
use crate::domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};

// ── Recording mode helpers ────────────────────────────────────────────────────

/// Convert user-facing recording mode strings into a `SensorRecordSpec`.
pub(crate) fn record_modes_to_spec(modes: &[String]) -> SensorRecordSpec {
    let mut fields = vec![SensorRecordField::Pressure];
    for s in modes {
        match s.as_str() {
            "p" => {}
            "p_max" => fields.push(SensorRecordField::PressureMax),
            "p_min" => fields.push(SensorRecordField::PressureMin),
            "p_rms" => fields.push(SensorRecordField::PressureRms),
            "p_final" => fields.push(SensorRecordField::PressureFinal),
            "all" => {
                fields.push(SensorRecordField::PressureMax);
                fields.push(SensorRecordField::PressureMin);
                fields.push(SensorRecordField::PressureRms);
                fields.push(SensorRecordField::PressureFinal);
            }
            "ux" => fields.push(SensorRecordField::VelocityX),
            "uy" => fields.push(SensorRecordField::VelocityY),
            "uz" => fields.push(SensorRecordField::VelocityZ),
            "ux_max" => fields.push(SensorRecordField::VelocityMaxX),
            "uy_max" => fields.push(SensorRecordField::VelocityMaxY),
            "uz_max" => fields.push(SensorRecordField::VelocityMaxZ),
            "ux_min" => fields.push(SensorRecordField::VelocityMinX),
            "uy_min" => fields.push(SensorRecordField::VelocityMinY),
            "uz_min" => fields.push(SensorRecordField::VelocityMinZ),
            "ux_rms" => fields.push(SensorRecordField::VelocityRmsX),
            "uy_rms" => fields.push(SensorRecordField::VelocityRmsY),
            "uz_rms" => fields.push(SensorRecordField::VelocityRmsZ),
            "ux_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredX),
            "uy_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredY),
            "uz_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredZ),
            "Ix" => fields.push(SensorRecordField::IntensityX),
            "Iy" => fields.push(SensorRecordField::IntensityY),
            "Iz" => fields.push(SensorRecordField::IntensityZ),
            "I_avg_x" => fields.push(SensorRecordField::IntensityAvgX),
            "I_avg_y" => fields.push(SensorRecordField::IntensityAvgY),
            "I_avg_z" => fields.push(SensorRecordField::IntensityAvgZ),
            _ => {}
        }
    }
    SensorRecordSpec::from_fields(&fields)
}

/// Convert user-facing recording mode strings into `RecordingMode` variants
/// (used by the simpler FDTD recorder path).
pub(crate) fn recording_modes_from_strings(modes: &[String]) -> Vec<RecordingMode> {
    modes
        .iter()
        .filter_map(|s| match s.as_str() {
            "p_max" => Some(RecordingMode::MaxPressure),
            "p_min" => Some(RecordingMode::MinPressure),
            "p_rms" => Some(RecordingMode::RmsPressure),
            "p_final" => Some(RecordingMode::FinalPressure),
            "all" => Some(RecordingMode::AllStatistics),
            _ => None,
        })
        .collect()
}

// ── Data trimming helpers ─────────────────────────────────────────────────────

/// Trim the initial recorder sample (record-start offset) from owned data.
pub(crate) fn trim_initial_recorder_sample(
    recorded_data: ndarray::Array2<f64>,
    time_steps: usize,
    record_start_index: usize,
) -> ndarray::Array2<f64> {
    let start = record_start_index.max(1).min(time_steps);
    let skip = start.saturating_sub(1);
    if recorded_data.ncols() > time_steps {
        recorded_data.slice(ndarray::s![.., skip..time_steps]).to_owned()
    } else {
        recorded_data.slice(ndarray::s![.., skip..]).to_owned()
    }
}

/// Trim the initial recorder sample (record-start offset) from a view.
pub(crate) fn trim_initial_recorder_view(
    recorded_data: ndarray::ArrayView2<'_, f64>,
    time_steps: usize,
    record_start_index: usize,
) -> ndarray::Array2<f64> {
    let start = record_start_index.max(1).min(time_steps);
    let skip = start.saturating_sub(1);
    if recorded_data.ncols() > time_steps {
        recorded_data.slice(ndarray::s![.., skip..time_steps]).to_owned()
    } else {
        recorded_data.slice(ndarray::s![.., skip..]).to_owned()
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Return the next power of two ≥ `n`.
pub(crate) fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}
