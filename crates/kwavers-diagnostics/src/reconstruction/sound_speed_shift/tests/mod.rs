//! Value-semantic tests for the speed-of-sound shift reconstruction pipeline.
//!
//! Sub-modules are grouped by the code path under test:
//! - `forward`     — forward prediction sign contract
//! - `dense`       — dense Tikhonov/H1 PCG reconstruction and workspace reuse
//! - `sparse`      — deterministic sparse sampling and L1 proximal localization
//! - `lsqr`        — matrix-free LSQR damped reconstruction
//! - `validation`  — invalid configuration rejection, operator geometry, sensitivity

mod dense;
mod forward;
mod lsqr;
mod sparse;
mod validation;

use super::{
    predict_sound_speed_time_shifts, reconstruct_sound_speed_shift,
    reconstruct_sound_speed_shift_with_workspace, ShiftPrior, ShiftPropagation, ShiftSampling,
    ShiftSensitivity, SoundSpeedShiftConfig, SoundSpeedShiftSample, SoundSpeedShiftWorkspace,
    SOUND_SPEED_SHIFT_MODEL,
};
use kwavers_solver::inverse::same_aperture::PlanarPoint;

pub(super) fn horizontal_samples(y_values: &[f64]) -> Vec<SoundSpeedShiftSample> {
    y_values.iter().map(|y| horizontal_sample(*y)).collect()
}

pub(super) fn horizontal_sample(y_m: f64) -> SoundSpeedShiftSample {
    SoundSpeedShiftSample::new(
        PlanarPoint { x_m: -0.004, y_m },
        PlanarPoint { x_m: 0.004, y_m },
        0.0,
    )
}

pub(super) fn vertical_sample(x_m: f64) -> SoundSpeedShiftSample {
    SoundSpeedShiftSample::new(
        PlanarPoint { x_m, y_m: -0.004 },
        PlanarPoint { x_m, y_m: 0.004 },
        0.0,
    )
}

pub(super) fn attach_time_shifts(
    samples: &[SoundSpeedShiftSample],
    time_shifts: &[f64],
) -> Vec<SoundSpeedShiftSample> {
    samples
        .iter()
        .zip(time_shifts.iter())
        .map(|(sample, time_shift_s)| SoundSpeedShiftSample {
            time_shift_s: *time_shift_s,
            ..*sample
        })
        .collect()
}
