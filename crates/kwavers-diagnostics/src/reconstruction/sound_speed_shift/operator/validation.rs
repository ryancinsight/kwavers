//! Input-domain validation for speed-shift operator construction.

use leto::Array2;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftSample};

pub(super) fn validate_inputs(
    samples: &[SoundSpeedShiftSample],
    active_mask: &Array2<bool>,
    config: SoundSpeedShiftConfig,
) -> KwaversResult<()> {
    config
        .sampling
        .validate()
        .map_err(KwaversError::InvalidInput)?;
    validate_positive_finite("reference sound speed", config.reference_sound_speed_m_s)?;
    validate_positive_finite("spacing", config.spacing_m)?;
    validate_nonnegative_finite("Tikhonov weight", config.tikhonov_weight)?;
    validate_nonnegative_finite("smoothness weight", config.smoothness_weight)?;
    validate_nonnegative_finite("sparsity weight", config.sparsity_weight)?;
    config
        .propagation
        .validate()
        .map_err(KwaversError::InvalidInput)?;
    config
        .sensitivity
        .validate(config.spacing_m)
        .map_err(KwaversError::InvalidInput)?;
    if config.iterations == 0 {
        return Err(KwaversError::InvalidInput(
            "Speed-shift reconstruction requires at least one iteration".to_owned(),
        ));
    }
    if samples.is_empty() {
        return Err(KwaversError::InvalidInput(
            "Speed-shift reconstruction requires at least one measurement".to_owned(),
        ));
    }
    if active_mask.is_empty() || !active_mask.iter().any(|value| *value) {
        return Err(KwaversError::InvalidInput(
            "Speed-shift reconstruction requires a nonempty active mask".to_owned(),
        ));
    }
    for (idx, sample) in samples.iter().enumerate() {
        validate_point(
            "transmitter",
            idx,
            sample.transmitter.x_m,
            sample.transmitter.y_m,
        )?;
        validate_point("receiver", idx, sample.receiver.x_m, sample.receiver.y_m)?;
        if !sample.time_shift_s.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Speed-shift sample {idx} has nonfinite time shift"
            )));
        }
        let dx = sample.receiver.x_m - sample.transmitter.x_m;
        let dy = sample.receiver.y_m - sample.transmitter.y_m;
        if dx.hypot(dy) <= f64::EPSILON {
            return Err(KwaversError::InvalidInput(format!(
                "Speed-shift sample {idx} has coincident transmitter and receiver"
            )));
        }
    }
    Ok(())
}

fn validate_positive_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

fn validate_nonnegative_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and nonnegative, got {value}"
        )))
    }
}

fn validate_point(name: &str, idx: usize, x_m: f64, y_m: f64) -> KwaversResult<()> {
    if x_m.is_finite() && y_m.is_finite() {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "Speed-shift sample {idx} has nonfinite {name} point"
        )))
    }
}
