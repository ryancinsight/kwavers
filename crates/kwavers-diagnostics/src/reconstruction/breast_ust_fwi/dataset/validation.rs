//! Input validation for `BreastUstPstdDatasetConfig` and acquisition parameters.

use super::{BreastUstPstdDatasetConfig, PSTD_DATASET_CFL_LIMIT};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

pub(super) fn validate_config(config: &BreastUstPstdDatasetConfig) -> KwaversResult<()> {
    if !config.spacing_m.is_finite() || config.spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {}",
            config.spacing_m
        )));
    }
    if !config.time_step_s.is_finite() || config.time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {}",
            config.time_step_s
        )));
    }
    if config.cycles_per_frequency == 0 {
        return Err(KwaversError::InvalidInput(
            "cycles_per_frequency must be positive".to_owned(),
        ));
    }
    if config.frequency_bin_cycles == 0 || config.frequency_bin_cycles > config.cycles_per_frequency
    {
        return Err(KwaversError::InvalidInput(format!(
            "frequency_bin_cycles must be in 1..={}, got {}",
            config.cycles_per_frequency, config.frequency_bin_cycles
        )));
    }
    if !config.source_amplitude_pa.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "source_amplitude_pa must be finite, got {}",
            config.source_amplitude_pa
        )));
    }
    if !config.density_kg_m3.is_finite() || config.density_kg_m3 <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "density_kg_m3 must be positive and finite, got {}",
            config.density_kg_m3
        )));
    }
    Ok(())
}

pub(super) fn validate_sound_speed(sound_speed_m_s: &Array3<f64>) -> KwaversResult<()> {
    if sound_speed_m_s.is_empty() {
        return Err(KwaversError::InvalidInput(
            "sound_speed_m_s volume must not be empty".to_owned(),
        ));
    }
    for &speed in sound_speed_m_s {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_frequencies(frequencies_hz: &[f64], dt: f64) -> KwaversResult<()> {
    if frequencies_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "frequencies_hz must not be empty".to_owned(),
        ));
    }
    let nyquist = 0.5 / dt;
    for &frequency_hz in frequencies_hz {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "frequency must be positive and finite, got {frequency_hz}"
            )));
        }
        if frequency_hz >= nyquist {
            return Err(KwaversError::InvalidInput(format!(
                "frequency {frequency_hz} Hz must be below Nyquist {nyquist} Hz"
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_cfl(
    sound_speed_m_s: &Array3<f64>,
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<()> {
    let max_sound_speed = sound_speed_m_s
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let cfl = max_sound_speed * config.time_step_s / config.spacing_m;
    if cfl > PSTD_DATASET_CFL_LIMIT {
        return Err(KwaversError::InvalidInput(format!(
            "PSTD acquisition CFL {cfl:.6} exceeds limit {PSTD_DATASET_CFL_LIMIT:.6}"
        )));
    }
    Ok(())
}
