//! Homogeneous direct-field diagnostics for breast UST PSTD acquisition.
//!
//! This module owns the clinical diagnostic that compares a homogeneous PSTD
//! acquisition against three closed-form or finite-grid reference operators:
//! a continuous outgoing point-source Green function, the same Green function
//! after the PSTD source `kappa` filter, and the periodic finite-grid PSTD
//! modal recurrence.
//!
//! # Theorem
//! For a spatially constant sound speed `c`, uniform spacing `dx`, and disabled
//! boundary absorption, the PSTD update is diagonal in the periodic Fourier
//! basis. The modal recurrence in [`predict`] applies exactly the same
//! k-space source filter and leapfrog propagator to a grid source mask, then
//! computes the same rectangular first-harmonic bin used by
//! [`super::generate_breast_ust_pstd_frequency_dataset`].

mod grid;
mod metrics;
mod predict;

#[cfg(test)]
mod tests;

use super::{generate_breast_ust_pstd_frequency_dataset, BreastUstPstdDatasetConfig};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use ndarray::Array3;

pub const BREAST_UST_HOMOGENEOUS_DIRECT_FIELD_DIAGNOSTIC_MODEL: &str =
    "clinical_breast_ust_homogeneous_direct_field_diagnostic";

/// Scaled residual and receiver-topology diagnostics for one reference operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstDirectFieldDiagnostics {
    /// Row-wise complex-scaled normalized residual over every receiver channel.
    pub normalized_l2_residual: f64,
    /// Arithmetic mean of per-frequency/per-transmit normalized residuals.
    pub row_normalized_l2_residual_mean: f64,
    /// Complex-scaled normalized residual over co-located source receiver channels.
    pub active_only_normalized_l2_residual: f64,
    /// Complex-scaled normalized residual over non-source receiver channels.
    pub passive_only_normalized_l2_residual: f64,
    /// Maximum coefficient of variation of source-scale magnitudes by frequency.
    pub source_scale_magnitude_coefficient_of_variation: f64,
    /// Maximum wrapped phase span of source-scale estimates by frequency.
    pub source_scale_phase_span_rad: f64,
    /// Number of frequency/transmit/receiver samples classified as active channels.
    pub active_pair_count: usize,
    /// RMS wrapped phase error on co-located source receiver channels.
    pub active_self_channel_phase_error_rms_rad: f64,
    /// Maximum absolute wrapped phase error on co-located source receiver channels.
    pub active_self_channel_phase_error_max_abs_rad: f64,
    /// RMS natural-log amplitude error on co-located source receiver channels.
    pub active_self_channel_log_amplitude_error_rms: f64,
    /// Number of frequency/transmit/receiver samples classified as passive channels.
    pub passive_pair_count: usize,
    /// Minimum source-receiver distance among passive channels.
    pub passive_range_min_m: f64,
    /// Maximum source-receiver distance among passive channels.
    pub passive_range_max_m: f64,
    /// RMS wrapped phase error on passive receiver channels.
    pub passive_phase_error_rms_rad: f64,
    /// Maximum absolute wrapped phase error on passive receiver channels.
    pub passive_phase_error_max_abs_rad: f64,
    /// RMS natural-log amplitude error on passive receiver channels.
    pub passive_log_amplitude_error_rms: f64,
}

/// Homogeneous PSTD parity diagnostics for all direct-field references.
#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstHomogeneousDirectFieldDiagnostics {
    pub point_source: BreastUstDirectFieldDiagnostics,
    pub source_kappa_filtered: BreastUstDirectFieldDiagnostics,
    pub source_kappa_filtered_residual_delta: f64,
    /// Passive-channel residual change from point Green to source-kappa Green.
    ///
    /// Negative values mean the source-kappa reference explains more passive
    /// receiver energy after row-wise source scaling.
    pub source_kappa_filtered_passive_residual_delta: f64,
    pub pstd_periodic: BreastUstDirectFieldDiagnostics,
    pub pstd_periodic_residual_delta: f64,
    /// Passive-channel residual change from point Green to finite-grid PSTD Green.
    ///
    /// This is the direct clinical diagnostic for the passive propagation
    /// contract because it excludes co-located active source/receiver channels.
    pub pstd_periodic_passive_residual_delta: f64,
    pub model_family: &'static str,
}

/// Compare homogeneous PSTD observations with direct-field reference operators.
///
/// # Contract
/// The sound-speed volume must be spatially homogeneous and CPML must be
/// disabled. Those constraints make the periodic PSTD modal recurrence a
/// mathematically valid reference for the dataset generator.
///
/// # Errors
/// Returns an error when the medium is not homogeneous, boundary absorption is
/// enabled, acquisition generation fails, or any diagnostic row has zero energy.
pub fn diagnose_breast_ust_homogeneous_direct_field(
    homogeneous_sound_speed_m_s: &Array3<f64>,
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    config: BreastUstPstdDatasetConfig,
) -> KwaversResult<BreastUstHomogeneousDirectFieldDiagnostics> {
    if config.cpml_thickness_cells != 0 {
        return Err(KwaversError::InvalidInput(
            "homogeneous direct-field diagnostics require cpml_thickness_cells = 0".into(),
        ));
    }
    let reference_speed_m_s = homogeneous_sound_speed(homogeneous_sound_speed_m_s)?;
    let dataset = generate_breast_ust_pstd_frequency_dataset(
        homogeneous_sound_speed_m_s,
        array,
        frequencies_hz,
        config,
    )?;

    let point = predict::point_source_observation_cube(
        array,
        frequencies_hz,
        reference_speed_m_s,
        config.spacing_m,
    )?;
    let source_kappa = predict::source_kappa_filtered_observation_cube(
        array,
        frequencies_hz,
        reference_speed_m_s,
        config.spacing_m,
        homogeneous_sound_speed_m_s.dim(),
        config.time_step_s,
    )?;
    let pstd_periodic = predict::pstd_periodic_observation_cube(
        array,
        frequencies_hz,
        reference_speed_m_s,
        config.spacing_m,
        homogeneous_sound_speed_m_s.dim(),
        config.time_step_s,
        &dataset.time_steps_per_frequency,
        &dataset.frequency_bin_start_steps_per_frequency,
        config.source_amplitude_pa,
    )?;

    let point_source = metrics::diagnostics_for_prediction(
        &point,
        &dataset.observed_pressure,
        frequencies_hz,
        config,
        &dataset.time_steps_per_frequency,
        &dataset.frequency_bin_start_steps_per_frequency,
        array,
    )?;
    let source_kappa_filtered = metrics::diagnostics_for_prediction(
        &source_kappa,
        &dataset.observed_pressure,
        frequencies_hz,
        config,
        &dataset.time_steps_per_frequency,
        &dataset.frequency_bin_start_steps_per_frequency,
        array,
    )?;
    let pstd_periodic = metrics::diagnostics_for_prediction(
        &pstd_periodic,
        &dataset.observed_pressure,
        frequencies_hz,
        config,
        &dataset.time_steps_per_frequency,
        &dataset.frequency_bin_start_steps_per_frequency,
        array,
    )?;

    Ok(BreastUstHomogeneousDirectFieldDiagnostics {
        source_kappa_filtered_residual_delta: source_kappa_filtered.normalized_l2_residual
            - point_source.normalized_l2_residual,
        source_kappa_filtered_passive_residual_delta: source_kappa_filtered
            .passive_only_normalized_l2_residual
            - point_source.passive_only_normalized_l2_residual,
        pstd_periodic_residual_delta: pstd_periodic.normalized_l2_residual
            - point_source.normalized_l2_residual,
        pstd_periodic_passive_residual_delta: pstd_periodic.passive_only_normalized_l2_residual
            - point_source.passive_only_normalized_l2_residual,
        point_source,
        source_kappa_filtered,
        pstd_periodic,
        model_family: BREAST_UST_HOMOGENEOUS_DIRECT_FIELD_DIAGNOSTIC_MODEL,
    })
}

fn homogeneous_sound_speed(sound_speed_m_s: &Array3<f64>) -> KwaversResult<f64> {
    let mut iter = sound_speed_m_s.iter().copied();
    let Some(reference) = iter.next() else {
        return Err(KwaversError::InvalidInput(
            "homogeneous sound-speed volume must not be empty".into(),
        ));
    };
    if !reference.is_finite() || reference <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "sound speed must be positive and finite, got {reference}"
        )));
    }
    for speed in iter {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
        if speed != reference {
            return Err(KwaversError::InvalidInput(
                "homogeneous direct-field diagnostics require one constant sound speed".into(),
            ));
        }
    }
    Ok(reference)
}
