use super::excitation::source_excitation_diagnostics_with_receiver_policy;
use super::residual::{
    scaled_observation_residual_metrics_by_policy, BreastUstReceiverChannelPolicy,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use kwavers_math::fft::Complex64;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct BreastUstForwardOperatorPrediction<'a> {
    pub model: &'a str,
    pub pressure: &'a Array3<Complex64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstForwardOperatorModelDiagnostics {
    pub model: String,
    pub normalized_l2_residual: f64,
    pub row_normalized_l2_residual_mean: f64,
    pub source_scale_magnitude_coefficient_of_variation: f64,
    pub source_scale_phase_span_rad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstForwardOperatorEquivalenceDiagnostics {
    pub model_count: usize,
    pub receiver_channel_policy: BreastUstReceiverChannelPolicy,
    pub best_model: String,
    pub best_normalized_l2_residual: f64,
    pub worst_model: String,
    pub worst_normalized_l2_residual: f64,
    pub residual_spread: f64,
    pub per_model: Vec<BreastUstForwardOperatorModelDiagnostics>,
}

pub fn forward_operator_equivalence_diagnostics(
    predictions_by_model: &[BreastUstForwardOperatorPrediction<'_>],
    observed: &Array3<Complex64>,
    frequencies_hz: &[f64],
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
) -> KwaversResult<BreastUstForwardOperatorEquivalenceDiagnostics> {
    forward_operator_equivalence_diagnostics_with_receiver_policy(
        predictions_by_model,
        observed,
        frequencies_hz,
        source_amplitude_pa,
        time_step_s,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
        BreastUstReceiverChannelPolicy::All,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn forward_operator_equivalence_diagnostics_with_receiver_policy(
    predictions_by_model: &[BreastUstForwardOperatorPrediction<'_>],
    observed: &Array3<Complex64>,
    frequencies_hz: &[f64],
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<BreastUstForwardOperatorEquivalenceDiagnostics> {
    if predictions_by_model.is_empty() {
        return Err(KwaversError::InvalidInput(
            "predictions_by_model must not be empty".into(),
        ));
    }
    validate_model_names(predictions_by_model)?;

    let mut per_model = predictions_by_model
        .iter()
        .map(|prediction| {
            let residual = scaled_observation_residual_metrics_by_policy(
                prediction.pressure,
                observed,
                receiver_channel_policy,
            )?;
            let excitation = source_excitation_diagnostics_with_receiver_policy(
                prediction.pressure,
                observed,
                frequencies_hz,
                source_amplitude_pa,
                time_step_s,
                time_steps_per_frequency,
                frequency_bin_start_steps_per_frequency,
                receiver_channel_policy,
            )?;
            Ok(BreastUstForwardOperatorModelDiagnostics {
                model: prediction.model.to_owned(),
                normalized_l2_residual: residual.normalized_l2_residual,
                row_normalized_l2_residual_mean: residual.row_normalized_l2_residual_mean,
                source_scale_magnitude_coefficient_of_variation: excitation
                    .max_source_scale_magnitude_coefficient_of_variation,
                source_scale_phase_span_rad: excitation.max_source_scale_phase_span_rad,
            })
        })
        .collect::<KwaversResult<Vec<_>>>()?;

    per_model.sort_by(|left, right| {
        left.normalized_l2_residual
            .total_cmp(&right.normalized_l2_residual)
            .then_with(|| left.model.cmp(&right.model))
    });
    let best = per_model.first().expect("nonempty model diagnostics");
    let worst = per_model.last().expect("nonempty model diagnostics");
    Ok(BreastUstForwardOperatorEquivalenceDiagnostics {
        model_count: per_model.len(),
        receiver_channel_policy,
        best_model: best.model.clone(),
        best_normalized_l2_residual: best.normalized_l2_residual,
        worst_model: worst.model.clone(),
        worst_normalized_l2_residual: worst.normalized_l2_residual,
        residual_spread: worst.normalized_l2_residual - best.normalized_l2_residual,
        per_model,
    })
}

pub(super) fn validate_model_names(
    predictions_by_model: &[BreastUstForwardOperatorPrediction<'_>],
) -> KwaversResult<()> {
    let mut seen = HashSet::with_capacity(predictions_by_model.len());
    for prediction in predictions_by_model {
        if prediction.model.trim().is_empty() {
            return Err(KwaversError::InvalidInput(
                "operator model names must not be empty".into(),
            ));
        }
        if !seen.insert(prediction.model) {
            return Err(KwaversError::InvalidInput(format!(
                "duplicate operator model name: {}",
                prediction.model
            )));
        }
    }
    Ok(())
}
