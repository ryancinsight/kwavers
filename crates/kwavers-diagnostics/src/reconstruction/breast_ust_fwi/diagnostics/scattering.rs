//! Scattering-increment diagnostics for Ali 2025 heterogeneous PSTD parity.
//!
//! The direct-field theorem is already checked by homogeneous PSTD diagnostics.
//! This module isolates the residual term that remains after each
//! frequency/transmit row is calibrated by the same complex source scale as the
//! homogeneous baseline.
//!
//! The scale-decomposition diagnostics report both calibration domains:
//! homogeneous-baseline scale for the scattering increment theorem and
//! model-specific least-squares scale for full-field operator equivalence.
//! The model-scale increment residual uses the same model-specific scale but
//! normalizes by the homogeneous-baseline observed increment energy. This keeps
//! all increment residuals in one denominator domain while exposing the
//! model-calibrated increment norm separately for source-drift diagnosis.

use super::operator::{validate_model_names, BreastUstForwardOperatorPrediction};
use super::residual::{
    row_scale_selected, scaled_observation_residual_metrics_by_policy, validate_observation_pair,
    validate_ring_channel_policy_shape, BreastUstReceiverChannelPolicy,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array3};
use num_complex::Complex64;

/// Per-model finite-window scattering-increment residual.
#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstScatteringIncrementModelDiagnostics {
    pub model: String,
    pub predicted_increment_l2_norm: f64,
    pub increment_residual_l2_norm: f64,
    pub normalized_increment_residual: f64,
    pub row_normalized_increment_residual_mean: f64,
    pub row_normalized_increment_residual_max: f64,
    pub increment_energy_ratio: f64,
    pub baseline_scaled_full_field_normalized_residual: f64,
    pub model_scaled_full_field_normalized_residual: f64,
    pub model_scaled_observed_increment_l2_norm: f64,
    pub model_scaled_increment_residual_l2_norm: f64,
    pub model_scaled_normalized_increment_residual: f64,
    pub model_scaled_increment_energy_ratio: f64,
    pub source_scale_relative_drift_mean: f64,
    pub source_scale_relative_drift_max: f64,
    pub source_scale_phase_drift_mean_abs_rad: f64,
    pub source_scale_phase_drift_max_abs_rad: f64,
}

/// Residual decomposition after homogeneous direct-field calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstScatteringIncrementDiagnostics {
    pub model_count: usize,
    pub receiver_channel_policy: BreastUstReceiverChannelPolicy,
    pub direct_field_normalized_l2_residual: f64,
    pub direct_field_scaled_residual_l2_norm: f64,
    pub observed_increment_l2_norm: f64,
    pub best_model: String,
    pub best_normalized_increment_residual: f64,
    pub best_model_scaled_increment_model: String,
    pub best_model_scaled_normalized_increment_residual: f64,
    pub worst_model: String,
    pub worst_normalized_increment_residual: f64,
    pub increment_residual_spread: f64,
    pub per_model: Vec<BreastUstScatteringIncrementModelDiagnostics>,
}

/// Compare each candidate's scattered field against the observed finite-window
/// increment after homogeneous source-scale calibration.
///
/// For each frequency/transmit row, let `alpha` be the complex least-squares
/// scale minimizing `||alpha*d0 - d_obs||` over the selected receivers. The
/// observed scattering increment is `d_obs - alpha*d0`; the model increment is
/// `alpha*(d_model - d0)`. This diagnostic reports the normalized L2 residual
/// between those two increments.
///
/// # Errors
/// Returns an error when observation shapes differ, model names are empty or
/// repeated, a receiver policy is incompatible with the array shape, or the
/// calibrated observed scattering increment has zero energy.
pub fn scattering_increment_diagnostics(
    homogeneous_baseline: &Array3<Complex64>,
    predictions_by_model: &[BreastUstForwardOperatorPrediction<'_>],
    observed: &Array3<Complex64>,
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<BreastUstScatteringIncrementDiagnostics> {
    if predictions_by_model.is_empty() {
        return Err(KwaversError::InvalidInput(
            "predictions_by_model must not be empty".into(),
        ));
    }
    validate_model_names(predictions_by_model)?;
    validate_observation_pair(homogeneous_baseline, observed)?;
    let (_, transmission_count, receiver_count) = observed.dim();
    validate_ring_channel_policy_shape(
        transmission_count,
        receiver_count,
        receiver_channel_policy,
    )?;
    for prediction in predictions_by_model {
        validate_observation_pair(prediction.pressure, observed)?;
    }

    let direct_field = scaled_observation_residual_metrics_by_policy(
        homogeneous_baseline,
        observed,
        receiver_channel_policy,
    )?;
    let mut per_model = predictions_by_model
        .iter()
        .map(|prediction| {
            scattering_increment_model_diagnostics(
                homogeneous_baseline,
                prediction,
                observed,
                receiver_channel_policy,
            )
        })
        .collect::<KwaversResult<Vec<_>>>()?;

    per_model.sort_by(|left, right| {
        left.normalized_increment_residual
            .total_cmp(&right.normalized_increment_residual)
            .then_with(|| left.model.cmp(&right.model))
    });
    let best = per_model.first().expect("nonempty scattering diagnostics");
    let worst = per_model.last().expect("nonempty scattering diagnostics");
    let best_model_scaled = per_model
        .iter()
        .min_by(|left, right| {
            left.model_scaled_normalized_increment_residual
                .total_cmp(&right.model_scaled_normalized_increment_residual)
                .then_with(|| left.model.cmp(&right.model))
        })
        .expect("nonempty scattering diagnostics");

    Ok(BreastUstScatteringIncrementDiagnostics {
        model_count: per_model.len(),
        receiver_channel_policy,
        direct_field_normalized_l2_residual: direct_field.normalized_l2_residual,
        direct_field_scaled_residual_l2_norm: direct_field.scaled_residual_l2_norm,
        observed_increment_l2_norm: direct_field.scaled_residual_l2_norm,
        best_model: best.model.clone(),
        best_normalized_increment_residual: best.normalized_increment_residual,
        best_model_scaled_increment_model: best_model_scaled.model.clone(),
        best_model_scaled_normalized_increment_residual: best_model_scaled
            .model_scaled_normalized_increment_residual,
        worst_model: worst.model.clone(),
        worst_normalized_increment_residual: worst.normalized_increment_residual,
        increment_residual_spread: worst.normalized_increment_residual
            - best.normalized_increment_residual,
        per_model,
    })
}

fn scattering_increment_model_diagnostics(
    homogeneous_baseline: &Array3<Complex64>,
    prediction: &BreastUstForwardOperatorPrediction<'_>,
    observed: &Array3<Complex64>,
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<BreastUstScatteringIncrementModelDiagnostics> {
    let (frequency_count, transmission_count, receiver_count) = observed.dim();
    let mut observed_full_field_norm_sq = 0.0;
    let mut observed_increment_norm_sq = 0.0;
    let mut predicted_increment_norm_sq = 0.0;
    let mut residual_norm_sq = 0.0;
    let mut baseline_scaled_full_field_residual_sq = 0.0;
    let mut model_scaled_full_field_residual_sq = 0.0;
    let mut model_scaled_observed_increment_norm_sq = 0.0;
    let mut model_scaled_predicted_increment_norm_sq = 0.0;
    let mut model_scaled_increment_residual_sq = 0.0;
    let mut row_residual_sum = 0.0;
    let mut row_residual_max = 0.0;
    let mut scale_relative_drift_sum = 0.0;
    let mut scale_relative_drift_max = 0.0;
    let mut scale_phase_drift_sum = 0.0;
    let mut scale_phase_drift_max = 0.0;
    let mut row_count = 0usize;

    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmission_count {
            let baseline_row = homogeneous_baseline.slice(s![frequency_index, transmit_index, ..]);
            let observed_row = observed.slice(s![frequency_index, transmit_index, ..]);
            let prediction_row = prediction
                .pressure
                .slice(s![frequency_index, transmit_index, ..]);
            let baseline_scale = row_scale_selected(baseline_row, observed_row, |receiver| {
                receiver_channel_policy.selects(transmit_index, receiver, transmission_count)
            })?;
            let model_scale = row_scale_selected(prediction_row, observed_row, |receiver| {
                receiver_channel_policy.selects(transmit_index, receiver, transmission_count)
            })?;
            let baseline_scale_norm = baseline_scale.norm();
            if baseline_scale_norm <= f64::EPSILON {
                return Err(KwaversError::InvalidInput(
                    "baseline source scale has zero magnitude for a selected row".into(),
                ));
            }
            let model_scale_norm = model_scale.norm();
            if model_scale_norm <= f64::EPSILON {
                return Err(KwaversError::InvalidInput(
                    "model source scale has zero magnitude for a selected row".into(),
                ));
            }
            let scale_relative_drift = (model_scale - baseline_scale).norm() / baseline_scale_norm;
            let scale_ratio = model_scale / baseline_scale;
            let scale_phase_drift = scale_ratio.im.atan2(scale_ratio.re).abs();
            scale_relative_drift_sum += scale_relative_drift;
            scale_relative_drift_max = f64::max(scale_relative_drift_max, scale_relative_drift);
            scale_phase_drift_sum += scale_phase_drift;
            scale_phase_drift_max = f64::max(scale_phase_drift_max, scale_phase_drift);

            let mut row_observed_increment_sq = 0.0;
            let mut row_residual_sq = 0.0;
            for receiver_index in 0..receiver_count {
                if !receiver_channel_policy.selects(
                    transmit_index,
                    receiver_index,
                    transmission_count,
                ) {
                    continue;
                }
                let baseline =
                    homogeneous_baseline[[frequency_index, transmit_index, receiver_index]];
                let prediction_value =
                    prediction.pressure[[frequency_index, transmit_index, receiver_index]];
                let observed_value = observed[[frequency_index, transmit_index, receiver_index]];
                let observed_increment = observed_value - baseline_scale * baseline;
                let predicted_increment = baseline_scale * (prediction_value - baseline);
                let residual = predicted_increment - observed_increment;
                let baseline_scaled_full_field_residual =
                    baseline_scale * prediction_value - observed_value;
                let model_scaled_full_field_residual =
                    model_scale * prediction_value - observed_value;
                let model_scaled_observed_increment = observed_value - model_scale * baseline;
                let model_scaled_predicted_increment = model_scale * (prediction_value - baseline);
                let model_scaled_increment_residual =
                    model_scaled_predicted_increment - model_scaled_observed_increment;
                observed_full_field_norm_sq += observed_value.norm_sqr();
                row_observed_increment_sq += observed_increment.norm_sqr();
                predicted_increment_norm_sq += predicted_increment.norm_sqr();
                row_residual_sq += residual.norm_sqr();
                baseline_scaled_full_field_residual_sq +=
                    baseline_scaled_full_field_residual.norm_sqr();
                model_scaled_full_field_residual_sq += model_scaled_full_field_residual.norm_sqr();
                model_scaled_observed_increment_norm_sq +=
                    model_scaled_observed_increment.norm_sqr();
                model_scaled_predicted_increment_norm_sq +=
                    model_scaled_predicted_increment.norm_sqr();
                model_scaled_increment_residual_sq += model_scaled_increment_residual.norm_sqr();
            }
            if row_observed_increment_sq <= f64::EPSILON {
                return Err(KwaversError::InvalidInput(
                    "observed scattering increment has zero energy for a selected row".into(),
                ));
            }
            observed_increment_norm_sq += row_observed_increment_sq;
            residual_norm_sq += row_residual_sq;
            let row_residual = (row_residual_sq / row_observed_increment_sq).sqrt();
            row_residual_sum += row_residual;
            row_residual_max = f64::max(row_residual_max, row_residual);
            row_count += 1;
        }
    }

    if observed_full_field_norm_sq <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "selected observed full field has zero energy".into(),
        ));
    }
    let observed_increment_l2_norm = observed_increment_norm_sq.sqrt();
    let observed_full_field_l2_norm = observed_full_field_norm_sq.sqrt();
    let model_scaled_observed_increment_l2_norm = model_scaled_observed_increment_norm_sq.sqrt();
    Ok(BreastUstScatteringIncrementModelDiagnostics {
        model: prediction.model.to_owned(),
        predicted_increment_l2_norm: predicted_increment_norm_sq.sqrt(),
        increment_residual_l2_norm: residual_norm_sq.sqrt(),
        normalized_increment_residual: residual_norm_sq.sqrt() / observed_increment_l2_norm,
        row_normalized_increment_residual_mean: row_residual_sum / row_count as f64,
        row_normalized_increment_residual_max: row_residual_max,
        increment_energy_ratio: predicted_increment_norm_sq.sqrt() / observed_increment_l2_norm,
        baseline_scaled_full_field_normalized_residual: baseline_scaled_full_field_residual_sq
            .sqrt()
            / observed_full_field_l2_norm,
        model_scaled_full_field_normalized_residual: model_scaled_full_field_residual_sq.sqrt()
            / observed_full_field_l2_norm,
        model_scaled_observed_increment_l2_norm,
        model_scaled_increment_residual_l2_norm: model_scaled_increment_residual_sq.sqrt(),
        model_scaled_normalized_increment_residual: model_scaled_increment_residual_sq.sqrt()
            / observed_increment_l2_norm,
        model_scaled_increment_energy_ratio: model_scaled_predicted_increment_norm_sq.sqrt()
            / observed_increment_l2_norm,
        source_scale_relative_drift_mean: scale_relative_drift_sum / row_count as f64,
        source_scale_relative_drift_max: scale_relative_drift_max,
        source_scale_phase_drift_mean_abs_rad: scale_phase_drift_sum / row_count as f64,
        source_scale_phase_drift_max_abs_rad: scale_phase_drift_max,
    })
}
