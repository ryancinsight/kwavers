use super::super::diagnostics::{
    row_scale, scaled_observation_residual_metrics_by_receiver, source_excitation_diagnostics,
};
use super::grid::distance_m;
use super::BreastUstDirectFieldDiagnostics;
use crate::reconstruction::breast_ust_fwi::BreastUstPstdDatasetConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use ndarray::{s, Array3};
use kwavers_math::fft::Complex64;

pub(super) fn diagnostics_for_prediction(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    frequencies_hz: &[f64],
    config: BreastUstPstdDatasetConfig,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
    array: &MultiRowRingArray,
) -> KwaversResult<BreastUstDirectFieldDiagnostics> {
    validate_array_shape(predicted.dim(), array)?;
    if !config.source_amplitude_pa.is_finite() || config.source_amplitude_pa <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "source_amplitude_pa must be positive and finite, got {}",
            config.source_amplitude_pa
        )));
    }
    if !config.time_step_s.is_finite() || config.time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {}",
            config.time_step_s
        )));
    }

    let residual =
        scaled_observation_residual_metrics_by_receiver(predicted, observed, |_, _, _| true)?;
    let active = scaled_observation_residual_metrics_by_receiver(
        predicted,
        observed,
        |_, transmit, receiver| receiver % array.circumferential_elements() == transmit,
    )?;
    let passive = scaled_observation_residual_metrics_by_receiver(
        predicted,
        observed,
        |_, transmit, receiver| receiver % array.circumferential_elements() != transmit,
    )?;
    let source_scale = source_excitation_diagnostics(
        predicted,
        observed,
        frequencies_hz,
        config.source_amplitude_pa,
        config.time_step_s,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
    )?;
    let active_pair = topology_pair_errors(predicted, observed, array, ReceiverClass::Active)?;
    let passive_pair = topology_pair_errors(predicted, observed, array, ReceiverClass::Passive)?;

    Ok(BreastUstDirectFieldDiagnostics {
        normalized_l2_residual: residual.normalized_l2_residual,
        row_normalized_l2_residual_mean: residual.row_normalized_l2_residual_mean,
        active_only_normalized_l2_residual: active.normalized_l2_residual,
        passive_only_normalized_l2_residual: passive.normalized_l2_residual,
        source_scale_magnitude_coefficient_of_variation: source_scale
            .max_source_scale_magnitude_coefficient_of_variation,
        source_scale_phase_span_rad: source_scale.max_source_scale_phase_span_rad,
        active_pair_count: active_pair.count,
        active_self_channel_phase_error_rms_rad: active_pair.phase_error_rms_rad,
        active_self_channel_phase_error_max_abs_rad: active_pair.phase_error_max_abs_rad,
        active_self_channel_log_amplitude_error_rms: active_pair.log_amplitude_error_rms,
        passive_pair_count: passive_pair.count,
        passive_range_min_m: passive_pair.range_min_m,
        passive_range_max_m: passive_pair.range_max_m,
        passive_phase_error_rms_rad: passive_pair.phase_error_rms_rad,
        passive_phase_error_max_abs_rad: passive_pair.phase_error_max_abs_rad,
        passive_log_amplitude_error_rms: passive_pair.log_amplitude_error_rms,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReceiverClass {
    Active,
    Passive,
}

#[derive(Clone, Copy, Debug)]
struct TopologyPairErrors {
    count: usize,
    range_min_m: f64,
    range_max_m: f64,
    phase_error_rms_rad: f64,
    phase_error_max_abs_rad: f64,
    log_amplitude_error_rms: f64,
}

fn topology_pair_errors(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    array: &MultiRowRingArray,
    receiver_class: ReceiverClass,
) -> KwaversResult<TopologyPairErrors> {
    let (frequency_count, transmission_count, receiver_count) = predicted.dim();
    let circumferential_elements = array.circumferential_elements();
    let mut phase_sq = 0.0;
    let mut phase_max = 0.0;
    let mut log_amp_sq = 0.0;
    let mut range_min = f64::INFINITY;
    let mut range_max = f64::NEG_INFINITY;
    let mut count = 0usize;

    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmission_count {
            let predicted_row = predicted.slice(s![frequency_index, transmit_index, ..]);
            let observed_row = observed.slice(s![frequency_index, transmit_index, ..]);
            let scale = row_scale(predicted_row, observed_row)?;
            for receiver_index in 0..receiver_count {
                let is_active = receiver_index % circumferential_elements == transmit_index;
                if (receiver_class == ReceiverClass::Active) != is_active {
                    continue;
                }
                let modeled = scale * predicted[[frequency_index, transmit_index, receiver_index]];
                let measured = observed[[frequency_index, transmit_index, receiver_index]];
                if modeled.norm() <= f64::EPSILON || measured.norm() <= f64::EPSILON {
                    return Err(KwaversError::InvalidInput(
                        match receiver_class {
                            ReceiverClass::Active => {
                                "active self-channel amplitude must be nonzero"
                            }
                            ReceiverClass::Passive => "passive pair amplitude must be nonzero",
                        }
                        .into(),
                    ));
                }
                let phase_error = (modeled / measured).arg();
                let log_amp_error = (modeled.norm() / measured.norm()).ln();
                phase_sq += phase_error * phase_error;
                phase_max = f64::max(phase_max, phase_error.abs());
                log_amp_sq += log_amp_error * log_amp_error;
                let source_index = (receiver_index / circumferential_elements)
                    * circumferential_elements
                    + transmit_index;
                let range = distance_m(
                    array.elements()[receiver_index],
                    array.elements()[source_index],
                );
                range_min = f64::min(range_min, range);
                range_max = f64::max(range_max, range);
                count += 1;
            }
        }
    }

    if count == 0 {
        return Err(KwaversError::InvalidInput(
            match receiver_class {
                ReceiverClass::Active => {
                    "active self-channel diagnostics require at least one active receiver"
                }
                ReceiverClass::Passive => {
                    "passive diagnostics require at least one passive receiver"
                }
            }
            .into(),
        ));
    }
    Ok(TopologyPairErrors {
        count,
        range_min_m: range_min,
        range_max_m: range_max,
        phase_error_rms_rad: (phase_sq / count as f64).sqrt(),
        phase_error_max_abs_rad: phase_max,
        log_amplitude_error_rms: (log_amp_sq / count as f64).sqrt(),
    })
}

fn validate_array_shape(
    observation_shape: (usize, usize, usize),
    array: &MultiRowRingArray,
) -> KwaversResult<()> {
    let (_, transmissions, receivers) = observation_shape;
    if transmissions != array.circumferential_elements() || receivers != array.element_count() {
        return Err(KwaversError::DimensionMismatch(format!(
            "observation shape {:?} does not match array topology",
            observation_shape
        )));
    }
    Ok(())
}
