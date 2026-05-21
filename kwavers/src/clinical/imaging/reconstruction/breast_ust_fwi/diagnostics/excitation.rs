use super::residual::{
    row_scale_selected, validate_observation_pair, validate_ring_channel_policy_shape,
    BreastUstReceiverChannelPolicy,
};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array3};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstSourceExcitationFrequencyDiagnostics {
    pub frequency_hz: f64,
    pub tone_bin_magnitude: f64,
    pub tone_bin_phase_rad: f64,
    pub mean_source_scale_magnitude: f64,
    pub source_scale_magnitude_coefficient_of_variation: f64,
    pub source_scale_phase_circular_variance: f64,
    pub source_scale_phase_span_rad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstSourceExcitationDiagnostics {
    pub frequency_count: usize,
    pub transmission_count: usize,
    pub source_amplitude_pa: f64,
    pub max_source_scale_magnitude_coefficient_of_variation: f64,
    pub max_source_scale_phase_circular_variance: f64,
    pub max_source_scale_phase_span_rad: f64,
    pub per_frequency: Vec<BreastUstSourceExcitationFrequencyDiagnostics>,
}

pub fn source_excitation_diagnostics(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    frequencies_hz: &[f64],
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
) -> KwaversResult<BreastUstSourceExcitationDiagnostics> {
    source_excitation_diagnostics_with_receiver_policy(
        predicted,
        observed,
        frequencies_hz,
        source_amplitude_pa,
        time_step_s,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
        BreastUstReceiverChannelPolicy::All,
    )
}

pub(crate) fn source_excitation_diagnostics_with_receiver_policy(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    frequencies_hz: &[f64],
    source_amplitude_pa: f64,
    time_step_s: f64,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<BreastUstSourceExcitationDiagnostics> {
    validate_observation_pair(predicted, observed)?;
    validate_frequency_metadata(
        predicted.dim().0,
        frequencies_hz,
        time_steps_per_frequency,
        frequency_bin_start_steps_per_frequency,
    )?;
    if !source_amplitude_pa.is_finite() || source_amplitude_pa <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "source_amplitude_pa must be positive and finite, got {source_amplitude_pa}"
        )));
    }
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {time_step_s}"
        )));
    }

    let transmission_count = predicted.dim().1;
    validate_ring_channel_policy_shape(
        transmission_count,
        predicted.dim().2,
        receiver_channel_policy,
    )?;
    let mut per_frequency = Vec::with_capacity(frequencies_hz.len());
    for (frequency_index, &frequency_hz) in frequencies_hz.iter().enumerate() {
        let bin_coefficient = sine_frequency_bin_coefficient(
            frequency_hz,
            time_step_s,
            time_steps_per_frequency[frequency_index],
            frequency_bin_start_steps_per_frequency[frequency_index],
        )?;
        let normalization = source_amplitude_pa * bin_coefficient;
        let mut magnitudes = Vec::with_capacity(transmission_count);
        let mut phases = Vec::with_capacity(transmission_count);
        for transmit_index in 0..transmission_count {
            let predicted_row = predicted.slice(s![frequency_index, transmit_index, ..]);
            let observed_row = observed.slice(s![frequency_index, transmit_index, ..]);
            let scale = row_scale_selected(predicted_row, observed_row, |receiver_index| {
                receiver_channel_policy.selects(transmit_index, receiver_index, transmission_count)
            })? / normalization;
            magnitudes.push(scale.norm());
            phases.push(scale.arg());
        }
        let mean_magnitude = mean(&magnitudes)?;
        let circular_variance = 1.0 - circular_resultant(&phases)?;
        per_frequency.push(BreastUstSourceExcitationFrequencyDiagnostics {
            frequency_hz,
            tone_bin_magnitude: bin_coefficient.norm(),
            tone_bin_phase_rad: bin_coefficient.arg(),
            mean_source_scale_magnitude: mean_magnitude,
            source_scale_magnitude_coefficient_of_variation: stddev(&magnitudes, mean_magnitude)?
                / mean_magnitude.max(f64::EPSILON),
            source_scale_phase_circular_variance: circular_variance,
            source_scale_phase_span_rad: phase_span_rad(&phases)?,
        });
    }

    Ok(BreastUstSourceExcitationDiagnostics {
        frequency_count: predicted.dim().0,
        transmission_count,
        source_amplitude_pa,
        max_source_scale_magnitude_coefficient_of_variation: per_frequency
            .iter()
            .map(|row| row.source_scale_magnitude_coefficient_of_variation)
            .fold(f64::NEG_INFINITY, f64::max),
        max_source_scale_phase_circular_variance: per_frequency
            .iter()
            .map(|row| row.source_scale_phase_circular_variance)
            .fold(f64::NEG_INFINITY, f64::max),
        max_source_scale_phase_span_rad: per_frequency
            .iter()
            .map(|row| row.source_scale_phase_span_rad)
            .fold(f64::NEG_INFINITY, f64::max),
        per_frequency,
    })
}

pub fn sine_frequency_bin_coefficient(
    frequency_hz: f64,
    time_step_s: f64,
    total_steps: usize,
    start_sample: usize,
) -> KwaversResult<Complex64> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "frequency_hz must be positive and finite, got {frequency_hz}"
        )));
    }
    if !time_step_s.is_finite() || time_step_s <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_step_s must be positive and finite, got {time_step_s}"
        )));
    }
    if total_steps == 0 || start_sample >= total_steps {
        return Err(KwaversError::InvalidInput(
            "start_sample must lie inside the trace".into(),
        ));
    }
    let mut coefficient = Complex64::new(0.0, 0.0);
    for sample in start_sample..total_steps {
        let phase = TAU * frequency_hz * sample as f64 * time_step_s;
        coefficient += phase.sin() * Complex64::new(phase.cos(), -phase.sin());
    }
    coefficient *= 2.0 / (total_steps - start_sample) as f64;
    if coefficient.norm() <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "sine frequency-bin coefficient has zero energy".into(),
        ));
    }
    Ok(coefficient)
}

pub(crate) fn phase_span_rad(phases: &[f64]) -> KwaversResult<f64> {
    let Some((&first, rest)) = phases.split_first() else {
        return Err(KwaversError::InvalidInput(
            "phase vector must not be empty".into(),
        ));
    };
    let mut previous_raw = first;
    let mut offset = 0.0;
    let mut min_phase = first;
    let mut max_phase = first;
    for &phase in rest {
        let delta = phase - previous_raw;
        if delta > PI {
            offset -= TAU;
        } else if delta < -PI {
            offset += TAU;
        }
        let unwrapped = phase + offset;
        min_phase = f64::min(min_phase, unwrapped);
        max_phase = f64::max(max_phase, unwrapped);
        previous_raw = phase;
    }
    Ok(max_phase - min_phase)
}

fn validate_frequency_metadata(
    frequency_count: usize,
    frequencies_hz: &[f64],
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
) -> KwaversResult<()> {
    if frequencies_hz.len() != frequency_count {
        return Err(KwaversError::DimensionMismatch(
            "frequencies_hz length must match observation frequency axis".into(),
        ));
    }
    if time_steps_per_frequency.len() != frequency_count
        || frequency_bin_start_steps_per_frequency.len() != frequency_count
    {
        return Err(KwaversError::DimensionMismatch(
            "frequency metadata length must match observation frequency axis".into(),
        ));
    }
    for ((&frequency_hz, &total_steps), &start_step) in frequencies_hz
        .iter()
        .zip(time_steps_per_frequency)
        .zip(frequency_bin_start_steps_per_frequency)
    {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "frequency_hz entries must be positive and finite, got {frequency_hz}"
            )));
        }
        if total_steps == 0 || start_step >= total_steps {
            return Err(KwaversError::InvalidInput(
                "frequency bin start must be smaller than total steps".into(),
            ));
        }
    }
    Ok(())
}

fn mean(values: &[f64]) -> KwaversResult<f64> {
    if values.is_empty() {
        return Err(KwaversError::InvalidInput(
            "mean requires at least one value".into(),
        ));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn stddev(values: &[f64], mean: f64) -> KwaversResult<f64> {
    Ok((values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt())
}

fn circular_resultant(phases: &[f64]) -> KwaversResult<f64> {
    if phases.is_empty() {
        return Err(KwaversError::InvalidInput(
            "phase vector must not be empty".into(),
        ));
    }
    let mean_cos = phases.iter().map(|phase| phase.cos()).sum::<f64>() / phases.len() as f64;
    let mean_sin = phases.iter().map(|phase| phase.sin()).sum::<f64>() / phases.len() as f64;
    Ok((mean_cos * mean_cos + mean_sin * mean_sin).sqrt())
}
