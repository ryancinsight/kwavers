//! Time-domain Delay-and-Sum (DAS) beamforming — core algorithm.

use crate::analysis::signal_processing::beamforming::time_domain::delay_reference::{
    relative_delays_s, DelayReference,
};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Apply time-domain DAS given **absolute** propagation delays and an explicit delay reference.
///
/// # Algorithm
///
/// This function:
/// 1. Validates invariants (data shape, finite values, array lengths)
/// 2. Converts absolute delays `τᵢ` into relative delays `Δτᵢ = τᵢ - τᵣₑ𝒻`
/// 3. Applies integer sample shifts `kᵢ = round(Δτᵢ * fs)` and sums weighted samples
///
/// # Parameters
///
/// - `sensor_data`: RF data, shape `(n_elements, 1, n_samples)`
/// - `sampling_frequency_hz`: sampling rate `fs` in Hz
/// - `delays_s`: absolute TOF delays `τᵢ` in seconds, length `n_elements`
/// - `weights`: apodization weights (real), length `n_elements`
/// - `reference`: delay datum selection policy
///
/// # Returns
///
/// Beamformed output, shape `(1, 1, n_samples)`
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn delay_and_sum(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array3<f64>> {
    let (n_elements, channels, n_samples) = sensor_data.dim();

    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das expects sensor_data shape (n_elements, 1, n_samples); got channels={channels}. \
             RF data should have a single channel dimension."
        )));
    }
    if n_elements == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "time_domain::das requires n_elements > 0 and n_samples > 0".to_owned(),
        ));
    }
    if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das requires sampling_frequency_hz to be finite and > 0; got {sampling_frequency_hz}"
        )));
    }
    if delays_s.len() != n_elements {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das: delays_s length ({}) must equal n_elements ({n_elements})",
            delays_s.len()
        )));
    }
    if weights.len() != n_elements {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das: weights length ({}) must equal n_elements ({n_elements})",
            weights.len()
        )));
    }

    let rel_delays_s = relative_delays_s(delays_s, reference)?;

    let mut shifts: Vec<isize> = Vec::with_capacity(n_elements);
    for (i, &dt) in rel_delays_s.iter().enumerate() {
        let k = (dt * sampling_frequency_hz).round();
        if !k.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "time_domain::das: relative_delay[{i}] = {dt} s produced non-finite sample shift \
                 (fs = {sampling_frequency_hz} Hz)"
            )));
        }
        shifts.push(k as isize);
    }

    let min_shift = shifts.iter().copied().min().unwrap_or(0);
    let offset = -min_shift;

    let mut output = Array3::<f64>::zeros((1, 1, n_samples));

    for elem_idx in 0..n_elements {
        let effective_shift = shifts[elem_idx] + offset;
        let eff = if effective_shift < 0 {
            0usize
        } else {
            effective_shift as usize
        };

        if eff >= n_samples {
            continue;
        }

        let w = weights[elem_idx];
        for t in eff..n_samples {
            output[[0, 0, t - eff]] += sensor_data[[elem_idx, 0, t]] * w;
        }
    }

    Ok(output)
}

/// Recommended default delay reference for transient SRP-DAS localization: sensor index 0.
pub const DEFAULT_DELAY_REFERENCE: DelayReference = DelayReference::SensorIndex(0);
