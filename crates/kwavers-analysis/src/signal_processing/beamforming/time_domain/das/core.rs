//! Time-domain Delay-and-Sum (DAS) beamforming — core algorithm.

use crate::signal_processing::beamforming::time_domain::delay_reference::{
    relative_delays_s, DelayReference,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    Array2,
    Array3,
};

/// Validate the common DAS input contract and return `(n_elements, n_samples)`.
///
/// Shared by [`align_channels`] and every consumer so the shape/finiteness rules
/// live in exactly one place.
fn validate_das_inputs(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
) -> KwaversResult<(usize, usize)> {
    let [n_elements, channels, n_samples] = sensor_data.shape();

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
    Ok((n_elements, n_samples))
}

/// Delay-align per-element RF data onto a common time base **without** summing.
///
/// This is the single source of truth for the alignment step shared by
/// [`delay_and_sum`] and coherence-factor weighting
/// ([`super::super::coherence`]). Element `i` is shifted so that its
/// time-of-flight contribution lands at the same output index as every other
/// element; the earliest-arriving (most negative) relative shift defines output
/// index 0 (`offset = -min_shift`), matching [`delay_and_sum`] exactly.
///
/// # Returns
///
/// Aligned matrix of shape `(n_elements, n_samples)` where `aligned[[i, j]]`
/// is element `i`'s sample that contributes to output index `j` (zero-padded
/// past the end of the record). Apodization is intentionally **not** applied —
/// coherence estimators (Mallart & Fink 1994) operate on the raw aperture data,
/// and [`delay_and_sum`] applies its weights during summation.
///
/// # Errors
/// - [`KwaversError::InvalidInput`] on shape/length/finiteness contract violations.
pub fn align_channels(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array2<f64>> {
    let (n_elements, n_samples) =
        validate_das_inputs(sensor_data, sampling_frequency_hz, delays_s, weights)?;

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

    // `offset = -min_shift` guarantees `eff >= 0` for every element, so the
    // earliest arrival maps to output index 0.
    let min_shift = shifts.iter().copied().min().unwrap_or(0);
    let offset = -min_shift;

    let mut aligned = Array2::<f64>::zeros((n_elements, n_samples));
    for elem_idx in 0..n_elements {
        let eff = (shifts[elem_idx] + offset).max(0) as usize;
        if eff >= n_samples {
            continue;
        }
        for t in eff..n_samples {
            aligned[[elem_idx, t - eff]] = sensor_data[[elem_idx, 0, t]];
        }
    }

    Ok(aligned)
}

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
    let aligned = align_channels(
        sensor_data,
        sampling_frequency_hz,
        delays_s,
        weights,
        reference,
    )?;
    Ok(sum_aligned(&aligned, weights))
}

/// Apodized coherent sum over a delay-aligned aperture matrix.
///
/// SSOT for the summation step: `output[j] = Σᵢ wᵢ · aligned[[i, j]]`, shape
/// `(1, 1, n_samples)`. Shared by [`delay_and_sum`] and coherence-factor
/// weighting so the two never drift. `weights.len()` must equal
/// `aligned.shape()[0]` (guaranteed by [`align_channels`] validation upstream).
#[must_use]
pub fn sum_aligned(aligned: &Array2<f64>, weights: &[f64]) -> Array3<f64> {
    let [n_elements, n_samples] = aligned.shape();
    let mut output = Array3::<f64>::zeros((1, 1, n_samples));
    for elem_idx in 0..n_elements {
        let w = weights[elem_idx];
        for j in 0..n_samples {
            output[[0, 0, j]] += aligned[[elem_idx, j]] * w;
        }
    }
    output
}

/// Recommended default delay reference for transient SRP-DAS localization: sensor index 0.
pub const DEFAULT_DELAY_REFERENCE: DelayReference = DelayReference::SensorIndex(0);
