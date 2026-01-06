//! Time-domain Delay-and-Sum (DAS) beamforming with explicit delay reference policy.
//!
//! # Field jargon / intent
//! - **DAS** is also called *conventional beamforming* or *shift-and-sum*.
//! - For time-domain DAS, you must choose a **delay reference / delay datum**:
//!   the absolute propagation delays `τ_i(p) = ||x_i - p|| / c` are only defined up to
//!   an additive constant (the unknown emission time).
//! - This module makes that reference explicit by depending on the shared
//!   `time_domain::delay_reference` policy and converting absolute TOF delays into
//!   **relative delays** `Δτ_i = τ_i - τ_ref` prior to sample shifting.
//!
//! # Why this exists (SSOT + correctness)
//! The legacy `BeamformingProcessor::delay_and_sum_with` performs an implicit
//! **latest-arrival alignment** (`τ_ref = max τ`). That is a valid convention for
//! beamforming output rendering, but it is *not a universal default* for localization
//! scoring objectives (SRP-DAS) and can make scores point-invariant under certain
//! synthetic data models if the reference is not modeled consistently.
//!
//! By separating (1) reference selection (policy) from (2) application (math),
//! we avoid silent convention mismatches.
//!
//! # Data model
//! - `sensor_data` is shaped `(n_elements, 1, n_samples)`.
//! - `delays_s` are **absolute** propagation delays in seconds, one per element.
//! - Output is shaped `(1, 1, n_samples)`.
//!
//! # Discretization
//! Delays are applied by integer sample shifts using `round(Δτ * fs)`.
//! This is a deliberate, explicit trade-off; fractional delay filtering is not
//! implemented in this module.

use crate::error::{KwaversError, KwaversResult};
use crate::sensor::beamforming::time_domain::delay_reference::{relative_delays_s, DelayReference};
use ndarray::Array3;

/// Apply time-domain DAS given **absolute** propagation delays and an explicit delay reference.
///
/// This function:
/// 1. Validates invariants.
/// 2. Converts absolute delays `τ_i` into relative delays `Δτ_i = τ_i - τ_ref`.
/// 3. Applies integer sample shifts `k_i = round(Δτ_i * fs)` and sums weighted samples.
///
/// # Parameters
/// - `sensor_data`: `(n_elements, 1, n_samples)`
/// - `sampling_frequency_hz`: `fs` in Hz
/// - `delays_s`: absolute delays `τ_i` in seconds, length `n_elements`
/// - `weights`: real weights, length `n_elements`
/// - `reference`: delay datum selection policy
///
/// # Output
/// `(1, 1, n_samples)`
///
/// # Notes
/// - Only **non-negative** relative delays are supported without truncation.
///   When `Δτ_i < 0`, the required shift would be negative; this implementation handles that
///   by shifting the output index instead (equivalently, delaying other channels).
///   This is safe and keeps the operation well-defined, but it changes the effective
///   time origin; callers should prefer a reference that keeps relative delays mostly ≥ 0
///   (e.g., `EarliestArrival`).
pub fn delay_and_sum_time_domain_with_reference(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array3<f64>> {
    let (n_elements, channels, n_samples) = sensor_data.dim();

    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das expects sensor_data shape (n_elements, 1, n_samples); got channels={channels}"
        )));
    }
    if n_elements == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "time_domain::das requires n_elements > 0 and n_samples > 0".to_string(),
        ));
    }
    if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(
            "time_domain::das requires sampling_frequency_hz to be finite and > 0".to_string(),
        ));
    }
    if delays_s.len() != n_elements || weights.len() != n_elements {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das invalid delays/weights: delays={}, weights={}, n_elements={n_elements}",
            delays_s.len(),
            weights.len()
        )));
    }

    let rel_delays_s = relative_delays_s(delays_s, reference)?;

    // Convert to integer sample shifts. We allow negative shifts by tracking min/max.
    let mut shifts: Vec<isize> = Vec::with_capacity(n_elements);
    for &dt in &rel_delays_s {
        let k = (dt * sampling_frequency_hz).round();
        if !k.is_finite() {
            return Err(KwaversError::InvalidInput(
                "time_domain::das produced non-finite sample shift".to_string(),
            ));
        }
        shifts.push(k as isize);
    }

    // We want an output array of length n_samples. For negative shifts, we can only produce a
    // truncated / shifted output without resizing time. To keep the contract stable, we:
    // - compute a global offset so that the smallest shift maps to 0
    // - apply all channels with a non-negative effective shift
    let min_shift = shifts.iter().copied().min().unwrap_or(0);
    let offset = -min_shift; // makes (shift + offset) >= 0 for all

    let mut output = Array3::<f64>::zeros((1, 1, n_samples));

    for elem_idx in 0..n_elements {
        let effective_shift = shifts[elem_idx] + offset;
        let eff = if effective_shift < 0 {
            // Should be impossible by construction, but keep it defensive.
            0usize
        } else {
            effective_shift as usize
        };

        if eff >= n_samples {
            continue;
        }

        let w = weights[elem_idx];
        for t in eff..n_samples {
            // read sensor at t, write at t - eff
            output[[0, 0, t - eff]] += sensor_data[[elem_idx, 0, t]] * w;
        }
    }

    Ok(output)
}

/// Recommended common default for transient SRP-DAS localization:
/// use a fixed reference sensor index 0.
///
/// This keeps the reference deterministic and matches typical array-processing practice.
pub const DEFAULT_DELAY_REFERENCE: DelayReference = DelayReference::SensorIndex(0);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn das_aligns_impulses_with_reference_sensor_0() {
        // Two sensors, impulses occur at times consistent with relative delays to sensor 0.
        let fs = 10.0;
        let n = 8usize;

        // delays: τ0 = 1.0s (ref), τ1 = 1.2s => Δτ1 = 0.2s => 2 samples shift for sensor1
        let delays = vec![1.0, 1.2];
        let weights = vec![1.0, 1.0];

        let mut x = Array3::<f64>::zeros((2, 1, n));

        // Put an impulse at t=3 for sensor0; for sensor1, impulse at t=3+2=5
        x[[0, 0, 3]] = 1.0;
        x[[1, 0, 5]] = 1.0;

        let y = delay_and_sum_time_domain_with_reference(
            &x,
            fs,
            &delays,
            &weights,
            DelayReference::SensorIndex(0),
        )
        .unwrap();

        // After alignment, both should land on the same output sample (t=3).
        assert!((y[[0, 0, 3]] - 2.0).abs() < 1e-12);
    }
}
