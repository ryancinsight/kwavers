//! Time-domain Delay-and-Sum (DAS) beamforming with explicit delay reference policy.
//!
//! # Mathematical Foundation
//!
//! **Delay-and-Sum (DAS)** beamforming is the foundational array processing algorithm:
//!
//! ```text
//! y(t) = Σᵢ wᵢ · xᵢ(t - Δτᵢ)
//! ```
//!
//! where:
//! - `xᵢ(t)` = received signal at sensor i
//! - `wᵢ` = apodization weight for sensor i (typically normalized)
//! - `Δτᵢ` = relative time delay (steering delay)
//! - `y(t)` = beamformed output
//!
//! # Delay Reference Policy
//!
//! For time-domain DAS, you must choose a **delay reference / delay datum**.
//! The absolute propagation delays `τᵢ(p) = ||xᵢ - p|| / c` are only defined up to
//! an additive constant (the unknown emission time).
//!
//! This implementation makes the reference **explicit** by depending on the
//! `delay_reference` module and converting absolute TOF delays into
//! **relative delays** `Δτᵢ = τᵢ - τᵣₑ𝒻` prior to sample shifting.
//!
//! ## Why Explicit Reference Matters
//!
//! Legacy implementations often perform an implicit **latest-arrival alignment**
//! (`τᵣₑ𝒻 = max τ`). While valid for beamforming output rendering, this is
//! **not a universal default** for localization scoring objectives (SRP-DAS)
//! and can make scores point-invariant under certain synthetic data models
//! if the reference is not modeled consistently.
//!
//! By separating:
//! 1. **Reference selection** (policy in `delay_reference`)
//! 2. **Application** (math in this module)
//!
//! we avoid silent convention mismatches.
//!
//! # Data Model
//!
//! - `sensor_data`: shape `(n_elements, 1, n_samples)`
//!   - First dimension: sensor/element index
//!   - Second dimension: channel (always 1 for RF data)
//!   - Third dimension: time samples
//! - `delays_s`: **absolute** propagation delays in seconds, length `n_elements`
//! - `weights`: real apodization weights, length `n_elements`
//! - Output: shape `(1, 1, n_samples)`
//!
//! # Discretization
//!
//! Delays are applied by **integer sample shifts** using `round(Δτ * fs)`.
//! This is a deliberate, explicit trade-off:
//! - ✅ Simple, fast, exact for integer delays
//! - ❌ Quantization error for fractional delays
//!
//! **Fractional delay filtering** (e.g., sinc interpolation, FIR filters) is **not**
//! implemented in this basic DAS module. For sub-sample accuracy, use:
//! - Upsampling + integer shifts
//! - Frequency-domain beamforming (phase shifts)
//! - Dedicated fractional delay filter module (future)
//!
//! # Literature References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.
//!   (Chapter 2: Spatial and Temporal Sampling)
//! - Brandstein, M., & Ward, D. (2001). *Microphone Arrays*. Springer.
//!   (Chapter 5: Beamforming)
//! - DiBiase, J. H. (2000). *A High-Accuracy, Low-Latency Technique for Talker Localization
//!   in Reverberant Environments Using Microphone Arrays*. PhD Thesis, Brown University.
//!   (Steered Response Power - Phase Transform, SRP-PHAT)
//!
//! # Migration Note
//!
//! This module was migrated from `domain::sensor::beamforming::time_domain::das` to
//! `analysis::signal_processing::beamforming::time_domain::das` as part of the
//! architectural purification effort (ADR 003). The API remains unchanged.

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
///
/// # Errors
///
/// - If `sensor_data` does not have shape `(n_elements, 1, n_samples)` with valid dimensions
/// - If `sampling_frequency_hz` is not finite and positive
/// - If `delays_s` or `weights` have incorrect length (must equal `n_elements`)
/// - If delay reference cannot be resolved (see [`DelayReference::resolve_reference_delay_s`])
/// - If discretization produces non-finite sample shifts
///
/// # Handling Negative Shifts
///
/// When `Δτᵢ < 0`, the required shift would be negative (sensor arrives earlier than reference).
/// This implementation handles negative shifts by computing a global offset:
/// - Find minimum shift: `min_shift = minᵢ kᵢ`
/// - Apply offset: `effective_shift[i] = kᵢ - min_shift ≥ 0`
///
/// This is **safe and mathematically correct**, but it changes the effective time origin.
/// To minimize negative shifts, prefer a reference that produces mostly positive delays,
/// such as `EarliestArrival`.
///
/// # Performance
///
/// - **Time Complexity**: O(n_elements × n_samples)
/// - **Space Complexity**: O(n_samples) for output array
/// - **Parallelization**: Currently single-threaded; GPU version available with `gpu` feature
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::time_domain::{
///     delay_and_sum, DelayReference
/// };
/// use ndarray::Array3;
///
/// // Two sensors, 8 time samples, 10 Hz sampling rate
/// let mut sensor_data = Array3::<f64>::zeros((2, 1, 8));
/// sensor_data[[0, 0, 3]] = 1.0; // Impulse at t=3 for sensor 0
/// sensor_data[[1, 0, 5]] = 1.0; // Impulse at t=5 for sensor 1
///
/// let fs = 10.0; // Hz
/// let delays = vec![1.0, 1.2]; // seconds (sensor 1 is 0.2s = 2 samples later)
/// let weights = vec![1.0, 1.0]; // Equal weights
///
/// let output = delay_and_sum(
///     &sensor_data,
///     fs,
///     &delays,
///     &weights,
///     DelayReference::SensorIndex(0),
/// )?;
///
/// // After alignment, both impulses should sum at output t=3
/// assert!((output[[0, 0, 3]] - 2.0).abs() < 1e-12);
/// ```
///
/// # See Also
///
/// - [`DelayReference`]: Policy for selecting delay reference
/// - [`relative_delays_s`]: Utility for computing relative delays
pub fn delay_and_sum(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array3<f64>> {
    let (n_elements, channels, n_samples) = sensor_data.dim();

    // Validate data shape
    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das expects sensor_data shape (n_elements, 1, n_samples); got channels={channels}. \
             RF data should have a single channel dimension."
        )));
    }
    if n_elements == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "time_domain::das requires n_elements > 0 and n_samples > 0".to_string(),
        ));
    }

    // Validate sampling frequency
    if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::das requires sampling_frequency_hz to be finite and > 0; got {sampling_frequency_hz}"
        )));
    }

    // Validate array lengths
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

    // Convert absolute delays to relative delays using explicit reference policy
    let rel_delays_s = relative_delays_s(delays_s, reference)?;

    // Convert to integer sample shifts
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

    // Compute global offset to handle negative shifts
    // We want an output array of length n_samples. For negative shifts, we compute
    // a global offset so that the smallest shift maps to 0.
    let min_shift = shifts.iter().copied().min().unwrap_or(0);
    let offset = -min_shift; // makes (shift + offset) >= 0 for all

    // Initialize output
    let mut output = Array3::<f64>::zeros((1, 1, n_samples));

    // Apply delay-and-sum with integer sample shifts
    for elem_idx in 0..n_elements {
        let effective_shift = shifts[elem_idx] + offset;

        // Convert to usize (defensive: should be non-negative by construction)
        let eff = if effective_shift < 0 {
            // Should be impossible by construction, but keep it defensive
            0usize
        } else {
            effective_shift as usize
        };

        // Skip if shift moves all samples out of bounds
        if eff >= n_samples {
            continue;
        }

        let w = weights[elem_idx];

        // Apply weighted, shifted samples
        // Read from sensor_data[elem_idx, 0, t] and write to output[0, 0, t - eff]
        for t in eff..n_samples {
            output[[0, 0, t - eff]] += sensor_data[[elem_idx, 0, t]] * w;
        }
    }

    Ok(output)
}

/// Recommended common default for transient SRP-DAS localization:
/// use a fixed reference sensor index 0.
///
/// This keeps the reference deterministic and matches typical array-processing practice.
///
/// # Rationale
///
/// - **Deterministic**: Same reference for all focal points
/// - **Reproducible**: Results do not depend on data-dependent min/max
/// - **Standard**: Matches literature conventions (Van Trees, Brandstein)
/// - **SRP-DAS Correctness**: Ensures point-dependent scoring
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::time_domain::{
///     delay_and_sum, DEFAULT_DELAY_REFERENCE
/// };
///
/// let output = delay_and_sum(
///     &sensor_data,
///     fs,
///     &delays,
///     &weights,
///     DEFAULT_DELAY_REFERENCE,
/// )?;
/// ```
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

        let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect("das should succeed");

        // After alignment, both should land on the same output sample (t=3).
        assert!(
            (y[[0, 0, 3]] - 2.0).abs() < 1e-12,
            "both impulses should sum at t=3, got {}",
            y[[0, 0, 3]]
        );
    }

    #[test]
    fn das_handles_earliest_arrival_reference() {
        let fs = 10.0;
        let n = 8usize;

        // delays: sensor 2 is earliest (0.9s), sensor 0 is 1.0s, sensor 1 is 1.1s
        let delays = vec![1.0, 1.1, 0.9];
        let weights = vec![1.0, 1.0, 1.0];

        let mut x = Array3::<f64>::zeros((3, 1, n));

        // Place impulses according to absolute delays
        x[[0, 0, 3]] = 1.0; // sensor 0 at t=3
        x[[1, 0, 4]] = 1.0; // sensor 1 at t=4 (1 sample later)
        x[[2, 0, 2]] = 1.0; // sensor 2 at t=2 (1 sample earlier)

        let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::EarliestArrival)
            .expect("das should succeed");

        // With earliest arrival reference, all should align to sensor 2's arrival
        // Sensor 2 has zero relative delay, sensors 0 and 1 are delayed
        // Expected alignment at t=2
        assert!(
            (y[[0, 0, 2]] - 3.0).abs() < 1e-12,
            "all impulses should sum at t=2 with EarliestArrival, got {}",
            y[[0, 0, 2]]
        );
    }

    #[test]
    fn das_handles_negative_relative_delays() {
        let fs = 10.0;
        let n = 10usize;

        // Reference sensor 0 at 1.0s, sensor 1 arrives earlier at 0.8s
        // Relative delay for sensor 1: 0.8 - 1.0 = -0.2s = -2 samples
        let delays = vec![1.0, 0.8];
        let weights = vec![1.0, 1.0];

        let mut x = Array3::<f64>::zeros((2, 1, n));

        // Sensor 0 impulse at t=5
        x[[0, 0, 5]] = 1.0;
        // Sensor 1 impulse at t=3 (arrives 2 samples earlier)
        x[[1, 0, 3]] = 1.0;

        let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect("das should handle negative shifts");

        // After alignment with offset, both should sum
        // The implementation adds an offset to make all shifts non-negative
        // Output should have both aligned
        let sum: f64 = y.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-12,
            "total energy should be preserved, got {sum}"
        );
    }

    #[test]
    fn das_applies_apodization_weights() {
        let fs = 10.0;
        let n = 8usize;

        let delays = vec![1.0, 1.0]; // Same delay (no relative shift)
        let weights = vec![0.5, 1.5]; // Different weights

        let mut x = Array3::<f64>::zeros((2, 1, n));
        x[[0, 0, 3]] = 2.0; // sensor 0: value 2.0
        x[[1, 0, 3]] = 2.0; // sensor 1: value 2.0

        let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect("das should succeed");

        // Expected: 0.5 * 2.0 + 1.5 * 2.0 = 1.0 + 3.0 = 4.0
        assert!(
            (y[[0, 0, 3]] - 4.0).abs() < 1e-12,
            "weighted sum should be 4.0, got {}",
            y[[0, 0, 3]]
        );
    }

    #[test]
    fn das_rejects_invalid_channel_dimension() {
        let x = Array3::<f64>::zeros((2, 2, 8)); // Invalid: 2 channels
        let fs = 10.0;
        let delays = vec![1.0, 1.0];
        let weights = vec![1.0, 1.0];

        let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject non-1 channel dimension");

        assert!(
            err.to_string().contains("channels"),
            "error should mention channels"
        );
    }

    #[test]
    fn das_rejects_empty_arrays() {
        let x = Array3::<f64>::zeros((0, 1, 8));
        let fs = 10.0;
        let delays: Vec<f64> = vec![];
        let weights: Vec<f64> = vec![];

        let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject empty arrays");

        assert!(err.to_string().contains("n_elements > 0"));
    }

    #[test]
    fn das_rejects_invalid_sampling_frequency() {
        let x = Array3::<f64>::zeros((2, 1, 8));
        let delays = vec![1.0, 1.0];
        let weights = vec![1.0, 1.0];

        let err = delay_and_sum(&x, 0.0, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject fs=0");
        assert!(err.to_string().contains("finite"));

        let err = delay_and_sum(&x, -10.0, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject negative fs");
        assert!(err.to_string().contains("> 0"));

        let err = delay_and_sum(
            &x,
            f64::NAN,
            &delays,
            &weights,
            DelayReference::SensorIndex(0),
        )
        .expect_err("should reject NaN fs");
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn das_rejects_mismatched_array_lengths() {
        let x = Array3::<f64>::zeros((3, 1, 8));
        let fs = 10.0;
        let delays = vec![1.0, 1.0]; // Wrong length
        let weights = vec![1.0, 1.0, 1.0];

        let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject mismatched delays length");
        assert!(err.to_string().contains("delays_s length"));

        let delays = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0]; // Wrong length

        let err = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect_err("should reject mismatched weights length");
        assert!(err.to_string().contains("weights length"));
    }

    #[test]
    fn das_preserves_zero_input() {
        let x = Array3::<f64>::zeros((2, 1, 8));
        let fs = 10.0;
        let delays = vec![1.0, 1.2];
        let weights = vec![1.0, 1.0];

        let y = delay_and_sum(&x, fs, &delays, &weights, DelayReference::SensorIndex(0))
            .expect("das should succeed");

        let sum: f64 = y.iter().sum();
        assert!(
            sum.abs() < 1e-14,
            "zero input should produce zero output, got sum={sum}"
        );
    }

    #[test]
    fn default_delay_reference_is_sensor_zero() {
        assert_eq!(DEFAULT_DELAY_REFERENCE, DelayReference::SensorIndex(0));
    }
}
