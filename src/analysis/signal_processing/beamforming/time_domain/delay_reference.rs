//! Time-domain delay reference policy and utilities.
//!
//! # Mathematical Foundation
//!
//! In array signal processing, geometric **propagation delays** are the **time-of-flight (TOF)**
//! values:
//!
//! ```text
//! τᵢ(p) = ||xᵢ - p|| / c
//! ```
//!
//! for sensor `i` at position `xᵢ`, candidate point `p`, and sound speed `c`.
//!
//! Time-domain beamforming (e.g., **delay-and-sum**, DAS) generally requires **relative delays**
//! with respect to an explicit **delay datum / reference**:
//!
//! ```text
//! Δτᵢ(p) = τᵢ(p) - τᵣₑ𝒻(p)
//! ```
//!
//! The choice of `τᵣₑ𝒻` is a *policy* decision and must be explicit to avoid silent divergence,
//! especially for **SRP-DAS** (steered response power) localization.
//!
//! # Architectural Intent (Single Source of Truth)
//!
//! This module provides the **canonical delay reference policy** for all time-domain beamforming
//! in kwavers. It is placed in the **analysis layer** because:
//!
//! 1. **Delay reference is an analysis decision**, not a domain primitive
//! 2. **Physics layer** provides sound speed `c`
//! 3. **Domain layer** provides sensor positions `xᵢ`
//! 4. **Analysis layer** combines these to compute delays and apply reference policy
//!
//! # Default Policy
//!
//! The most common, deterministic reference for time-domain transient localization is a **fixed
//! reference sensor** (often element 0). This module provides `DelayReference::SensorIndex(0)`
//! as the recommended default, ensuring reproducible results across different datasets and
//! array configurations.
//!
//! ## Alternative Policies
//!
//! - **EarliestArrival**: `τᵣₑ𝒻 = minᵢ τᵢ(p)` — aligns to first wavefront arrival
//! - **LatestArrival**: `τᵣₑ𝒻 = maxᵢ τᵢ(p)` — aligns to last wavefront arrival
//! - **SensorIndex(k)**: `τᵣₑ𝒻 = τₖ(p)` — aligns to sensor k (deterministic)
//!
//! # Invariants
//!
//! 1. All absolute delays must be **finite and non-negative** (physical TOF constraint)
//! 2. Reference selection must yield a **valid, finite delay**
//! 3. Relative delays `Δτᵢ = τᵢ - τᵣₑ𝒻` **may be negative** (sensor closer than reference)
//!
//! # Discretization Note
//!
//! This module deliberately does **not** perform sample-rate discretization. Converting
//! seconds to samples (and choosing interpolation vs integer shifts) is a separate concern
//! handled by the beamforming algorithms themselves.
//!
//! # Literature References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley. (Chapter 2: Temporal and Spatial Sampling)
//! - Brandstein, M., & Ward, D. (2001). *Microphone Arrays*. Springer. (Chapter 4: Time Delay Estimation)
//!
//! # Migration Note
//!
//! This module was migrated from `domain::sensor::beamforming::time_domain::delay_reference`
//! to `analysis::signal_processing::beamforming::time_domain::delay_reference` as part of
//! the architectural purification effort (ADR 003). The API remains unchanged.

use crate::core::error::{KwaversError, KwaversResult};

/// Policy for selecting a delay datum / reference when converting absolute TOF delays to relative delays.
///
/// This is used by time-domain steering (DAS / SRP-DAS) to define the alignment convention.
///
/// # Design Rationale
///
/// Making the delay reference **explicit** prevents subtle bugs:
/// - Different papers use different conventions (earliest, latest, center element)
/// - Implicit conventions can make SRP-DAS scoring point-invariant (incorrect)
/// - Explicit policy ensures reproducibility and correctness
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::time_domain::DelayReference;
///
/// // Recommended: deterministic reference sensor
/// let ref_policy = DelayReference::SensorIndex(0);
///
/// // Alternative: earliest arrival (data-dependent)
/// let ref_policy = DelayReference::EarliestArrival;
///
/// // Apply to absolute delays
/// let absolute_delays = vec![0.010, 0.011, 0.009]; // seconds
/// let relative_delays = ref_policy.compute_relative_delays(&absolute_delays)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayReference {
    /// Use the sensor at the given index as the reference (`τᵣₑ𝒻 = τ[index]`).
    ///
    /// This is the **recommended default** for deterministic transient localization.
    /// Using a fixed sensor (typically element 0) ensures:
    /// - Reproducible results across runs
    /// - Point-dependent SRP-DAS scoring
    /// - Alignment with standard array processing literature
    SensorIndex(usize),

    /// Use the earliest arrival as the reference (`τᵣₑ𝒻 = minᵢ τᵢ`).
    ///
    /// Common when interpreting a source emission time relative to first arrival.
    /// Note: This is **data-dependent** and changes for each focal point.
    EarliestArrival,

    /// Use the latest arrival as the reference (`τᵣₑ𝒻 = maxᵢ τᵢ`).
    ///
    /// This matches some legacy DAS implementations where the latest
    /// channel aligns to `t = 0` (zero-delay reference).
    /// Note: This is **data-dependent** and changes for each focal point.
    LatestArrival,
}

impl DelayReference {
    /// Recommended default delay reference for time-domain transient localization.
    ///
    /// Returns `DelayReference::SensorIndex(0)`, which:
    /// - Is deterministic (not data-dependent)
    /// - Matches typical array processing conventions
    /// - Ensures reproducible SRP-DAS scoring
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let default_ref = DelayReference::recommended_default();
    /// assert_eq!(default_ref, DelayReference::SensorIndex(0));
    /// ```
    #[must_use]
    pub const fn recommended_default() -> Self {
        Self::SensorIndex(0)
    }

    /// Resolve the reference delay `τᵣₑ𝒻` from an absolute delay vector.
    ///
    /// # Parameters
    ///
    /// - `delays_s`: Absolute time-of-flight delays in seconds (one per sensor)
    ///
    /// # Returns
    ///
    /// The reference delay `τᵣₑ𝒻` in seconds.
    ///
    /// # Errors
    ///
    /// - If `delays_s` is empty
    /// - If any delay is non-finite or negative (violates TOF invariant)
    /// - If `SensorIndex(i)` is out of bounds
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// SensorIndex(k):    τᵣₑ𝒻 = τₖ
    /// EarliestArrival:   τᵣₑ𝒻 = minᵢ τᵢ
    /// LatestArrival:     τᵣₑ𝒻 = maxᵢ τᵢ
    /// ```
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let delays = vec![0.010, 0.011, 0.009]; // seconds
    /// let ref_delay = DelayReference::SensorIndex(0)
    ///     .resolve_reference_delay_s(&delays)?;
    /// assert_eq!(ref_delay, 0.010);
    /// ```
    pub fn resolve_reference_delay_s(self, delays_s: &[f64]) -> KwaversResult<f64> {
        if delays_s.is_empty() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: delays_s must be non-empty".to_string(),
            ));
        }

        // Validate TOF invariant: all delays must be finite and non-negative
        for (i, &tau) in delays_s.iter().enumerate() {
            if !tau.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "DelayReference: delay[{i}] = {tau} is non-finite (TOF must be finite)"
                )));
            }
            if tau < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "DelayReference: delay[{i}] = {tau} is negative (TOF must be non-negative)"
                )));
            }
        }

        let tau_ref = match self {
            DelayReference::SensorIndex(idx) => delays_s.get(idx).copied().ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "DelayReference::SensorIndex({idx}) out of bounds for delays_s (len={})",
                    delays_s.len()
                ))
            })?,
            DelayReference::EarliestArrival => {
                // Safe: delays_s is non-empty and all finite
                *delays_s
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            }
            DelayReference::LatestArrival => {
                // Safe: delays_s is non-empty and all finite
                *delays_s
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            }
        };

        // Defensive: verify resolved reference is finite
        if !tau_ref.is_finite() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: resolved τᵣₑ𝒻 is non-finite".to_string(),
            ));
        }

        Ok(tau_ref)
    }

    /// Compute relative delays `Δτᵢ = τᵢ - τᵣₑ𝒻` from absolute delays.
    ///
    /// # Parameters
    ///
    /// - `delays_s`: Absolute time-of-flight delays in seconds
    ///
    /// # Returns
    ///
    /// Vector of relative delays in seconds. Note: **may contain negative values**
    /// if a sensor is closer to the source than the reference sensor.
    ///
    /// # Errors
    ///
    /// Same as [`resolve_reference_delay_s`](Self::resolve_reference_delay_s).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let delays = vec![0.010, 0.011, 0.009]; // seconds
    /// let relative = DelayReference::SensorIndex(0)
    ///     .compute_relative_delays(&delays)?;
    /// // relative = [0.0, 0.001, -0.001]
    /// ```
    pub fn compute_relative_delays(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau - tau_ref).collect())
    }

    /// Compute alignment shifts `Δτₐₗᵢ𝓰ₙ,ᵢ = τᵣₑ𝒻 - τᵢ` from absolute delays.
    ///
    /// This is the shift you apply to *advance* earlier channels so arrivals align to the reference.
    /// Note that `Δτₐₗᵢ𝓰ₙ,ᵢ = -(τᵢ - τᵣₑ𝒻)`.
    ///
    /// # Parameters
    ///
    /// - `delays_s`: Absolute time-of-flight delays in seconds
    ///
    /// # Returns
    ///
    /// Vector of alignment shifts in seconds (opposite sign of relative delays).
    ///
    /// # Errors
    ///
    /// Same as [`resolve_reference_delay_s`](Self::resolve_reference_delay_s).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let delays = vec![0.010, 0.011, 0.009]; // seconds
    /// let shifts = DelayReference::SensorIndex(0)
    ///     .compute_alignment_shifts(&delays)?;
    /// // shifts = [0.0, -0.001, 0.001]
    /// // Apply shifts: sensor[1] delays by 0.001s, sensor[2] advances by 0.001s
    /// ```
    pub fn compute_alignment_shifts(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau_ref - tau).collect())
    }
}

/// Convert absolute TOF delays `τᵢ` to relative delays `Δτᵢ = τᵢ - τᵣₑ𝒻` using the given reference policy.
///
/// This is a convenience function equivalent to `reference.compute_relative_delays(delays_s)`.
///
/// # Parameters
///
/// - `delays_s`: Absolute propagation delays in seconds (one per sensor)
/// - `reference`: Delay reference policy
///
/// # Returns
///
/// Relative delays in seconds. May be negative if a sensor arrives earlier than the reference.
///
/// # Errors
///
/// - If `delays_s` violates TOF invariants (finite, non-negative)
/// - If reference cannot be resolved (e.g., index out of bounds)
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::time_domain::{
///     relative_delays_s, DelayReference
/// };
///
/// let delays = vec![0.010, 0.011, 0.009];
/// let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0))?;
/// assert_eq!(rel.len(), 3);
/// assert!((rel[0] - 0.0).abs() < 1e-15);
/// assert!((rel[1] - 0.001).abs() < 1e-15);
/// assert!((rel[2] - (-0.001)).abs() < 1e-15);
/// ```
pub fn relative_delays_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    reference.compute_relative_delays(delays_s)
}

/// Convert absolute TOF delays `τᵢ` to alignment shifts `Δτₐₗᵢ𝓰ₙ,ᵢ = τᵣₑ𝒻 - τᵢ`.
///
/// This is a convenience function equivalent to `reference.compute_alignment_shifts(delays_s)`.
///
/// # Parameters
///
/// - `delays_s`: Absolute propagation delays in seconds (one per sensor)
/// - `reference`: Delay reference policy
///
/// # Returns
///
/// Alignment shifts in seconds (opposite sign of relative delays).
///
/// # Errors
///
/// Same as [`relative_delays_s`].
///
/// # Mathematical Relationship
///
/// ```text
/// alignment_shifts[i] = -relative_delays[i]
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::time_domain::{
///     alignment_shifts_s, relative_delays_s, DelayReference
/// };
///
/// let delays = vec![0.010, 0.011, 0.009];
/// let ref_policy = DelayReference::SensorIndex(0);
///
/// let rel = relative_delays_s(&delays, ref_policy)?;
/// let shifts = alignment_shifts_s(&delays, ref_policy)?;
///
/// for i in 0..delays.len() {
///     assert!((shifts[i] + rel[i]).abs() < 1e-15);
/// }
/// ```
pub fn alignment_shifts_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    reference.compute_alignment_shifts(delays_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_index_reference_is_deterministic() {
        let delays = vec![0.010, 0.011, 0.009];
        let tau_ref = DelayReference::SensorIndex(0)
            .resolve_reference_delay_s(&delays)
            .expect("should resolve reference");
        assert!((tau_ref - 0.010).abs() < 1e-15);
    }

    #[test]
    fn earliest_and_latest_reference() {
        let delays = vec![0.010, 0.011, 0.009];
        let tau_min = DelayReference::EarliestArrival
            .resolve_reference_delay_s(&delays)
            .expect("should resolve min");
        let tau_max = DelayReference::LatestArrival
            .resolve_reference_delay_s(&delays)
            .expect("should resolve max");
        assert!((tau_min - 0.009).abs() < 1e-15);
        assert!((tau_max - 0.011).abs() < 1e-15);
    }

    #[test]
    fn relative_delays_match_definition() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0))
            .expect("should compute relative delays");
        assert_eq!(rel.len(), 3);
        assert!(
            (rel[0] - 0.0).abs() < 1e-15,
            "sensor 0 should have zero delay"
        );
        assert!(
            (rel[1] - 0.001).abs() < 1e-15,
            "sensor 1 should be 1ms later"
        );
        assert!(
            (rel[2] - (-0.001)).abs() < 1e-15,
            "sensor 2 should be 1ms earlier"
        );
    }

    #[test]
    fn alignment_shifts_are_negative_relative_delays() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel =
            relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("compute relative");
        let shifts =
            alignment_shifts_s(&delays, DelayReference::SensorIndex(0)).expect("compute alignment");
        for i in 0..delays.len() {
            assert!(
                (shifts[i] + rel[i]).abs() < 1e-15,
                "shift[{i}] should be -relative[{i}]"
            );
        }
    }

    #[test]
    fn rejects_negative_delays() {
        let delays = vec![0.01, -0.01]; // Negative delay violates TOF
        let err = DelayReference::EarliestArrival
            .resolve_reference_delay_s(&delays)
            .expect_err("should reject negative delay");
        let msg = err.to_string();
        assert!(
            msg.contains("negative") || msg.contains("non-negative"),
            "error message should mention negative delays: {msg}"
        );
    }

    #[test]
    fn rejects_non_finite_delays() {
        let delays = vec![0.01, f64::NAN];
        let err = DelayReference::EarliestArrival
            .resolve_reference_delay_s(&delays)
            .expect_err("should reject NaN");
        let msg = err.to_string();
        assert!(
            msg.contains("finite") || msg.contains("non-finite"),
            "error message should mention finite: {msg}"
        );

        let delays = vec![0.01, f64::INFINITY];
        let err = DelayReference::LatestArrival
            .resolve_reference_delay_s(&delays)
            .expect_err("should reject infinity");
        let msg = err.to_string();
        assert!(
            msg.contains("finite") || msg.contains("non-finite"),
            "error message should mention finite: {msg}"
        );
    }

    #[test]
    fn rejects_out_of_bounds_sensor_index() {
        let delays = vec![0.01, 0.02];
        let err = DelayReference::SensorIndex(5)
            .resolve_reference_delay_s(&delays)
            .expect_err("should reject out-of-bounds index");
        assert!(
            err.to_string().contains("out of bounds"),
            "error should mention out of bounds"
        );
    }

    #[test]
    fn rejects_empty_delays() {
        let delays: Vec<f64> = vec![];
        let err = DelayReference::SensorIndex(0)
            .resolve_reference_delay_s(&delays)
            .expect_err("should reject empty delays");
        assert!(
            err.to_string().contains("empty"),
            "error should mention empty"
        );
    }

    #[test]
    fn recommended_default_is_sensor_zero() {
        assert_eq!(
            DelayReference::recommended_default(),
            DelayReference::SensorIndex(0)
        );
    }

    #[test]
    fn relative_delays_can_be_negative() {
        // Sensor 2 arrives first (closest to source)
        let delays = vec![0.010, 0.011, 0.008];
        let rel =
            relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("compute relative");
        assert!(rel[2] < 0.0, "sensor 2 should have negative relative delay");
        assert!((rel[2] - (-0.002)).abs() < 1e-15);
    }

    #[test]
    fn earliest_arrival_makes_all_relative_delays_non_negative() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel =
            relative_delays_s(&delays, DelayReference::EarliestArrival).expect("compute relative");
        for (i, &r) in rel.iter().enumerate() {
            assert!(
                r >= -1e-14,
                "relative delay[{i}] = {r} should be non-negative with EarliestArrival"
            );
        }
    }

    #[test]
    fn latest_arrival_makes_all_relative_delays_non_positive() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel =
            relative_delays_s(&delays, DelayReference::LatestArrival).expect("compute relative");
        for (i, &r) in rel.iter().enumerate() {
            assert!(
                r <= 1e-14,
                "relative delay[{i}] = {r} should be non-positive with LatestArrival"
            );
        }
    }
}
