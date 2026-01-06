//! Time-domain delay reference policy and utilities.
//!
//! # Field jargon
//! In array signal processing, geometric **propagation delays** are the **time-of-flight (TOF)**
//! values
//!
//! `τ_i(p) = ||x_i - p|| / c`
//!
//! for sensor `i` and candidate point `p`.
//!
//! Time-domain beamforming (e.g., **delay-and-sum**, DAS) generally requires **relative delays**
//! with respect to an explicit **delay datum / reference**:
//!
//! `Δτ_i(p) = τ_i(p) - τ_ref(p)`
//!
//! The choice of `τ_ref` is a *policy* decision and must be explicit to avoid silent divergence,
//! especially for **SRP-DAS** (steered response power) localization.
//!
//! # Default policy
//! The most common, deterministic reference for time-domain transient localization is a fixed
//! reference sensor (often element 0). This module therefore provides `DelayReference::SensorIndex(0)`
//! as the recommended default.
//!
//! # Invariants
//! - All delays must be finite and non-negative (TOF).
//! - Reference selection must yield a valid index (for `SensorIndex`) and a finite delay.
//! - Relative delays are computed as `Δτ_i = τ_i - τ_ref` and may be negative.
//!
//! # Notes
//! This module deliberately does **not** perform any sample-rate discretization. Converting
//! seconds to samples (and choosing interpolation vs integer shifts) is a separate concern.

use crate::error::{KwaversError, KwaversResult};

/// Policy for selecting a delay datum / reference when converting absolute TOF delays to relative delays.
///
/// This is used by time-domain steering (DAS / SRP-DAS) to define the alignment convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayReference {
    /// Use the sensor at the given index as the reference (`τ_ref = τ[index]`).
    ///
    /// This is the recommended default for deterministic transient localization.
    SensorIndex(usize),

    /// Use the earliest arrival as the reference (`τ_ref = min_i τ_i`).
    ///
    /// Common when interpreting a source emission time relative to first arrival.
    EarliestArrival,

    /// Use the latest arrival as the reference (`τ_ref = max_i τ_i`).
    ///
    /// This matches the current legacy convention in some DAS implementations where the latest
    /// channel aligns to `t = 0`.
    LatestArrival,
}

impl DelayReference {
    /// Recommended default delay reference for time-domain transient localization.
    ///
    /// This keeps scoring deterministic and avoids dependence on max/min conventions.
    #[must_use]
    pub const fn recommended_default() -> Self {
        Self::SensorIndex(0)
    }

    /// Resolve the reference delay `τ_ref` from an absolute delay vector.
    ///
    /// # Errors
    /// - If `delays_s` is empty.
    /// - If any selected delay is non-finite or negative.
    /// - If `SensorIndex(i)` is out of bounds.
    pub fn resolve_reference_delay_s(self, delays_s: &[f64]) -> KwaversResult<f64> {
        if delays_s.is_empty() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: delays_s must be non-empty".to_string(),
            ));
        }

        // Validate finiteness and non-negativity up front (TOF invariant).
        if delays_s.iter().any(|d| !d.is_finite() || *d < 0.0) {
            return Err(KwaversError::InvalidInput(
                "DelayReference: delays_s must be finite and non-negative (TOF)".to_string(),
            ));
        }

        let tau_ref = match self {
            DelayReference::SensorIndex(idx) => delays_s.get(idx).copied().ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "DelayReference::SensorIndex({idx}) out of bounds for delays_s (len={})",
                    delays_s.len()
                ))
            })?,
            DelayReference::EarliestArrival => {
                delays_s.iter().copied().fold(f64::INFINITY, f64::min)
            }
            DelayReference::LatestArrival => {
                delays_s.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            }
        };

        if !tau_ref.is_finite() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: resolved τ_ref is non-finite".to_string(),
            ));
        }
        Ok(tau_ref)
    }
}

/// Convert absolute TOF delays `τ_i` to relative delays `Δτ_i = τ_i - τ_ref` using the given reference policy.
///
/// # Errors
/// - If `delays_s` violates TOF invariants (finite, non-negative) or reference cannot be resolved.
///
/// # Output
/// Relative delays may be negative if a sensor arrives earlier than the reference.
pub fn relative_delays_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    let tau_ref = reference.resolve_reference_delay_s(delays_s)?;
    Ok(delays_s.iter().map(|&tau| tau - tau_ref).collect())
}

/// Convert absolute TOF delays `τ_i` to alignment shifts `Δτ_align,i = τ_ref - τ_i`.
///
/// This is the shift you apply to *advance* earlier channels so arrivals align to the reference.
/// Note that `Δτ_align,i = - (τ_i - τ_ref)`.
///
/// # Errors
/// Same as [`relative_delays_s`].
pub fn alignment_shifts_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    let tau_ref = reference.resolve_reference_delay_s(delays_s)?;
    Ok(delays_s.iter().map(|&tau| tau_ref - tau).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_index_reference_is_deterministic() {
        let delays = vec![0.010, 0.011, 0.009];
        let tau_ref = DelayReference::SensorIndex(0)
            .resolve_reference_delay_s(&delays)
            .expect("ref");
        assert!((tau_ref - 0.010).abs() < 1e-15);
    }

    #[test]
    fn earliest_and_latest_reference() {
        let delays = vec![0.010, 0.011, 0.009];
        let tau_min = DelayReference::EarliestArrival
            .resolve_reference_delay_s(&delays)
            .expect("min");
        let tau_max = DelayReference::LatestArrival
            .resolve_reference_delay_s(&delays)
            .expect("max");
        assert!((tau_min - 0.009).abs() < 1e-15);
        assert!((tau_max - 0.011).abs() < 1e-15);
    }

    #[test]
    fn relative_delays_match_definition() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("rel");
        assert_eq!(rel.len(), 3);
        assert!((rel[0] - 0.0).abs() < 1e-15);
        assert!((rel[1] - 0.001).abs() < 1e-15);
        assert!((rel[2] - (-0.001)).abs() < 1e-15);
    }

    #[test]
    fn alignment_shifts_are_negative_relative_delays() {
        let delays = vec![0.010, 0.011, 0.009];
        let rel = relative_delays_s(&delays, DelayReference::SensorIndex(0)).expect("rel");
        let shifts = alignment_shifts_s(&delays, DelayReference::SensorIndex(0)).expect("shifts");
        for i in 0..delays.len() {
            assert!((shifts[i] + rel[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn rejects_negative_or_non_finite_delays() {
        let delays = vec![0.01, -0.01];
        let err = DelayReference::EarliestArrival
            .resolve_reference_delay_s(&delays)
            .expect_err("should error");
        let msg = err.to_string();
        assert!(
            msg.contains("non-negative") || msg.contains("finite"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn rejects_out_of_bounds_sensor_index() {
        let delays = vec![0.01, 0.02];
        let err = DelayReference::SensorIndex(5)
            .resolve_reference_delay_s(&delays)
            .expect_err("should error");
        assert!(err.to_string().contains("out of bounds"));
    }
}
