//! DelayReference enum and methods.

use crate::core::error::{KwaversError, KwaversResult};

/// Policy for selecting a delay datum / reference when converting absolute TOF delays to relative delays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayReference {
    /// Use the sensor at the given index as the reference (`τᵣₑ𝒻 = τ[index]`).
    SensorIndex(usize),

    /// Use the earliest arrival as the reference (`τᵣₑ𝒻 = minᵢ τᵢ`).
    EarliestArrival,

    /// Use the latest arrival as the reference (`τᵣₑ𝒻 = maxᵢ τᵢ`).
    LatestArrival,
}

impl DelayReference {
    /// Recommended default delay reference for time-domain transient localization.
    #[must_use]
    pub const fn recommended_default() -> Self {
        Self::SensorIndex(0)
    }

    /// Resolve the reference delay `τᵣₑ𝒻` from an absolute delay vector.
    pub fn resolve_reference_delay_s(self, delays_s: &[f64]) -> KwaversResult<f64> {
        if delays_s.is_empty() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: delays_s must be non-empty".to_string(),
            ));
        }

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
            DelayReference::EarliestArrival => *delays_s
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            DelayReference::LatestArrival => *delays_s
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        };

        if !tau_ref.is_finite() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: resolved τᵣₑ𝒻 is non-finite".to_string(),
            ));
        }

        Ok(tau_ref)
    }

    /// Compute relative delays `Δτᵢ = τᵢ - τᵣₑ𝒻` from absolute delays.
    pub fn compute_relative_delays(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau - tau_ref).collect())
    }

    /// Compute alignment shifts `Δτₐₗᵢ𝓰ₙ,ᵢ = τᵣₑ𝒻 - τᵢ` from absolute delays.
    pub fn compute_alignment_shifts(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau_ref - tau).collect())
    }
}
