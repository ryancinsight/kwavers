//! Convenience free functions for delay reference computation.

use crate::core::error::KwaversResult;

use super::policy::DelayReference;

/// Convert absolute TOF delays to relative delays using the given reference policy.
pub fn relative_delays_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    reference.compute_relative_delays(delays_s)
}

/// Convert absolute TOF delays to alignment shifts `Δτₐₗᵢ𝓰ₙ,ᵢ = τᵣₑ𝒻 - τᵢ`.
pub fn alignment_shifts_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    reference.compute_alignment_shifts(delays_s)
}
