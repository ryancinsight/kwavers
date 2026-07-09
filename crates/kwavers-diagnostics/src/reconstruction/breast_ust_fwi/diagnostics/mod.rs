//! Ali 2025 breast-FWI diagnostic metrics.
//!
//! The clinical layer owns replication diagnostics that bind acquisition
//! topology, PSTD frequency-bin metadata, and published Table-1 parity gates.
//! Python callers receive these values through PyO3 and remain responsible only
//! for orchestration, report serialization, and plotting.

mod excitation;
mod identifiability;
mod operator;
mod reconstruction;
mod residual;
mod scattering;

#[cfg(test)]
mod tests;

pub use excitation::{
    sine_frequency_bin_coefficient, source_excitation_diagnostics,
    BreastUstSourceExcitationDiagnostics, BreastUstSourceExcitationFrequencyDiagnostics,
};
pub use identifiability::{
    acquisition_identifiability, BreastUstAcquisitionIdentifiability, BreastUstSourceScalingPolicy,
};
pub use operator::{
    forward_operator_equivalence_diagnostics,
    forward_operator_equivalence_diagnostics_with_receiver_policy,
    BreastUstForwardOperatorEquivalenceDiagnostics, BreastUstForwardOperatorModelDiagnostics,
    BreastUstForwardOperatorPrediction,
};
pub use reconstruction::{
    reconstruction_metrics, table1_parity, BreastUstReconstructionMetrics, BreastUstTable1Parity,
};
pub use residual::{
    passive_receiver_mask, scaled_observation_residual_metrics,
    source_channel_residual_diagnostics, source_receiver_mask, BreastUstReceiverChannelPolicy,
    BreastUstScaledObservationResidualMetrics, BreastUstSourceChannelResidualDiagnostics,
};
pub(crate) use residual::{row_scale, scaled_observation_residual_metrics_by_receiver};
pub use scattering::{
    scattering_increment_diagnostics, BreastUstScatteringIncrementDiagnostics,
    BreastUstScatteringIncrementModelDiagnostics,
};

use super::BreastUstPstdDatasetConfig;
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use leto::Array3;
use kwavers_math::fft::Complex64;

/// Combined diagnostic report for one predicted/observed observation cube pair.
#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstObservationPairDiagnostics {
    pub forward_consistency: BreastUstScaledObservationResidualMetrics,
    pub source_channel_consistency: BreastUstSourceChannelResidualDiagnostics,
    pub source_excitation: BreastUstSourceExcitationDiagnostics,
}

/// Compute all Ali 2025 observation-pair diagnostics from Rust-owned formulas.
///
/// # Errors
/// Returns an error when observation cubes, array topology, or PSTD
/// frequency-bin metadata violate the diagnostic contracts.
pub fn diagnose_breast_ust_observation_pair(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    array: &MultiRowRingArray,
    frequencies_hz: &[f64],
    config: BreastUstPstdDatasetConfig,
    time_steps_per_frequency: &[usize],
    frequency_bin_start_steps_per_frequency: &[usize],
) -> KwaversResult<BreastUstObservationPairDiagnostics> {
    Ok(BreastUstObservationPairDiagnostics {
        forward_consistency: scaled_observation_residual_metrics(predicted, observed, None)?,
        source_channel_consistency: source_channel_residual_diagnostics(
            predicted,
            observed,
            array.circumferential_elements(),
            array.rows(),
        )?,
        source_excitation: source_excitation_diagnostics(
            predicted,
            observed,
            frequencies_hz,
            config.source_amplitude_pa,
            config.time_step_s,
            time_steps_per_frequency,
            frequency_bin_start_steps_per_frequency,
        )?,
    })
}
