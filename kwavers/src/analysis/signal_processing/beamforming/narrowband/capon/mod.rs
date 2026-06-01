//! Narrowband Capon/MVDR spatial spectrum (point-steered) for localization.
//!
//! # Field jargon
//! - **MVDR** (Minimum Variance Distortionless Response) is also known as the **Capon beamformer**.
//! - The **Capon spatial spectrum** evaluates candidate look directions/points by
//!
//! `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
//!
//! where:
//! - `R` is the sample covariance matrix of the array snapshots
//! - `a(p)` is the (complex) steering vector for candidate point `p`
//!
//! # Design constraints (SSOT)
//! - This module is the canonical location for Capon/MVDR spatial spectrum computation.
//! - It lives in the analysis layer and uses domain primitives for geometry and covariance.
//!
//! # Numerical notes
//! - Diagonal loading is applied as `R_loaded = R + δ I` with `δ >= 0`.
//!
//! # Invariants / validation
//! - `frequency_hz` must be finite and > 0.
//! - `sound_speed` must be finite and > 0.
//! - `diagonal_loading` must be finite and >= 0.
//! - `sensor_data` must have shape `(n_sensors, 1, n_samples)` with `n_samples > 0`.

use crate::analysis::signal_processing::beamforming::covariance::CovarianceEstimator;
use crate::analysis::signal_processing::beamforming::narrowband::snapshots::SnapshotSelection;
use crate::analysis::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::{DEFAULT_DIAGONAL_LOADING, MHZ_TO_HZ};
use crate::core::error::{KwaversError, KwaversResult};

mod spectrum;
mod spectrum_complex;
#[cfg(test)]
mod tests;

pub use spectrum::capon_spatial_spectrum_point;
pub use spectrum_complex::capon_spatial_spectrum_point_complex_baseband;

/// Configuration for the narrowband Capon/MVDR spatial spectrum.
#[derive(Debug, Clone)]
pub struct CaponSpectrumConfig {
    /// Narrowband frequency (Hz) at which the steering vector is evaluated.
    pub frequency_hz: f64,
    /// Speed of sound (m/s).
    pub sound_speed: f64,
    /// Diagonal loading factor (δ >= 0).
    pub diagonal_loading: f64,
    /// Covariance estimation policy.
    pub covariance: CovarianceEstimator,
    /// Steering vector model.
    pub steering: SteeringVectorMethod,
    /// Sampling frequency (Hz) of `sensor_data`.
    ///
    /// Required for complex snapshot extraction.
    /// If `None`, complex-narrowband scorers will reject with an explicit error.
    pub sampling_frequency_hz: Option<f64>,
    /// Snapshot formation policy for the canonical complex narrowband path.
    ///
    /// - `Some(SnapshotSelection::Explicit(_))`: uses exactly that method.
    /// - `Some(SnapshotSelection::Auto(_))`: deterministically selects a literature-aligned method.
    /// - `None`: auto-derives a conservative scenario from `frequency_hz` and `sampling_frequency_hz`.
    pub snapshot_selection: Option<SnapshotSelection>,
    /// Legacy snapshot stride (samples) for analytic-signal complex baseband snapshots.
    ///
    /// Retained for compatibility; prefer `snapshot_selection` for advanced usage.
    pub baseband_snapshot_step_samples: Option<usize>,
}

impl CaponSpectrumConfig {
    /// Validate invariants.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: frequency_hz must be finite and > 0".to_owned(),
            ));
        }
        if !self.sound_speed.is_finite() || self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: sound_speed must be finite and > 0".to_owned(),
            ));
        }
        if !self.diagonal_loading.is_finite() || self.diagonal_loading < 0.0 {
            return Err(KwaversError::InvalidInput(
                "CaponSpectrumConfig: diagonal_loading must be finite and >= 0".to_owned(),
            ));
        }
        if let Some(fs) = self.sampling_frequency_hz {
            if !fs.is_finite() || fs <= 0.0 {
                return Err(KwaversError::InvalidInput(
                    "CaponSpectrumConfig: sampling_frequency_hz must be finite and > 0 (when provided)".to_owned(),
                ));
            }
        }
        if let Some(step) = self.baseband_snapshot_step_samples {
            if step == 0 {
                return Err(KwaversError::InvalidInput(
                    "CaponSpectrumConfig: baseband_snapshot_step_samples must be >= 1 (when provided)".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for CaponSpectrumConfig {
    fn default() -> Self {
        Self {
            frequency_hz: MHZ_TO_HZ,
            sound_speed: SOUND_SPEED_WATER_SIM,
            diagonal_loading: DEFAULT_DIAGONAL_LOADING,
            covariance: CovarianceEstimator::default(),
            steering: SteeringVectorMethod::PlaneWave,
            sampling_frequency_hz: None,
            snapshot_selection: None,
            baseband_snapshot_step_samples: None,
        }
    }
}
