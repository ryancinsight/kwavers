//! Types and configuration for beamforming-based localization.

use crate::analysis::signal_processing::beamforming::time_domain::DelayReference;
use crate::analysis::signal_processing::beamforming::utils::steering::SteeringVectorMethod;
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingCoreConfig;
use ndarray::Array3;

/// Covariance / snapshot domain policy for narrowband MVDR/Capon scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvdrCovarianceDomain {
    /// Estimate a real covariance from real-valued time samples.
    RealTimeSamples,
    /// Complex baseband snapshots with Hermitian covariance.
    ///
    /// - `snapshot_step_samples` controls snapshot stride (>= 1).
    ComplexBaseband {
        /// Snapshot stride in samples (>= 1).
        snapshot_step_samples: usize,
    },
}

impl Default for MvdrCovarianceDomain {
    fn default() -> Self {
        Self::ComplexBaseband {
            snapshot_step_samples: 1,
        }
    }
}

/// Beamforming scorer selection for localization grid search.
#[derive(Debug, Clone, PartialEq)]
pub enum LocalizationBeamformingMethod {
    /// **SRP-DAS** (Steered Response Power with time-domain Delay-and-Sum).
    SrpDasTimeDomain {
        /// Delay datum / delay reference policy.
        delay_reference: DelayReference,
    },

    /// **Capon/MVDR spatial spectrum** (narrowband, adaptive).
    CaponMvdrSpectrum {
        /// Narrowband frequency (Hz) at which the steering vector is evaluated.
        frequency_hz: f64,
        /// Diagonal loading (δ ≥ 0) added to the covariance matrix for robustness.
        diagonal_loading: f64,
        /// Steering model used to build `a(p)`.
        steering: SteeringVectorMethod,
        /// Covariance / snapshot domain policy for MVDR/Capon scoring.
        covariance_domain: MvdrCovarianceDomain,
    },
}

/// Search space definition for beamforming-based localization.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchGrid {
    /// Cubic grid centered on the array centroid.
    CenteredCube {
        /// Radius (m) around centroid.
        search_radius_m: f64,
        /// Step size (m).
        grid_resolution_m: f64,
        /// Minimum number of points per axis.
        min_points_per_axis: usize,
    },
    /// Explicit list of candidate points (meters).
    ExplicitPoints {
        /// Candidate points `[x, y, z]` in meters.
        points_m: Vec<[f64; 3]>,
    },
}

/// Configuration for beamforming grid-search localization.
#[derive(Debug, Clone)]
pub struct LocalizationBeamformSearchConfig {
    /// Shared beamforming core configuration (SSOT for physics + numerics).
    pub core: BeamformingCoreConfig,
    /// Scorer used to evaluate each candidate point.
    pub method: LocalizationBeamformingMethod,
    /// Search grid definition.
    pub grid: SearchGrid,
    /// If true, normalize scores by number of sensors.
    pub normalize_by_sensor_count: bool,
}

impl LocalizationBeamformSearchConfig {
    /// Validate configuration invariants.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if !self.core.sound_speed.is_finite() || self.core.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.sound_speed must be finite and > 0"
                    .to_owned(),
            ));
        }
        if !self.core.sampling_frequency.is_finite() || self.core.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.sampling_frequency must be finite and > 0"
                    .to_owned(),
            ));
        }
        if !self.core.reference_frequency.is_finite() || self.core.reference_frequency < 0.0 {
            return Err(KwaversError::InvalidInput(
                "LocalizationBeamformSearchConfig: core.reference_frequency must be finite and >= 0".to_owned(),
            ));
        }

        match &self.method {
            LocalizationBeamformingMethod::SrpDasTimeDomain { delay_reference: _ } => {}
            LocalizationBeamformingMethod::CaponMvdrSpectrum {
                frequency_hz,
                diagonal_loading,
                steering: _,
                covariance_domain,
            } => {
                if !frequency_hz.is_finite() || *frequency_hz <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.frequency_hz must be finite and > 0".to_owned(),
                    ));
                }
                if !diagonal_loading.is_finite() || *diagonal_loading < 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.diagonal_loading must be finite and >= 0".to_owned(),
                    ));
                }
                match covariance_domain {
                    MvdrCovarianceDomain::RealTimeSamples => {}
                    MvdrCovarianceDomain::ComplexBaseband {
                        snapshot_step_samples,
                    } => {
                        if *snapshot_step_samples == 0 {
                            return Err(KwaversError::InvalidInput(
                                "LocalizationBeamformSearchConfig: CaponMvdrSpectrum.covariance_domain.ComplexBaseband.snapshot_step_samples must be >= 1".to_owned(),
                            ));
                        }
                    }
                }
            }
        }

        match &self.grid {
            SearchGrid::CenteredCube {
                search_radius_m,
                grid_resolution_m,
                min_points_per_axis,
            } => {
                if !search_radius_m.is_finite() || *search_radius_m <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: search_radius_m must be finite and > 0"
                            .to_owned(),
                    ));
                }
                if !grid_resolution_m.is_finite() || *grid_resolution_m <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: grid_resolution_m must be finite and > 0"
                            .to_owned(),
                    ));
                }
                if *min_points_per_axis < 2 {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::CenteredCube: min_points_per_axis must be >= 2".to_owned(),
                    ));
                }
            }
            SearchGrid::ExplicitPoints { points_m } => {
                if points_m.is_empty() {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::ExplicitPoints: points_m must be non-empty".to_owned(),
                    ));
                }
                if points_m.iter().any(|p| p.iter().any(|v| !v.is_finite())) {
                    return Err(KwaversError::InvalidInput(
                        "SearchGrid::ExplicitPoints: all coordinates must be finite".to_owned(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Conservative default search config for small arrays and quick diagnostics.
    #[must_use]
    pub fn conservative_default() -> Self {
        Self {
            core: BeamformingCoreConfig::default(),
            method: LocalizationBeamformingMethod::SrpDasTimeDomain {
                delay_reference: DelayReference::recommended_default(),
            },
            grid: SearchGrid::CenteredCube {
                search_radius_m: 1.0,
                grid_resolution_m: 0.05,
                min_points_per_axis: 10,
            },
            normalize_by_sensor_count: true,
        }
    }
}

impl Default for LocalizationBeamformSearchConfig {
    fn default() -> Self {
        Self::conservative_default()
    }
}

/// Dedicated input type for beamforming-based localization.
///
/// Shape: `(n_sensors, 1, n_samples)`.
#[derive(Debug, Clone)]
pub struct BeamformingLocalizationInput {
    /// Raw sensor time-series data shaped `(n_sensors, 1, n_samples)`.
    pub sensor_data: Array3<f64>,
    /// Sampling frequency in Hz.
    pub sampling_frequency: f64,
}

impl BeamformingLocalizationInput {
    /// Validate invariants for the beamforming-localization input.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self, expected_sensors: usize) -> KwaversResult<()> {
        let (n_sensors, channels, n_samples) = self.sensor_data.dim();
        if n_sensors != expected_sensors {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: sensor_data n_sensors ({n_sensors}) does not match expected ({expected_sensors})"
            )));
        }
        if channels != 1 {
            return Err(KwaversError::InvalidInput(format!(
                "BeamformingLocalizationInput: expected channels=1, got {channels}"
            )));
        }
        if n_samples == 0 {
            return Err(KwaversError::InvalidInput(
                "BeamformingLocalizationInput: n_samples must be > 0".to_owned(),
            ));
        }
        if !self.sampling_frequency.is_finite() || self.sampling_frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "BeamformingLocalizationInput: sampling_frequency must be finite and > 0"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}
