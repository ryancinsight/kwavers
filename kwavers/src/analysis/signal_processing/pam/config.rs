use super::{ApodizationType, PamBeamformingMethod};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingCoreConfig;

#[derive(Debug, Clone)]
pub struct PamBeamformingConfig {
    pub core: BeamformingCoreConfig,
    pub method: PamBeamformingMethod,
    pub frequency_range: (f64, f64),
    pub spatial_resolution: f64,
    pub apodization: ApodizationType,
    pub focal_point: [f64; 3],
}

impl PamBeamformingConfig {
    /// Validate.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        let (f_min, f_max) = self.frequency_range;

        if !(f_min.is_finite() && f_max.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be finite".to_owned(),
            ));
        }
        if f_min < 0.0 || f_max < 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: frequency_range must be non-negative".to_owned(),
            ));
        }
        if f_min > f_max {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: require f_min <= f_max".to_owned(),
            ));
        }

        if !self.spatial_resolution.is_finite() || self.spatial_resolution <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: spatial_resolution must be finite and > 0".to_owned(),
            ));
        }

        if self.focal_point.iter().any(|v| !v.is_finite()) {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: focal_point must be finite".to_owned(),
            ));
        }

        match self.method {
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => {
                if !diagonal_loading.is_finite() || diagonal_loading < 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: diagonal_loading must be finite and >= 0"
                            .to_owned(),
                    ));
                }
            }
            PamBeamformingMethod::Music { num_sources } => {
                if num_sources == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: MUSIC requires num_sources >= 1".to_owned(),
                    ));
                }
            }
            PamBeamformingMethod::EigenspaceMinVariance {
                signal_subspace_dimension,
            } => {
                if signal_subspace_dimension == 0 {
                    return Err(KwaversError::InvalidInput(
                        "PAM beamforming config: ESMV requires signal_subspace_dimension >= 1"
                            .to_owned(),
                    ));
                }
            }
            PamBeamformingMethod::DelayAndSum | PamBeamformingMethod::TimeExposureAcoustics => {}
        }

        Ok(())
    }

    #[must_use]
    pub fn reference_frequency_midpoint(&self) -> f64 {
        let (f_min, f_max) = self.frequency_range;
        0.5 * (f_min + f_max)
    }
}

impl Default for PamBeamformingConfig {
    fn default() -> Self {
        Self {
            core: BeamformingCoreConfig::default(),
            method: PamBeamformingMethod::DelayAndSum,
            frequency_range: (20e3, 10e6),
            spatial_resolution: 1e-3,
            apodization: ApodizationType::Hamming,
            focal_point: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PAMConfig {
    pub beamforming: PamBeamformingConfig,
    pub frequency_bands: Vec<(f64, f64)>,
    pub integration_time: f64,
    pub threshold: f64,
    pub enable_harmonic_analysis: bool,
    pub enable_broadband_analysis: bool,
}

impl Default for PAMConfig {
    fn default() -> Self {
        Self {
            beamforming: PamBeamformingConfig::default(),
            frequency_bands: vec![(20e3, 100e3), (100e3, 500e3), (500e3, 2e6), (2e6, 10e6)],
            integration_time: 0.1,
            threshold: 1e-6,
            enable_harmonic_analysis: true,
            enable_broadband_analysis: true,
        }
    }
}
