use super::{ApodizationType, PamBeamformingMethod};
use aequitas::systems::si::quantities::{Frequency, Length, Time};
use aequitas::systems::si::units::{Hertz, Meter, Second};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_transducer::beamforming::BeamformingCoreConfig;

#[derive(Debug, Clone)]
pub struct PamBeamformingConfig {
    pub core: BeamformingCoreConfig,
    pub method: PamBeamformingMethod,
    pub frequency_range: (Frequency<f64>, Frequency<f64>),
    pub spatial_resolution: Length<f64>,
    pub apodization: ApodizationType,
    pub focal_point: [Length<f64>; 3],
}

impl PamBeamformingConfig {
    /// Validate.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        let (f_min, f_max) = self.frequency_range;
        let f_min = f_min.in_unit::<Hertz>();
        let f_max = f_max.in_unit::<Hertz>();

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

        let spatial_resolution = self.spatial_resolution.in_unit::<Meter>();
        if !spatial_resolution.is_finite() || spatial_resolution <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM beamforming config: spatial_resolution must be finite and > 0".to_owned(),
            ));
        }

        if self
            .focal_point
            .iter()
            .any(|v| !v.in_unit::<Meter>().is_finite())
        {
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
    pub fn reference_frequency_midpoint(&self) -> Frequency<f64> {
        let (f_min, f_max) = self.frequency_range;
        Frequency::from_unit::<Hertz>(0.5 * (f_min.in_unit::<Hertz>() + f_max.in_unit::<Hertz>()))
    }
}

impl Default for PamBeamformingConfig {
    fn default() -> Self {
        Self {
            core: BeamformingCoreConfig::default(),
            method: PamBeamformingMethod::DelayAndSum,
            frequency_range: (
                Frequency::from_unit::<Hertz>(20e3),
                Frequency::from_unit::<Hertz>(10.0 * MHZ_TO_HZ),
            ),
            spatial_resolution: Length::from_unit::<Meter>(1e-3),
            apodization: ApodizationType::Hamming,
            focal_point: [Length::from_unit::<Meter>(0.0); 3],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PAMConfig {
    pub beamforming: PamBeamformingConfig,
    pub frequency_bands: Vec<(Frequency<f64>, Frequency<f64>)>,
    pub integration_time: Time<f64>,
    pub threshold: f64,
    pub enable_harmonic_analysis: bool,
    pub enable_broadband_analysis: bool,
}

impl PAMConfig {
    /// Validate the complete passive-acoustic mapping configuration.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when the beamforming policy,
    /// frequency bands, integration time, or representation threshold is
    /// outside its contract.
    pub fn validate(&self) -> KwaversResult<()> {
        self.beamforming.validate()?;
        if self.frequency_bands.is_empty() {
            return Err(KwaversError::InvalidInput(
                "PAM configuration requires at least one frequency band".to_owned(),
            ));
        }
        for (f_min, f_max) in &self.frequency_bands {
            let f_min = f_min.in_unit::<Hertz>();
            let f_max = f_max.in_unit::<Hertz>();
            if !f_min.is_finite() || !f_max.is_finite() || f_min < 0.0 || f_max < f_min {
                return Err(KwaversError::InvalidInput(
                    "PAM frequency bands must be finite, non-negative, and ordered".to_owned(),
                ));
            }
        }
        let integration_time = self.integration_time.in_unit::<Second>();
        if !integration_time.is_finite() || integration_time <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM integration time must be finite and positive".to_owned(),
            ));
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(KwaversError::InvalidInput(
                "PAM representation threshold must be finite and non-negative".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for PAMConfig {
    fn default() -> Self {
        Self {
            beamforming: PamBeamformingConfig::default(),
            frequency_bands: vec![
                (
                    Frequency::from_unit::<Hertz>(20e3),
                    Frequency::from_unit::<Hertz>(100e3),
                ),
                (
                    Frequency::from_unit::<Hertz>(100e3),
                    Frequency::from_unit::<Hertz>(500e3),
                ),
                (
                    Frequency::from_unit::<Hertz>(500e3),
                    Frequency::from_unit::<Hertz>(2.0 * MHZ_TO_HZ),
                ),
                (
                    Frequency::from_unit::<Hertz>(2.0 * MHZ_TO_HZ),
                    Frequency::from_unit::<Hertz>(10.0 * MHZ_TO_HZ),
                ),
            ],
            integration_time: Time::from_unit::<Second>(0.1),
            threshold: 1e-6,
            enable_harmonic_analysis: true,
            enable_broadband_analysis: true,
        }
    }
}
