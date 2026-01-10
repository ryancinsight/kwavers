//! Passive Acoustic Mapping (PAM) for cavitation detection and sonoluminescence
//!
//! This module implements passive acoustic mapping techniques for detecting and
//! mapping cavitation fields and sonoluminescence events using arbitrary sensor
//! array geometries.
//!
//! ## Architectural Note (SSOT Enforcement)
//!
//! PAM provides **no** independent beamforming algorithm implementations.
//! Beamforming algorithms and numerical primitives are owned by
//! `crate::sensor::beamforming` (single source of truth). PAM owns:
//! - beamforming *policy* (method selection, apodization, focal point, bands)
//! - map construction and post-processing (TEA, band power integration, etc.)
//!
//! This preserves a deep vertical separation of concern.
//!
//! ## Literature References
//!
//! 1. **Gyöngy & Coussios (2010)**: "Passive spatial mapping of inertial cavitation
//!    during HIFU exposure", IEEE Trans. Biomed. Eng.
//! 2. **Haworth et al. (2012)**: "Passive imaging with pulsed ultrasound insonations",
//!    J. Acoust. Soc. Am.
//! 3. **Coviello et al. (2015)**: "Passive acoustic mapping utilizing optimal beamforming
//!    in ultrasound therapy monitoring", J. Acoust. Soc. Am.

pub mod beamforming_config;
pub mod geometry;
pub mod mapping;

pub use beamforming_config::{ApodizationType, PamBeamformingConfig, PamBeamformingMethod};
pub use geometry::{ArrayElement, ArrayGeometry};
pub use mapping::{PAMConfig, PAMProcessor};

use crate::domain::core::error::{KwaversError, KwaversResult};
use crate::domain::sensor::beamforming::BeamformingProcessor;
use ndarray::{Array3, Axis};

/// Main PAM interface
#[derive(Debug)]
pub struct PassiveAcousticMapper {
    processor: PAMProcessor,
    beamformer: BeamformingProcessor,
}

impl PassiveAcousticMapper {
    /// Create a new passive acoustic mapper
    pub fn new(config: PAMConfig, geometry: ArrayGeometry) -> KwaversResult<Self> {
        config.beamforming.validate()?;

        let element_positions = geometry.element_positions();
        let core_cfg = config.beamforming.clone().into();
        let beamformer = BeamformingProcessor::new(core_cfg, element_positions);
        let processor = PAMProcessor::new(config)?;

        Ok(Self {
            processor,
            beamformer,
        })
    }

    /// Process sensor data to create PAM image
    pub fn process(
        &mut self,
        sensor_data: &Array3<f64>,
        sample_rate: f64,
    ) -> KwaversResult<Array3<f64>> {
        let config = self.processor.config();

        // Ensure config invariants (no silent divergence).
        config.beamforming.validate()?;

        let delays = self
            .beamformer
            .compute_delays(config.beamforming.focal_point);

        let beamformed = match config.beamforming.method {
            PamBeamformingMethod::DelayAndSum => {
                let weights = vec![1.0; self.beamformer.num_sensors()];
                self.beamformer
                    .delay_and_sum_with(sensor_data, sample_rate, &delays, &weights)?
            }
            PamBeamformingMethod::CaponDiagonalLoading { diagonal_loading } => self
                .beamformer
                .mvdr_unsteered_weights_time_series(sensor_data, diagonal_loading)?,
            PamBeamformingMethod::Music { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: MUSIC is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping."
                        .to_string(),
                ));
            }
            PamBeamformingMethod::EigenspaceMinVariance { .. } => {
                return Err(KwaversError::InvalidInput(
                    "PAM beamforming: EigenspaceMinVariance is not yet wired to the shared subspace implementation. Use DelayAndSum or CaponDiagonalLoading for PAM mapping."
                        .to_string(),
                ));
            }
            PamBeamformingMethod::TimeExposureAcoustics => {
                // TEA = ∫ (DAS(t))² dt, computed here to keep the invariant explicit.
                let weights = vec![1.0; self.beamformer.num_sensors()];
                let das = self.beamformer.delay_and_sum_with(
                    sensor_data,
                    sample_rate,
                    &delays,
                    &weights,
                )?;

                let mut squared = das.clone();
                squared.mapv_inplace(|x| x * x);

                let integrated = squared.sum_axis(Axis(2));
                let (nx, ny) = (integrated.shape()[0], integrated.shape()[1]);

                let mut tea = Array3::<f64>::zeros((nx, ny, 1));
                for ix in 0..nx {
                    for iy in 0..ny {
                        tea[[ix, iy, 0]] = integrated[[ix, iy]];
                    }
                }

                tea
            }
        };

        self.processor.process(&beamformed)
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &PAMConfig {
        self.processor.config()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        config.beamforming.validate()?;

        // Rebuild the shared beamformer from SSOT-derived config.
        let element_positions: Vec<[f64; 3]> = self.beamformer.sensor_positions().to_vec();
        let core_cfg = config.beamforming.clone().into();
        self.beamformer = BeamformingProcessor::new(core_cfg, element_positions);

        self.processor.set_config(config)?;
        Ok(())
    }
}
