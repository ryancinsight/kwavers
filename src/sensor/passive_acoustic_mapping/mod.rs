//! Passive Acoustic Mapping (PAM) for cavitation detection and sonoluminescence
//!
//! This module implements passive acoustic mapping techniques for detecting and
//! mapping cavitation fields and sonoluminescence events using arbitrary sensor
//! array geometries.
//!
//! ## Literature References
//!
//! 1. **Gyöngy & Coussios (2010)**: "Passive spatial mapping of inertial cavitation
//!    during HIFU exposure", IEEE Trans. Biomed. Eng.
//! 2. **Haworth et al. (2012)**: "Passive imaging with pulsed ultrasound insonations",
//!    J. Acoust. Soc. Am.
//! 3. **Coviello et al. (2015)**: "Passive acoustic mapping utilizing optimal beamforming
//!    in ultrasound therapy monitoring", J. Acoust. Soc. Am.

pub mod beamforming;
pub mod geometry;
pub mod mapping;
pub mod plugin;

pub use beamforming::{Beamformer, BeamformingConfig, BeamformingMethod};
pub use geometry::{ArrayElement, ArrayGeometry};
pub use mapping::{PAMConfig, PAMProcessor};
pub use plugin::PAMPlugin;

use crate::error::KwaversResult;
use ndarray::Array3;

/// Main PAM interface
#[derive(Debug)]
pub struct PassiveAcousticMapper {
    processor: PAMProcessor,
    beamformer: Beamformer,
}

impl PassiveAcousticMapper {
    /// Create a new passive acoustic mapper
    pub fn new(config: PAMConfig, geometry: ArrayGeometry) -> KwaversResult<Self> {
        let beamformer = Beamformer::new(geometry, config.beamforming.clone())?;
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
        // Beamform the data
        let beamformed = self.beamformer.beamform(sensor_data, sample_rate)?;

        // Process to extract cavitation map
        self.processor.process(&beamformed)
    }

    /// Get the current configuration
    pub fn config(&self) -> &PAMConfig {
        self.processor.config()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PAMConfig) -> KwaversResult<()> {
        self.processor.set_config(config.clone())?;
        self.beamformer.set_config(config.beamforming)?;
        Ok(())
    }
}
