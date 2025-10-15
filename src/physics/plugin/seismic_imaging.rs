//! Seismic Imaging Plugin
//!
//! Unified plugin interface for RTM and FWI algorithms
//! Refactored following GRASP principles - maintains backward compatibility
//! Based on Claerbout (1985): "Imaging the Earth's Interior"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::plugin::{PluginMetadata, PluginState};
use ndarray::Array3;

mod fwi;
mod parameters;
mod rtm;

// Re-export types for backward compatibility and direct access
pub use parameters::{
    BoundaryType, ConvergenceCriteria, FwiParameters, ImagingCondition, MigrationAperture,
    RegularizationParameters, RtmSettings, StorageStrategy, TaperFunction,
};

// Re-export processors for direct usage
pub use fwi::FwiProcessor;
pub use rtm::RtmProcessor;

/// Seismic Imaging Plugin
/// Provides RTM and FWI capabilities for subsurface imaging
/// Now follows GRASP principles with modular architecture
#[derive(Debug)]
pub struct SeismicImagingPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// RTM processor
    rtm_processor: Option<RtmProcessor>,
    /// FWI processor  
    fwi_processor: Option<FwiProcessor>,
    /// Migration aperture
    aperture: Option<MigrationAperture>,
}

impl Default for SeismicImagingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl SeismicImagingPlugin {
    /// Create new seismic imaging plugin
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "seismic_imaging".to_string(),
                name: "Seismic Imaging".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "RTM and FWI for subsurface imaging".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            rtm_processor: None,
            fwi_processor: None,
            aperture: None,
        }
    }

    /// Configure RTM settings
    pub fn configure_rtm(&mut self, settings: RtmSettings) {
        self.rtm_processor = Some(RtmProcessor::new(settings));
        self.state = PluginState::Configured;
    }

    /// Configure FWI parameters
    pub fn configure_fwi(&mut self, parameters: FwiParameters) {
        self.fwi_processor = Some(FwiProcessor::new(parameters));
        self.state = PluginState::Configured;
    }

    /// Perform Reverse Time Migration
    /// Based on Baysal et al. (1983): "Reverse time migration"
    pub fn reverse_time_migration(
        &mut self,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let processor = self.rtm_processor.as_ref().ok_or_else(|| {
            crate::error::KwaversError::Physics(crate::error::PhysicsError::InvalidConfiguration {
                parameter: "rtm_processor".to_string(),
                reason: "RTM processor not configured".to_string(),
            })
        })?;

        self.state = PluginState::Running;
        let result = processor.migrate(source_wavefield, receiver_wavefield, grid);
        self.state = PluginState::Initialized;
        result
    }

    /// Perform Full Waveform Inversion
    /// Full Waveform Inversion implementation
    /// Based on Tarantola (1984): "Inversion of seismic reflection data in the acoustic approximation"
    /// Reference: Geophysics, 49(8), 1259-1266
    pub fn full_waveform_inversion(
        &mut self,
        observed_data: &Array3<f64>,
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let processor = self.fwi_processor.as_ref().ok_or_else(|| {
            crate::error::KwaversError::Physics(crate::error::PhysicsError::InvalidConfiguration {
                parameter: "fwi_processor".to_string(),
                reason: "FWI processor not configured".to_string(),
            })
        })?;

        self.state = PluginState::Running;
        let result = processor.invert(observed_data, initial_model, grid);
        self.state = PluginState::Initialized;
        result
    }

    /// Set migration aperture
    pub fn set_aperture(&mut self, aperture: MigrationAperture) {
        self.aperture = Some(aperture);
    }

    /// Get plugin metadata
    #[must_use]
    pub fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    /// Get plugin state
    #[must_use]
    pub fn state(&self) -> &PluginState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = SeismicImagingPlugin::new();
        assert_eq!(plugin.metadata.id, "seismic_imaging");
        assert!(matches!(plugin.state, PluginState::Initialized));
    }

    #[test]
    fn test_rtm_configuration() {
        let mut plugin = SeismicImagingPlugin::new();
        let settings = RtmSettings::default();

        plugin.configure_rtm(settings);
        assert!(plugin.rtm_processor.is_some());
        assert!(matches!(plugin.state, PluginState::Configured));
    }

    #[test]
    fn test_fwi_configuration() {
        let mut plugin = SeismicImagingPlugin::new();
        let parameters = FwiParameters::default();

        plugin.configure_fwi(parameters);
        assert!(plugin.fwi_processor.is_some());
        assert!(matches!(plugin.state, PluginState::Configured));
    }

    #[test]
    fn test_rtm_without_configuration() {
        let mut plugin = SeismicImagingPlugin::new();
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let field = Array3::ones((10, 10, 10));

        let result = plugin.reverse_time_migration(&field, &field, &grid);
        assert!(result.is_err());
    }
}
