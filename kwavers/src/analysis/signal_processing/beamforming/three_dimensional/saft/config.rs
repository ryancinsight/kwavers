//! SAFT configuration.

use super::super::{config, Beamforming3dApodizationWindow};

/// SAFT configuration parameters
#[derive(Debug, Clone)]
pub struct SaftConfig {
    /// Number of virtual sources for synthetic aperture
    pub virtual_sources: usize,
    /// Apodization window for sidelobe suppression
    pub apodization: Beamforming3dApodizationWindow,
    /// Coherence factor weighting enabled
    pub coherence_factor_enabled: bool,
    /// F-number for dynamic focusing
    pub f_number: f64,
}

impl Default for SaftConfig {
    fn default() -> Self {
        Self {
            virtual_sources: 100,
            apodization: config::Beamforming3dApodizationWindow::Hamming,
            coherence_factor_enabled: true,
            f_number: 1.5,
        }
    }
}
