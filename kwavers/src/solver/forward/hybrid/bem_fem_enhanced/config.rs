//! Configuration for enhanced BEM-FEM coupling with Burton-Miller support.

use super::super::BemFemCouplingConfig;
use crate::core::constants::fundamental::SOUND_SPEED_AIR;
use crate::solver::forward::bem::burton_miller::BurtonMillerConfig;

/// Enhanced BEM-FEM coupling configuration with Burton-Miller support.
#[derive(Debug, Clone)]
pub struct EnhancedBemFemConfig {
    /// Standard coupling configuration.
    pub base_config: BemFemCouplingConfig,

    /// Burton-Miller coupling parameter (automatically computed as -i/k).
    pub burton_miller_config: Option<BurtonMillerConfig>,

    /// Enable adaptive interface refinement.
    pub adaptive_refinement: bool,

    /// Maximum refinement level for adaptive procedure.
    pub max_refinement_level: usize,

    /// Target error for adaptive refinement.
    pub target_interface_error: f64,

    /// Minimum element size (for mesh generation).
    pub min_element_size: f64,

    /// Maximum element size (for mesh generation).
    pub max_element_size: f64,

    /// Validation frequency range (Hz) for spurious resonance testing.
    pub validation_frequencies: Option<Vec<f64>>,
}

impl Default for EnhancedBemFemConfig {
    fn default() -> Self {
        Self {
            base_config: BemFemCouplingConfig::default(),
            burton_miller_config: Some(BurtonMillerConfig::new(1000.0, SOUND_SPEED_AIR)),
            adaptive_refinement: true,
            max_refinement_level: 5,
            target_interface_error: 1e-4,
            min_element_size: 1e-3,
            max_element_size: 1e-1,
            validation_frequencies: None,
        }
    }
}

impl EnhancedBemFemConfig {
    /// Create configuration for sphere-in-half-space validation problem.
    #[must_use]
    pub fn for_sphere_validation(_radius: f64, frequency: f64) -> Self {
        let mut config = Self::default();

        let wavelength = SOUND_SPEED_AIR / frequency;
        config.min_element_size = wavelength / 10.0;
        config.max_element_size = wavelength / 4.0;
        config.burton_miller_config = Some(BurtonMillerConfig::new(frequency, SOUND_SPEED_AIR));
        config.adaptive_refinement = true;
        config.target_interface_error = 1e-5;
        config.validation_frequencies = Some(vec![frequency]);

        config
    }

    /// Enable Burton-Miller formulation with specified frequency.
    #[must_use]
    pub fn with_burton_miller(mut self, frequency: f64, sound_speed: f64) -> Self {
        self.burton_miller_config = Some(BurtonMillerConfig::new(frequency, sound_speed));
        self
    }

    /// Disable Burton-Miller (use standard BEM — for comparison).
    #[must_use]
    pub fn without_burton_miller(mut self) -> Self {
        self.burton_miller_config = None;
        self
    }

    /// Enable or disable adaptive interface refinement.
    #[must_use]
    pub fn with_adaptive_refinement(mut self, enabled: bool) -> Self {
        self.adaptive_refinement = enabled;
        self
    }

    /// Set target interface error for adaptive refinement.
    #[must_use]
    pub fn with_target_error(mut self, error: f64) -> Self {
        self.target_interface_error = error;
        self
    }
}
