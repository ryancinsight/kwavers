//! Source factory for creating acoustic sources
//!
//! Follows Creator pattern for source instantiation

use crate::error::{KwaversResult, ConfigError};
use crate::grid::Grid;
use crate::source::Source;
use crate::constants::physics;

/// Source configuration
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub source_type: String,
    pub position: (f64, f64, f64),
    pub amplitude: f64,
    pub frequency: f64,
    pub radius: Option<f64>,
    pub focus: Option<(f64, f64, f64)>,
    pub num_elements: Option<usize>,
    pub signal_type: String,
    pub phase: f64,
}

impl SourceConfig {
    /// Validate source configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.amplitude <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "amplitude".to_string(),
                value: self.amplitude.to_string(),
                constraint: "Amplitude must be positive".to_string(),
            }.into());
        }

        if self.frequency <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Frequency must be positive".to_string(),
            }.into());
        }

        if let Some(radius) = self.radius {
            if radius <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    parameter: "radius".to_string(),
                    value: radius.to_string(),
                    constraint: "Radius must be positive".to_string(),
                }.into());
            }
        }

        Ok(())
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            source_type: "gaussian".to_string(),
            position: (0.0, 0.0, 0.0),
            amplitude: physics::STANDARD_PRESSURE_AMPLITUDE,
            frequency: physics::DEFAULT_ULTRASOUND_FREQUENCY,
            radius: Some(physics::STANDARD_BEAM_WIDTH),
            focus: None,
            num_elements: None,
            signal_type: "sine".to_string(),
            phase: 0.0,
        }
    }
}

/// Factory for creating sources
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration
    pub fn create_source(config: &SourceConfig, grid: &Grid) -> KwaversResult<Box<dyn Source>> {
        config.validate()?;
        
        // Implementation would create appropriate source type based on config
        // For now, return a placeholder error
        use crate::error::KwaversError;
        Err(KwaversError::NotImplemented("Source creation to be implemented".to_string()))
    }
}