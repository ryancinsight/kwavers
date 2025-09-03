//! Source factory for creating acoustic sources
//!
//! Follows Creator pattern for source instantiation

use crate::error::{ConfigError, KwaversResult};
use crate::grid::Grid;
use crate::physics::constants::{
    DEFAULT_ULTRASOUND_FREQUENCY, STANDARD_BEAM_WIDTH, STANDARD_PRESSURE_AMPLITUDE,
};
use crate::signal::{Signal, SineWave};
use crate::source::{PointSource, Source};
use std::sync::Arc;

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
            }
            .into());
        }

        if self.frequency <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Frequency must be positive".to_string(),
            }
            .into());
        }

        if let Some(radius) = self.radius {
            if radius <= 0.0 {
                return Err(ConfigError::InvalidValue {
                    parameter: "radius".to_string(),
                    value: radius.to_string(),
                    constraint: "Radius must be positive".to_string(),
                }
                .into());
            }
        }

        Ok(())
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            source_type: "point".to_string(),
            position: (0.0, 0.0, 0.0),
            amplitude: STANDARD_PRESSURE_AMPLITUDE,
            frequency: DEFAULT_ULTRASOUND_FREQUENCY,
            radius: Some(STANDARD_BEAM_WIDTH),
            focus: None,
            num_elements: None,
            signal_type: "sine".to_string(),
            phase: 0.0,
        }
    }
}

/// Factory for creating sources
#[derive(Debug)]
pub struct SourceFactory;

impl SourceFactory {
    /// Create a source from configuration
    pub fn create_source(config: &SourceConfig, _grid: &Grid) -> KwaversResult<Box<dyn Source>> {
        config.validate()?;

        // Create signal based on configuration
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(
            config.frequency,
            config.amplitude,
            config.phase,
        ));

        // Create appropriate source type based on config
        if config.source_type.as_str() == "point" {
            let source = PointSource::new(config.position, signal);
            Ok(Box::new(source))
        } else {
            // For now, default to point source for unrecognized types
            let source = PointSource::new(config.position, signal);
            Ok(Box::new(source))
        }
    }

    /// Create a point source at specified location
    #[must_use]
    pub fn create_point_source(
        x: f64,
        y: f64,
        z: f64,
        amplitude: f64,
        frequency: f64,
    ) -> Box<dyn Source> {
        let signal: Arc<dyn Signal> = Arc::new(SineWave::new(frequency, amplitude, 0.0));
        Box::new(PointSource::new((x, y, z), signal))
    }
}
