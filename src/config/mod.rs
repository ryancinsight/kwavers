// config/mod.rs

// Validation configuration system
pub mod validation;

// Re-export the validation system
pub use validation::ValidationConfig;

// No longer directly used here: Medium, Arc
use log::debug;
use std::fs;
// Removed Arc

use serde::{Deserialize, Serialize};

/// Field validation configuration - single source of truth for all field limits
/// Follows SSOT and DRY principles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldValidationConfig {
    /// Pressure field limits
    pub pressure: FieldLimits,
    /// Temperature field limits  
    pub temperature: FieldLimits,
    /// Light intensity limits
    pub light: FieldLimits,
    /// Velocity field limits
    pub velocity: FieldLimits,
    /// Stress field limits
    pub stress: FieldLimits,
}

/// Field validation limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldLimits {
    pub min: f64,
    pub max: f64,
    pub warn_min: Option<f64>,
    pub warn_max: Option<f64>,
}

impl Default for FieldValidationConfig {
    fn default() -> Self {
        Self {
            pressure: FieldLimits {
                min: -1e9,  // -1 GPa
                max: 1e9,   // 1 GPa
                warn_min: Some(-1e8),
                warn_max: Some(1e8),
            },
            temperature: FieldLimits {
                min: 0.0,     // Absolute zero
                max: 10000.0, // 10,000 K
                warn_min: Some(200.0),
                warn_max: Some(5000.0),
            },
            light: FieldLimits {
                min: 0.0,
                max: 1e12,  // 1 TW/m²
                warn_min: None,
                warn_max: Some(1e10),
            },
            velocity: FieldLimits {
                min: -10000.0,  // -10 km/s
                max: 10000.0,   // 10 km/s
                warn_min: Some(-5000.0),
                warn_max: Some(5000.0),
            },
            stress: FieldLimits {
                min: -1e10,  // -10 GPa
                max: 1e10,   // 10 GPa
                warn_min: Some(-1e9),
                warn_max: Some(1e9),
            },
        }
    }
}

impl FieldValidationConfig {
    /// Load from file
    pub fn from_file(path: &str) -> Result<Self, std::io::Error> {
        let contents = std::fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
    
    /// Save to file
    pub fn to_file(&self, path: &str) -> Result<(), std::io::Error> {
        let contents = toml::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        std::fs::write(path, contents)
    }
}

pub mod output;
pub mod simulation;
pub mod source;

pub use output::OutputConfig;
pub use simulation::{SimulationConfig, MediumConfig};
pub use source::SourceConfig;

#[derive(Debug)]
pub struct Config {
    pub simulation: SimulationConfig,
    pub source: SourceConfig,
    pub output: OutputConfig,
    // Removed cached components: grid, time, source_instance, medium_instance, pml
}

impl Config {
    pub fn from_file(filename: &str) -> Result<Self, String> {
        debug!("Loading config from {}", filename);
        let contents = fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read {}: {}", filename, e))?;

        let simulation: SimulationConfig = toml::from_str(&contents)
            .map_err(|e| format!("Simulation config parse error: {}", e))?;
        let source: SourceConfig =
            toml::from_str(&contents).map_err(|e| format!("Source config parse error: {}", e))?;
        let output: OutputConfig =
            toml::from_str(&contents).map_err(|e| format!("Output config parse error: {}", e))?;

        Ok(Self {
            simulation,
            source,
            output,
        })
    }
    // Removed initialize_components method
    // Removed getter methods for grid, time, source, medium, pml
}
