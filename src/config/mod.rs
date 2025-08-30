// config/mod.rs

// Validation configuration system
pub mod validation;

// Re-export the validation system including FieldValidationConfig
pub use validation::{FieldLimits, FieldValidationConfig, ValidationConfig};

// No longer directly used here: Medium, Arc
use log::debug;
use std::fs;
// Removed Arc

use serde::{Deserialize, Serialize};

// Field validation configuration has been moved to validation module for better organization
// This follows proper separation of concerns - validation logic belongs in validation module

pub mod output;
pub mod simulation;
pub mod source;

pub use output::OutputConfig;
pub use simulation::{MediumConfig, SimulationConfig};
pub use source::SourceConfig;

#[derive(Debug, Serialize, Deserialize, Default]
pub struct Config {
    pub simulation: SimulationConfig,
    pub source: SourceConfig,
    pub output: OutputConfig,
    // Removed cached components: grid, time, source_instance, medium_instance, pml
}

// Helper struct for efficient TOML parsing - parse once, not thrice!
#[derive(Deserialize)]
struct TomlConfigHelper {
    simulation: SimulationConfig,
    source: SourceConfig,
    output: OutputConfig,
}

// Helper struct for TOML serialization with references to avoid cloning
#[derive(Serialize)]
struct TomlSerializeHelper<'a> {
    simulation: &'a SimulationConfig,
    source: &'a SourceConfig,
    output: &'a OutputConfig,
}

impl Config {
    /// Load configuration from TOML file - parses ONCE for efficiency
    pub fn from_file(filename: &str) -> Result<Self, crate::error::ConfigError> {
        use crate::error::ConfigError;

        debug!("Loading config from {}", filename);
        let contents = fs::read_to_string(filename).map_err(|_e| ConfigError::FileNotFound {
            path: filename.to_string(),
        })?;

        // Parse the ENTIRE file ONCE into the helper struct
        let helper: TomlConfigHelper =
            toml::from_str(&contents).map_err(|e| ConfigError::ParseError {
                line: 0, // toml error doesnt provide line info in newer versions
                message: e.to_string(),
            })?;

        Ok(Self {
            simulation: helper.simulation,
            source: helper.source,
            output: helper.output,
        })
    }

    /// Save configuration to TOML file for reproducibility
    pub fn to_file(&self, filename: &str) -> Result<(), crate::error::ConfigError> {
        use crate::error::ConfigError;

        let helper = TomlSerializeHelper {
            simulation: &self.simulation,
            source: &self.source,
            output: &self.output,
        };

        let contents = toml::to_string_pretty(&helper).map_err(|e| ConfigError::ParseError {
            line: 0,
            message: format!("Failed to serialize config: {}", e),
        })?;

        fs::write(filename, contents).map_err(|_e| ConfigError::FileNotFound {
            path: filename.to_string(),
        })?;

        debug!("Saved config to {}", filename);
        Ok(())
    }
    // Removed initialize_components method
    // Removed getter methods for grid, time, source, medium, pml
}
