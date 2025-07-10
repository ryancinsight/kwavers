// config/mod.rs

// No longer directly used here: Medium, Arc
use log::debug;
use std::fs;
// Removed Arc

pub mod output;
pub mod simulation;
pub mod source;

pub use output::OutputConfig;
pub use simulation::SimulationConfig;
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
