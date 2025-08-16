//! Simulation factory module for creating and configuring simulations
//!
//! This module provides a structured approach to building acoustic simulations
//! with proper configuration, validation, and component initialization.

// Submodules
pub mod config;
pub mod builder;
pub mod setup;
pub mod results;
pub mod validation;

// Re-export main types
pub use config::{
    SimulationConfig, GridConfig, MediumConfig, MediumType,
    PhysicsConfig, PhysicsModelConfig, PhysicsModelType,
    TimeConfig, SourceConfig
};
pub use builder::SimulationBuilder;
pub use setup::SimulationSetup;
pub use results::{SimulationResults, TimestepData};
pub use validation::ValidationConfig;

use crate::error::KwaversResult;

/// Main factory for creating simulations
pub struct SimulationFactory;

impl SimulationFactory {
    /// Create a new simulation with the given configuration
    pub fn create(config: SimulationConfig) -> KwaversResult<SimulationSetup> {
        let builder = SimulationBuilder::new()
            .with_config(config)
            .validate()?;
        
        builder.build()
    }
    
    /// Create a simulation with default configuration
    pub fn create_default() -> KwaversResult<SimulationSetup> {
        Self::create(SimulationConfig::default())
    }
}