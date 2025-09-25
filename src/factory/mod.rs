//! Factory patterns for creating simulation components
//!
//! Deep hierarchical organization following GRASP principles:
//! - Information Expert: Objects that have the information needed to fulfill a responsibility
//! - Creator: Objects responsible for creating other objects they use
//! - Controller: Objects that coordinate and control system operations
//! - Low Coupling: Minimize dependencies between objects
//! - High Cohesion: Keep related functionality together

// Deep hierarchical component organization
pub mod component {
    pub mod grid;
    pub mod medium;
    pub mod physics;
}

// Legacy flat modules for backward compatibility
pub mod config;
pub mod grid;
pub mod medium;
pub mod physics;
pub mod source;
pub mod time;
pub mod validation;

// Re-export main types - maintaining backward compatibility
pub use config::{ConfigBuilder, SimulationConfig};
pub use grid::{GridConfig, GridFactory};
pub use medium::{MediumConfig, MediumFactory, MediumType};
pub use physics::{PhysicsConfig, PhysicsFactory, PhysicsModelConfig, PhysicsModelType};
pub use source::{SourceConfig, SourceFactory};
pub use time::{TimeConfig, TimeFactory};
pub use validation::{ConfigValidator, ValidationConfig};

// New hierarchical exports
pub use component::grid as hierarchical_grid;
pub use component::medium as hierarchical_medium;
pub use component::physics as hierarchical_physics;

use crate::error::KwaversResult;

/// Main simulation factory for creating complete simulation setups
/// Follows Controller pattern from GRASP
#[derive(Debug)]
pub struct SimulationFactory;

impl SimulationFactory {
    /// Create a new simulation from configuration
    pub fn create_simulation(config: SimulationConfig) -> KwaversResult<SimulationComponents> {
        // Validate configuration
        ConfigValidator::validate(&config)?;

        // Create components using domain-specific factories
        let grid = GridFactory::create_grid(&config.grid)?;
        let medium = MediumFactory::create_medium(&config.medium, &grid)?;
        let physics = PhysicsFactory::create_physics(&config.physics)?;
        let time = TimeFactory::create_time(&config.time, &grid)?;
        let source = SourceFactory::create_source(&config.source, &grid)?;

        Ok(SimulationComponents {
            grid,
            medium,
            physics,
            time,
            source,
        })
    }
}

/// Container for all simulation components
#[derive(Debug)]
pub struct SimulationComponents {
    pub grid: crate::grid::Grid,
    pub medium: Box<dyn crate::medium::Medium>,
    pub physics: crate::physics::plugin::PluginManager,
    pub time: crate::time::Time,
    pub source: Box<dyn crate::source::Source>,
}
