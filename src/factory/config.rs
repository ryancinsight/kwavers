//! Configuration types and builders for simulation components
//!
//! Follows SSOT principle - single source of truth for configuration

use crate::error::{KwaversResult, ConfigError};
use std::collections::HashMap;

/// Main simulation configuration
/// Follows SSOT principle - single source of truth for configuration
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid configuration
    pub grid: super::GridConfig,
    /// Medium configuration
    pub medium: super::MediumConfig,
    /// Physics models to include
    pub physics: super::PhysicsConfig,
    /// Time stepping configuration
    pub time: super::TimeConfig,
    /// Source configuration
    pub source: super::SourceConfig,
    /// Validation settings
    pub validation: super::ValidationConfig,
}

/// Builder for simulation configuration
/// Follows Builder pattern for complex object construction
pub struct ConfigBuilder {
    config: SimulationConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder with defaults
    pub fn new() -> Self {
        Self {
            config: SimulationConfig {
                grid: super::GridConfig::default(),
                medium: super::MediumConfig::default(),
                physics: super::PhysicsConfig::default(),
                time: super::TimeConfig::default(),
                source: super::SourceConfig::default(),
                validation: super::ValidationConfig::default(),
            },
        }
    }
    
    /// Set grid configuration
    pub fn with_grid(mut self, grid: super::GridConfig) -> Self {
        self.config.grid = grid;
        self
    }
    
    /// Set medium configuration
    pub fn with_medium(mut self, medium: super::MediumConfig) -> Self {
        self.config.medium = medium;
        self
    }
    
    /// Set physics configuration
    pub fn with_physics(mut self, physics: super::PhysicsConfig) -> Self {
        self.config.physics = physics;
        self
    }
    
    /// Set time configuration
    pub fn with_time(mut self, time: super::TimeConfig) -> Self {
        self.config.time = time;
        self
    }
    
    /// Set source configuration
    pub fn with_source(mut self, source: super::SourceConfig) -> Self {
        self.config.source = source;
        self
    }
    
    /// Set validation configuration
    pub fn with_validation(mut self, validation: super::ValidationConfig) -> Self {
        self.config.validation = validation;
        self
    }
    
    /// Build the final configuration
    pub fn build(self) -> KwaversResult<SimulationConfig> {
        // Validate the complete configuration
        super::ConfigValidator::validate(&self.config)?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}