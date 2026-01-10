//! Physics factory - Simulation orchestration
//!
//! Creates the physics plugin manager from configuration.

use crate::simulation::manager::PhysicsManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Physics configuration with comprehensive validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub models: Vec<PhysicsModelConfig>,
    pub global_parameters: HashMap<String, f64>,
    pub plugin_paths: Vec<String>,
}

impl PhysicsConfig {
    /// Create new physics configuration
    pub fn new() -> Self {
        Self {
            models: vec![PhysicsModelConfig::default()],
            global_parameters: HashMap::new(),
            plugin_paths: Vec::new(),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::core::error::KwaversResult<()> {
        use crate::core::error::ConfigError;

        if self.models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "models".to_string(),
                value: "empty".to_string(),
                constraint: "At least one physics model is required".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder for physics model configuration
/// This should be moved from physics::factory::models
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhysicsModelConfig {
    pub enabled: bool,
    pub parameters: HashMap<String, f64>,
}

/// Main physics factory interface
#[derive(Debug)]
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Create physics plugin manager from configuration
    pub fn create_physics(
        config: &PhysicsConfig,
    ) -> crate::core::error::KwaversResult<crate::solver::plugin::PluginManager> {
        // Validate through specialized validator
        config.validate()?;

        // Build through specialized manager
        PhysicsManager::build(config)
    }
}
