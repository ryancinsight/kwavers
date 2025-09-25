//! Physics configuration management
//!
//! Configuration validation and management following SOLID principles

use serde::{Deserialize, Serialize};
use super::models::PhysicsModelConfig;

/// Physics configuration with comprehensive validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub models: Vec<PhysicsModelConfig>,
    pub global_parameters: std::collections::HashMap<String, f64>,
    pub plugin_paths: Vec<String>,
}

impl PhysicsConfig {
    /// Create new physics configuration
    pub fn new() -> Self {
        Self {
            models: vec![PhysicsModelConfig::default()],
            global_parameters: std::collections::HashMap::new(),
            plugin_paths: Vec::new(),
        }
    }
    
    /// Add physics model
    pub fn add_model(mut self, model: PhysicsModelConfig) -> Self {
        self.models.push(model);
        self
    }
    
    /// Set global parameter
    pub fn set_parameter(mut self, key: String, value: f64) -> Self {
        self.global_parameters.insert(key, value);
        self
    }
    
    /// Validate configuration - maintains backward compatibility
    pub fn validate(&self) -> crate::error::KwaversResult<()> {
        use crate::error::ConfigError;
        
        if self.models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "models".to_string(),
                value: "empty".to_string(),
                constraint: "At least one physics model is required".to_string(),
            }.into());
        }
        
        // Validate each model has reasonable parameters
        for (i, model) in self.models.iter().enumerate() {
            if !model.enabled {
                continue;
            }
            
            // Model-specific validation can be added here
            if model.parameters.is_empty() && matches!(
                model.model_type,
                super::models::PhysicsModelType::NonlinearAcoustics { .. }
            ) {
                return Err(ConfigError::InvalidValue {
                    parameter: format!("models[{}].parameters", i),
                    value: "empty".to_string(),
                    constraint: "Nonlinear models require parameters".to_string(),
                }.into());
            }
        }
        
        Ok(())
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self::new()
    }
}