//! Physics factory for creating physics models
//!
//! Follows Creator pattern for physics component instantiation

use crate::error::{KwaversResult, ConfigError};
use crate::physics::plugin::PluginManager;
use crate::constants::physics as phys_const;
use std::collections::HashMap;

/// Physics configuration
#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    pub models: Vec<PhysicsModelConfig>,
    pub frequency: f64,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct PhysicsModelConfig {
    pub model_type: PhysicsModelType,
    pub enabled: bool,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum PhysicsModelType {
    AcousticWave,
    ThermalDiffusion,
    Cavitation,
    KuznetsovWave,
    ElasticWave,
    LightDiffusion,
    Chemical,
}

impl PhysicsConfig {
    /// Validate physics configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.frequency <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Frequency must be positive".to_string(),
            }.into());
        }

        if self.models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "physics models".to_string(),
                value: "empty".to_string(),
                constraint: "At least one physics model must be specified".to_string(),
            }.into());
        }

        let enabled_models: Vec<_> = self.models.iter()
            .filter(|m| m.enabled)
            .collect();

        if enabled_models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "enabled physics models".to_string(),
                value: "none".to_string(),
                constraint: "At least one physics model must be enabled".to_string(),
            }.into());
        }

        Ok(())
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            models: vec![
                PhysicsModelConfig {
                    model_type: PhysicsModelType::AcousticWave,
                    enabled: true,
                    parameters: HashMap::new(),
                }
            ],
            frequency: phys_const::DEFAULT_ULTRASOUND_FREQUENCY,
            parameters: HashMap::new(),
        }
    }
}

/// Factory for creating physics models
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Create physics models from configuration
    pub fn create_physics(config: &PhysicsConfig) -> KwaversResult<PluginManager> {
        config.validate()?;
        
        let mut manager = PluginManager::new();
        
        for model_config in &config.models {
            if !model_config.enabled {
                continue;
            }
            
            match model_config.model_type {
                PhysicsModelType::AcousticWave => {
                    // Register acoustic wave plugin
                    // manager.register_acoustic_wave()?;
                }
                PhysicsModelType::KuznetsovWave => {
                    // Register Kuznetsov wave plugin
                    // manager.register_kuznetsov()?;
                }
                PhysicsModelType::ThermalDiffusion => {
                    // Register thermal diffusion plugin
                    // manager.register_thermal()?;
                }
                _ => {
                    // Other physics models
                }
            }
        }
        
        Ok(manager)
    }
}