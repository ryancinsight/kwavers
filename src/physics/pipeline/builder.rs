// src/physics/pipeline/builder.rs
//! Builder pattern for constructing physics pipelines
//! 
//! This module provides a fluent API for building complex physics pipelines
//! with proper configuration and validation.

use crate::error::{KwaversResult, PhysicsError};
use crate::physics::core::{PhysicsEffect, PhysicsSystem, EffectSystem};
use crate::physics::pipeline::{PhysicsPipeline, PipelineConfig};
use std::collections::HashMap;

/// Builder for physics pipelines
pub struct PipelineBuilder {
    /// Pipeline name
    name: String,
    /// Effects to add
    effects: Vec<Box<dyn PhysicsEffect>>,
    /// Systems to add
    systems: Vec<Box<dyn PhysicsSystem>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Global parameters
    parameters: HashMap<String, f64>,
    /// Plugin paths to load
    plugins: Vec<String>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            name: "default".to_string(),
            effects: Vec::new(),
            systems: Vec::new(),
            config: PipelineConfig::default(),
            parameters: HashMap::new(),
            plugins: Vec::new(),
        }
    }
    
    /// Set the pipeline name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// Add an effect to the pipeline
    pub fn add_effect(mut self, effect: Box<dyn PhysicsEffect>) -> Self {
        self.effects.push(effect);
        self
    }
    
    /// Add multiple effects
    pub fn add_effects(mut self, effects: Vec<Box<dyn PhysicsEffect>>) -> Self {
        self.effects.extend(effects);
        self
    }
    
    /// Add a system to the pipeline
    pub fn add_system(mut self, system: Box<dyn PhysicsSystem>) -> Self {
        self.systems.push(system);
        self
    }
    
    /// Add a plugin by path
    pub fn add_plugin(mut self, path: impl Into<String>) -> Self {
        self.plugins.push(path.into());
        self
    }
    
    /// Set a configuration option
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Enable parallel execution
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.config.parallel_execution = enabled;
        self
    }
    
    /// Enable event processing
    pub fn with_event_processing(mut self, enabled: bool) -> Self {
        self.config.event_processing = enabled;
        self
    }
    
    /// Enable performance tracking
    pub fn with_performance_tracking(mut self, enabled: bool) -> Self {
        self.config.performance_tracking = enabled;
        self
    }
    
    /// Enable automatic optimization
    pub fn with_auto_optimization(mut self, enabled: bool) -> Self {
        self.config.auto_optimize = enabled;
        self
    }
    
    /// Add a parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: f64) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Add multiple parameters
    pub fn with_parameters(mut self, params: HashMap<String, f64>) -> Self {
        self.parameters.extend(params);
        self
    }
    
    /// Use a scheduler
    pub fn with_scheduler(self, _scheduler: impl Into<String>) -> Self {
        // In a full implementation, this would configure the scheduler
        self
    }
    
    /// Build the pipeline
    pub fn build(self) -> KwaversResult<PhysicsPipeline> {
        let mut pipeline = PhysicsPipeline::new(self.name);
        pipeline.configure(self.config);
        
        // Add parameters
        for (key, value) in self.parameters {
            pipeline.set_parameter(key, value);
        }
        
        // Load plugins
        for plugin_path in self.plugins {
            // In a full implementation, this would dynamically load plugins
            log::info!("Would load plugin from: {}", plugin_path);
        }
        
        // Add systems
        for system in self.systems {
            pipeline.add_system(system);
        }
        
        // Add effects to a default system if no systems were added
        if !self.effects.is_empty() {
            if pipeline.systems.is_empty() {
                let mut system = EffectSystem::new("default");
                for effect in self.effects {
                    system.add_effect(effect)?;
                }
                pipeline.add_system(Box::new(system));
            } else {
                // Add to existing systems
                for effect in self.effects {
                    pipeline.add_effect(effect)?;
                }
            }
        }
        
        // Validate the pipeline
        self.validate_pipeline(&pipeline)?;
        
        Ok(pipeline)
    }
    
    /// Validate the constructed pipeline
    fn validate_pipeline(&self, pipeline: &PhysicsPipeline) -> KwaversResult<()> {
        let stats = pipeline.statistics();
        
        if stats.num_systems == 0 {
            return Err(PhysicsError::InvalidConfiguration {
                component: "PipelineBuilder".to_string(),
                reason: "Pipeline must have at least one system".to_string(),
            }.into());
        }
        
        Ok(())
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset pipeline configurations
pub mod presets {
    use super::*;
    use crate::physics::effects::optical::SonoluminescenceEffect;
    use crate::physics::effects::optical::sonoluminescence::SonoluminescenceParameters;
    
    /// Create a basic acoustic simulation pipeline
    pub fn acoustic() -> PipelineBuilder {
        PipelineBuilder::new()
            .with_name("acoustic")
            .with_parameter("frequency", 1e6)
            .with_parameter("sound_speed", 1500.0)
    }
    
    /// Create a sonoluminescence simulation pipeline
    pub fn sonoluminescence() -> PipelineBuilder {
        PipelineBuilder::new()
            .with_name("sonoluminescence")
            .add_effect(Box::new(SonoluminescenceEffect::new(
                SonoluminescenceParameters::default()
            )))
            .with_event_processing(true)
            .with_parameter("frequency", 20e3)
    }
    
    /// Create a full multiphysics pipeline
    pub fn multiphysics() -> PipelineBuilder {
        PipelineBuilder::new()
            .with_name("multiphysics")
            .with_parallel_execution(true)
            .with_event_processing(true)
            .with_performance_tracking(true)
            .with_auto_optimization(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builder_basic() {
        let pipeline = PipelineBuilder::new()
            .with_name("test")
            .with_parameter("frequency", 1e6)
            .build();
        
        assert!(pipeline.is_ok());
    }
    
    #[test]
    fn test_builder_presets() {
        let pipeline = presets::acoustic().build();
        assert!(pipeline.is_ok());
        
        let pipeline = presets::sonoluminescence().build();
        assert!(pipeline.is_ok());
    }
}