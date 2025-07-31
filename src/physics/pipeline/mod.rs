// src/physics/pipeline/mod.rs
//! Physics pipeline for orchestrating effect execution
//! 
//! This module provides the infrastructure for building and executing
//! physics simulation pipelines with proper dependency management.

mod builder;
mod scheduler;
mod optimizer;

pub use builder::PipelineBuilder;
pub use scheduler::{EffectScheduler, DependencyGraph};
pub use optimizer::PipelineOptimizer;

use crate::error::{KwaversResult, PhysicsError};
use crate::physics::core::{
    PhysicsEffect, EffectId, EffectContext, EventBus,
    PhysicsSystem, SystemContext, EffectSystem,
};
use crate::physics::state::PhysicsState;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Physics pipeline for managing and executing effects
pub struct PhysicsPipeline {
    /// Name of this pipeline
    name: String,
    /// Systems containing effects
    systems: Vec<Box<dyn PhysicsSystem>>,
    /// Event bus for inter-effect communication
    event_bus: Arc<EventBus>,
    /// Global parameters
    parameters: HashMap<String, f64>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Pipeline configuration
    config: PipelineConfig,
}

/// Configuration for the physics pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable parallel execution where possible
    pub parallel_execution: bool,
    /// Enable event processing between effects
    pub event_processing: bool,
    /// Maximum events to process per step
    pub max_events_per_step: usize,
    /// Enable performance tracking
    pub performance_tracking: bool,
    /// Enable automatic optimization
    pub auto_optimize: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            event_processing: true,
            max_events_per_step: 1000,
            performance_tracking: true,
            auto_optimize: false,
        }
    }
}

impl PhysicsPipeline {
    /// Create a new physics pipeline
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            systems: Vec::new(),
            event_bus: Arc::new(EventBus::new()),
            parameters: HashMap::new(),
            metrics: HashMap::new(),
            config: PipelineConfig::default(),
        }
    }
    
    /// Create a pipeline builder
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::new()
    }
    
    /// Configure the pipeline
    pub fn configure(&mut self, config: PipelineConfig) {
        self.config = config;
    }
    
    /// Add a parameter
    pub fn set_parameter(&mut self, key: impl Into<String>, value: f64) {
        self.parameters.insert(key.into(), value);
    }
    
    /// Add a system to the pipeline
    pub fn add_system(&mut self, system: Box<dyn PhysicsSystem>) {
        self.systems.push(system);
    }
    
    /// Add an effect to the default system
    pub fn add_effect(&mut self, effect: Box<dyn PhysicsEffect>) -> KwaversResult<()> {
        // Find or create default system
        if self.systems.is_empty() {
            let mut system = EffectSystem::new("default");
            system.add_effect(effect)?;
            self.systems.push(Box::new(system));
        } else {
            // Add to first effect system found
            for system in &mut self.systems {
                if let Some(effect_system) = system.as_any_mut().downcast_mut::<EffectSystem>() {
                    effect_system.add_effect(effect)?;
                    return Ok(());
                }
            }
            
            // No effect system found, create one
            let mut system = EffectSystem::new("additional");
            system.add_effect(effect)?;
            self.systems.push(Box::new(system));
        }
        
        Ok(())
    }
    
    /// Update the pipeline for one time step
    pub fn update(
        &mut self,
        state: &mut PhysicsState,
        dt: f64,
        time: f64,
        step: usize,
    ) -> KwaversResult<()> {
        let start = Instant::now();
        
        // Create system context
        let mut context = SystemContext::new(time, dt, step);
        context.event_bus = self.event_bus.clone();
        context.parameters = self.parameters.clone();
        
        // Execute systems
        for system in &mut self.systems {
            let system_start = Instant::now();
            system.process(state, &context)?;
            
            if self.config.performance_tracking {
                let elapsed = system_start.elapsed().as_secs_f64();
                self.metrics.insert(
                    format!("{}_time", system.name()),
                    elapsed,
                );
            }
        }
        
        // Process events if enabled
        if self.config.event_processing {
            self.event_bus.process_events()?;
        }
        
        // Track overall time
        if self.config.performance_tracking {
            let total_time = start.elapsed().as_secs_f64();
            self.metrics.insert("total_time".to_string(), total_time);
            self.metrics.insert("step_count".to_string(), step as f64);
        }
        
        Ok(())
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
    
    /// Get the event bus
    pub fn event_bus(&self) -> &Arc<EventBus> {
        &self.event_bus
    }
    
    /// Optimize the pipeline based on performance data
    pub fn optimize(&mut self) -> KwaversResult<()> {
        if !self.config.auto_optimize {
            return Ok(());
        }
        
        // Use optimizer to reorder effects
        let optimizer = PipelineOptimizer::new();
        optimizer.optimize_systems(&mut self.systems, &self.metrics)?;
        
        Ok(())
    }
    
    /// Get pipeline statistics
    pub fn statistics(&self) -> PipelineStatistics {
        let mut stats = PipelineStatistics::default();
        
        stats.num_systems = self.systems.len();
        
        // Count effects in each system
        for system in &self.systems {
            if let Some(effect_system) = system.as_any().downcast_ref::<EffectSystem>() {
                stats.num_effects += effect_system.effects_ordered().len();
            }
        }
        
        // Event statistics
        stats.events_processed = self.event_bus.history().len();
        stats.events_pending = self.event_bus.pending_count();
        
        // Performance statistics
        if let Some(&total_time) = self.metrics.get("total_time") {
            stats.average_step_time = total_time;
        }
        
        stats
    }
}

/// Statistics about the pipeline
#[derive(Debug, Default)]
pub struct PipelineStatistics {
    /// Number of systems
    pub num_systems: usize,
    /// Number of effects
    pub num_effects: usize,
    /// Events processed
    pub events_processed: usize,
    /// Events pending
    pub events_pending: usize,
    /// Average step time
    pub average_step_time: f64,
}

// Helper trait for downcasting
trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: PhysicsSystem + 'static> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::composable::FieldType;
    
    // Mock effect for testing
    #[derive(Debug)]
    struct MockEffect {
        id: EffectId,
    }
    
    impl PhysicsEffect for MockEffect {
        fn id(&self) -> &EffectId {
            &self.id
        }
        
        fn category(&self) -> crate::physics::core::EffectCategory {
            crate::physics::core::EffectCategory::Custom
        }
        
        fn required_fields(&self) -> Vec<FieldType> {
            vec![]
        }
        
        fn provided_fields(&self) -> Vec<FieldType> {
            vec![]
        }
        
        fn update(&mut self, _state: &mut PhysicsState, _context: &EffectContext) -> KwaversResult<()> {
            Ok(())
        }
    }
    
    #[test]
    fn test_pipeline_creation() {
        let mut pipeline = PhysicsPipeline::new("test");
        
        let effect = Box::new(MockEffect {
            id: EffectId::from("test_effect"),
        });
        
        pipeline.add_effect(effect).unwrap();
        
        let stats = pipeline.statistics();
        assert_eq!(stats.num_systems, 1);
        assert_eq!(stats.num_effects, 1);
    }
}