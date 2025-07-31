// src/physics/core/effect.rs
//! Base trait and types for physics effects
//! 
//! This module defines the fundamental PhysicsEffect trait that all physics
//! effects must implement, following SOLID and CUPID principles.

use crate::error::KwaversResult;
use crate::physics::composable::{FieldType, ValidationResult};
use crate::physics::core::event::PhysicsEvent;
use crate::physics::state::PhysicsState;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt::Debug;

/// Unique identifier for a physics effect
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EffectId(pub String);

impl EffectId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl From<&str> for EffectId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Category of physics effects for organization and scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectCategory {
    Wave,
    Particle,
    Thermal,
    Optical,
    Chemical,
    Mechanical,
    Custom,
}

/// Context provided to physics effects during execution
#[derive(Debug, Clone)]
pub struct EffectContext {
    /// Current simulation time
    pub time: f64,
    /// Time step
    pub dt: f64,
    /// Simulation step number
    pub step: usize,
    /// Global parameters
    pub parameters: HashMap<String, f64>,
    /// Available fields
    pub available_fields: Vec<FieldType>,
}

impl EffectContext {
    pub fn new(time: f64, dt: f64, step: usize) -> Self {
        Self {
            time,
            dt,
            step,
            parameters: HashMap::new(),
            available_fields: Vec::new(),
        }
    }
    
    /// Add a parameter to the context
    pub fn with_parameter(mut self, key: impl Into<String>, value: f64) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }
    
    /// Set available fields
    pub fn with_fields(mut self, fields: Vec<FieldType>) -> Self {
        self.available_fields = fields;
        self
    }
}

/// Serializable state of a physics effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectState {
    /// Effect identifier
    pub id: EffectId,
    /// Effect-specific state data
    pub data: serde_json::Value,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Core trait for all physics effects
/// 
/// This trait follows SOLID principles:
/// - Single Responsibility: Each effect handles one physics phenomenon
/// - Open/Closed: New effects can be added without modifying existing code
/// - Liskov Substitution: All effects are interchangeable through this trait
/// - Interface Segregation: Minimal required methods with optional extensions
/// - Dependency Inversion: Effects depend on abstractions, not concrete types
pub trait PhysicsEffect: Send + Sync + Debug {
    /// Unique identifier for this effect
    fn id(&self) -> &EffectId;
    
    /// Category of this effect for scheduling and organization
    fn category(&self) -> EffectCategory;
    
    /// Human-readable name
    fn name(&self) -> &str {
        &self.id().0
    }
    
    /// Description of what this effect does
    fn description(&self) -> &str {
        "A physics effect"
    }
    
    // Dependencies and requirements
    
    /// Other effects that must run before this one
    fn required_effects(&self) -> Vec<EffectId> {
        Vec::new()
    }
    
    /// Field types this effect needs to read
    fn required_fields(&self) -> Vec<FieldType>;
    
    /// Field types this effect will write/modify
    fn provided_fields(&self) -> Vec<FieldType>;
    
    /// Optional fields that enhance this effect if available
    fn optional_fields(&self) -> Vec<FieldType> {
        Vec::new()
    }
    
    // Lifecycle methods
    
    /// Initialize the effect before simulation starts
    fn initialize(&mut self, context: &EffectContext) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Validate effect configuration and state
    fn validate(&self, context: &EffectContext) -> ValidationResult {
        let mut result = ValidationResult::new();
        
        // Check if required fields are available
        for field in self.required_fields() {
            if !context.available_fields.contains(&field) {
                result.add_error(format!(
                    "Required field {:?} not available for effect {}",
                    field,
                    self.name()
                ));
            }
        }
        
        result
    }
    
    /// Update the physics state for one time step
    fn update(&mut self, state: &mut PhysicsState, context: &EffectContext) -> KwaversResult<()>;
    
    /// Clean up resources when effect is no longer needed
    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Reset effect to initial state
    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }
    
    // Event handling
    
    /// Handle incoming physics events
    fn handle_event(&mut self, event: &PhysicsEvent) -> KwaversResult<()> {
        Ok(())
    }
    
    /// Get events to emit after update
    fn emit_events(&self) -> Vec<PhysicsEvent> {
        Vec::new()
    }
    
    // State serialization
    
    /// Save current state for checkpointing or analysis
    fn save_state(&self) -> KwaversResult<EffectState> {
        Ok(EffectState {
            id: self.id().clone(),
            data: serde_json::Value::Null,
            metadata: HashMap::new(),
        })
    }
    
    /// Restore state from a checkpoint
    fn load_state(&mut self, state: EffectState) -> KwaversResult<()> {
        Ok(())
    }
    
    // Performance and diagnostics
    
    /// Get performance metrics
    fn metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
    
    /// Priority for scheduling (lower = higher priority)
    fn priority(&self) -> i32 {
        0
    }
    
    /// Whether this effect can run in parallel with others
    fn is_thread_safe(&self) -> bool {
        true
    }
    
    /// Whether this effect is currently enabled
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Extension trait for plugin-based effects
pub trait PluginEffect: PhysicsEffect {
    /// Plugin metadata
    fn metadata(&self) -> &crate::physics::plugin::PluginMetadata;
    
    /// Configure the plugin
    fn configure(&mut self, config: serde_json::Value) -> KwaversResult<()>;
    
    /// Get plugin capabilities
    fn capabilities(&self) -> EffectCapabilities;
}

/// Capabilities that a plugin effect can have
#[derive(Debug, Clone, Default)]
pub struct EffectCapabilities {
    /// Can handle GPU acceleration
    pub gpu_accelerated: bool,
    /// Supports adaptive time stepping
    pub adaptive_timestep: bool,
    /// Can be checkpointed
    pub checkpointable: bool,
    /// Supports multi-scale modeling
    pub multiscale: bool,
    /// Can be visualized
    pub visualizable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_effect_id() {
        let id1 = EffectId::new("test");
        let id2 = EffectId::from("test");
        assert_eq!(id1, id2);
    }
    
    #[test]
    fn test_effect_context() {
        let context = EffectContext::new(1.0, 0.01, 100)
            .with_parameter("frequency", 1e6)
            .with_fields(vec![FieldType::Pressure, FieldType::Temperature]);
        
        assert_eq!(context.time, 1.0);
        assert_eq!(context.dt, 0.01);
        assert_eq!(context.step, 100);
        assert_eq!(context.parameters.get("frequency"), Some(&1e6));
        assert_eq!(context.available_fields.len(), 2);
    }
}