// src/physics/plugin/adapters.rs
//! Adapters to wrap existing physics components as plugins
//! 
//! This module provides adapters that allow existing PhysicsComponent implementations
//! to be used as plugins, following the Adapter pattern from GRASP.

use super::{PhysicsPlugin, PluginMetadata, PluginConfig, PluginContext};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::composable::{PhysicsComponent, PhysicsContext as ComposableContext, FieldType, ValidationResult};
use ndarray::Array4;
use std::collections::HashMap;
use std::fmt::Debug;

/// Adapter to convert a PhysicsComponent into a PhysicsPlugin
#[derive(Debug)]
pub struct ComponentPluginAdapter<C: PhysicsComponent> {
    component: C,
    metadata: PluginMetadata,
}

impl<C: PhysicsComponent> ComponentPluginAdapter<C> {
    /// Create a new adapter for a physics component
    pub fn new(component: C, metadata: PluginMetadata) -> Self {
        Self { component, metadata }
    }
}

impl<C: PhysicsComponent + Debug + 'static> PhysicsPlugin for ComponentPluginAdapter<C> {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        self.component.dependencies()
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        self.component.output_fields()
    }
    
    fn initialize(
        &mut self,
        _config: Option<Box<dyn PluginConfig>>,
        _grid: &Grid,
        _medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Components are typically initialized in their constructors
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        // Convert PluginContext to ComposableContext
        let composable_context = ComposableContext::new(context.frequency);
        
        // Call the component's apply method
        self.component.apply(fields, grid, medium, dt, t, &composable_context)
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.component.get_metrics()
    }
    
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult {
        self.component.validate_configuration(grid, medium)
    }
}

/// Factory functions for creating plugin adapters for common components
pub mod factories {
    use super::*;
    use crate::physics::composable::{AcousticWaveComponent, ThermalDiffusionComponent};
    
    /// Create a plugin adapter for the acoustic wave component
    pub fn acoustic_wave_plugin(id: String) -> ComponentPluginAdapter<AcousticWaveComponent> {
        let metadata = PluginMetadata {
            id: id.clone(),
            name: "Acoustic Wave Propagation".to_string(),
            version: "1.0.0".to_string(),
            description: "k-space pseudospectral acoustic wave propagation".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        
        ComponentPluginAdapter::new(
            AcousticWaveComponent::new(id),
            metadata,
        )
    }
    
    /// Create a plugin adapter for the thermal diffusion component
    pub fn thermal_diffusion_plugin(id: String) -> ComponentPluginAdapter<ThermalDiffusionComponent> {
        let metadata = PluginMetadata {
            id: id.clone(),
            name: "Thermal Diffusion".to_string(),
            version: "1.0.0".to_string(),
            description: "Pennes' bioheat equation solver".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        
        ComponentPluginAdapter::new(
            ThermalDiffusionComponent::new(id),
            metadata,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::factories::*;
    
    #[test]
    fn test_acoustic_wave_adapter() {
        let plugin = acoustic_wave_plugin("test_acoustic".to_string());
        
        assert_eq!(plugin.metadata().id, "test_acoustic");
        assert_eq!(plugin.metadata().name, "Acoustic Wave Propagation");
        assert_eq!(plugin.required_fields(), vec![FieldType::Pressure]);
        assert_eq!(plugin.provided_fields(), vec![FieldType::Pressure]);
    }
    
    #[test]
    fn test_thermal_diffusion_adapter() {
        let plugin = thermal_diffusion_plugin("test_thermal".to_string());
        
        assert_eq!(plugin.metadata().id, "test_thermal");
        assert_eq!(plugin.metadata().name, "Thermal Diffusion");
        assert_eq!(plugin.required_fields(), vec![FieldType::Pressure]);
        assert_eq!(plugin.provided_fields(), vec![FieldType::Temperature]);
    }
}