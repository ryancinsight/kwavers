// src/physics/plugin/adapters.rs
//! Adapters to wrap existing physics components as plugins
//! 
//! This module provides adapters that allow existing PhysicsComponent implementations
//! to be used as plugins, following the Adapter pattern from GRASP.

use super::{PhysicsPlugin, PluginMetadata, PluginContext, PluginState};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::composable::{PhysicsComponent, FieldType, ValidationResult};
use ndarray::Array4;
use std::collections::HashMap;
use std::fmt::Debug;

/// Adapter that wraps a PhysicsComponent to implement the PhysicsPlugin trait
/// 
/// This follows the Adapter pattern to allow existing components to work
/// with the plugin system without modification.
#[derive(Debug)]
pub struct ComponentPluginAdapter {
    component: Box<dyn PhysicsComponent>,
    metadata: PluginMetadata,
    state: PluginState,
}

impl ComponentPluginAdapter {
    /// Create a new adapter for a physics component
    pub fn new(component: Box<dyn PhysicsComponent>, metadata: PluginMetadata) -> Self {
        Self {
            component,
            metadata,
            state: PluginState::Created,
        }
    }
}

impl PhysicsPlugin for ComponentPluginAdapter {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        self.component.required_fields()
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        self.component.provided_fields()
    }
    
    fn initialize(
        &mut self,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        self.component.initialize(grid, medium)?;
        self.state = PluginState::Initialized;
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        self.state = PluginState::Running;
        match self.component.update(fields, grid, medium, dt, t) {
            Ok(()) => {
                self.state = PluginState::Initialized;
                Ok(())
            }
            Err(e) => {
                self.state = PluginState::Error;
                Err(e)
            }
        }
    }
    
    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        Ok(())
    }
    
    fn performance_metrics(&self) -> HashMap<String, f64> {
        self.component.performance_metrics()
    }
    
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult {
        self.component.validate(grid, medium)
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(Self {
            component: self.component.clone_component(),
            metadata: self.metadata.clone(),
            state: PluginState::Created,
        })
    }
}

/// Factory functions for creating plugin adapters for common components
pub mod factories {
    use super::*;
    use crate::physics::composable::{AcousticWaveComponent, ThermalDiffusionComponent};
    
    /// Create a plugin adapter for the acoustic wave component
    pub fn acoustic_wave_plugin(id: String) -> ComponentPluginAdapter {
        let metadata = PluginMetadata {
            id: id.clone(),
            name: "Acoustic Wave Propagation".to_string(),
            version: "1.0.0".to_string(),
            description: "k-space pseudospectral acoustic wave propagation".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        
        ComponentPluginAdapter::new(
            Box::new(AcousticWaveComponent::new(id)),
            metadata,
        )
    }
    
    /// Create a plugin adapter for the thermal diffusion component
    pub fn thermal_diffusion_plugin(id: String) -> ComponentPluginAdapter {
        let metadata = PluginMetadata {
            id: id.clone(),
            name: "Thermal Diffusion".to_string(),
            version: "1.0.0".to_string(),
            description: "Pennes' bioheat equation solver".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        
        ComponentPluginAdapter::new(
            Box::new(ThermalDiffusionComponent::new(id)),
            metadata,
        )
    }
}

// TODO: Enable these tests after adding Debug trait to physics components
#[cfg(test)]
#[cfg(feature = "test_debug_components")]
mod adapter_tests {
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