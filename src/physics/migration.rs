//! Migration helpers for transitioning from PhysicsComponent to PhysicsPlugin
//!
//! This module provides adapters and utilities to help migrate existing code
//! from the deprecated PhysicsComponent system to the new PhysicsPlugin system.

use crate::physics::composable::{PhysicsComponent, PhysicsContext, FieldType as OldFieldType};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginState, PluginContext};
use crate::physics::field_mapping::{UnifiedFieldType, migrate_field_type};
use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::Array4;
use std::any::Any;

/// Adapter to use a PhysicsComponent as a PhysicsPlugin
pub struct ComponentToPluginAdapter {
    component: Box<dyn PhysicsComponent>,
    metadata: PluginMetadata,
    state: PluginState,
}

impl ComponentToPluginAdapter {
    /// Create a new adapter for a PhysicsComponent
    pub fn new(component: Box<dyn PhysicsComponent>) -> Self {
        let id = component.component_id().to_string();
        let metadata = PluginMetadata {
            id: id.clone(),
            name: format!("{} (Migrated)", id),
            version: "1.0.0".to_string(),
            description: format!("Migrated from PhysicsComponent: {}", id),
            author: "Migration Tool".to_string(),
            license: "MIT".to_string(),
        };
        
        Self {
            component,
            metadata,
            state: PluginState::Created,
        }
    }
}

impl PhysicsPlugin for ComponentToPluginAdapter {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state.clone()
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        self.component.required_fields()
            .into_iter()
            .filter_map(|old| migrate_field_type(&old))
            .collect()
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        self.component.provided_fields()
            .into_iter()
            .filter_map(|old| migrate_field_type(&old))
            .collect()
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
        _plugin_context: &PluginContext,
    ) -> KwaversResult<()> {
        // Create a PhysicsContext for the component
        let component_context = PhysicsContext::new();
        
        self.state = PluginState::Running;
        self.component.update(fields, grid, medium, dt, t, &component_context)?;
        Ok(())
    }
    
    fn finalize(&mut self) -> KwaversResult<()> {
        self.component.finalize()?;
        self.state = PluginState::Stopped;
        Ok(())
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(Self {
            component: self.component.clone_component(),
            metadata: self.metadata.clone(),
            state: self.state.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Helper function to migrate a vector of PhysicsComponents to PhysicsPlugins
pub fn migrate_components_to_plugins(
    components: Vec<Box<dyn PhysicsComponent>>
) -> Vec<Box<dyn PhysicsPlugin>> {
    components.into_iter()
        .map(|comp| {
            Box::new(ComponentToPluginAdapter::new(comp)) as Box<dyn PhysicsPlugin>
        })
        .collect()
}

/// Migration guide documentation
pub mod migration_guide {
    //! # Migration Guide: PhysicsComponent to PhysicsPlugin
    //!
    //! ## Why Migrate?
    //! 
    //! The PhysicsComponent system is being deprecated in favor of the more
    //! feature-complete PhysicsPlugin system. The plugin system offers:
    //! - Better dependency management
    //! - Parallel execution support
    //! - Event-driven architecture
    //! - Factory and registry patterns
    //!
    //! ## Quick Migration
    //!
    //! For existing PhysicsComponent implementations, use the adapter:
    //!
    //! ```rust
    //! use kwavers::physics::migration::ComponentToPluginAdapter;
    //! 
    //! let component = MyPhysicsComponent::new();
    //! let plugin = ComponentToPluginAdapter::new(Box::new(component));
    //! plugin_manager.register(Box::new(plugin))?;
    //! ```
    //!
    //! ## Full Migration
    //!
    //! To fully migrate to the plugin system:
    //!
    //! 1. Implement `PhysicsPlugin` instead of `PhysicsComponent`
    //! 2. Use `UnifiedFieldType` instead of the old `FieldType`
    //! 3. Use `PluginContext` instead of `PhysicsContext`
    //! 4. Register with `PluginManager` instead of `PhysicsPipeline`
    //!
    //! ## Field Type Migration
    //!
    //! Old FieldType -> UnifiedFieldType mapping:
    //! - `FieldType::Pressure` -> `UnifiedFieldType::Pressure`
    //! - `FieldType::Temperature` -> `UnifiedFieldType::Temperature`
    //! - `FieldType::Velocity` -> `UnifiedFieldType::VelocityX/Y/Z`
    //! - `FieldType::Light` -> `UnifiedFieldType::LightFluence`
    //! - `FieldType::Chemical` -> `UnifiedFieldType::ChemicalConcentration`
    //! - `FieldType::Cavitation` -> `UnifiedFieldType::BubbleRadius`
    //!
    //! ## Example Migration
    //!
    //! Before:
    //! ```rust
    //! impl PhysicsComponent for MyComponent {
    //!     fn update(&mut self, fields: &mut Array4<f64>, ...) -> KwaversResult<()> {
    //!         let pressure = fields.index_axis(Axis(0), 0); // WRONG INDEX!
    //!         // ...
    //!     }
    //! }
    //! ```
    //!
    //! After:
    //! ```rust
    //! impl PhysicsPlugin for MyPlugin {
    //!     fn update(&mut self, fields: &mut Array4<f64>, ...) -> KwaversResult<()> {
    //!         use kwavers::physics::field_mapping::UnifiedFieldType;
    //!         let pressure_idx = UnifiedFieldType::Pressure.index();
    //!         let pressure = fields.index_axis(Axis(0), pressure_idx); // CORRECT!
    //!         // ...
    //!     }
    //! }
    //! ```
}