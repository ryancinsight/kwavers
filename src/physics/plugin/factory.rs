//! Plugin factory and registry
//!
//! This module provides factory patterns for creating and managing plugins.

use super::{Plugin, PluginMetadata};
use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::grid::Grid;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Factory for creating physics plugins
pub trait PluginFactory: Send + Sync {
    /// Create a new plugin instance
    fn create(
        &self,
        config: Box<dyn Any + Send + Sync>,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Plugin>>;

    /// Get the plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Validate configuration before creation
    fn validate_config(&self, config: &dyn Any) -> KwaversResult<()>;
}

/// Type-safe plugin factory implementation
#[derive(Debug)]
pub struct TypedPluginFactory<F, C, P>
where
    F: Fn(C, &Grid) -> KwaversResult<P> + Send + Sync,
    C: Clone + Send + Sync + 'static,
    P: Plugin + 'static,
{
    create_fn: F,
    metadata: PluginMetadata,
    _phantom: std::marker::PhantomData<(C, P)>,
}

impl<F, C, P> TypedPluginFactory<F, C, P>
where
    F: Fn(C, &Grid) -> KwaversResult<P> + Send + Sync,
    C: Clone + Send + Sync + 'static,
    P: Plugin + 'static,
{
    pub fn new(metadata: PluginMetadata, create_fn: F) -> Self {
        Self {
            create_fn,
            metadata,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, C, P> PluginFactory for TypedPluginFactory<F, C, P>
where
    F: Fn(C, &Grid) -> KwaversResult<P> + Send + Sync,
    C: Clone + Send + Sync + 'static,
    P: Plugin + 'static,
{
    fn create(
        &self,
        config: Box<dyn Any + Send + Sync>,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Plugin>> {
        let config = config
            .downcast::<C>()
            .map_err(|_| ValidationError::FieldValidation {
                field: "config".to_string(),
                value: "unknown".to_string(),
                constraint: "Invalid configuration type".to_string(),
            })?;

        let plugin = (self.create_fn)(*config, grid)?;
        Ok(Box::new(plugin))
    }

    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn validate_config(&self, config: &dyn Any) -> KwaversResult<()> {
        config.downcast_ref::<C>().ok_or_else(|| {
            KwaversError::Validation(ValidationError::FieldValidation {
                field: "config".to_string(),
                value: "unknown".to_string(),
                constraint: "Invalid configuration type".to_string(),
            })
        })?;
        Ok(())
    }
}

/// Registry for plugin factories
#[derive(Debug)]
pub struct PluginRegistry {
    factories: HashMap<String, Arc<dyn PluginFactory>>,
    metadata_cache: HashMap<String, PluginMetadata>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            metadata_cache: HashMap::new(),
        }
    }

    /// Register a plugin factory
    pub fn register<F: PluginFactory + 'static>(
        &mut self,
        id: &str,
        factory: F,
    ) -> KwaversResult<()> {
        if self.factories.contains_key(id) {
            return Err(ValidationError::FieldValidation {
                field: "plugin_id".to_string(),
                value: id.to_string(),
                constraint: "Plugin ID must be unique (already registered)".to_string(),
            }
            .into());
        }

        let metadata = factory.metadata().clone();
        self.metadata_cache.insert(id.to_string(), metadata);
        self.factories.insert(id.to_string(), Arc::new(factory));
        Ok(())
    }

    /// Create a plugin instance
    pub fn create(
        &self,
        id: &str,
        config: Box<dyn Any + Send + Sync>,
        grid: &Grid,
    ) -> KwaversResult<Box<dyn Plugin>> {
        let factory = self
            .factories
            .get(id)
            .ok_or_else(|| ValidationError::FieldValidation {
                field: "plugin_id".to_string(),
                value: id.to_string(),
                constraint: "Plugin must be registered".to_string(),
            })?;

        factory.create(config, grid)
    }

    /// Get plugin metadata
    pub fn get_metadata(&self, id: &str) -> Option<&PluginMetadata> {
        self.metadata_cache.get(id)
    }

    /// List all registered plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.metadata_cache.values().collect()
    }

    /// Check if a plugin is registered
    pub fn has_plugin(&self, id: &str) -> bool {
        self.factories.contains_key(id)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}
