//! Physics component factory - Deep hierarchical organization
//!
//! Domain-driven decomposition for physics model management:
//! - Models: Physics model type definitions and variants
//! - Config: Configuration validation and management
//! - Manager: Plugin manager construction and coordination

pub mod models;
pub mod config;
pub mod manager;

// Re-export main types
pub use models::{PhysicsModelConfig, PhysicsModelType};
pub use config::PhysicsConfig;
pub use manager::PhysicsManager;

/// Main physics factory interface
#[derive(Debug)]
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Create physics plugin manager from configuration
    pub fn create_physics(
        config: &PhysicsConfig
    ) -> crate::error::KwaversResult<crate::physics::plugin::PluginManager> {
        // Validate through specialized validator
        config.validate()?;
        
        // Build through specialized manager
        PhysicsManager::build(config)
    }
}