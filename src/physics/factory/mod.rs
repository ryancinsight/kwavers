//! Physics component factory - Deep hierarchical organization
//!
//! Domain-driven decomposition for physics model management:
//! - Models: Physics model type definitions and variants
//! - Config: Configuration validation and management
//! - Manager: Plugin manager construction and coordination

pub mod config;
pub mod manager;
pub mod models;

// Re-export main types
pub use config::PhysicsConfig;
pub use manager::PhysicsManager;
pub use models::{PhysicsModelConfig, PhysicsModelType};

/// Main physics factory interface
#[derive(Debug)]
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Create physics plugin manager from configuration
    pub fn create_physics(
        config: &PhysicsConfig,
    ) -> crate::domain::core::error::KwaversResult<crate::solver::plugin::PluginManager> {
        // Validate through specialized validator
        config.validate()?;

        // Build through specialized manager
        PhysicsManager::build(config)
    }
}
