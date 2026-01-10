//! Physics component factory - Deep hierarchical organization
//!
//! Domain-driven decomposition for physics model management:
//! - Models: Physics model type definitions and variants
//! - Config: Configuration validation and management
//! - Manager: Plugin manager construction and coordination

pub mod config;
pub mod config;
pub mod models;

// Re-export main types
pub use config::PhysicsConfig;
// pub use manager::PhysicsManager; // Moved to simulation
pub use models::{PhysicsModelConfig, PhysicsModelType};

// PhysicsFactory moved to crate::simulation::factory::PhysicsFactory
