//! Simulation orchestration module
//!
//! Provides the core simulation loop and orchestration of components.

pub mod builder;
pub mod configuration;
pub mod core;
pub mod factory; // Moved from physics
pub mod manager; // Moved from physics/factory
pub mod parameters;
pub mod setup; // New setup module
               // pub mod components; // Removed, moved to domain
               // pub mod environment; // (Removed, moved to domain)
               // pub mod factory; // Removed
pub mod imaging;
pub mod modalities;
pub mod therapy; // Keep this as it's re-exported below

// Re-exports
pub use crate::domain::sensor::recorder;
pub use builder::ConfigurationBuilder;
pub use configuration::Configuration;
pub use core::{CoreSimulation, SimulationBuilder, SimulationResult, SimulationStatistics};
pub use manager::PhysicsManager;
pub use modalities::{PhotoacousticParameters, PhotoacousticResult, PhotoacousticSimulator};
pub use parameters::SimulationParameters;
pub use setup::{SimulationComponents, SimulationSetup};
