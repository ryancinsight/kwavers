//! Simulation orchestration module
//!
//! Provides the core simulation loop and orchestration of components.

pub mod backends; // Solver backend adapters (simulation → solver interface)
pub mod builder;
pub mod configuration;
pub mod core;
pub mod factory; // Moved from physics
pub mod manager; // Moved from physics/factory
pub mod multi_physics; // Multi-physics coupling orchestration
pub mod parameters;
pub mod setup; // New setup module
               // pub mod components; // Removed, moved to domain
               // pub mod environment; // (Removed, moved to domain)
               // pub mod factory; // Removed
pub mod imaging;
pub mod modalities;
pub mod photoacoustics;
pub mod solver_adapters;
pub mod solver_factory;
pub mod therapy; // Keep this as it's re-exported below

// Re-exports
pub use crate::domain::sensor::recorder;
pub use builder::ConfigurationBuilder;
pub use configuration::Configuration;
pub use core::{CoreSimulation, CoreSimulationStatistics, SimulationBuilder, SimulationResult};
pub use manager::PhysicsManager;
pub use modalities::{PhotoacousticParameters, PhotoacousticResult, PhotoacousticSimulator};
pub use multi_physics::{
    CoupledPhysicsSolver, MultiPhysicsConfig, MultiPhysicsFieldCoupler, SimulationCouplingStrategy,
    SimulationMultiPhysicsSolver, SimulationPhysicsDomain,
};
pub use parameters::{
    OutputFieldType, OutputFormat, OutputParameters, PerformanceParameters, SimulationParameters,
};
pub use photoacoustics::PhotoacousticRunner;
pub use setup::{SimulationComponents, SimulationSetup};
