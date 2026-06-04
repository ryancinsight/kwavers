//! Simulation orchestration module
//!
//! Provides the core simulation loop and orchestration of components.

pub mod backends;
pub mod builder;
pub mod configuration;
pub mod configs;
pub mod core;
pub mod dispatch;
pub mod factory;
pub mod imaging;
pub mod io;
pub mod manager;
pub mod modalities;
pub mod multi_physics;
pub mod parameters;
pub mod photoacoustics;
pub mod runner;
pub mod setup;
pub mod solver_adapters;
pub mod solver_factory;
pub mod therapy;
pub mod types;

// Re-exports
pub use kwavers_receiver::recorder;
pub use builder::ConfigurationBuilder;
pub use configs::{HelmholtzConfig, NonlinearConfig, PmlConfig, PoroelasticConfig, ThermalConfig};
pub use configuration::Configuration;
pub use core::{CoreSimulation, CoreSimulationStatistics, SimulationBuilder, SimulationResult};
pub use runner::{SimulationRunner};
pub use types::{extract_full_grid_stats, FullGridStats, SimulationRunRequest, SimulationRunResult};
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
