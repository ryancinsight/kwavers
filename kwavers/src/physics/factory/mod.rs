//! Physics capability factory
//!
//! Strongly-typed configuration of which physics models are enabled, plus
//! the catalog that translates that configuration into a populated plugin
//! manager. This is the unified-enablement entry point for the solver: the
//! caller declares capabilities (FDTD vs PSTD acoustics, thermal diffusion,
//! elastic waves, etc.) and the catalog hands back a `PluginManager` ready
//! for the time loop.
//!
//! ## Layers
//!
//! - [`models`]: capability enums (`PhysicsModelType`, `AcousticSolver`,
//!   `PhysicsBoundaryCondition`, `NonlinearEquation`, `BubbleModel`).
//! - [`config`]: top-level `PhysicsConfig` aggregating enabled capabilities
//!   plus global parameters and external plugin-path hooks.
//! - [`catalog`]: the [`PhysicsCatalog`] dispatcher mapping each
//!   `PhysicsModelType` to its concrete plugin constructor.

pub mod catalog;
pub mod config;
pub mod models;

pub use catalog::PhysicsCatalog;
pub use config::PhysicsConfig;
pub use models::{
    AcousticSolver, BubbleModel, NonlinearEquation, PhysicsBoundaryCondition, PhysicsModelConfig,
    PhysicsModelType,
};
