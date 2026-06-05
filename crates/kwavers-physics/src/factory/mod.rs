//! Physics capability descriptors
//!
//! Strongly-typed configuration of which physics models are enabled. This is
//! the solver-independent capability vocabulary: the caller declares
//! capabilities (FDTD vs PSTD acoustics, thermal diffusion, elastic waves,
//! etc.) as data.
//!
//! The capability→plugin dispatcher (`PhysicsCatalog`) that turns this config
//! into a populated `PluginManager` lives in
//! [`crate::solver::plugin::catalog`]: it constructs concrete solver plugins,
//! so it depends on `solver` and belongs in the solver layer. Keeping it there
//! (rather than here) preserves the unidirectional `solver → physics`
//! dependency: `physics` holds the descriptors, `solver` consumes them.
//!
//! ## Layers
//!
//! - [`models`]: capability enums (`PhysicsModelType`, `AcousticSolver`,
//!   `PhysicsBoundaryCondition`, `NonlinearEquation`, `BubbleModel`).
//! - [`config`]: top-level `PhysicsConfig` aggregating enabled capabilities
//!   plus global parameters and external plugin-path hooks.

pub mod config;
pub mod models;

pub use config::PhysicsConfig;
pub use models::{
    AcousticSolver, BubbleModel, NonlinearEquation, PhysicsBoundaryCondition, PhysicsModelConfig,
    PhysicsModelType,
};
