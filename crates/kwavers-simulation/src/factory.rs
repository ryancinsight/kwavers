//! Simulation-layer entry point for building a populated plugin manager.
//!
//! This module is a thin façade over [`kwavers_solver::plugin::PhysicsCatalog`].
//! Capability types ([`PhysicsConfig`], [`PhysicsModelConfig`],
//! [`PhysicsModelType`]) live in `kwavers_physics::factory` (SSOT); the catalog
//! that turns them into a `PluginManager` lives in `kwavers_solver::plugin`
//! (it constructs solver plugins). Both are re-exported here so callers in the
//! simulation layer can keep a single `crate::factory` import surface.

pub use kwavers_physics::factory::{
    AcousticSolver, BubbleModel, NonlinearEquation, PhysicsConfig, PhysicsModelConfig,
    PhysicsModelType,
};
pub use kwavers_solver::plugin::PhysicsCatalog;

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_solver::plugin::PluginManager;

/// Simulation-layer factory entry point.
///
/// Wraps [`PhysicsCatalog`] to provide the canonical
/// `Configuration → PluginManager` construction path used by
/// [`crate::manager::PhysicsManager`].
#[derive(Debug)]
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Build a plugin manager from validated configuration and runtime context.
    ///
    /// `dt` is the global integrator timestep. `medium` and `grid` are
    /// borrowed only during plugin construction.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_physics(
        config: &PhysicsConfig,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<PluginManager> {
        PhysicsCatalog::build(config, grid, medium, dt)
    }
}
