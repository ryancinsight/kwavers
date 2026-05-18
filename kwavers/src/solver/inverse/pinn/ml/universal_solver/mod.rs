//! Universal PINN Solver for Multi-Physics Applications.
//!
//! SRP split:
//! - `types`        — all public data types
//! - `solver`       — `UniversalPINNSolver` struct definition
//! - `constructors` — `new`, `with_all_domains`, `with_cavitation_sonoluminescence_coupling`
//! - `training`     — `solve_physics_domain`, collocation, model init, training loop
//! - `accessors`    — registration, configuration, query, orchestration

mod accessors;
mod constructors;
mod solver;
#[cfg(test)]
mod tests;
mod training;
mod types;

pub use solver::UniversalPINNSolver;
pub use types::{
    ConvergenceInfo, DomainInfo, EarlyStoppingConfig, GeometricFeature, Geometry2D,
    LineSearchMethod, MultiDomainTrainingResult, PhysicsSolution, UniversalSolverLrSchedule,
    UniversalSolverMemoryStats, UniversalSolverOptimizerType, UniversalSolverStats,
    UniversalTrainingConfig,
};
