//! Base solver trait
//!
//! This module defines the fundamental solver trait that all
//! solver implementations must implement.

use crate::feature::SolverFeature;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_receiver::GridSensorSet;
use kwavers_source::Source;
use leto::Array3;
use std::fmt::Debug;

/// Fundamental solver trait
pub trait Solver: Debug + Send + Sync {
    /// Get the solver name
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn name(&self) -> &str;

    /// Initialize the solver with grid and medium
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()>;

    /// Add a source to the solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()>;

    /// Add a grid sensor probe set to the solver
    ///
    /// This is the canonical high-level sensor representation in Kwavers and is
    /// compatible with multi-physics (acoustics + optics).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn add_sensor(&mut self, sensor: &GridSensorSet) -> KwaversResult<()>;

    /// Run the simulation for specified number of steps
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn run(&mut self, num_steps: usize) -> KwaversResult<()>;

    /// Advance the simulation by one time step.
    ///
    /// FWI and other inversion drivers need single-step control to interleave
    /// source injection, sensor recording, and wavefield checkpointing. Default
    /// impl forwards to `self.run(1)`; concrete solvers should override with
    /// their inherent `step_forward` to skip the per-call init / dispatch
    /// overhead of `run`.
    ///
    /// # Errors
    /// - Returns [`Err`] if the step fails (typically a numerical NaN or
    ///   internal-state precondition violation).
    fn step_forward(&mut self) -> KwaversResult<()> {
        self.run(1)
    }

    /// Get the current pressure field
    fn pressure_field(&self) -> &Array3<f64>;

    /// Owned snapshot of the recorded sensor pressure history, if the solver
    /// has been configured with a sensor mask or sensor set.
    ///
    /// Shape `(N_receivers, N_time_samples)` matching the per-solver sensor
    /// recorder. Returns `None` when no sensor is configured or no samples
    /// have been recorded yet. Default impl returns `None` so solvers without
    /// integrated sensor recording can satisfy the trait without behavioral
    /// change.
    ///
    /// FWI / RTM drivers consume this through trait dispatch instead of
    /// downcasting to the concrete solver type. Allocating once at the end of
    /// a run is acceptable; callers that want zero-copy access should hold a
    /// concrete-typed reference and use the inherent view accessors.
    fn recorded_sensor_pressure(&self) -> Option<leto::Array2<f64>> {
        None
    }

    /// Get the current velocity fields
    fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>);

    /// Get solver statistics
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn statistics(&self) -> SolverStatistics;

    /// Check if solver supports a specific feature
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn supports_feature(&self, feature: SolverFeature) -> bool;

    /// Enable a solver feature
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn enable_feature(&mut self, feature: SolverFeature, enable: bool) -> KwaversResult<()>;
}

/// Solver statistics
#[derive(Debug, Clone, Default)]
pub struct SolverStatistics {
    pub total_steps: usize,
    pub current_step: usize,
    pub computation_time: std::time::Duration,
    pub memory_usage: usize,
    pub max_pressure: f64,
    pub max_velocity: f64,
}
