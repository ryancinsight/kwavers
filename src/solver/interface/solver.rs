//! Base solver trait
//!
//! This module defines the fundamental solver trait that all
//! solver implementations must implement.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::Source;
use crate::solver::interface::feature::SolverFeature;
use std::fmt::Debug;

/// Fundamental solver trait
pub trait Solver: Debug + Send + Sync {
    /// Get the solver name
    fn name(&self) -> &str;

    /// Initialize the solver with grid and medium
    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()>;

    /// Add a source to the solver
    fn add_source(&mut self, source: Box<dyn Source>) -> KwaversResult<()>;

    /// Add a grid sensor probe set to the solver
    ///
    /// This is the canonical high-level sensor representation in Kwavers and is
    /// compatible with multi-physics (acoustics + optics).
    fn add_sensor(&mut self, sensor: &GridSensorSet) -> KwaversResult<()>;

    /// Run the simulation for specified number of steps
    fn run(&mut self, num_steps: usize) -> KwaversResult<()>;

    /// Get the current pressure field
    fn pressure_field(&self) -> &ndarray::Array3<f64>;

    /// Get the current velocity fields
    fn velocity_fields(
        &self,
    ) -> (
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
    );

    /// Get solver statistics
    fn statistics(&self) -> SolverStatistics;

    /// Check if solver supports a specific feature
    fn supports_feature(&self, feature: SolverFeature) -> bool;

    /// Enable a solver feature
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
