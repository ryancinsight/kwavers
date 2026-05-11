//! `SimulationBuilder` — fluent builder for `CoreSimulation`.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::Source;
use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::feature::SolverFeature;
use std::sync::Arc;

use super::controller::CoreSimulation;

/// Simulation builder for fluent configuration
#[derive(Debug)]
pub struct SimulationBuilder<'a, M: Medium> {
    grid: Option<Grid>,
    medium: Option<&'a M>,
    sources: Vec<Arc<dyn Source>>,
    sensors: Vec<GridSensorSet>,
    feature_config: crate::solver::feature::FeatureBasedConfig,
    solver_config: SolverConfiguration,
}

impl<'a, M: Medium> SimulationBuilder<'a, M> {
    /// Create a new simulation builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            grid: None,
            medium: None,
            sources: Vec::new(),
            sensors: Vec::new(),
            feature_config: crate::solver::feature::FeatureBasedConfig::default(),
            solver_config: SolverConfiguration::default(),
        }
    }

    /// Set the computational grid
    pub fn with_grid(mut self, grid: Grid) -> Self {
        self.grid = Some(grid);
        self
    }

    /// Set the medium
    pub fn with_medium(mut self, medium: &'a M) -> Self {
        self.medium = Some(medium);
        self
    }

    /// Add a source
    pub fn with_source(mut self, source: Arc<dyn Source>) -> Self {
        self.sources.push(source);
        self
    }

    /// Add a sensor
    pub fn with_sensor(mut self, sensor: GridSensorSet) -> Self {
        self.sensors.push(sensor);
        self
    }

    /// Enable a feature
    pub fn with_feature(mut self, feature: SolverFeature) -> Self {
        self.feature_config.enabled_features.enable(feature);
        self
    }

    /// Set fallback behavior
    pub fn with_fallback_behavior(
        mut self,
        behavior: crate::solver::feature::FallbackBehavior,
    ) -> Self {
        self.feature_config.fallback_behavior = behavior;
        self
    }

    /// Set solver configuration
    pub fn with_solver_config(mut self, config: SolverConfiguration) -> Self {
        self.solver_config = config;
        self
    }

    /// Set solver type
    pub fn with_solver_type(mut self, solver_type: SolverType) -> Self {
        self.solver_config.solver_type = solver_type;
        self
    }

    /// Build the simulation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn build(self) -> KwaversResult<CoreSimulation<'a, M>> {
        let grid = self
            .grid
            .clone()
            .ok_or_else(|| "Grid must be specified".to_owned())?;
        let medium = self
            .medium
            .ok_or_else(|| "Medium must be specified".to_owned())?;

        // Use console reporter by default
        let progress_reporter: Box<dyn crate::solver::interface::ProgressReporter> =
            Box::new(crate::solver::interface::ConsoleProgressReporter::default());

        let mut simulation =
            CoreSimulation::new(grid, medium, self.sources, self.sensors, progress_reporter)?;

        simulation.set_solver_config(self.solver_config);

        for feature in self.feature_config.enabled_features.enabled_features() {
            let feature_enum = Self::parse_feature_name_static(feature)?;
            simulation.enable_feature(feature_enum)?;
        }

        Ok(simulation)
    }

    /// Helper to parse feature names back to enum (static version)
    fn parse_feature_name_static(name: &str) -> Result<SolverFeature, String> {
        match name {
            "Reconstruction" => Ok(SolverFeature::Reconstruction),
            "Time Reversal" => Ok(SolverFeature::TimeReversal),
            "Adaptive Mesh Refinement" => Ok(SolverFeature::AdaptiveMeshRefinement),
            "GPU Acceleration" => Ok(SolverFeature::GpuAcceleration),
            "Detailed Logging" => Ok(SolverFeature::DetailedLogging),
            "Validation Mode" => Ok(SolverFeature::ValidationMode),
            "High Precision" => Ok(SolverFeature::HighPrecision),
            "Multi-Threaded" => Ok(SolverFeature::MultiThreaded),
            "Memory Optimization" => Ok(SolverFeature::MemoryOptimization),
            "Experimental Features" => Ok(SolverFeature::ExperimentalFeatures),
            _ => Err(format!("Unknown feature: {}", name)),
        }
    }
}

impl<'a, M: Medium> Default for SimulationBuilder<'a, M> {
    fn default() -> Self {
        Self::new()
    }
}
