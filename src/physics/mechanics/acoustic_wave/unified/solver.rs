//! Core unified acoustic solver implementation

use crate::{error::KwaversResult, grid::Grid, medium::Medium};
use ndarray::Array3;
use std::time::Instant;

use super::config::{AcousticModelType, AcousticSolverConfig};
use super::{kuznetsov::KuznetsovSolver, westervelt::WesterveltSolver};

/// Unified acoustic solver that dispatches to model-specific implementations
pub struct UnifiedAcousticSolver {
    /// Configuration
    config: AcousticSolverConfig,

    /// Grid
    grid: Grid,

    /// Model-specific solver
    solver: Box<dyn AcousticSolver>,

    /// Performance metrics
    metrics: SolverMetrics,
}

impl std::fmt::Debug for UnifiedAcousticSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedAcousticSolver")
            .field("config", &self.config)
            .field("grid", &self.grid)
            .field("solver", &"<dyn AcousticSolver>")
            .field("metrics", &self.metrics)
            .finish()
    }
}

/// Trait for model-specific acoustic solvers
pub trait AcousticSolver: Send + Sync {
    /// Update the acoustic field for one time step
    fn update(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Get solver name for logging
    fn name(&self) -> &str;

    /// Check stability condition
    fn check_stability(&self, dt: f64, grid: &Grid, max_sound_speed: f64) -> KwaversResult<()>;
}

/// Performance metrics for the solver
#[derive(Debug, Default)]
struct SolverMetrics {
    total_steps: usize,
    total_time: f64,
    fft_time: f64,
    nonlinear_time: f64,
}

impl UnifiedAcousticSolver {
    /// Create a new unified acoustic solver
    pub fn new(config: AcousticSolverConfig, grid: Grid) -> KwaversResult<Self> {
        config.validate()?;

        // Create model-specific solver
        let solver: Box<dyn AcousticSolver> = match config.model_type {
            AcousticModelType::Linear => {
                // Linear solver was removed due to fundamental flaws
                // Use Westervelt with zero nonlinearity as linear approximation
                let mut linear_config = config.clone();
                linear_config.model_type = AcousticModelType::Westervelt;
                Box::new(WesterveltSolver::new(linear_config, grid.clone())?)
            }
            AcousticModelType::Westervelt => {
                Box::new(WesterveltSolver::new(config.clone(), grid.clone())?)
            }
            AcousticModelType::Kuznetsov => {
                Box::new(KuznetsovSolver::new(config.clone(), grid.clone())?)
            }
        };

        Ok(Self {
            config,
            grid,
            solver,
            metrics: SolverMetrics::default(),
        })
    }

    /// Update the acoustic wave field
    pub fn update_wave(
        &mut self,
        pressure: &mut Array3<f64>,
        medium: &dyn Medium,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        // Delegate to model-specific solver
        self.solver
            .update(pressure, medium, source_term, &self.grid, dt)?;

        // Update metrics
        self.metrics.total_steps += 1;
        self.metrics.total_time += start.elapsed().as_secs_f64();

        Ok(())
    }

    /// Get performance metrics
    pub fn metrics(&self) -> &SolverMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = SolverMetrics::default();
    }
}
