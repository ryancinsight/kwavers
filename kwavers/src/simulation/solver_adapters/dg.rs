//! High-level adapter for the nodal discontinuous-Galerkin time-domain core.
//!
//! # Contract
//!
//! The current DG core advances a one-dimensional scalar conservation law over a
//! tensor with shape `(n_elements, n_nodes, n_variables)`. This adapter exposes
//! that real computation through the simulation `Solver` trait for the scalar
//! acoustic pressure case:
//!
//! - `grid.nx` is the number of DG elements.
//! - `grid.ny` is the nodal count per element and must equal `p + 1`.
//! - `grid.nz` must be `1`, the pressure scalar variable.
//!
//! # Theorem
//!
//! For a pressure field whose restriction to each DG element is represented at
//! the configured GLL nodes, `run(n)` applies `n` compositions of the core
//! semi-discrete DG operator with the configured SSP-RK3 or Forward-Euler time
//! integrator. No projection, flux, or CFL logic is duplicated here.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::{sound_speed_at, Medium};
use crate::domain::sensor::GridSensorSet;
use crate::domain::source::Source;
use crate::solver::config::SolverConfiguration;
use crate::solver::feature::SolverFeature;
use crate::solver::forward::pstd::dg::{DGConfig, DGOperations, DGSolver, NumericalSolver};
use crate::solver::interface::{Solver, SolverStatistics};
use ndarray::Array3;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Simulation-facing DG solver over the existing DG numerical core.
#[derive(Debug)]
pub struct DgSimulationSolver {
    core: DGSolver,
    grid: Grid,
    dt: f64,
    pressure: Array3<f64>,
    ux: Array3<f64>,
    uy: Array3<f64>,
    uz: Array3<f64>,
    current_step: usize,
    computation_time: Duration,
}

impl DgSimulationSolver {
    /// Construct the DG adapter from a high-level solver configuration.
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new<M: Medium>(
        config: &SolverConfiguration,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<Self> {
        let polynomial_order = config.spatial_order;
        let expected_nodes = polynomial_order + 1;

        if grid.nz != 1 {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "DiscontinuousGalerkin simulation adapter currently supports one scalar variable; expected grid.nz = 1, got {}",
                grid.nz
            )));
        }

        if grid.ny != expected_nodes {
            return Err(KwaversError::InvalidInput(format!(
                "DiscontinuousGalerkin grid layout requires grid.ny = spatial_order + 1 = {expected_nodes}, got {}",
                grid.ny
            )));
        }

        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "DiscontinuousGalerkin dt must be finite and positive, got {}",
                config.dt
            )));
        }

        let sound_speed = sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "DiscontinuousGalerkin sound speed must be finite and positive, got {sound_speed}"
            )));
        }

        let dg_config = DGConfig {
            polynomial_order,
            sound_speed,
            ..DGConfig::default()
        };
        let core = DGSolver::new(dg_config, Arc::new(grid.clone()))?;
        let max_dt = core.max_stable_dt(grid);
        if config.dt > max_dt {
            return Err(KwaversError::InvalidInput(format!(
                "DiscontinuousGalerkin dt={} exceeds CFL limit {} for p={} and c={}",
                config.dt, max_dt, polynomial_order, sound_speed
            )));
        }

        let shape = (grid.nx, grid.ny, grid.nz);
        Ok(Self {
            core,
            grid: grid.clone(),
            dt: config.dt,
            pressure: Array3::zeros(shape),
            ux: Array3::zeros(shape),
            uy: Array3::zeros(shape),
            uz: Array3::zeros(shape),
            current_step: 0,
            computation_time: Duration::ZERO,
        })
    }

    #[cfg(test)]
    fn pressure_field_mut(&mut self) -> &mut Array3<f64> {
        &mut self.pressure
    }
}

impl Solver for DgSimulationSolver {
    fn name(&self) -> &str {
        "DiscontinuousGalerkin"
    }

    fn initialize(&mut self, grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        if (grid.nx, grid.ny, grid.nz) != (self.grid.nx, self.grid.ny, self.grid.nz) {
            return Err(KwaversError::DimensionMismatch(format!(
                "DG adapter initialized for ({}, {}, {}), got ({}, {}, {})",
                self.grid.nx, self.grid.ny, self.grid.nz, grid.nx, grid.ny, grid.nz
            )));
        }
        Ok(())
    }

    fn add_source(&mut self, _source: Box<dyn Source>) -> KwaversResult<()> {
        Err(KwaversError::FeatureNotAvailable(
            "DiscontinuousGalerkin adapter does not yet define a Source-to-DG projection contract".to_owned(),
        ))
    }

    fn add_sensor(&mut self, sensor: &GridSensorSet) -> KwaversResult<()> {
        for point in sensor.points() {
            if point.i >= self.grid.nx || point.j >= self.grid.ny || point.k >= self.grid.nz {
                return Err(KwaversError::InvalidInput(format!(
                    "DG sensor point ({}, {}, {}) is outside grid ({}, {}, {})",
                    point.i, point.j, point.k, self.grid.nx, self.grid.ny, self.grid.nz
                )));
            }
        }
        Ok(())
    }

    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        let start = Instant::now();
        for _ in 0..num_steps {
            if !self.core.has_modal_coefficients() {
                let coefficients = self.core.project_to_basis(&self.pressure)?;
                self.core.initialize_modal_coefficients(self.grid.nx, 1);
                *self.core.modal_coefficients_mut().ok_or_else(|| {
                    KwaversError::InternalError(
                        "DG modal coefficient allocation failed".to_owned(),
                    )
                })? = coefficients;
            }

            self.core.solve_step(&mut self.pressure, self.dt)?;
            let coefficients = self
                .core
                .modal_coefficients()
                .ok_or_else(|| {
                    KwaversError::InternalError(
                        "DG modal coefficients missing after time step".to_owned(),
                    )
                })?
                .clone();
            self.pressure = self.core.reconstruct_from_basis(&coefficients)?;
            self.current_step += 1;
        }
        self.computation_time += start.elapsed();
        Ok(())
    }

    fn pressure_field(&self) -> &Array3<f64> {
        &self.pressure
    }

    fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    fn statistics(&self) -> SolverStatistics {
        let max_pressure = self.pressure.iter().fold(0.0_f64, |m, &p| m.max(p.abs()));
        SolverStatistics {
            total_steps: self.current_step,
            current_step: self.current_step,
            computation_time: self.computation_time,
            memory_usage: 4 * self.pressure.len() * std::mem::size_of::<f64>(),
            max_pressure,
            max_velocity: 0.0,
        }
    }

    fn supports_feature(&self, feature: SolverFeature) -> bool {
        matches!(feature, SolverFeature::ValidationMode)
    }

    fn enable_feature(&mut self, feature: SolverFeature, enable: bool) -> KwaversResult<()> {
        if enable && !self.supports_feature(feature) {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "DiscontinuousGalerkin adapter does not support {feature:?}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::homogeneous::HomogeneousMedium;
    use crate::solver::config::SolverType;

    #[test]
    fn dg_adapter_advances_input_sensitive_pressure_field() {
        let grid = Grid::new(3, 3, 1, 1.0e-3, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 10.0, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::DiscontinuousGalerkin,
            spatial_order: 2,
            dt: 1.0e-6,
            ..SolverConfiguration::default()
        };
        let mut solver = DgSimulationSolver::new(&config, &grid, &medium).unwrap();
        solver.pressure_field_mut()[[1, 1, 0]] = 1.0;
        let initial = solver.pressure_field().clone();

        solver.run(1).unwrap();

        let stats = solver.statistics();
        assert_eq!(stats.current_step, 1);
        assert!(stats.max_pressure > 0.0);
        let l1_delta: f64 = solver
            .pressure_field()
            .iter()
            .zip(initial.iter())
            .map(|(&after, &before)| (after - before).abs())
            .sum();
        assert!(l1_delta > 0.0);
    }

    #[test]
    fn dg_adapter_rejects_layout_that_cannot_match_dg_nodes() {
        let grid = Grid::new(3, 4, 1, 1.0e-3, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 10.0, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::DiscontinuousGalerkin,
            spatial_order: 2,
            dt: 1.0e-6,
            ..SolverConfiguration::default()
        };

        let error = DgSimulationSolver::new(&config, &grid, &medium).unwrap_err();

        assert!(matches!(error, KwaversError::InvalidInput(_)));
    }
}
