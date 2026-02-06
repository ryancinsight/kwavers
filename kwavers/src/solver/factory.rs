//! Solver Factory for automated solver selection
//!
//! This module provides the `SolverFactory` which creates the appropriate solver
//! instance based on configuration and problem characteristics.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::forward::fdtd::FdtdConfig;
use crate::solver::forward::hybrid::config::HybridConfig;
use crate::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use crate::solver::forward::{FdtdSolver, HybridSolver};
use crate::solver::interface::Solver;

use log::info;

/// Factory for creating solver instances
#[derive(Debug)]
pub struct SolverFactory;

impl SolverFactory {
    /// Create a solver based on the provided configuration
    pub fn create_solver(
        solver_type: SolverType,
        _config: SolverConfiguration, // Using unified config
        grid: &Grid,
        medium: &dyn Medium,
        // Optional specific configs could be passed here or embedded in a larger config struct
        // For simplicity, we use default or generic configs for now, but in a real app this would likely take a `SolverConfigs` struct
    ) -> KwaversResult<Box<dyn Solver>> {
        let selected_type = if solver_type == SolverType::Auto {
            Self::select_best_solver(grid, medium)
        } else {
            solver_type
        };

        info!("Creating solver of type: {:?}", selected_type);

        match selected_type {
            SolverType::FDTD => {
                // Initialize default FDTD config from generic config
                // In a real scenario, we would map generic config to specific config
                let fdtd_config = FdtdConfig::default();
                // We need to construct the solver.
                // Note: The constructors might require specific source types or other params.
                // Assuming we can add sources later.
                let source = GridSource::default(); // Placeholder source
                let solver = FdtdSolver::new(fdtd_config, grid, medium, source)?;
                Ok(Box::new(solver))
            }
            SolverType::PSTD => {
                let pstd_config = PSTDConfig::default();
                let source = GridSource::default();
                let solver = PSTDSolver::new(pstd_config, grid.clone(), medium, source)?;
                Ok(Box::new(solver))
            }
            SolverType::Hybrid => {
                let hybrid_config = HybridConfig::default();
                let solver = HybridSolver::new(hybrid_config, grid, medium)?;
                Ok(Box::new(solver))
            }
            SolverType::Auto => unreachable!("Auto should have been resolved"),
            _ => Err(crate::core::error::KwaversError::NotImplemented(
                "Solver type not yet supported in factory".to_string(),
            )),
        }
    }

    /// Analyze the problem to select the best solver
    fn select_best_solver(grid: &Grid, _medium: &dyn Medium) -> SolverType {
        // Heuristic 1: Check for heterogeneity
        // If medium is highly heterogeneous, FDTD might be better or Hybrid.
        // If smooth, PSTD is much more efficient.

        // Simulating analysis...
        let is_heterogeneous = false; // logic would check medium properties variance

        // Heuristic 2: Grid size
        // For very large grids, PSTD's lower points-per-wavelength requirement is huge advantage.
        let large_grid = grid.nx * grid.ny * grid.nz > 1_000_000;

        if large_grid && !is_heterogeneous {
            info!("Auto-selection: Choosing PSTD for large homogeneous grid");
            SolverType::PSTD
        } else if is_heterogeneous {
            info!("Auto-selection: Choosing Hybrid for heterogeneous medium");
            SolverType::Hybrid
        } else {
            // Default fallback
            SolverType::FDTD
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::homogeneous::HomogeneousMedium;
    use crate::solver::config::SolverConfiguration;

    #[test]
    fn test_solver_factory_creation() {
        let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();

        // Use a simple homogeneous medium
        let medium = HomogeneousMedium::water(&grid);

        let config = SolverConfiguration {
            max_steps: 10,
            dt: 1e-6,
            ..Default::default()
        };

        // Test creation with Auto selection
        let result = SolverFactory::create_solver(SolverType::Auto, config.clone(), &grid, &medium);
        assert!(
            result.is_ok(),
            "Factory failed to create solver (Auto): {:?}",
            result.err()
        );

        // Test creation with explicit FDTD
        let result_fdtd = SolverFactory::create_solver(SolverType::FDTD, config, &grid, &medium);
        assert!(
            result_fdtd.is_ok(),
            "Factory failed to create solver (FDTD): {:?}",
            result_fdtd.err()
        );
    }
}
