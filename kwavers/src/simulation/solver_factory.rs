//! Simulation-owned concrete solver assembly.
//!
//! The solver layer owns selection policy over abstract descriptors. This module
//! owns concrete assembly because it is the orchestration boundary that binds
//! domain objects to numerical implementations.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::{density_at, sound_speed_at, AcousticProperties, CoreMedium, Medium};
use crate::domain::source::GridSource;
use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::factory::SolverFactory;
use crate::solver::forward::fdtd::FdtdConfig;
use crate::solver::forward::hybrid::config::HybridConfig;
use crate::solver::forward::pstd::{PSTDConfig, PSTDSolver};
use crate::solver::forward::{FdtdSolver, HybridSolver};
use crate::solver::interface::factory::{FactoryConfiguration, GridParameters, MediumParameters};
use crate::solver::interface::Solver;

/// Concrete assembly boundary for simulation-driven solver creation.
#[derive(Debug, Default)]
pub struct SimulationSolverFactory;

#[derive(Debug)]
struct GridDescriptor<'a>(&'a Grid);

impl GridParameters for GridDescriptor<'_> {
    fn nx(&self) -> usize {
        self.0.nx
    }

    fn ny(&self) -> usize {
        self.0.ny
    }

    fn nz(&self) -> usize {
        self.0.nz
    }

    fn dx(&self) -> f64 {
        self.0.dx
    }

    fn dy(&self) -> f64 {
        self.0.dy
    }

    fn dz(&self) -> f64 {
        self.0.dz
    }
}

#[derive(Debug)]
struct MediumDescriptor<'a, M: Medium> {
    medium: &'a M,
    grid: &'a Grid,
}

impl<M: Medium> MediumParameters for MediumDescriptor<'_, M> {
    fn sound_speed(&self, x: f64, y: f64, z: f64) -> f64 {
        sound_speed_at(self.medium, x, y, z, self.grid)
    }

    fn density(&self, x: f64, y: f64, z: f64) -> f64 {
        density_at(self.medium, x, y, z, self.grid)
    }

    fn heterogeneity(&self) -> f64 {
        if CoreMedium::is_homogeneous(self.medium) {
            0.0
        } else {
            1.0
        }
    }

    fn absorption(&self, frequency: f64) -> f64 {
        AcousticProperties::absorption_coefficient(self.medium, 0.0, 0.0, 0.0, self.grid, frequency)
    }
}

impl SimulationSolverFactory {
    /// Create a concrete solver for a simulation.
    pub fn create_solver<M: Medium>(
        solver_type: SolverType,
        _config: SolverConfiguration,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<Box<dyn Solver>> {
        let grid_descriptor = GridDescriptor(grid);
        let medium_descriptor = MediumDescriptor { medium, grid };
        SolverFactory::validate_memory_budget(&grid_descriptor, &FactoryConfiguration::default())?;

        let selected_type =
            SolverFactory::resolve_solver_type(solver_type, &grid_descriptor, &medium_descriptor);

        match selected_type {
            SolverType::FDTD => {
                let solver =
                    FdtdSolver::new(FdtdConfig::default(), grid, medium, GridSource::default())?;
                Ok(Box::new(solver))
            }
            SolverType::PSTD => {
                let solver = PSTDSolver::new(
                    PSTDConfig::default(),
                    grid.clone(),
                    medium,
                    GridSource::default(),
                )?;
                Ok(Box::new(solver))
            }
            SolverType::Hybrid => {
                let solver = HybridSolver::new(HybridConfig::default(), grid, medium)?;
                Ok(Box::new(solver))
            }
            SolverType::Auto => unreachable!("Auto solver type must be resolved before assembly"),
            unsupported => Err(KwaversError::NotImplemented(format!(
                "Solver type not supported by simulation factory: {unsupported:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::homogeneous::HomogeneousMedium;

    #[test]
    fn grid_descriptor_preserves_grid_dimensions_and_spacing() {
        let grid = Grid::new(4, 5, 6, 1.0e-3, 2.0e-3, 3.0e-3).unwrap();
        let descriptor = GridDescriptor(&grid);

        assert_eq!(descriptor.nx(), 4);
        assert_eq!(descriptor.ny(), 5);
        assert_eq!(descriptor.nz(), 6);
        assert_eq!(descriptor.total_points(), 120);
        assert_eq!(descriptor.dx(), 1.0e-3);
        assert_eq!(descriptor.dy(), 2.0e-3);
        assert_eq!(descriptor.dz(), 3.0e-3);
        assert_eq!(
            descriptor.characteristic_size().to_bits(),
            (6.0_f64 * 3.0e-3).to_bits()
        );
    }

    #[test]
    fn medium_descriptor_preserves_acoustic_values() {
        let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(998.2, 1482.0, &grid);
        let descriptor = MediumDescriptor {
            medium: &medium,
            grid: &grid,
        };

        assert_eq!(descriptor.density(0.0, 0.0, 0.0), 998.2);
        assert_eq!(descriptor.sound_speed(0.0, 0.0, 0.0), 1482.0);
        assert_eq!(descriptor.heterogeneity(), 0.0);
    }
}
