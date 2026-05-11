//! Simulation-owned concrete solver assembly.
//!
//! The solver layer owns selection policy over abstract descriptors. This module
//! owns concrete assembly because it is the orchestration boundary that binds
//! domain objects to numerical implementations.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::{density_at, sound_speed_at, AcousticProperties, CoreMedium, Medium};
use crate::domain::source::GridSource;
use crate::simulation::solver_adapters::DgSimulationSolver;
use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::factory::SolverFactory;
use crate::solver::forward::fdtd::FdtdConfig;
use crate::solver::forward::hybrid::config::HybridConfig;
use crate::solver::forward::pstd::config::KSpaceMethod;
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
    /// # Errors
    /// - Returns [`KwaversError::FeatureNotAvailable`] if the precondition for a FeatureNotAvailable-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn create_solver<M: Medium>(
        solver_type: SolverType,
        config: SolverConfiguration,
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
                let solver = FdtdSolver::new(
                    fdtd_config_from(&config),
                    grid,
                    medium,
                    GridSource::default(),
                )?;
                Ok(Box::new(solver))
            }
            SolverType::PSTD => {
                let solver = PSTDSolver::new(
                    pstd_config_from(&config, KSpaceMethod::StandardPSTD),
                    grid.clone(),
                    medium,
                    GridSource::default(),
                )?;
                Ok(Box::new(solver))
            }
            SolverType::KSpace => {
                let solver = PSTDSolver::new(
                    pstd_config_from(&config, KSpaceMethod::FullKSpace),
                    grid.clone(),
                    medium,
                    GridSource::default(),
                )?;
                Ok(Box::new(solver))
            }
            SolverType::Hybrid => {
                let solver = HybridSolver::new(hybrid_config_from(&config), grid, medium)?;
                Ok(Box::new(solver))
            }
            SolverType::Auto => unreachable!("Auto solver type must be resolved before assembly"),
            SolverType::DiscontinuousGalerkin => {
                Ok(Box::new(DgSimulationSolver::new(&config, grid, medium)?))
            }
            SolverType::FEM => Err(KwaversError::FeatureNotAvailable(
                "Simulation factory cannot assemble FEM from Grid until a real Grid-to-TetrahedralMesh generator and frequency-domain source/boundary contract are available".to_owned(),
            )),
        }
    }
}

fn fdtd_config_from(config: &SolverConfiguration) -> FdtdConfig {
    FdtdConfig {
        spatial_order: config.spatial_order,
        cfl_factor: config.cfl,
        enable_gpu_acceleration: config.enable_gpu,
        nt: config.max_steps,
        dt: config.dt,
        ..FdtdConfig::default()
    }
}

fn pstd_config_from(config: &SolverConfiguration, kspace_method: KSpaceMethod) -> PSTDConfig {
    PSTDConfig {
        nt: config.max_steps,
        dt: config.dt,
        kspace_method,
        ..PSTDConfig::default()
    }
}

fn hybrid_config_from(config: &SolverConfiguration) -> HybridConfig {
    HybridConfig {
        pstd_config: pstd_config_from(config, KSpaceMethod::StandardPSTD),
        fdtd_config: fdtd_config_from(config),
        validation: crate::solver::forward::hybrid::config::ValidationConfig {
            enable_validation: config.validation_mode,
            ..Default::default()
        },
        ..HybridConfig::default()
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

    #[test]
    fn assembles_kspace_solver_through_full_kspace_pstd() {
        let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(998.2, 1482.0, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::KSpace,
            max_steps: 2,
            dt: 1.0e-8,
            ..SolverConfiguration::default()
        };

        let mut solver =
            SimulationSolverFactory::create_solver(SolverType::KSpace, config, &grid, &medium)
                .unwrap();

        assert_eq!(solver.name(), "PSTD");
        solver.run(0).unwrap();
        assert_eq!(solver.pressure_field().dim(), (4, 4, 4));
    }

    #[test]
    fn assembles_discontinuous_galerkin_solver_for_valid_layout() {
        let grid = Grid::new(3, 3, 1, 1.0e-3, 1.0, 1.0).unwrap();
        let medium = HomogeneousMedium::from_minimal(998.2, 10.0, &grid);
        let config = SolverConfiguration {
            solver_type: SolverType::DiscontinuousGalerkin,
            spatial_order: 2,
            dt: 1.0e-6,
            ..SolverConfiguration::default()
        };

        let mut solver = SimulationSolverFactory::create_solver(
            SolverType::DiscontinuousGalerkin,
            config,
            &grid,
            &medium,
        )
        .unwrap();

        assert_eq!(solver.name(), "DiscontinuousGalerkin");
        solver.run(1).unwrap();
        assert_eq!(solver.statistics().current_step, 1);
        assert_eq!(solver.pressure_field().dim(), (3, 3, 1));
    }

    #[test]
    fn reports_unavailable_fem_grid_assembly_without_not_implemented() {
        let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium = HomogeneousMedium::from_minimal(998.2, 1482.0, &grid);
        let config = SolverConfiguration::default();

        let error = SimulationSolverFactory::create_solver(SolverType::FEM, config, &grid, &medium)
            .unwrap_err();

        assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
    }
}
