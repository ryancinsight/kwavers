//! Simulation-owned concrete solver assembly.
//!
//! The solver layer owns selection policy over abstract descriptors. This module
//! owns concrete assembly because it is the orchestration boundary that binds
//! domain objects to numerical implementations.

use crate::solver_adapters::DgSimulationSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::{density_at, sound_speed_at, AcousticProperties, CoreMedium, Medium};
use kwavers_solver::config::{SolverConfiguration, SolverType};
use kwavers_solver::factory::SolverFactoryRegistry;
use kwavers_solver::forward::fdtd::FdtdConfig;
use kwavers_solver::forward::hybrid::config::HybridConfig;
use kwavers_solver::forward::pstd::config::KSpaceMethod;
use kwavers_solver::forward::pstd::{PSTDConfig, PSTDSolver};
use kwavers_solver::interface::factory::{
    FactoryConfiguration, FactoryGridParameters, FactoryMediumParameters,
};
use kwavers_solver::interface::Solver;
use kwavers_solver::{FdtdSolver, HybridSolver};
use kwavers_source::GridSource;

/// Concrete assembly boundary for simulation-driven solver creation.
#[derive(Debug, Default)]
pub struct SimulationSolverFactory;

#[derive(Debug)]
struct GridDescriptor<'a>(&'a Grid);

impl FactoryGridParameters for GridDescriptor<'_> {
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

impl<M: Medium> FactoryMediumParameters for MediumDescriptor<'_, M> {
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
        SolverFactoryRegistry::validate_memory_budget(
            &grid_descriptor,
            &FactoryConfiguration::default(),
        )?;

        let selected_type = SolverFactoryRegistry::resolve_solver_type(
            solver_type,
            &grid_descriptor,
            &medium_descriptor,
        );

        match selected_type {
            SolverType::FDTD => {
                let mut solver = FdtdSolver::new(
                    fdtd_config_from(&config),
                    grid,
                    medium,
                    GridSource::default(),
                )?;
                // Hoist CPML configuration: apply absorbing boundary before
                // boxing so callers receive a fully configured `Box<dyn Solver>`
                // without needing to downcast to FdtdSolver.
                if let Some(ref abc) = config.absorbing_boundary {
                    solver.enable_cpml(abc.cpml.clone(), config.dt, abc.max_sound_speed)?;
                }
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
            #[cfg(feature = "gpu")]
            SolverType::PstdGpu => {
                use crate::solver_adapters::GpuPstdSimulationAdapter;
                Ok(Box::new(GpuPstdSimulationAdapter::new(&config, grid, medium)?))
            }
            #[cfg(not(feature = "gpu"))]
            SolverType::PstdGpu => Err(KwaversError::FeatureNotAvailable(
                "SolverType::PstdGpu requires the `gpu` Cargo feature".to_owned(),
            )),
            _ => Err(KwaversError::FeatureNotAvailable(format!(
                "SolverType::{selected_type:?} not supported by SimulationSolverFactory"
            ))),
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
        validation: kwavers_solver::forward::hybrid::config::HybridValidationConfig {
            enable_validation: config.validation_mode,
            ..Default::default()
        },
        ..HybridConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{
        DENSITY_WATER, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER,
    };
    use kwavers_medium::homogeneous::HomogeneousMedium;

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
        let medium = HomogeneousMedium::from_minimal(DENSITY_WATER, SOUND_SPEED_WATER, &grid);
        let descriptor = MediumDescriptor {
            medium: &medium,
            grid: &grid,
        };

        assert_eq!(descriptor.density(0.0, 0.0, 0.0), DENSITY_WATER);
        assert_eq!(descriptor.sound_speed(0.0, 0.0, 0.0), SOUND_SPEED_WATER);
        assert_eq!(descriptor.heterogeneity(), 0.0);
    }

    #[test]
    fn assembles_kspace_solver_through_full_kspace_pstd() {
        let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
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
        let medium = HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, 10.0, &grid);
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
    fn assembles_fdtd_with_cpml_via_absorbing_boundary_config() {
        use kwavers_boundary::cpml::CPMLConfig;
        use kwavers_solver::config::AbsorbingBoundaryConfig;

        // Grid must be large enough for the CPML thickness.
        // CPMLConfig::with_thickness(4) → 4 cells each side.
        // 16 × 16 × 16 with dx = 1e-3 is well-conditioned.
        let grid = Grid::new(16, 16, 16, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);

        // dt satisfies CFL: dx / (sqrt(3) * c) ≈ 1e-3 / (1.732 * 1500) ≈ 3.85e-7 s.
        // Use 1e-7 to stay well inside the stability region.
        let dt = 1.0e-7_f64;
        let config = SolverConfiguration {
            solver_type: SolverType::FDTD,
            max_steps: 2,
            dt,
            cfl: 0.3,
            absorbing_boundary: Some(AbsorbingBoundaryConfig {
                cpml: CPMLConfig::with_thickness(4),
                max_sound_speed: SOUND_SPEED_WATER,
            }),
            ..SolverConfiguration::default()
        };

        let mut solver =
            SimulationSolverFactory::create_solver(SolverType::FDTD, config, &grid, &medium)
                .unwrap();

        // Run two steps through the CPML-enabled FDTD to confirm the boundary
        // code path executes without panic or NaN propagation.
        solver.run(2).unwrap();

        let p = solver.pressure_field();
        // Value-semantic: every pressure cell must be finite after 2 steps
        // in a medium with no source and CPML boundaries.
        assert!(
            p.iter().all(|v| v.is_finite()),
            "CPML FDTD produced non-finite pressure after 2 steps"
        );
        // Shape must match the grid dimensions unchanged.
        assert_eq!(p.dim(), (16, 16, 16));
    }

    #[test]
    fn reports_unavailable_fem_grid_assembly_without_not_implemented() {
        let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER, &grid);
        let config = SolverConfiguration::default();

        let error = SimulationSolverFactory::create_solver(SolverType::FEM, config, &grid, &medium)
            .unwrap_err();

        assert!(matches!(error, KwaversError::FeatureNotAvailable(_)));
    }
}
