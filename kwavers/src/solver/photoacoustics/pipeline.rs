use super::acoustic::AcousticBenchmarkCase;
use super::optical::OpticalBenchmarkCase;
use super::reconstruction::ReconstructionBenchmarkCase;
use super::source::SourceValidationCase;
use super::{
    validate_photoacoustic_simulation, AcousticForwardModel, DiffusionOpticalSolver,
    MonteCarloOpticalSolver, OpticalForwardModel, OpticalSolveResult,
    PhotoacousticReconstructionModel, PhotoacousticSourceModel,
};
use crate::core::error::KwaversResult;
use crate::domain::imaging::photoacoustic::{
    OpticalModel, PhotoacousticScenario, PhotoacousticSimulation, PhotoacousticValidationReport,
};
use ndarray::Array3;

/// Reusable canonical pipeline workspace.
///
/// This struct makes stage ownership explicit for performance and memory audits:
/// the optical fluence, initial pressure, and reconstruction buffers are the
/// canonical steady-state scratch objects of the retained CPU path.
#[derive(Debug, Clone)]
pub struct PhotoacousticWorkspace {
    pub optical_fluence: Array3<f64>,
    pub initial_pressure: Array3<f64>,
    pub reconstruction: Array3<f64>,
}

/// Deterministic validation case descriptor for the canonical vertical.
#[derive(Debug, Clone)]
pub struct PhotoacousticValidationCase {
    pub name: &'static str,
    pub wavelength_nm: f64,
    pub optical_model: OpticalModel,
    pub expected_sensor_count: usize,
}

/// Benchmark descriptor for the canonical photoacoustic pipeline.
#[derive(Debug, Clone)]
pub struct PhotoacousticBenchmarkCase {
    pub name: &'static str,
    pub grid_size: [usize; 3],
    pub optical_model: OpticalModel,
    pub num_time_steps: usize,
}

#[derive(Debug, Default)]
pub struct PhotoacousticPipeline {
    source_model: PhotoacousticSourceModel,
    acoustic_model: AcousticForwardModel,
    reconstruction_model: PhotoacousticReconstructionModel,
    diffusion_solver: DiffusionOpticalSolver,
    monte_carlo_solver: MonteCarloOpticalSolver,
}

impl PhotoacousticPipeline {
    #[must_use]
    pub fn workspace_for(&self, scenario: &PhotoacousticScenario) -> PhotoacousticWorkspace {
        let shape = (scenario.grid.nx, scenario.grid.ny, scenario.grid.nz);
        PhotoacousticWorkspace {
            optical_fluence: Array3::zeros(shape),
            initial_pressure: Array3::zeros(shape),
            reconstruction: Array3::zeros(shape),
        }
    }

    #[must_use]
    pub fn validation_case_for(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> PhotoacousticValidationCase {
        PhotoacousticValidationCase {
            name: "canonical_photoacoustic_pipeline",
            wavelength_nm: scenario.wavelength_nm,
            optical_model: scenario.config.optical_model,
            expected_sensor_count: scenario.sensor_positions_m.len(),
        }
    }

    #[must_use]
    pub fn benchmark_case_for(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> PhotoacousticBenchmarkCase {
        PhotoacousticBenchmarkCase {
            name: "canonical_photoacoustic_pipeline",
            grid_size: [scenario.grid.nx, scenario.grid.ny, scenario.grid.nz],
            optical_model: scenario.config.optical_model,
            num_time_steps: scenario.config.acoustic.num_time_steps,
        }
    }

    fn optical_solver(&self, model: OpticalModel) -> &dyn OpticalForwardModel {
        match model {
            OpticalModel::Diffusion => &self.diffusion_solver,
            OpticalModel::MonteCarlo => &self.monte_carlo_solver,
        }
    }
    /// Compute fluence.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_fluence(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> KwaversResult<OpticalSolveResult> {
        self.optical_solver(scenario.config.optical_model)
            .solve(scenario)
    }
    /// Simulate.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn simulate(
        &self,
        scenario: &PhotoacousticScenario,
    ) -> KwaversResult<(PhotoacousticSimulation, PhotoacousticValidationReport)> {
        let _workspace = self.workspace_for(scenario);
        let _validation_case = self.validation_case_for(scenario);
        let _benchmark_case = self.benchmark_case_for(scenario);
        let _optical_benchmark = OpticalBenchmarkCase {
            name: "optical_forward_stage",
            model: scenario.config.optical_model,
            grid_size: [scenario.grid.nx, scenario.grid.ny, scenario.grid.nz],
        };
        let _acoustic_benchmark = AcousticBenchmarkCase {
            name: "acoustic_forward_stage",
            grid_size: [scenario.grid.nx, scenario.grid.ny, scenario.grid.nz],
            num_time_steps: scenario.config.acoustic.num_time_steps,
        };
        let _reconstruction_benchmark = ReconstructionBenchmarkCase {
            name: "reconstruction_stage",
            geometry: "dispatch",
            grid_size: [scenario.grid.nx, scenario.grid.ny, scenario.grid.nz],
        };
        let _source_validation = SourceValidationCase {
            name: "thermoelastic_source_stage",
            expected_relation: "p0 = Gamma * mu_a * Phi",
        };
        let optical = self.compute_fluence(scenario)?;
        let initial_pressure = self
            .source_model
            .compute_initial_pressure(scenario, &optical.fluence)?;
        let (pressure_fields, time_points) =
            self.acoustic_model.propagate(scenario, &initial_pressure)?;
        let signals =
            self.reconstruction_model
                .sample_signals(scenario, &pressure_fields, &time_points)?;
        let reconstruction = self.reconstruction_model.reconstruct(scenario, &signals)?;

        let simulation = PhotoacousticSimulation {
            optical_fluence: optical.fluence,
            initial_pressure,
            pressure_fields,
            time_points,
            signals,
            reconstruction,
        };
        let report = validate_photoacoustic_simulation(scenario, &simulation)?;
        Ok((simulation, report))
    }
}
