use super::{OpticalForwardModel, OpticalSolveResult, OpticalWorkspace};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::imaging::photoacoustic::{
    IlluminationGeometry, MonteCarloModelConfig, OpticalModel, PhotoacousticScenario,
};
use kwavers_physics::optics::monte_carlo::config::SimulationConfig;
use kwavers_physics::optics::monte_carlo::result::MCResult;
use kwavers_physics::optics::monte_carlo::solver::MonteCarloSolver;
use kwavers_physics::optics::monte_carlo::source::PhotonSource;
use ndarray::Array3;

/// Monte Carlo optical transport solver for the canonical photoacoustic vertical.
///
/// # Governing Equation
/// This stage solves radiative transport stochastically by propagating photon
/// packets through the heterogeneous optical map and tallying deposited energy.
///
/// # Numerical Realization
/// The retained Monte Carlo solver owns photon tracking; this adapter performs
/// scenario-to-solver translation and normalizes the resulting fluence field
/// against the incident surface fluence.
#[derive(Debug, Default)]
pub struct MonteCarloOpticalSolver;

impl MonteCarloOpticalSolver {
    fn photon_source(illumination: IlluminationGeometry) -> PhotonSource {
        match illumination {
            IlluminationGeometry::PencilBeam {
                origin_m,
                direction,
            } => PhotonSource::pencil_beam(origin_m, direction),
            IlluminationGeometry::IsotropicPoint { origin_m } => PhotonSource::isotropic(origin_m),
        }
    }

    fn mc_config(config: &MonteCarloModelConfig) -> SimulationConfig {
        SimulationConfig::default()
            .num_photons(config.photon_count)
            .max_steps(config.max_steps)
            .russian_roulette_threshold(config.russian_roulette_threshold)
            .boundary_reflection(config.boundary_reflection)
    }
}

impl OpticalForwardModel for MonteCarloOpticalSolver {
    fn model_kind(&self) -> OpticalModel {
        OpticalModel::MonteCarlo
    }

    fn solve(&self, scenario: &PhotoacousticScenario) -> KwaversResult<OpticalSolveResult> {
        let mut workspace = OpticalWorkspace::new(scenario.optical_map.dimensions);
        let mc_solver = MonteCarloSolver::new(scenario.grid.clone(), scenario.optical_map.clone());
        let source = Self::photon_source(scenario.config.illumination);
        let result = mc_solver
            .simulate(&source, &Self::mc_config(&scenario.config.monte_carlo))
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        workspace.fluence = mc_result_to_array(&result, scenario.config.incident_fluence_j_m2)?;
        Ok(OpticalSolveResult {
            model: OpticalModel::MonteCarlo,
            fluence: workspace.fluence,
        })
    }
}

fn mc_result_to_array(result: &MCResult, incident_fluence_j_m2: f64) -> KwaversResult<Array3<f64>> {
    let dims = result.dimensions();
    let normalized = result.normalized_fluence();
    let scaled: Vec<f64> = normalized
        .into_iter()
        .map(|value| value * incident_fluence_j_m2)
        .collect();
    Array3::from_shape_vec((dims.nx, dims.ny, dims.nz), scaled).map_err(|e| {
        KwaversError::DimensionMismatch(format!(
            "failed to reshape Monte Carlo fluence into 3D field: {e}"
        ))
    })
}
