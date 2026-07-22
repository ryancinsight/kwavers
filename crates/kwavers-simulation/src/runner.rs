//! Simulation runner — thin solver dispatch orchestration.
//!
//! The runner delegates to per-solver dispatch modules under
//! [`dispatch`]. All shared types live in
//! [`types`](super::types).

use crate::dispatch;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::config::SolverType;
use kwavers_source::Source as KwaversSource;

use crate::types::{SimulationRunRequest, SimulationRunResult};

/// Owns solver dispatch for simulation requests.
#[derive(Debug, Default)]
pub struct SimulationRunner;

impl SimulationRunner {
    /// Dispatch and run a simulation based on the request config.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` for unsupported solver/config combos.
    /// - Propagates solver creation and runtime errors.
    pub fn run(
        req: &SimulationRunRequest<'_>,
        sources: Vec<Box<dyn KwaversSource>>,
    ) -> KwaversResult<SimulationRunResult> {
        match req.solver_type {
            // ── Time-domain acoustic ──────────────────────────────────────────
            SolverType::FDTD => dispatch::fdtd::run(req, sources),
            SolverType::PSTD | SolverType::Hybrid => {
                if let Some(thermal) = req.thermal {
                    dispatch::pstd::run_with_thermal(req, sources, thermal)
                } else {
                    dispatch::pstd::run(req, sources)
                }
            }
            #[cfg(feature = "gpu")]
            SolverType::PstdGpu => Err(KwaversError::FeatureNotAvailable(
                "SimulationRunner does not yet map its full request contract onto GpuPstdSimulationAdapter; construct the GPU adapter directly instead of silently running CPU PSTD".to_owned(),
            )),
            #[cfg(not(feature = "gpu"))]
            SolverType::PstdGpu => Err(KwaversError::FeatureNotAvailable(
                "SolverType::PstdGpu requires the `gpu` Cargo feature".to_owned(),
            )),

            // ── Frequency-domain ─────────────────────────────────────────────
            SolverType::Helmholtz => dispatch::helmholtz::run(req),
            SolverType::BEM => dispatch::bem::run(req),

            // ── Discontinuous Galerkin ────────────────────────────────────────
            SolverType::DG => dispatch::dg::run(req),

            // ── Elastic ──────────────────────────────────────────────────────
            SolverType::Elastic => dispatch::elastic::run(req),
            SolverType::ElasticPSTD => dispatch::elastic_pstd::run(req),

            // ── Nonlinear acoustics ───────────────────────────────────────────
            SolverType::Nonlinear => dispatch::nonlinear::run(req),

            // ── Poroelastic ───────────────────────────────────────────────────
            SolverType::Poroelastic => dispatch::poroelastic::run(req),

            // ── Analytical ────────────────────────────────────────────────────
            SolverType::RayleighSommerfeld => dispatch::rayleigh_sommerfeld::run(req),

            // ── Unsupported — these are plugin/external types ─────────────────
            SolverType::KSpace => Err(KwaversError::InvalidInput(
                "Simulation.run expects SolverType::PSTD with k-space correction; use KSpaceCorrectionMode".into(),
            )),
            SolverType::DiscontinuousGalerkin | SolverType::FEM => Err(KwaversError::InvalidInput(
                "Use SolverType::DG (DiscontinuousGalerkin) or SolverType::Helmholtz (FEM)".into(),
            )),
            SolverType::Auto => Err(KwaversError::InvalidInput(
                "SolverType::Auto is not supported — select a concrete solver type".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationRunner;
    use crate::types::SimulationRunRequest;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_solver::config::SolverType;
    use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
    use kwavers_solver::forward::pstd::config::CompatibilityMode;
    use kwavers_source::GridSource;

    fn gpu_request<'a>(grid: &'a Grid, medium: &'a HomogeneousMedium) -> SimulationRunRequest<'a> {
        SimulationRunRequest {
            grid,
            medium,
            time_steps: 8,
            dt: 1.0e-7,
            solver_type: SolverType::PstdGpu,
            pml: None,
            helmholtz: None,
            nonlinear: None,
            thermal: None,
            poroelastic: None,
            compatibility_mode: CompatibilityMode::Optimal,
            kspace_correction: KSpaceCorrectionMode::None,
            axisymmetric: false,
            grid_source: GridSource::default(),
            sensor_mask: None,
            transducer_ordered_indices: None,
            record_modes: Vec::new(),
            record_start_index: 0,
            transducers_for_rs: &[],
            elastic_velocity_source: None,
            elastic_ivp_axis: None,
        }
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn gpu_selection_without_feature_returns_feature_error() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let request = gpu_request(&grid, &medium);

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("GPU solver selection must not silently dispatch CPU PSTD");

        assert_eq!(
            error.to_string(),
            "Feature not available: SolverType::PstdGpu requires the `gpu` Cargo feature"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_without_runner_mapping_returns_feature_error() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let request = gpu_request(&grid, &medium);

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("GPU solver selection must not silently dispatch CPU PSTD");

        assert_eq!(
            error.to_string(),
            "Feature not available: SimulationRunner does not yet map its full request contract onto GpuPstdSimulationAdapter; construct the GPU adapter directly instead of silently running CPU PSTD"
        );
    }
}