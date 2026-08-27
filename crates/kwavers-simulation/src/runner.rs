//! Simulation runner — thin solver dispatch orchestration.
//!
//! The runner delegates to per-solver dispatch modules under
//! [`dispatch`]. All shared types live in
//! [`types`](super::types).

use crate::dispatch;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::config::{FftBackend, SolverType};
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
            SolverType::PSTD if req.fft_backend == FftBackend::Leto => {
                if let Some(thermal) = req.thermal {
                    dispatch::pstd::run_with_thermal(req, sources, thermal)
                } else {
                    dispatch::pstd::run(req, sources)
                }
            }
            SolverType::Hybrid if req.fft_backend == FftBackend::Hephaestus => {
                Err(KwaversError::FeatureNotAvailable(
                    "Hybrid does not yet support FftBackend::Hephaestus".to_owned(),
                ))
            }
            SolverType::Hybrid => {
                if let Some(thermal) = req.thermal {
                    dispatch::pstd::run_with_thermal(req, sources, thermal)
                } else {
                    dispatch::pstd::run(req, sources)
                }
            }
            #[cfg(feature = "gpu")]
            SolverType::PSTD => dispatch::pstd::run_gpu(req, sources),
            #[cfg(not(feature = "gpu"))]
            SolverType::PSTD => Err(KwaversError::FeatureNotAvailable(
                "FftBackend::Hephaestus requires the `gpu` Cargo feature".to_owned(),
            )),

            // ── Frequency-domain ─────────────────────────────────────────────
            SolverType::Helmholtz => dispatch::helmholtz::run(req),
            SolverType::BEM => dispatch::bem::run(req),

            // ── Discontinuous Galerkin ────────────────────────────────────────
            SolverType::DG => dispatch::dg::run(req),

            // ── Elastic ──────────────────────────────────────────────────────
            SolverType::Elastic => dispatch::elastic::run(req),
            SolverType::ElasticPSTD if req.fft_backend == FftBackend::Hephaestus => {
                Err(KwaversError::FeatureNotAvailable(
                    "ElasticPSTD does not yet support FftBackend::Hephaestus".to_owned(),
                ))
            }
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
    use kwavers_solver::config::{FftBackend, SolverType};
    use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
    use kwavers_solver::forward::pstd::config::CompatibilityMode;
    use kwavers_source::GridSource;

    fn gpu_request<'a>(grid: &'a Grid, medium: &'a HomogeneousMedium) -> SimulationRunRequest<'a> {
        SimulationRunRequest {
            grid,
            medium,
            time_steps: 8,
            dt: 1.0e-7,
            solver_type: SolverType::PSTD,
            fft_backend: FftBackend::Hephaestus,
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
            "Feature not available: FftBackend::Hephaestus requires the `gpu` Cargo feature"
        );
    }

    #[test]
    fn hybrid_rejects_hephaestus_instead_of_running_cpu() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.solver_type = SolverType::Hybrid;

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("Hybrid must not silently ignore the selected FFT backend");

        assert_eq!(
            error.to_string(),
            "Feature not available: Hybrid does not yet support FftBackend::Hephaestus"
        );
    }

    #[test]
    fn elastic_pstd_rejects_hephaestus_instead_of_running_cpu() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.solver_type = SolverType::ElasticPSTD;

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("ElasticPSTD must not silently ignore the selected FFT backend");

        assert_eq!(
            error.to_string(),
            "Feature not available: ElasticPSTD does not yet support FftBackend::Hephaestus"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_rejects_unsupported_thermal_coupling_before_device_acquisition() {
        use crate::configs::ThermalConfig;

        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let thermal = ThermalConfig::default();
        let mut request = gpu_request(&grid, &medium);
        request.thermal = Some(&thermal);

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("Hephaestus PSTD does not yet implement thermal coupling");

        assert_eq!(
            error.to_string(),
            "Feature not available: Hephaestus PSTD does not support coupled thermal propagation"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_rejects_sensor_shape_before_device_acquisition() {
        use leto::Array3;

        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.sensor_mask = Some(Array3::from_elem([8, 7, 8], false));

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("sensor geometry must fail before device acquisition");

        assert_eq!(
            error.to_string(),
            "Dimension mismatch: sensor_mask shape [8, 7, 8]; expected [8, 8, 8]"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a GPU device; run on the scheduled self-hosted GPU job"]
    fn gpu_selection_executes_hephaestus_without_cpu_fallback() {
        use leto::{Array2, Array3};

        let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.time_steps = 3;
        let mut source_mask = Array3::zeros([7, 4, 3]);
        source_mask[[3, 2, 1]] = 1.0;
        request.grid_source = GridSource {
            p_mask: Some(source_mask),
            p_signal: Some(
                Array2::from_shape_vec([1, 3], vec![0.0, 1.0, 0.0])
                    .expect("source signal matches the requested steps"),
            ),
            ..GridSource::new_empty()
        };
        let mut sensor_mask = Array3::from_elem([7, 4, 3], false);
        sensor_mask[[3, 2, 1]] = true;
        request.sensor_mask = Some(sensor_mask);

        let result = SimulationRunner::run(&request, Vec::new())
            .expect("Hephaestus executes the selected PSTD request");

        assert_eq!(result.sensor_data.shape(), [1, 3]);
        assert!(result.sensor_data.iter().all(|value| value.is_finite()));
        assert!(result.sensor_data.iter().any(|&value| value != 0.0));
    }
}
