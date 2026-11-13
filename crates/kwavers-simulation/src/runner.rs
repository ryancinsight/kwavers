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
    /// - Returns `KwaversError::FeatureNotAvailable` for `SolverType::Hybrid`
    ///   under every `FftBackend`: the runner has no path that hands a
    ///   request's grid source and sensor recording to `HybridSolver`, and it
    ///   does not answer with PSTD.
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
            SolverType::Hybrid => Err(KwaversError::FeatureNotAvailable(
                "SolverType::Hybrid is not supported by SimulationRunner: HybridSolver builds \
                 its inner solvers with an empty grid source and keeps them private, so the \
                 request's grid source and sensor recording cannot reach them; select \
                 SolverType::PSTD or SolverType::FDTD"
                    .to_owned(),
            )),
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
    use kwavers_source::{GridSource, SourceMode};

    const HYBRID_REFUSAL: &str = "Feature not available: SolverType::Hybrid is not supported by \
         SimulationRunner: HybridSolver builds its inner solvers with an empty grid source and \
         keeps them private, so the request's grid source and sensor recording cannot reach \
         them; select SolverType::PSTD or SolverType::FDTD";

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

    fn leto_request<'a>(grid: &'a Grid, medium: &'a HomogeneousMedium) -> SimulationRunRequest<'a> {
        SimulationRunRequest {
            fft_backend: FftBackend::Leto,
            sensor_mask: Some(leto::Array3::from_elem([8, 8, 8], true)),
            ..gpu_request(grid, medium)
        }
    }

    /// A pressure source of mode `Dirichlet` at the grid centre, for the
    /// backend selected: the Leto solver runs it, and the Hephaestus
    /// provider has no such source.
    fn dirichlet_source_request<'a>(
        grid: &'a Grid,
        medium: &'a HomogeneousMedium,
        fft_backend: FftBackend,
    ) -> SimulationRunRequest<'a> {
        let mut mask = leto::Array3::zeros([8, 8, 8]);
        mask[[4, 4, 4]] = 1.0;
        SimulationRunRequest {
            fft_backend,
            sensor_mask: Some(leto::Array3::from_elem([8, 8, 8], true)),
            grid_source: GridSource {
                p_mask: Some(mask),
                p_signal: Some(
                    leto::Array2::from_shape_vec([1, 8], vec![1.0; 8])
                        .expect("one signal row of eight steps"),
                ),
                p_mode: SourceMode::Dirichlet,
                ..GridSource::new_empty()
            },
            ..gpu_request(grid, medium)
        }
    }

    /// `f64` prints these values as `0`, `-0.0000001`, `NaN` and `inf`.
    #[test]
    fn leto_selection_refuses_a_time_step_that_is_not_finite_and_positive() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        for (dt, printed) in [
            (0.0, "0"),
            (-1.0e-7, "-0.0000001"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "inf"),
        ] {
            let mut request = leto_request(&grid, &medium);
            request.dt = dt;

            let error = SimulationRunner::run(&request, Vec::new())
                .expect_err("the PSTD solver cannot advance by this time step");

            assert_eq!(
                error.to_string(),
                format!("Invalid input: PSTDConfig.dt must be finite and positive; got {printed}")
            );
        }
    }

    #[test]
    fn leto_selection_refuses_zero_time_steps() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = leto_request(&grid, &medium);
        request.time_steps = 0;

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("a PSTD run of zero steps records nothing");

        assert_eq!(
            error.to_string(),
            "Invalid input: PSTD requires time_steps > 0; got 0"
        );
    }

    /// The request the two refusal tests start from runs.
    #[test]
    fn leto_selection_runs_the_request_the_refusals_start_from() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let request = leto_request(&grid, &medium);

        let result = SimulationRunner::run(&request, Vec::new())
            .expect("eight steps at 100 ns on an 8x8x8 grid run");

        assert_eq!(result.sensor_data.shape(), [512, 8]);
    }

    /// The request of `gpu_selection_reports_the_provider_source_rejection`
    /// runs on the Leto solver, so a fall-back to it would return a record.
    #[test]
    fn leto_selection_runs_a_dirichlet_pressure_source() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let request = dirichlet_source_request(&grid, &medium, FftBackend::Leto);

        let result = SimulationRunner::run(&request, Vec::new())
            .expect("the Leto solver runs a Dirichlet pressure source");

        assert_eq!(result.sensor_data.shape(), [512, 8]);
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

    /// The refusal names no FFT backend: it holds for every `FftBackend`.
    #[test]
    fn hybrid_rejects_hephaestus_instead_of_running_cpu() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.solver_type = SolverType::Hybrid;

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("Hybrid must not silently ignore the selected FFT backend");

        assert_eq!(error.to_string(), HYBRID_REFUSAL);
    }

    /// The request of `leto_selection_runs_the_request_the_refusals_start_from`
    /// changed only in its solver: the PSTD dispatch answers it with a record,
    /// so a Hybrid selection handed to that dispatch fails this test.
    #[test]
    fn hybrid_selection_is_refused_instead_of_answered_by_pstd() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = leto_request(&grid, &medium);
        request.solver_type = SolverType::Hybrid;

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("the PSTD dispatch must not answer a Hybrid request");

        assert_eq!(error.to_string(), HYBRID_REFUSAL);
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

    /// The provider refuses zero steps before it acquires a device. The
    /// Leto dispatch refuses them with another message, so handing back its
    /// outcome in place of the refusal fails this test.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_reports_the_provider_rejection() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.time_steps = 0;
        request.sensor_mask = Some(leto::Array3::from_elem([8, 8, 8], true));

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("zero time steps violate the GPU PSTD batch contract");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD requires time_steps > 0 and finite positive dt; got steps=0 dt=0.0000001"
        );
    }

    /// The provider refuses two absorption exponents while it prepares the
    /// medium snapshot; the Leto dispatch returns `Ok` for this request.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_reports_the_provider_medium_rejection() {
        use kwavers_medium::heterogeneous::HeterogeneousMedium;

        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let uniform = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut medium = HeterogeneousMedium::new(8, 8, 8, false);
        medium
            .sound_speed
            .iter_mut()
            .for_each(|value| *value = 1_500.0);
        medium.density.iter_mut().for_each(|value| *value = 1_000.0);
        medium.absorption.iter_mut().for_each(|value| *value = 0.2);
        medium.alpha_power.iter_mut().for_each(|value| *value = 1.3);
        medium.alpha_power[[0, 0, 1]] = 1.7;
        let request = SimulationRunRequest {
            medium: &medium,
            sensor_mask: Some(leto::Array3::from_elem([8, 8, 8], true)),
            ..gpu_request(&grid, &uniform)
        };

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("two absorption exponents violate the GPU PSTD medium contract");

        assert_eq!(
            error.to_string(),
            "Feature not available: Hephaestus PSTD requires one active absorption exponent; medium flat index 1 uses 1.7"
        );
    }

    /// The provider has no Dirichlet pressure source and refuses one while it
    /// assembles the source schedule, before it acquires a device. The Leto
    /// dispatch runs this request (`leto_selection_runs_a_dirichlet_pressure_source`),
    /// so a fall-back to it on this refusal returns a record.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_selection_reports_the_provider_source_rejection() {
        let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let request = dirichlet_source_request(&grid, &medium, FftBackend::Hephaestus);

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("the GPU PSTD provider has no Dirichlet pressure source");

        assert_eq!(
            error.to_string(),
            "Invalid input: GPU PSTD does not support Dirichlet pressure sources"
        );
    }

    /// The provider requests a `max_buffer_size` of 128 MiB (134217728
    /// bytes). The packed PML buffer of a 256x256x256 grid holds three `f32`
    /// values for each of 16777216 points: 201326592 bytes. With the sensor
    /// the Leto dispatch returns `Ok` for this request, so handing back its
    /// outcome in place of the refusal fails this test.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a GPU device"]
    fn gpu_selection_reports_a_plan_beyond_device_capacity() {
        use kwavers_core::error::KwaversError;

        let grid = Grid::new(256, 256, 256, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let mut request = gpu_request(&grid, &medium);
        request.time_steps = 1;
        let mut sensor_mask = leto::Array3::from_elem([256, 256, 256], false);
        sensor_mask[[128, 128, 128]] = true;
        request.sensor_mask = Some(sensor_mask);

        let error = SimulationRunner::run(&request, Vec::new())
            .expect_err("the packed PML buffer exceeds the buffer size the provider requests");

        assert!(
            matches!(error, KwaversError::ResourceLimitExceeded { .. }),
            "{error:?}"
        );
        assert_eq!(
            error.to_string(),
            "Resource limit exceeded: GPU PSTD buffer `pml_xyz` needs 201326592 bytes; the device enables max_buffer_size = 134217728 bytes"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a GPU device"]
    fn gpu_selection_executes_a_hephaestus_request_on_a_device() {
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
