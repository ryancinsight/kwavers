//! Request and source fixtures shared by the dispatch tests.

use crate::configs::NonlinearConfig;
use crate::types::SimulationRunRequest;
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_solver::config::SolverType;
use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers_solver::forward::pstd::config::CompatibilityMode;
use kwavers_source::grid_source::GridSource;

/// A four-step request with no sensor, source, PML or recording mode,
/// dispatched to the PSTD solver; a test overrides the fields it exercises.
pub(crate) fn request<'a>(
    grid: &'a Grid,
    medium: &'a HomogeneousMedium,
    nonlinear: &'a NonlinearConfig,
) -> SimulationRunRequest<'a> {
    SimulationRunRequest {
        grid,
        medium,
        time_steps: 4,
        dt: 1e-8,
        solver_type: SolverType::PSTD,
        fft_backend: kwavers_solver::config::FftBackend::Leto,
        pml: None,
        helmholtz: None,
        nonlinear: Some(nonlinear),
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

/// One pressure source at the grid centre of a 7x4x3 grid, so the run
/// propagates a field the absent sensors do not record.
pub(crate) fn centre_source() -> GridSource {
    let mut mask = leto::Array3::zeros([7, 4, 3]);
    mask[[3, 2, 1]] = 1.0;
    GridSource {
        p_mask: Some(mask),
        p_signal: Some(
            leto::Array2::from_shape_vec([1, 3], vec![0.0, 1.0, 0.0])
                .expect("one signal row of three steps"),
        ),
        ..GridSource::new_empty()
    }
}
