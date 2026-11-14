use super::{power_law_absorption_mode, prepare_solver, run};
use crate::configs::NonlinearConfig;
use crate::types::SimulationRunRequest;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::heterogeneous::HeterogeneousMedium;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::config::SolverType;
use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers_solver::forward::pstd::config::{CompatibilityMode, PowerLawDispersion};
use kwavers_source::grid_source::GridSource;

fn request<'a>(
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

/// The requested dispersion reaches the PSTD solver: at `y = 1` the causal
/// form is rejected with its alternative named, and absorption without
/// dispersion builds.
#[test]
fn power_law_dispersion_reaches_the_pstd_solver() {
    let grid = Grid::uniform(16, 1e-3).expect("valid uniform grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let unit_exponent = NonlinearConfig::default()
        .with_alpha_coeff(0.5)
        .with_alpha_power(1.0);

    let causal = request(&grid, &medium, &unit_exponent);
    let Err(KwaversError::Validation(ValidationError::ConstraintViolation { message })) =
        prepare_solver(&causal, Vec::new())
    else {
        panic!("the causal form must reject y = 1");
    };
    assert!(
        message.contains("PowerLawDispersion::Omitted"),
        "the rejection names its alternative: {message}"
    );

    let omitted_config = unit_exponent.with_power_law_dispersion(PowerLawDispersion::Omitted);
    let omitted = request(&grid, &medium, &omitted_config);
    let (_, solver_grid) =
        prepare_solver(&omitted, Vec::new()).expect("absorption without dispersion admits y = 1");
    assert_eq!(
        (solver_grid.nx, solver_grid.ny, solver_grid.nz),
        (16, 16, 16)
    );
}

fn heterogeneous_medium(absorbing_voxel: Option<[usize; 3]>) -> HeterogeneousMedium {
    const N: usize = 4;
    let mut medium = HeterogeneousMedium::new_acoustic_only(N, N, N, true);
    medium.sound_speed = leto::Array3::from_elem((N, N, N), 1500.0);
    medium.density = leto::Array3::from_elem((N, N, N), 1000.0);
    if let Some(index) = absorbing_voxel {
        medium.absorption[index] = 0.5;
    }
    medium
}

/// Absorption that is zero at the origin and positive elsewhere is
/// absorption: the solver is told to read the medium per voxel.
#[test]
fn a_medium_absorbing_away_from_the_origin_is_not_lossless() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
    let medium = heterogeneous_medium(Some([2, 1, 3]));
    assert_eq!(medium.absorption[[0, 0, 0]], 0.0);
    assert_eq!(
        power_law_absorption_mode(&medium, &grid, 0.0, 1.5),
        AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.5
        }
    );
}

#[test]
fn a_medium_without_absorption_is_lossless() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
    let medium = heterogeneous_medium(None);
    assert_eq!(
        power_law_absorption_mode(&medium, &grid, 0.0, 1.5),
        AbsorptionMode::Lossless
    );
}

#[test]
fn an_explicit_coefficient_outranks_the_medium() {
    let grid = Grid::new(4, 4, 4, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
    let medium = heterogeneous_medium(None);
    assert_eq!(
        power_law_absorption_mode(&medium, &grid, 0.75, 1.1),
        AbsorptionMode::PowerLaw {
            alpha_coeff: Some(0.75),
            alpha_power: 1.1
        }
    );
}

/// One pressure source at the grid centre of a 7x4x3 grid, so the run
/// propagates a field the absent sensors do not record.
fn centre_source() -> GridSource {
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

/// A request without a sensor mask, and one whose mask has no true element,
/// run and record nothing: `(0, time_steps)` for the three steps, and
/// `(0, 2)` when `record_start_index = 2` skips one sample, the columns
/// the same request with sensors would record. The result carries no other
/// record.
#[test]
fn a_request_without_sensors_returns_an_empty_record() {
    let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let nonlinear = NonlinearConfig::default();
    let no_true_element = leto::Array3::from_elem([7, 4, 3], false);

    for (mask, record_start_index, columns) in [
        (None, 0, 3),
        (Some(no_true_element.clone()), 0, 3),
        (Some(no_true_element), 2, 2),
    ] {
        let request = SimulationRunRequest {
            time_steps: 3,
            dt: 1.0e-7,
            grid_source: centre_source(),
            sensor_mask: mask,
            record_start_index,
            ..request(&grid, &medium, &nonlinear)
        };

        let result = run(&request, Vec::new()).expect("a request without sensors runs");

        assert_eq!(result.sensor_data.shape(), [0, columns]);
        assert!(result.ux_data.is_none() && result.uy_data.is_none() && result.uz_data.is_none());
        assert!(result.ix_data.is_none() && result.iy_data.is_none() && result.iz_data.is_none());
        assert!(result.stats.is_none() && result.velocity_stats.is_none());
    }
}

/// One request without a sensor mask through both FFT backends: each returns
/// the record with no rows, and neither returns an error.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU device"]
fn both_backends_return_an_empty_record_for_a_request_without_sensors() {
    use crate::runner::SimulationRunner;
    use kwavers_solver::config::FftBackend;

    let grid = Grid::new(7, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.0, 0.0, &grid);
    let nonlinear = NonlinearConfig::default();

    for fft_backend in [FftBackend::Leto, FftBackend::Hephaestus] {
        let request = SimulationRunRequest {
            fft_backend,
            time_steps: 3,
            dt: 1.0e-7,
            grid_source: centre_source(),
            ..request(&grid, &medium, &nonlinear)
        };

        let result =
            SimulationRunner::run(&request, Vec::new()).expect("a request without sensors runs");

        assert_eq!(result.sensor_data.shape(), [0, 3], "{fft_backend:?}");
    }
}
