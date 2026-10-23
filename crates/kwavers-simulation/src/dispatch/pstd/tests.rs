use super::prepare_solver;
use crate::configs::NonlinearConfig;
use crate::types::SimulationRunRequest;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
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
