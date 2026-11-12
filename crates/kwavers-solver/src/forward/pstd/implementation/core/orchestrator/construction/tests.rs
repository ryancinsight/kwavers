use crate::forward::pstd::config::PSTDConfig;
use crate::forward::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

fn construct(nt: usize) -> Result<PSTDSolver, KwaversError> {
    construct_with(PSTDConfig {
        nt,
        ..Default::default()
    })
}

fn construct_with(config: PSTDConfig) -> Result<PSTDSolver, KwaversError> {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("valid 8^3 grid");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    PSTDSolver::new(config, grid, &medium, GridSource::new_empty())
}

#[test]
fn a_step_count_with_no_room_for_the_initial_record_is_rejected() {
    // `nt + 1` recorded states: `usize::MAX` overflowed and panicked here.
    let Err(KwaversError::InvalidInput(message)) = construct(usize::MAX) else {
        panic!("nt = usize::MAX must be rejected as invalid input");
    };
    assert!(message.contains("PSTDConfig.nt"), "{message}");
    assert!(message.contains(&usize::MAX.to_string()), "{message}");
}

#[test]
fn the_largest_representable_step_count_is_accepted() {
    // The boundary: one below the limit leaves exactly room for the extra record.
    construct(usize::MAX - 1).expect("nt = usize::MAX - 1 leaves room for nt + 1 records");
}

/// `f64` prints zero as `0`, negative zero as `-0`, and the non-finite
/// values as `NaN`, `inf` and `-inf`; the refusal carries that text.
#[test]
fn a_time_step_that_is_not_finite_and_positive_is_rejected() {
    for (dt, printed) in [
        (0.0, "0"),
        (-0.0, "-0"),
        (-1.0e-7, "-0.0000001"),
        (f64::NAN, "NaN"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
    ] {
        let Err(KwaversError::InvalidInput(message)) = construct_with(PSTDConfig {
            dt,
            ..Default::default()
        }) else {
            panic!("dt = {dt} must be rejected as invalid input");
        };
        assert_eq!(
            message,
            format!("PSTDConfig.dt must be finite and positive; got {printed}")
        );
    }
}

/// The time step is checked before the step count, which the constructor
/// reads only when it sizes the recorder.
#[test]
fn the_time_step_is_checked_before_the_step_count() {
    let Err(KwaversError::InvalidInput(message)) = construct_with(PSTDConfig {
        nt: usize::MAX,
        dt: 0.0,
        ..Default::default()
    }) else {
        panic!("dt = 0 with nt = usize::MAX must be rejected as invalid input");
    };
    assert_eq!(message, "PSTDConfig.dt must be finite and positive; got 0");
}

#[test]
fn the_smallest_positive_time_step_is_accepted() {
    construct_with(PSTDConfig {
        dt: f64::MIN_POSITIVE,
        ..Default::default()
    })
    .expect("a finite positive dt passes the time step check");
}
