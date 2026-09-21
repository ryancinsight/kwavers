use crate::forward::pstd::config::PSTDConfig;
use crate::forward::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::error::KwaversError;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

fn construct(nt: usize) -> Result<PSTDSolver, KwaversError> {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).expect("valid 8^3 grid");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        nt,
        ..Default::default()
    };
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
