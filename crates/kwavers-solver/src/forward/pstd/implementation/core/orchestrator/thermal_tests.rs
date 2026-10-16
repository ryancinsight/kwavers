use crate::forward::pstd::config::{PSTDConfig, PowerLawDispersion};
use crate::forward::pstd::PSTDSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::{
    power_law_db_cm_to_np_omega_m, AbsorptionMode,
};
use kwavers_source::GridSource;
use std::f64::consts::TAU;

/// The heat-source absorption scales `α_SI` by `ω^y` with the exponent the
/// kernel resolved, here the medium's `y = 1`, not the configured 1.5.
#[test]
fn absorption_at_frequency_uses_the_resolved_exponent() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    medium.set_acoustic_properties(0.7, 1.0, 5.0).unwrap();
    let config = PSTDConfig {
        nt: 2,
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.5,
        },
        power_law_dispersion: PowerLawDispersion::Omitted,
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid, &medium, GridSource::new_empty()).unwrap();
    let omega = TAU * 2.0e6;
    solver.populate_alpha_np_m_at_frequency(omega);
    // ω¹ is exact, so α(ω) = α_SI · ω holds bit for bit.
    let expected = power_law_db_cm_to_np_omega_m(0.7, 1.0) * omega;
    let alpha = solver.alpha_np_m().unwrap();
    assert!(alpha.iter().all(|&value| value == expected));
}
