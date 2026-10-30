use crate::forward::pstd::config::{PSTDConfig, PowerLawDispersion};
use crate::forward::pstd::PSTDSolver;
use kwavers_core::constants::acoustic_parameters::DB_TO_NP;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_omega_m;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
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

/// With a per-voxel exponent each voxel scales by its own `ω^y(r)`.
#[test]
fn absorption_at_frequency_scales_each_voxel_by_its_exponent() {
    use kwavers_medium::heterogeneous::HeterogeneousMedium;

    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let background = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let mut medium = HeterogeneousMedium::from_homogeneous(&background, &grid).unwrap();
    medium.use_trilinear_interpolation = false;
    // Np/m storage (storage convention in `kwavers_medium::core`): the 0.7
    // dB/cm prefactor converted once, so `alpha_coefficient` keeps reporting
    // the same SI values and the expectations below are unchanged.
    medium.absorption.fill(0.7 * DB_TO_NP * 100.0);
    medium.alpha0.fill(0.7 * DB_TO_NP * 100.0);
    let exponent_at = |i: usize| if i < 4 { 1.3 } else { 1.7 };
    for ([i, _, _], value) in medium.alpha_power.indexed_iter_mut().unwrap() {
        *value = exponent_at(i);
    }
    let config = PSTDConfig {
        nt: 2,
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.3,
        },
        ..PSTDConfig::default()
    };
    let mut solver = PSTDSolver::new(config, grid, &medium, GridSource::new_empty()).unwrap();
    assert!(solver.absorption.as_ref().unwrap().strata.is_some());
    let omega = TAU * 2.0e6;
    solver.populate_alpha_np_m_at_frequency(omega);
    // α_SI and ω^y are formed by the same operations as in the solver, so each
    // voxel matches bit for bit; a configured-exponent scaling would put
    // ω^1.3 on the y = 1.7 half.
    let alpha = solver.alpha_np_m().unwrap();
    for ([i, _, _], &value) in alpha.indexed_iter() {
        let y = exponent_at(i);
        assert_eq!(value, power_law_db_cm_to_np_omega_m(0.7, y) * omega.powf(y));
    }
}
