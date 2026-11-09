//! Power-law exponent validation: the configured exponent is checked only
//! where an absorbing voxel uses it (KW-PSTD-UNIT-EXPONENT-BACKENDS).

use super::{test_k_mag, uniform_symbols};
use crate::forward::pstd::config::{PSTDConfig, PowerLawDispersion};
use crate::forward::pstd::physics::absorption::init::initialize_absorption_operators;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_omega_m;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;

/// k-Wave `alpha_mode = 'no_dispersion'`: the absorption term `τ` is unchanged,
/// the dispersion term `η` vanishes, and `y = 1` (rejected by the causal form,
/// where `tan(πy/2)` diverges) becomes admissible.
#[test]
fn omitted_dispersion_keeps_tau_and_zeroes_eta() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let kernel_for = |alpha_power: f64, dispersion: PowerLawDispersion| {
        let config = PSTDConfig {
            absorption_mode: AbsorptionMode::PowerLaw {
                alpha_coeff: Some(0.7),
                alpha_power,
            },
            power_law_dispersion: dispersion,
            ..PSTDConfig::default()
        };
        initialize_absorption_operators(
            &config,
            &grid,
            &medium,
            &k_mag,
            MHZ_TO_HZ,
            SOUND_SPEED_WATER_SIM,
        )
    };

    let causal = kernel_for(1.5, PowerLawDispersion::Causal)
        .unwrap()
        .unwrap();
    let omitted = kernel_for(1.5, PowerLawDispersion::Omitted)
        .unwrap()
        .unwrap();
    assert_eq!(omitted.tau[[3, 4, 5]], causal.tau[[3, 4, 5]]);
    assert_ne!(causal.eta[[3, 4, 5]], 0.0);
    assert!(omitted.eta.iter().all(|&value| value == 0.0));

    let rejection = kernel_for(1.0, PowerLawDispersion::Causal)
        .err()
        .expect("the causal form rejects y = 1");
    let KwaversError::Validation(ValidationError::ConstraintViolation { message }) = rejection
    else {
        panic!("expected a constraint violation, got {rejection:?}");
    };
    assert_eq!(
        message,
        "alpha_power must not be 1.0 with causal dispersion, where tan(pi*y/2) is singular; set PowerLawDispersion::Omitted to absorb without dispersion"
    );
    let unit = kernel_for(1.0, PowerLawDispersion::Omitted)
        .unwrap()
        .unwrap();
    // y = 1: τ = −2 α₀ c₀^(y−1) = −2 α₀, exact in floating point because c₀⁰ = 1.
    let alpha_0_si = power_law_db_cm_to_np_omega_m(0.7, 1.0);
    assert!(unit.tau.iter().all(|&value| value == -2.0 * alpha_0_si));
    assert!(unit.eta.iter().all(|&value| value == 0.0));
}

/// The configured exponent is validated only where it is used: a medium that
/// supplies every voxel's exponent leaves an unused configured `y = 1` alone,
/// so causal dispersion accepts it (KW-PSTD-UNIT-EXPONENT-BACKENDS).
#[test]
fn unused_configured_unit_exponent_is_accepted_under_causal_dispersion() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    medium.set_acoustic_properties(0.7, 1.5, 5.0).unwrap();
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.0,
        },
        power_law_dispersion: PowerLawDispersion::Causal,
        ..PSTDConfig::default()
    };
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .expect("an unused configured y = 1 is accepted")
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");
    // The operators carry the medium's y = 1.5, not the configured 1.0:
    // ∇₂ = |k|^0.5 off the DC bin, where y = 1 would give |k|⁰ = 1.
    assert_eq!(
        uniform_symbols(&kernel).1[[1, 0, 0]],
        k_mag[[1, 0, 0]].powf(0.5)
    );
}

/// A degenerate medium exponent falls back to the configured one, which is
/// validated at that use: falling back to `y = 1` under causal dispersion
/// still rejects, with the same constraint message.
#[test]
fn fallback_to_a_configured_unit_exponent_rejects_under_causal_dispersion() {
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
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.0,
        },
        power_law_dispersion: PowerLawDispersion::Causal,
        ..PSTDConfig::default()
    };
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let rejection = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .err()
    .expect("the fallback to y = 1 rejects under causal dispersion");
    let KwaversError::Validation(ValidationError::ConstraintViolation { message }) = rejection
    else {
        panic!("expected a constraint violation, got {rejection:?}");
    };
    assert_eq!(
        message,
        "alpha_power must not be 1.0 with causal dispersion, where tan(pi*y/2) is singular; set PowerLawDispersion::Omitted to absorb without dispersion"
    );
}

/// Falling back to a usable configured exponent resolves the medium
/// degeneracy: the operators carry the configured y = 1.5.
#[test]
fn fallback_to_a_usable_configured_exponent_resolves_the_medium_degeneracy() {
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
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.5,
        },
        power_law_dispersion: PowerLawDispersion::Causal,
        ..PSTDConfig::default()
    };
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .expect("the fallback to a usable y = 1.5 is accepted")
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");
    assert_eq!(
        uniform_symbols(&kernel).1[[1, 0, 0]],
        k_mag[[1, 0, 0]].powf(0.5)
    );
}

/// An explicit zero coefficient absorbs nothing, so the configured exponent
/// is never used and needs no validation.
#[test]
fn zero_explicit_coefficient_leaves_the_configured_exponent_unvalidated() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: Some(0.0),
            alpha_power: 1.0,
        },
        power_law_dispersion: PowerLawDispersion::Causal,
        ..PSTDConfig::default()
    };
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .expect("an unused y = 1 with zero absorption is accepted")
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");
    assert!(kernel.tau.iter().all(|&value| value == 0.0));
    assert!(kernel.eta.iter().all(|&value| value == 0.0));
}

/// A medium-supplied `y = 1` is a valid exponent once dispersion is omitted;
/// only the causal form treats it as degenerate and falls back to the
/// configured exponent.
#[test]
fn omitted_dispersion_takes_unit_exponent_from_the_medium() {
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
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.5,
        },
        power_law_dispersion: PowerLawDispersion::Omitted,
        ..PSTDConfig::default()
    };
    let k_mag = test_k_mag(8, 8, 8, 1e-3);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .unwrap()
    .unwrap();
    // The operators carry the same y = 1: ∇₂ = |k|⁰ = 1 and ∇₁ = |k|⁻¹ off the
    // DC bin, where the configured y = 1.5 would give |k|^0.5 and |k|^-0.5.
    assert_eq!(uniform_symbols(&kernel).1[[1, 0, 0]], 1.0);
    assert_eq!(
        uniform_symbols(&kernel).0[[1, 0, 0]],
        k_mag[[1, 0, 0]].powf(-1.0)
    );
    // y = 1 from the medium: τ = −2 α₀ c₀⁰ = −2 α₀ exactly. The configured
    // y = 1.5 would instead give −2 α₀(1.5) c₀^0.5.
    let alpha_0_si = power_law_db_cm_to_np_omega_m(0.7, 1.0);
    assert!(kernel.tau.iter().all(|&value| value == -2.0 * alpha_0_si));
    assert!(kernel.eta.iter().all(|&value| value == 0.0));
}
