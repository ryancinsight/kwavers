use super::{test_k_mag, zeros_k_mag};
use crate::core::constants::cavitation::VISCOSITY_WATER;
use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::solver::forward::pstd::config::PSTDConfig;
use crate::solver::forward::pstd::physics::absorption::init::initialize_absorption_operators;

#[test]
fn test_power_law_initialization() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
    medium.set_acoustic_properties(0.75, 1.5, 5.0).unwrap();
    let config = PSTDConfig {
        dt: 1e-7,
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.0,
            alpha_power: 1.5,
        },
        ..PSTDConfig::default()
    };

    let k_mag = zeros_k_mag(32, 32, 32);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .unwrap()
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");

    let expected_tau = -4.246_711_703_873_091e-8;
    let expected_eta = -6.370_067_555_809_639e-5;
    assert!(
        (kernel.tau[[0, 0, 0]] - expected_tau).abs() < 1e-20,
        "tau mismatch: got {}, expected {}",
        kernel.tau[[0, 0, 0]],
        expected_tau
    );
    assert!(
        (kernel.eta[[0, 0, 0]] - expected_eta).abs() < 1e-18,
        "eta mismatch: got {}, expected {}",
        kernel.eta[[0, 0, 0]],
        expected_eta
    );
    assert_eq!(kernel.nabla1[[0, 0, 0]], 0.0);
    assert_eq!(kernel.nabla2[[0, 0, 0]], 0.0);
}

#[test]
fn test_nabla_operators_correct_power() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
    let y = 1.5_f64;
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.5,
            alpha_power: y,
        },
        ..PSTDConfig::default()
    };

    let dk = 2.0 * std::f64::consts::PI / (8.0 * 1e-3);
    let k_mag = test_k_mag(8, 8, 8, dk);
    let k_at_1 = k_mag[[1, 0, 0]];

    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        0.0,
        SOUND_SPEED_WATER_SIM,
    )
    .unwrap()
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");

    let expected_n1 = k_at_1.powf(y - 2.0);
    let expected_n2 = k_at_1.powf(y - 1.0);
    assert!(
        (kernel.nabla1[[1, 0, 0]] - expected_n1).abs() < 1e-10 * expected_n1,
        "nabla1 mismatch: got {}, expected {}",
        kernel.nabla1[[1, 0, 0]],
        expected_n1
    );
    assert!(
        (kernel.nabla2[[1, 0, 0]] - expected_n2).abs() < 1e-10 * expected_n2,
        "nabla2 mismatch: got {}, expected {}",
        kernel.nabla2[[1, 0, 0]],
        expected_n2
    );
}

#[test]
fn test_absorption_model_physics_validation() {
    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.75,
            alpha_power: 1.5,
        },
        ..Default::default()
    };
    let mut medium = HomogeneousMedium::new(SOUND_SPEED_WATER_SIM, DENSITY_WATER_NOMINAL, 0.0, 0.0, &grid);
    medium.set_acoustic_properties(0.0, 1.5, 0.0).unwrap();
    let k_mag = zeros_k_mag(16, 16, 16);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        MHZ_TO_HZ,
        SOUND_SPEED_WATER_SIM,
    )
    .unwrap()
    .expect("PowerLaw mode must return Some(AbsorptionKernel)");

    assert!(
        (kernel.tau[[0, 0, 0]] - (-3.467_425_586_398_137e-8)).abs() < 1e-20,
        "tau mismatch: got {}",
        kernel.tau[[0, 0, 0]]
    );
    assert!(
        (kernel.eta[[0, 0, 0]] - (-3.467_425_586_398_137_6e-5)).abs() < 1e-18,
        "eta mismatch: got {}",
        kernel.eta[[0, 0, 0]]
    );
}

/// Test Stokes absorption coefficient initialisation against the classical formula.
///
/// # Reference derivation (Blackstock 2000, Fundamentals of Physical Acoustics, Eq. 10-13)
/// ```text
/// α_SI = (4η_s/3 + η_b) / (2ρ₀c₀³)   [Np/(rad/s)²/m]
/// τ    = −2 α_SI · c₀                  [Treeby & Cox (2010) Eq. 19, y=2]
/// η    = 0                              [tan(π) = 0, non-dispersive]
/// ```
/// # Panics
/// - Panics if `Stokes init must succeed`.
/// - Panics if `Stokes mode must return Some(AbsorptionKernel)`.
///
#[test]
fn test_stokes_absorption_tau_matches_classical_formula() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, 0.0, 0.0, &grid);
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Stokes,
        dt: 1e-7,
        ..PSTDConfig::default()
    };

    let dk = 2.0 * std::f64::consts::PI / (8.0 * 1e-3);
    let k_mag = test_k_mag(8, 8, 8, dk);
    let kernel = initialize_absorption_operators(
        &config,
        &grid,
        &medium,
        &k_mag,
        0.0,
        SOUND_SPEED_WATER_SIM,
    )
    .expect("Stokes init must succeed")
    .expect("Stokes mode must return Some(AbsorptionKernel)");

    // VISCOSITY_WATER = 1.002e-3 Pa·s (water at 20°C, CRC Handbook) — canonical SSOT value.
    // HomogeneousMedium::new() sets shear_viscosity = VISCOSITY_WATER, bulk = 2.5 × VISCOSITY_WATER.
    let eta_s = VISCOSITY_WATER;
    let eta_b = 2.5 * VISCOSITY_WATER;
    let rho0 = 1000.0_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let alpha_si = (4.0 * eta_s / 3.0 + eta_b) / (2.0 * rho0 * c0 * c0 * c0);
    let expected_tau = -2.0 * alpha_si * c0;

    for val in kernel.tau.iter() {
        assert!(
            (val - expected_tau).abs() < 1e-24 * expected_tau.abs().max(1e-30),
            "tau cell mismatch: got {val}, expected {expected_tau}"
        );
    }

    for val in kernel.eta.iter() {
        assert_eq!(*val, 0.0, "eta must be zero for Stokes (y=2) absorption");
    }

    assert_eq!(kernel.nabla1[[0, 0, 0]], 0.0, "DC nabla1 must be 0");
    assert_eq!(
        kernel.nabla1[[1, 0, 0]],
        1.0,
        "nabla1 must be 1 at non-DC modes (|k|^0 = 1)"
    );

    let expected_n2 = k_mag[[1, 0, 0]];
    assert!(
        (kernel.nabla2[[1, 0, 0]] - expected_n2).abs() < 1e-12 * expected_n2,
        "nabla2 mismatch: got {}, expected {}",
        kernel.nabla2[[1, 0, 0]],
        expected_n2
    );
}
