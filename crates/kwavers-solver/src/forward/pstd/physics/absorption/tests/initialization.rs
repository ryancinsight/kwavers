use super::{test_k_mag, zeros_k_mag};
use crate::forward::pstd::config::PSTDConfig;
use crate::forward::pstd::physics::absorption::init::initialize_absorption_operators;
use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;

#[test]
fn test_power_law_initialization() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
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
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let y = 1.5_f64;
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: 0.5,
            alpha_power: y,
        },
        ..PSTDConfig::default()
    };

    let dk = TWO_PI / (8.0 * 1e-3);
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

/// # Theorem: Treeby & Cox (2010) τ and η derivation for power-law absorption
///
/// Given α(f) = α_dB [dB/MHz^y/cm] and power y:
///   α_SI = power_law_db_cm_to_np_omega_m(α_dB, y)  [Np/(rad/s)^y / m]
///   τ(r) = −2 · α_SI · c₀^(y−1)                    [Treeby & Cox Eq. 19]
///   η(r) =  2 · α_SI · c₀^y · tan(π·y/2)           [Treeby & Cox Eq. 20]
///
/// When medium.alpha_coefficient() returns 0, init.rs falls through to the config
/// alpha_coeff (0.75).  The expected values below are derived from SSOT constants.
///
/// For config: alpha_coeff=0.75, y=1.5, c0=SOUND_SPEED_WATER_SIM≈1498 m/s:
///   α_SI = 5.482481235081536e-10  (see power_law_db_cm_to_np_omega_m)
///   τ    = −2 · 5.482e-10 · 1498^0.5 = −4.2467e-8
///   η    =  2 · 5.482e-10 · 1498^1.5 · tan(3π/4) = −2 · 5.482e-10 · 57973.6 · 1.0
///         = (negative, magnitude ≈ 6.357e-5)
#[test]
fn test_absorption_model_physics_validation() {
    use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
    use std::f64::consts::PI;

    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
    let alpha_coeff = 0.75_f64;
    let alpha_power = 1.5_f64;
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff,
            alpha_power,
        },
        ..Default::default()
    };
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    // Set medium alpha_coeff=0.0 so init falls through to config's 0.75.
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

    // Derive expected τ and η analytically from SSOT constants.
    let c0 = SOUND_SPEED_WATER_SIM;
    let y = alpha_power;
    let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_coeff, y);
    let expected_tau = -2.0 * alpha_0_si * c0.powf(y - 1.0);
    let expected_eta = 2.0 * alpha_0_si * c0.powf(y) * (PI * y / 2.0).tan();

    assert!(
        (kernel.tau[[0, 0, 0]] - expected_tau).abs() < 1e-20_f64.max(1e-12 * expected_tau.abs()),
        "tau mismatch: got {}, expected {}",
        kernel.tau[[0, 0, 0]],
        expected_tau
    );
    assert!(
        (kernel.eta[[0, 0, 0]] - expected_eta).abs() < 1e-18_f64.max(1e-12 * expected_eta.abs()),
        "eta mismatch: got {}, expected {}",
        kernel.eta[[0, 0, 0]],
        expected_eta
    );
    // Verify sign conventions: τ < 0 (absorbing), η < 0 for y=1.5 (tan(3π/4) = −1)
    assert!(expected_tau < 0.0, "τ must be negative (absorbing)");
    assert!(
        expected_eta < 0.0,
        "η must be negative for y=1.5 (tan(3π/4) = −1)"
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
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::Stokes,
        dt: 1e-7,
        ..PSTDConfig::default()
    };

    let dk = TWO_PI / (8.0 * 1e-3);
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
    let rho0 = DENSITY_WATER_NOMINAL;
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
