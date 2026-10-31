use super::{test_k_mag, uniform_symbols, zeros_k_mag};
use crate::forward::pstd::config::{PSTDConfig, PowerLawDispersion};
use crate::forward::pstd::physics::absorption::init::initialize_absorption_operators;
use kwavers_core::constants::cavitation::VISCOSITY_WATER;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::constants::DB_TO_NP;
use kwavers_core::error::{KwaversError, ValidationError};
use kwavers_grid::Grid;
use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_omega_m;
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
            alpha_coeff: None,
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
    assert_eq!(uniform_symbols(&kernel).0[[0, 0, 0]], 0.0);
    assert_eq!(uniform_symbols(&kernel).1[[0, 0, 0]], 0.0);
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
            alpha_coeff: Some(0.5),
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
        (uniform_symbols(&kernel).0[[1, 0, 0]] - expected_n1).abs() < 1e-10 * expected_n1,
        "nabla1 mismatch: got {}, expected {}",
        uniform_symbols(&kernel).0[[1, 0, 0]],
        expected_n1
    );
    assert!(
        (uniform_symbols(&kernel).1[[1, 0, 0]] - expected_n2).abs() < 1e-10 * expected_n2,
        "nabla2 mismatch: got {}, expected {}",
        uniform_symbols(&kernel).1[[1, 0, 0]],
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
    use kwavers_medium::absorption::power_law::power_law_db_cm_to_np_omega_m;
    use std::f64::consts::PI;

    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4).unwrap();
    let alpha_coeff = 0.75_f64;
    let alpha_power = 1.5_f64;
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: Some(alpha_coeff),
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

    assert_eq!(
        uniform_symbols(&kernel).0[[0, 0, 0]],
        0.0,
        "DC nabla1 must be 0"
    );
    assert_eq!(
        uniform_symbols(&kernel).0[[1, 0, 0]],
        1.0,
        "nabla1 must be 1 at non-DC modes (|k|^0 = 1)"
    );

    let expected_n2 = k_mag[[1, 0, 0]];
    assert!(
        (uniform_symbols(&kernel).1[[1, 0, 0]] - expected_n2).abs() < 1e-12 * expected_n2,
        "nabla2 mismatch: got {}, expected {}",
        uniform_symbols(&kernel).1[[1, 0, 0]],
        expected_n2
    );
}

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

/// Configurations saved before the dispersion option existed still load, as
/// the causal default.
#[test]
fn config_without_dispersion_field_deserializes_as_causal() {
    let mut value = serde_json::to_value(PSTDConfig::default()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("power_law_dispersion")
        .unwrap();
    let config: PSTDConfig = serde_json::from_value(value).unwrap();
    assert_eq!(config.power_law_dispersion, PowerLawDispersion::Causal);
}

/// A medium whose exponent range is within the strata tolerance takes the
/// midpoint of that range in every voxel, for τ, η, the operators and the
/// thermal exponent alike, rather than the exponent of one sample voxel.
#[test]
fn near_uniform_medium_exponent_resolves_to_its_midpoint_everywhere() {
    use super::super::ExponentSpectrum;
    use kwavers_medium::heterogeneous::HeterogeneousMedium;

    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let (alpha0, low, high) = (0.7, 1.5_f64, 1.5 + 5e-7);
    let background = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let mut medium = HeterogeneousMedium::from_homogeneous(&background, &grid)
        .expect("valid homogeneous optical properties");
    medium.use_trilinear_interpolation = false;
    // The `absorption` field stores SI base prefactors (Np/m at the reference
    // frequency), so 0.7 dB/(MHz^y cm) is filled boundary-converted, as the
    // medium's own constructors store it.
    medium.absorption.fill(alpha0 * DB_TO_NP * 100.0);
    medium.alpha0.fill(alpha0);
    for ([i, _, _], y) in medium
        .alpha_power
        .indexed_iter_mut()
        .expect("invariant: alpha_power field is materialized")
    {
        // Voxel [0, 0, 0] carries the low end, so a sample-voxel rule would
        // resolve to `low`, not the midpoint.
        *y = if i == 0 { low } else { high };
    }
    let config = PSTDConfig {
        absorption_mode: AbsorptionMode::PowerLaw {
            alpha_coeff: None,
            alpha_power: 1.2,
        },
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

    let midpoint = 0.5 * (low + high);
    let ExponentSpectrum::Uniform { exponent, .. } = &kernel.spectrum else {
        panic!("a range within the strata tolerance takes the uniform path");
    };
    assert_eq!(*exponent, midpoint);
    let (nabla1, nabla2) = uniform_symbols(&kernel);
    assert_eq!(nabla1[[1, 0, 0]], k_mag[[1, 0, 0]].powf(midpoint - 2.0));
    assert_eq!(nabla2[[1, 0, 0]], k_mag[[1, 0, 0]].powf(midpoint - 1.0));
    // Every voxel's coefficients use the same expression at the midpoint, so
    // they match bit for bit. A voxel's own exponent would scale τ's c₀^(y−1)
    // by c₀^(±Δy/2) ≈ 1 ± 2.5e-7 · ln c₀ ≈ 1 ± 1.8e-6, far above one rounding.
    let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha0, midpoint);
    let tau = -2.0 * alpha_0_si * SOUND_SPEED_WATER_SIM.powf(midpoint - 1.0);
    let eta = 2.0
        * alpha_0_si
        * SOUND_SPEED_WATER_SIM.powf(midpoint)
        * (std::f64::consts::PI * midpoint / 2.0).tan();
    assert!(kernel.tau.iter().all(|&value| value == tau));
    assert!(kernel.eta.iter().all(|&value| value == eta));
}
