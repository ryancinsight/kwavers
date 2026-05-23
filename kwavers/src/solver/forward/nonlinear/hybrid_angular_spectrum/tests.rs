//! Unit tests for the Hybrid Angular Spectrum module.

use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::core::constants::numerical::MHZ_TO_HZ;
use crate::domain::grid::Grid;
use ndarray::Array3;

use super::config::HASConfig;
use super::facade::HybridAngularSpectrum;

#[test]
fn test_has_config_default() {
    let config = HASConfig::default();
    assert_eq!(config.sound_speed, SOUND_SPEED_WATER_SIM);
    assert_eq!(config.density, 1000.0);
    assert!(config.nonlinearity > 0.0);
}

#[test]
fn test_has_config_validation() {
    assert!(HASConfig::new(-1.0, 1000.0, 6.0, 0.5, 2.0, 0.0001, MHZ_TO_HZ).is_err());
    assert!(HASConfig::new(SOUND_SPEED_WATER_SIM, -1.0, 6.0, 0.5, 2.0, 0.0001, MHZ_TO_HZ).is_err());
    assert!(HASConfig::new(SOUND_SPEED_WATER_SIM, 1000.0, 6.0, 0.5, 2.0, -0.0001, MHZ_TO_HZ).is_err());
}

/// Z = ρ·c (acoustic impedance definition, Pierce 1989 §1.5).
#[test]
fn test_impedance_calculation() {
    let config = HASConfig::default();
    assert_eq!(config.impedance(), SOUND_SPEED_WATER_SIM * 1000.0);
}

/// Power-law attenuation α(f) = α₀·(f/MHz)^y; ratio at 2× frequency = 2^y.
///
/// For y=2 (default): α(2 MHz)/α(1 MHz) = 4.0.
/// Reference: Szabo TL (1994). *J. Acoust. Soc. Am.* 96(1), 491–500.
#[test]
fn test_attenuation_frequency_dependence() {
    use crate::core::constants::acoustic_parameters::NP_TO_DB;
    let config = HASConfig::default();
    let atten_1mhz = config.attenuation_at_frequency(MHZ_TO_HZ);
    let atten_2mhz = config.attenuation_at_frequency(2.0 * MHZ_TO_HZ);
    let ratio = atten_2mhz / atten_1mhz;
    assert!(
        (ratio - 4.0).abs() < 1e-12,
        "power-law ratio at 2×frequency: expected 4.0 (y=2), got {ratio:.15}"
    );
    let expected_np_per_m = 0.5 * 100.0 / NP_TO_DB;
    let rel_err = (atten_1mhz - expected_np_per_m).abs() / expected_np_per_m;
    assert!(
        rel_err < 1e-12,
        "attenuation at 1 MHz: expected {expected_np_per_m:.6e} Np/m, got {atten_1mhz:.6e}"
    );
}

#[test]
fn test_hybrid_angular_spectrum_creation() {
    let grid = Grid::new(64, 64, 32, 0.001, 0.001, 0.001).unwrap();
    let _has = HybridAngularSpectrum::new(&grid, HASConfig::default()).unwrap();
}

#[test]
fn test_propagate_zero_distance_returns_input_unchanged() {
    let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
    let has = HybridAngularSpectrum::new(&grid, HASConfig::default()).unwrap();
    let pressure = Array3::from_shape_fn((8, 8, 4), |(i, j, k)| {
        (i as f64 + 2.0 * j as f64 + 3.0 * k as f64) * 1.0e3
    });

    let propagated = has.propagate(&pressure, 0.0).unwrap();
    assert_eq!(propagated, pressure);
}

/// z_shock = ρ₀c₀³ / (β·ω·p₀) (Hamilton & Blackstock 1998 §4.3 eq. 4.3.5).
#[test]
fn test_shock_formation_distance_matches_analytical_formula() {
    let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
    // nonlinearity stores B/A; β = 1 + B/(2A) is used inside shock_formation_distance
    let (rho0, c0, b_over_a, f_ref) = (1000.0_f64, SOUND_SPEED_WATER_SIM, 6.0_f64, MHZ_TO_HZ);
    let beta = 1.0 + b_over_a / 2.0; // = 4.0 for B/A = 6 (tissue)
    let config = HASConfig {
        sound_speed: c0,
        density: rho0,
        nonlinearity: b_over_a,
        reference_frequency: f_ref,
        ..HASConfig::default()
    };
    let has = HybridAngularSpectrum::new(&grid, config).unwrap();
    let p0 = 1.0e5_f64;
    let d_shock = has.shock_formation_distance(p0);
    let omega = 2.0 * std::f64::consts::PI * f_ref;
    let z_shock_analytic = rho0 * c0.powi(3) / (beta * omega * p0);
    let rel_err = (d_shock - z_shock_analytic).abs() / z_shock_analytic;
    assert!(
        rel_err < 1e-12,
        "shock_formation_distance: computed={d_shock:.6e} analytic={z_shock_analytic:.6e} rel_err={rel_err:.2e}"
    );
    assert!(
        (0.5..2.0).contains(&d_shock),
        "z_shock = {d_shock:.4} m should be in [0.5, 2.0] m for tissue B/A=6 at 1 MHz, 100 kPa"
    );
}

/// z_R = a² / λ = a²·f / c₀ (Goodman 2005 §4.2).
#[test]
fn test_rayleigh_distance_matches_analytical_formula() {
    let grid = Grid::new(4, 4, 4, 0.001, 0.001, 0.001).unwrap();
    let (c0, f_ref) = (SOUND_SPEED_WATER_SIM, MHZ_TO_HZ);
    let config = HASConfig {
        sound_speed: c0,
        reference_frequency: f_ref,
        ..HASConfig::default()
    };
    let has = HybridAngularSpectrum::new(&grid, config).unwrap();
    let aperture = 0.01_f64;
    let d_rayleigh = has.rayleigh_distance(aperture);
    let z_r_analytic = aperture * aperture / (c0 / f_ref);
    let rel_err = (d_rayleigh - z_r_analytic).abs() / z_r_analytic;
    assert!(
        rel_err < 1e-12,
        "rayleigh_distance: computed={d_rayleigh:.6e} analytic={z_r_analytic:.6e} rel_err={rel_err:.2e}"
    );
}
