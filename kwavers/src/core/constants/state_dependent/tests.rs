use super::*;
use crate::core::constants::fundamental::ATMOSPHERIC_PRESSURE;
use crate::core::constants::numerical::MPA_TO_PA;

#[test]
fn test_sound_speed_water_temperature_dependence() {
    let constants = StateDependentConstants::default();

    let c_20 = constants.sound_speed_water(20.0, ATMOSPHERIC_PRESSURE);
    let c_21 = constants.sound_speed_water(21.0, ATMOSPHERIC_PRESSURE);

    let dc_dt = c_21 - c_20;
    assert!(
        (dc_dt - 3.0).abs() < 1.0,
        "dc/dT should be ~3.0 m/s/K, got {}",
        dc_dt
    );

    let c_37 = constants.sound_speed_water(37.0, ATMOSPHERIC_PRESSURE);
    assert!(
        (c_37 - 1525.0).abs() < 10.0,
        "Sound speed at 37°C should be ~1525 m/s, got {}",
        c_37
    );
}

#[test]
fn test_dynamic_viscosity_water() {
    let constants = StateDependentConstants::default();

    let eta_20 = constants.dynamic_viscosity_water(20.0);
    assert!(
        (eta_20 - 1.002e-3).abs() < 1e-5,
        "Viscosity at 20°C should be ~1.002e-3 Pa·s, got {}",
        eta_20
    );

    let eta_37 = constants.dynamic_viscosity_water(37.0);
    assert!(
        (eta_37 - 0.692e-3).abs() < 5e-5,
        "Viscosity at 37°C should be ~0.692e-3 Pa·s, got {}",
        eta_37
    );

    assert!(
        eta_20 > eta_37,
        "Viscosity should decrease with temperature"
    );
}

#[test]
fn test_surface_tension_water() {
    let constants = StateDependentConstants::default();

    let sigma_20 = constants.surface_tension_water(20.0);
    assert!(
        (sigma_20 - 0.0728).abs() < 0.005,
        "Surface tension at 20°C should be ~0.0728 N/m, got {}",
        sigma_20
    );

    let sigma_100 = constants.surface_tension_water(100.0);
    assert!(
        (sigma_100 - 0.0589).abs() < 0.01,
        "Surface tension at 100°C should be ~0.0589 N/m, got {}",
        sigma_100
    );

    assert!(
        sigma_20 > sigma_100,
        "Surface tension should decrease with temperature"
    );
}

#[test]
fn test_nonlinear_parameter() {
    let constants = StateDependentConstants::default();

    let ba_20 = constants.nonlinear_parameter_water(20.0);
    assert!(
        (ba_20 - 5.0).abs() < 0.1,
        "B/A at 20°C should be ~5.0, got {}",
        ba_20
    );

    let ba_37 = constants.nonlinear_parameter_water(37.0);
    assert!(ba_37 > ba_20, "B/A should increase with temperature");
}

#[test]
fn test_acoustic_impedance() {
    let constants = StateDependentConstants::default();

    let z_20 = constants.acoustic_impedance_water(20.0, ATMOSPHERIC_PRESSURE);
    assert!(
        (z_20 - 1.48e6).abs() < 0.05e6,
        "Acoustic impedance at 20°C should be ~1.48 MRayl, got {}",
        z_20
    );
}

#[test]
fn test_cavitation_threshold() {
    let constants = StateDependentConstants::default();

    let p_thresh = constants.cavitation_threshold(20.0, 1e-6, ATMOSPHERIC_PRESSURE);

    assert!(
        p_thresh < 0.0,
        "Cavitation threshold should be negative (tension)"
    );
    assert!(
        p_thresh.abs() < MPA_TO_PA,
        "Cavitation threshold should be reasonable magnitude"
    );
}

#[test]
fn test_prandtl_number() {
    let constants = StateDependentConstants::default();

    let pr = constants.prandtl_number_water(20.0);
    assert!(
        (pr - 7.0).abs() < 2.0,
        "Prandtl number at 20°C should be ~7, got {}",
        pr
    );
}

#[test]
fn test_reynolds_number() {
    let constants = StateDependentConstants::default();

    let re = constants.reynolds_number_water(0.1, 0.01, 20.0);

    assert!(
        re > 500.0 && re < 2000.0,
        "Reynolds number should be in laminar regime, got {}",
        re
    );
}

#[test]
fn test_vft_viscosity_known_values() {
    let constants = StateDependentConstants::default();

    let eta_0 = constants.dynamic_viscosity_water(0.0);
    assert!(
        (eta_0 - 1.787e-3).abs() / 1.787e-3 < 0.03,
        "VFT viscosity at 0°C: expected ~1.787e-3, got {eta_0:.4e}"
    );

    let eta_20 = constants.dynamic_viscosity_water(20.0);
    assert!(
        (eta_20 - 1.002e-3).abs() / 1.002e-3 < 0.03,
        "VFT viscosity at 20°C: expected ~1.002e-3, got {eta_20:.4e}"
    );

    let eta_100 = constants.dynamic_viscosity_water(100.0);
    assert!(
        (eta_100 - 2.82e-4).abs() / 2.82e-4 < 0.03,
        "VFT viscosity at 100°C: expected ~2.82e-4, got {eta_100:.4e}"
    );

    let eta_50 = constants.dynamic_viscosity_water(50.0);
    assert!(eta_0 > eta_20, "η must decrease 0°C → 20°C");
    assert!(eta_20 > eta_50, "η must decrease 20°C → 50°C");
    assert!(eta_50 > eta_100, "η must decrease 50°C → 100°C");
}

#[test]
fn test_viscosity_arrhenius() {
    let eta = StateDependentConstants::viscosity_arrhenius(5e-5, 30_000.0, 300.0);
    let expected = 5e-5 * (30_000.0_f64 / (8.314_462_618 * 300.0)).exp();
    assert!(
        (eta - expected).abs() < 1e-10,
        "Arrhenius model must reproduce exact exponential, got {eta:.6e}"
    );

    let eta_hot = StateDependentConstants::viscosity_arrhenius(5e-5, 30_000.0, 500.0);
    assert!(
        eta > eta_hot,
        "Viscosity should decrease with temperature (Arrhenius)"
    );
}
