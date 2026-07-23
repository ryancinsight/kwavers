use super::*;
use eunomia::assert_relative_eq;
use kwavers_core::constants::fundamental::{
    DENSITY_TISSUE, SOUND_SPEED_TISSUE, SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use leto::Array3;

#[test]
fn test_acoustic_heating_source() {
    let source = AcousticHeatingSource::new(0.5, 1e4); // 500 Np/m, 10 kW/m²
    let power = source.power();
    assert_eq!(power, 10_000.0);
}

#[test]
fn test_heating_depth_attenuation() {
    let source = AcousticHeatingSource::new(0.5, 1e4);
    let power_0 = source.power_at_depth(0.0);
    let power_1cm = source.power_at_depth(0.01);

    assert_relative_eq!(power_0, 10_000.0, epsilon = 8.0 * f64::EPSILON);
    let expected_1cm = 10_000.0_f64 * (-0.01_f64).exp();
    // The typed expression performs two products plus exp; 16 ulps bounds the
    // first-order f64 rounding error without hiding a formula error.
    assert_relative_eq!(
        power_1cm,
        expected_1cm,
        epsilon = 16.0 * f64::EPSILON * expected_1cm.abs()
    );
    assert!(power_1cm < power_0);
}

#[test]
fn test_temperature_coefficients_soft_tissue() {
    let coeff = TemperatureCoefficients::soft_tissue();

    // Base properties at 37°C
    let c0 = SOUND_SPEED_TISSUE;
    let rho0 = DENSITY_TISSUE;
    let alpha0 = 0.5;

    // At 40°C (3°C higher)
    let c_40 = coeff.sound_speed(c0, 40.0, BODY_TEMPERATURE_C);
    let rho_40 = coeff.density(rho0, 40.0, BODY_TEMPERATURE_C);
    let alpha_40 = coeff.absorption(alpha0, 40.0, BODY_TEMPERATURE_C);

    // Sound speed increases
    assert!(c_40 > c0);
    // Density decreases
    assert!(rho_40 < rho0);
    // Absorption increases
    assert!(alpha_40 > alpha0);
}

#[test]
fn test_acoustic_streaming_velocity() {
    let streaming = AcousticStreaming::new(1e3, SOUND_SPEED_WATER_SIM, DENSITY_TISSUE); // 1 kW/m²
    let v = streaming.velocity();
    let expected_velocity = 1e3_f64 / (DENSITY_TISSUE * SOUND_SPEED_WATER_SIM.powi(2));
    assert_relative_eq!(
        v,
        expected_velocity,
        epsilon = 32.0 * f64::EPSILON * expected_velocity.abs()
    );

    let expected_power = 1e3_f64.powi(2) / (DENSITY_TISSUE * SOUND_SPEED_WATER_SIM.powi(3));
    assert_relative_eq!(
        streaming.power(),
        expected_power,
        epsilon = 32.0 * f64::EPSILON * expected_power.abs()
    );
}

#[test]
fn test_nonlinear_heating() {
    let nl = NonlinearHeating::new(
        5.0,                   // B/A = 5
        1e5,                   // 100 kPa
        SOUND_SPEED_WATER_SIM, // m/s
        DENSITY_TISSUE,        // kg/m³
        MHZ_TO_HZ,             // 1 MHz
    );
    let power = nl.power();
    assert!(power > 0.0);

    let shock = nl.shock_parameter();
    assert!(shock > 0.0);
}

#[test]
fn test_nonlinear_regime_detection() {
    // Linear regime
    let nl_linear =
        NonlinearHeating::new(5.0, 1e4, SOUND_SPEED_WATER_SIM, DENSITY_TISSUE, MHZ_TO_HZ);
    assert!(!nl_linear.is_nonlinear_significant());

    // Nonlinear regime
    let nl_nonlinear =
        NonlinearHeating::new(5.0, 5e5, SOUND_SPEED_WATER_SIM, DENSITY_TISSUE, MHZ_TO_HZ);
    assert!(nl_nonlinear.is_nonlinear_significant());
}

#[test]
fn test_thermal_acoustic_coupling() {
    let mut coupling =
        ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());
    coupling.initialize((5, 5, 5));

    let temperature = Array3::from_elem((5, 5, 5), BODY_TEMPERATURE_C);
    let intensity = Array3::from_elem((5, 5, 5), 1e4);

    coupling
        .update(&temperature, &intensity, BODY_TEMPERATURE_C, 0.1)
        .unwrap();

    let energy_density = coupling.total_energy_density();
    assert!(energy_density > 0.0);
}

#[test]
fn test_coupling_temperature_effects_on_properties() {
    let coupling = ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());

    let c0 = SOUND_SPEED_TISSUE;
    let c_hot = coupling.sound_speed_at_temperature(c0, 45.0, BODY_TEMPERATURE_C);
    let rho0 = DENSITY_TISSUE;
    let rho_hot = coupling.density_at_temperature(rho0, 45.0, BODY_TEMPERATURE_C);

    // Temperature increases both sound speed and decreases density
    assert!(c_hot > c0);
    assert!(rho_hot < rho0);
}

#[test]
fn test_temperature_coefficient_variants() {
    let soft = TemperatureCoefficients::soft_tissue();
    let water = TemperatureCoefficients::water();
    let blood = TemperatureCoefficients::blood();
    let bone = TemperatureCoefficients::bone();

    // Each should have different coefficients
    assert_ne!(soft.sound_speed_coeff, water.sound_speed_coeff);
    assert_ne!(blood.absorption_coeff, bone.absorption_coeff);
}

#[test]
fn test_acoustic_heating_zero_absorption() {
    let source = AcousticHeatingSource::new(0.0, 1e5);
    assert_eq!(source.power(), 0.0);
}

#[test]
fn test_acoustic_heating_zero_intensity() {
    let source = AcousticHeatingSource::new(0.5, 0.0);
    assert_eq!(source.power(), 0.0);
}
