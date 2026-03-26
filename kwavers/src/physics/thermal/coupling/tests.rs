use super::*;
use ndarray::Array3;

#[test]
fn test_acoustic_heating_source() {
    let source = AcousticHeatingSource::new(0.5, 1e4); // 500 Np/m, 10 kW/m²
    let power = source.power();
    assert!(power > 0.0);
}

#[test]
fn test_heating_depth_attenuation() {
    let source = AcousticHeatingSource::new(0.5, 1e4);
    let power_0 = source.power_at_depth(0.0);
    let power_1cm = source.power_at_depth(0.01);

    // Power should decrease with depth
    assert!(power_1cm < power_0);
}

#[test]
fn test_temperature_coefficients_soft_tissue() {
    let coeff = TemperatureCoefficients::soft_tissue();

    // Base properties at 37°C
    let c0 = 1540.0;
    let rho0 = 1050.0;
    let alpha0 = 0.5;

    // At 40°C (3°C higher)
    let c_40 = coeff.sound_speed(c0, 40.0, 37.0);
    let rho_40 = coeff.density(rho0, 40.0, 37.0);
    let alpha_40 = coeff.absorption(alpha0, 40.0, 37.0);

    // Sound speed increases
    assert!(c_40 > c0);
    // Density decreases
    assert!(rho_40 < rho0);
    // Absorption increases
    assert!(alpha_40 > alpha0);
}

#[test]
fn test_acoustic_streaming_velocity() {
    let streaming = AcousticStreaming::new(1e3, 1500.0, 1050.0); // 1 kW/m²
    let v = streaming.velocity();
    assert!(v > 0.0);
}

#[test]
fn test_nonlinear_heating() {
    let nl = NonlinearHeating::new(
        5.0, // B/A = 5
        1e5, // 100 kPa
        1500.0, 1050.0,
    );
    let power = nl.power();
    assert!(power > 0.0);

    let shock = nl.shock_parameter();
    assert!(shock > 0.0);
}

#[test]
fn test_nonlinear_regime_detection() {
    // Linear regime
    let nl_linear = NonlinearHeating::new(5.0, 1e4, 1500.0, 1050.0);
    assert!(!nl_linear.is_nonlinear_significant());

    // Nonlinear regime
    let nl_nonlinear = NonlinearHeating::new(5.0, 5e5, 1500.0, 1050.0);
    assert!(nl_nonlinear.is_nonlinear_significant());
}

#[test]
fn test_thermal_acoustic_coupling() {
    let mut coupling =
        ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());
    coupling.initialize((5, 5, 5));

    let temperature = Array3::from_elem((5, 5, 5), 37.0);
    let intensity = Array3::from_elem((5, 5, 5), 1e4);

    let result = coupling.update(&temperature, &intensity, 37.0, 0.1);
    assert!(result.is_ok());

    let energy = coupling.total_energy();
    assert!(energy > 0.0);
}

#[test]
fn test_coupling_temperature_effects_on_properties() {
    let coupling = ThermalAcousticCoupling::new(0.5, 1e4, TemperatureCoefficients::soft_tissue());

    let c0 = 1540.0;
    let c_hot = coupling.sound_speed_at_temperature(c0, 45.0, 37.0);
    let rho0 = 1050.0;
    let rho_hot = coupling.density_at_temperature(rho0, 45.0, 37.0);

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
