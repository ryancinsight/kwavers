use super::model::CherenkovModel;
use crate::core::constants::fundamental::SPEED_OF_LIGHT;
use approx;
use ndarray::Array1;
use std::f64::consts::PI;

#[test]
fn test_cherenkov_frank_tamm_spectral_distribution() {
    // Reference: Frank & Tamm (1937), Jackson Classical Electrodynamics
    let model = CherenkovModel::new(1.5, 100.0);

    // Test relativistic electron in water (n=1.5)
    let v_relativistic = 0.99 * SPEED_OF_LIGHT; // β ≈ 0.99
    let charge = 1.0; // Single electron

    // Calculate expected Cherenkov angle
    let expected_angle = (1.0f64 / (1.5 * 0.99)).acos(); // cosθ = 1/(nβ)
    let calculated_angle = model.cherenkov_angle(v_relativistic);

    approx::assert_relative_eq!(calculated_angle, expected_angle, epsilon = 1e-6);

    // Test spectral intensity scales as 1/ω
    let freq1 = 1e15; // Hz
    let freq2 = 2e15; // Hz

    let intensity1 = model.spectral_intensity(freq1, v_relativistic, charge);
    let intensity2 = model.spectral_intensity(freq2, v_relativistic, charge);

    // Should follow 1/ω dependence
    approx::assert_relative_eq!(intensity1 / intensity2, 2.0, epsilon = 1e-3);
}

#[test]
fn test_cherenkov_threshold_condition() {
    // Reference: Classical condition for Cherenkov radiation
    let model = CherenkovModel::new(1.33, 1.0); // Water refractive index

    let v_below = model.critical_velocity * 0.99;
    let v_above = model.critical_velocity * 1.01;

    assert!(!model.exceeds_threshold(v_below));
    assert!(model.exceeds_threshold(v_above));

    // No emission below threshold
    assert_eq!(model.spectral_intensity(1e15, v_below, 1.0), 0.0);

    // Emission above threshold
    assert!(model.spectral_intensity(1e15, v_above, 1.0) > 0.0);
}

#[test]
fn test_cherenkov_angle_variation() {
    // θ = arccos(1/(nβ)), angle increases with higher β
    let model = CherenkovModel::new(1.5, 1.0);

    let v1 = model.critical_velocity * 1.1; // Just above threshold
    let v2 = model.critical_velocity * 2.0; // Higher velocity

    let angle1 = model.cherenkov_angle(v1);
    let angle2 = model.cherenkov_angle(v2);

    assert!(angle2 > angle1); // Angle increases with velocity
    assert!(angle1 > 0.0 && angle1 < PI / 2.0);
    assert!(angle2 > 0.0 && angle2 < PI / 2.0);
}

#[test]
fn test_cherenkov_refractive_index_update() {
    // Empirical relation n(ρ,T) for compressed water
    let mut model = CherenkovModel::new(1.33, 1.0);

    // Ambient conditions
    model.update_refractive_index(1.0, 300.0);
    assert!((model.refractive_index - 1.33).abs() < 0.01);

    // High compression (ρ/ρ₀ = 5)
    model.update_refractive_index(5.0, 300.0);
    assert!(model.refractive_index > 1.4);

    // High temperature (should decrease n)
    model.update_refractive_index(1.0, 10000.0);
    assert!(model.refractive_index < 1.35);

    // Critical velocity should update accordingly
    let expected_critical = SPEED_OF_LIGHT / model.refractive_index;
    approx::assert_relative_eq!(model.critical_velocity, expected_critical, epsilon = 1e-10);
}

#[test]
fn test_cherenkov_power_density_scaling() {
    // Power density scales with charge density and threshold behavior
    let model = CherenkovModel::new(1.5, 10.0);
    let velocity = model.critical_velocity * 1.5;
    let temperature = 15000.0; // K

    let charge_density_1 = 1e3; // C/m³
    let charge_density_2 = 2e3; // C/m³

    let power1 = model.total_power_density(velocity, charge_density_1, temperature);
    let power2 = model.total_power_density(velocity, charge_density_2, temperature);

    // Should scale approximately with charge density
    assert!(power2 > power1);

    // No emission below threshold
    let power_below =
        model.total_power_density(model.critical_velocity * 0.9, charge_density_1, temperature);
    assert_eq!(power_below, 0.0);
}

#[test]
fn test_cherenkov_spectral_emission() {
    // Spectrum properties with UV/blue bias
    let model = CherenkovModel::new(1.5, 50.0);
    let velocity = model.critical_velocity * 2.0;

    let wavelengths = Array1::linspace(200e-9, 800e-9, 100); // Visible spectrum
    let spectrum = model.emission_spectrum(velocity, 1.0, &wavelengths);

    // Should have emission
    assert!(spectrum.sum() > 0.0);

    // Should be broader than single wavelength
    let peak_intensity = spectrum.fold(0.0f64, |max, &val| max.max(val));
    assert!(peak_intensity > 0.0);

    // Find peak wavelength
    let peak_idx = spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // Peak should be in UV/blue for Cherenkov in water
    let peak_wavelength = wavelengths[peak_idx];
    assert!((200e-9..=400e-9).contains(&peak_wavelength));
}
