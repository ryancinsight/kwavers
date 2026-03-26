use super::model::SkullAttenuation;
use super::types::BoneType;

#[test]
fn test_cortical_bone_properties() {
    let skull = SkullAttenuation::cortical();

    // Test absorption at 1 MHz
    let alpha_1mhz = skull.absorption_coefficient(1e6);
    assert!((alpha_1mhz - 60.0).abs() < 1.0); // Should be ~60 Np/m

    // Test frequency scaling (linear for n=1.0)
    let alpha_2mhz = skull.absorption_coefficient(2e6);
    assert!((alpha_2mhz / alpha_1mhz - 2.0).abs() < 0.01);
}

#[test]
fn test_cancellous_bone_properties() {
    let skull = SkullAttenuation::cancellous();

    // Lower base attenuation than cortical
    let alpha = skull.absorption_coefficient(1e6);
    assert!(alpha < 60.0);
    assert!(alpha > 20.0);

    // Higher scattering than cortical
    let scatter_canc = skull.scattering_coefficient(1e6);
    let scatter_cort = SkullAttenuation::cortical().scattering_coefficient(1e6);
    assert!(scatter_canc > scatter_cort);
}

#[test]
fn test_frequency_dependence() {
    let skull = SkullAttenuation::cortical();

    // Absorption increases with frequency
    let alpha_low = skull.absorption_coefficient(0.5e6);
    let alpha_high = skull.absorption_coefficient(3.0e6);
    assert!(alpha_high > alpha_low);

    // Scattering increases faster (f^4 regime at low freq)
    let scatter_low = skull.scattering_coefficient(0.5e6);
    let scatter_high = skull.scattering_coefficient(1.5e6);
    assert!(scatter_high > scatter_low);
}

#[test]
fn test_temperature_correction() {
    let skull = SkullAttenuation::cortical();

    // Higher temperature increases attenuation
    let factor_cold = skull.temperature_correction(20.0);
    let factor_body = skull.temperature_correction(37.0);
    let factor_hot = skull.temperature_correction(45.0);

    assert!(factor_cold < factor_body);
    assert!(factor_hot > factor_body);
    assert!((factor_body - 1.0).abs() < 0.01); // Should be ~1.0 at reference temp
}

#[test]
fn test_db_conversion() {
    let skull = SkullAttenuation::cortical();

    // At 1 MHz: α = 60 Np/m = 60 * 8.686 * 0.01 = 5.2 dB/cm
    let alpha_db = skull.attenuation_db_per_cm(1e6);
    assert!((alpha_db - 5.2).abs() < 0.5);
}

#[test]
fn test_mixed_bone_type() {
    let skull = SkullAttenuation::new(
        50.0,
        1.1,
        BoneType::Mixed {
            cortical_fraction: 0.5,
        },
    )
    .unwrap();

    // Scattering should be between cortical and cancellous
    let scatter = skull.scattering_coefficient(1e6);
    let cortical_scatter = 0.01;
    let cancellous_scatter = 0.1;

    assert!(scatter > cortical_scatter);
    assert!(scatter < cancellous_scatter);
    assert!((scatter - 0.055).abs() < 0.01); // Should be ~average
}

#[test]
fn test_total_coefficient_components() {
    let mut skull = SkullAttenuation::cortical();

    let freq = 1.5e6;
    let alpha_abs = skull.absorption_coefficient(freq);
    let alpha_scatter = skull.scattering_coefficient(freq);
    let alpha_total = skull.total_coefficient(freq);

    // Total should be sum of components
    assert!((alpha_total - (alpha_abs + alpha_scatter)).abs() < 1e-6);

    // Disable scattering
    skull.set_scattering(false);
    let alpha_no_scatter = skull.total_coefficient(freq);
    assert!((alpha_no_scatter - alpha_abs).abs() < 1e-6);
}
