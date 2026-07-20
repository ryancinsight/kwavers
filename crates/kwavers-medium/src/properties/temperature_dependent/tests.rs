use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::thermodynamic::{
    BODY_TEMPERATURE_K, KELVIN_OFFSET_C, ROOM_TEMPERATURE_K,
};

use super::{
    MaterialPropertiesAtT, TemperatureDependentAcoustic, TemperatureDependentMaterial,
    TemperatureDependentThermal,
};

#[test]
fn test_water_sound_speed_temperature_dependence() {
    let water = TemperatureDependentAcoustic::water();
    let t_ref = ROOM_TEMPERATURE_K;
    let t_test = 313.15;

    let c_ref = water.sound_speed(t_ref);
    let c_test = water.sound_speed(t_test);

    assert!(c_test > c_ref);

    let delta_c = c_test - c_ref;
    let expected_delta = c_ref * water.sound_speed_coefficient * (t_test - t_ref);
    assert!((delta_c - expected_delta).abs() < 1e-6);
}

#[test]
fn test_water_density_temperature_dependence() {
    let water = TemperatureDependentAcoustic::water();
    let rho_ref = water.density(ROOM_TEMPERATURE_K);
    let rho_hot = water.density(333.15);

    assert!(rho_hot < rho_ref);
    let delta_rho = rho_ref - rho_hot;
    assert!(delta_rho > 0.0 && delta_rho < 50.0);
}

#[test]
fn test_impedance_temperature_dependence() {
    let water = TemperatureDependentAcoustic::water();
    let z1 = water.impedance(ROOM_TEMPERATURE_K);
    let z2 = water.impedance(BODY_TEMPERATURE_K);

    assert!(z2 > z1);
    let expected_z2 = water.density(BODY_TEMPERATURE_K) * water.sound_speed(BODY_TEMPERATURE_K);
    assert!((z2 - expected_z2).abs() < 1e-6);
}

#[test]
fn test_absorption_temperature_and_frequency_dependence() {
    let tissue = TemperatureDependentAcoustic::soft_tissue();
    let t_ref = BODY_TEMPERATURE_K;
    let t_hot = 320.15;
    let freq = MHZ_TO_HZ;

    let alpha_ref = tissue.absorption(t_ref, freq);
    let alpha_hot = tissue.absorption(t_hot, freq);

    assert!(alpha_hot > alpha_ref);
    let alpha_2mhz = tissue.absorption(t_ref, 2.0 * MHZ_TO_HZ);
    assert!(alpha_2mhz > alpha_ref);
}

#[test]
fn test_thermal_conductivity_temperature_dependence() {
    let thermal = TemperatureDependentThermal::water();
    let base = super::super::ThermalPropertyData::water();
    let k_ref = thermal
        .conductivity(ROOM_TEMPERATURE_K)
        .expect("reference temperature is physical");
    let k_hot = thermal
        .conductivity(313.15)
        .expect("elevated temperature is physical");
    let delta = 313.15 - ROOM_TEMPERATURE_K;
    let expected_factor = (-1.0e-5 * delta).mul_add(delta, 0.002_f64.mul_add(delta, 1.0));
    let expected = base.conductivity() * expected_factor;
    assert_eq!(k_ref.to_bits(), base.conductivity().to_bits());
    assert_eq!(k_hot.to_bits(), expected.to_bits());
    assert!(k_hot > k_ref);
}

#[test]
fn thermal_response_rejects_non_finite_inputs() {
    let invalid_coefficient = TemperatureDependentThermal::new(
        super::super::ThermalPropertyData::water(),
        ROOM_TEMPERATURE_K,
        f64::NAN,
        0.0,
        0.0,
    );
    assert!(invalid_coefficient.is_err());

    let water = TemperatureDependentThermal::water();
    assert!(water.properties(f64::NAN).is_err());
}

#[test]
fn test_combined_material_properties() {
    let water = TemperatureDependentMaterial::water();
    let props = water
        .properties_at_temperature(300.0)
        .expect("300 K produces physical thermal properties");

    assert!(props.sound_speed > 1400.0 && props.sound_speed < 1600.0);
    assert!(props.density > 900.0 && props.density < 1100.0);
    assert!(props.impedance > 1.0e6 && props.impedance < 2.0e6);
    assert!(props.thermal_conductivity > 0.5 && props.thermal_conductivity < 0.7);
    let expected_diffusivity = props.thermal_conductivity / (props.density * props.specific_heat);
    assert_eq!(
        props.thermal_diffusivity.to_bits(),
        expected_diffusivity.to_bits()
    );
}

#[test]
fn test_duck_1990_water_data_validation() {
    let water = TemperatureDependentAcoustic::water();

    let c_20c = water.sound_speed(ROOM_TEMPERATURE_K);
    assert!((c_20c - 1481.0).abs() < 10.0);

    let c_37c = water.sound_speed(BODY_TEMPERATURE_K);
    let expected_37c = 1481.0 * (1.0 + 0.002 * 17.0);
    assert!((c_37c - expected_37c).abs() < 10.0);
}

#[test]
fn test_soft_tissue_properties_physiological_range() {
    let tissue = TemperatureDependentAcoustic::soft_tissue();

    for t_celsius in 35..=40 {
        let t_kelvin = t_celsius as f64 + KELVIN_OFFSET_C;
        let c = tissue.sound_speed(t_kelvin);
        let rho = tissue.density(t_kelvin);

        assert!(
            c > 1450.0 && c < 1600.0,
            "Sound speed {c} m/s out of range at {t_celsius} °C"
        );
        assert!(
            rho > 1000.0 && rho < 1100.0,
            "Density {rho} kg/m³ out of range at {t_celsius} °C"
        );
    }
}

#[test]
fn test_validation_physical_constraints() {
    let water = TemperatureDependentAcoustic::water();

    let result =
        TemperatureDependentAcoustic::new(water.base_properties, 100.0, 0.002, 2.1e-4, 0.02);
    assert!(result.is_err());

    let result = TemperatureDependentAcoustic::new(
        water.base_properties,
        ROOM_TEMPERATURE_K,
        0.02,
        2.1e-4,
        0.02,
    );
    assert!(result.is_err());
}

// Suppress unused import warning — MaterialPropertiesAtT is accessed by field in test_combined_material_properties
const _: fn() = || {
    let _: MaterialPropertiesAtT;
};
