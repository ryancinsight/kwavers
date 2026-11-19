use super::ivus_therapy_pressure_field;
use super::IvusTherapyPressureFieldInput;

#[test]
fn ivus_therapy_pressure_field_matches_sector_decay() {
    let catheter = 0.55e-3;
    let peak = 300.0e3;
    let azimuth = -0.72;
    let width = 0.50;
    let decay = 3.2e-3;
    let radius = [catheter, catheter + decay, catheter + decay];
    let theta = [azimuth, azimuth, azimuth + width];

    let pressure = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m: &radius,
        theta_rad: &theta,
        catheter_radius_m: catheter,
        peak_pressure_pa: peak,
        therapy_azimuth_rad: azimuth,
        sector_width_rad: width,
        attenuation_length_m: decay,
    })
    .unwrap();

    assert_eq!(pressure[0], 0.0);
    assert!((pressure[1] - peak / std::f64::consts::E).abs() < 1.0e-9);
    assert!((pressure[2] - peak * (-1.5_f64).exp()).abs() < 1.0e-9);
}

#[test]
fn ivus_therapy_pressure_field_rejects_invalid_inputs() {
    let err = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m: &[1.0],
        theta_rad: &[],
        catheter_radius_m: 0.0,
        peak_pressure_pa: 1.0,
        therapy_azimuth_rad: 0.0,
        sector_width_rad: 1.0,
        attenuation_length_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("radius_m length"));

    let err = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m: &[f64::NAN],
        theta_rad: &[0.0],
        catheter_radius_m: 0.0,
        peak_pressure_pa: 1.0,
        therapy_azimuth_rad: 0.0,
        sector_width_rad: 1.0,
        attenuation_length_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("radius_m[0] must be finite"));

    let err = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m: &[1.0],
        theta_rad: &[0.0],
        catheter_radius_m: 0.0,
        peak_pressure_pa: 1.0,
        therapy_azimuth_rad: 0.0,
        sector_width_rad: 0.0,
        attenuation_length_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("sector_width_rad"));
}
