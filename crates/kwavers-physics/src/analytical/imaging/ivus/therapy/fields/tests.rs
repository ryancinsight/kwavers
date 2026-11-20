use super::ivus_therapy_fields;
use super::IvusTherapyFieldsInput;
use crate::analytical::imaging::ivus::therapy::pressure::ivus_therapy_pressure_field;
use crate::analytical::imaging::ivus::therapy::pressure::IvusTherapyPressureFieldInput;
use crate::analytical::imaging::ivus::therapy::response::ivus_therapy_response;
use crate::analytical::imaging::ivus::therapy::response::IvusTherapyDose;
use crate::analytical::imaging::ivus::therapy::response::IvusTherapyResponseInput;
use crate::analytical::imaging::ivus::therapy::response::IvusTissueMasks;

#[test]
fn ivus_therapy_fields_matches_pressure_and_response_helpers() {
    let radius = [0.55e-3, 2.30e-3, 2.30e-3];
    let theta = [-0.72, -0.72, -0.72];
    let attenuation = [1.0, 1.0, 1.0];
    let eel = [false, true, true];
    let lumen = [false, false, false];
    let cap = [false, false, true];
    let lipid = [false, false, false];
    let plaque = [false, false, true];

    let fields = ivus_therapy_fields(IvusTherapyFieldsInput {
        radius_m: &radius,
        theta_rad: &theta,
        attenuation_db_cm_mhz: &attenuation,
        masks: IvusTissueMasks {
            eel_mask: &eel,
            lumen_mask: &lumen,
            fibrous_cap_mask: &cap,
            lipid_mask: &lipid,
            plaque_mask: &plaque,
        },
        dose: IvusTherapyDose {
            catheter_radius_m: 0.55e-3,
            therapy_frequency_hz: 2.0e6,
            therapy_duty_cycle: 0.25,
            therapy_sonication_s: 0.50,
            density_kg_m3: 1000.0,
            sound_speed_m_s: 1500.0,
            specific_heat_j_kg_k: 4000.0,
        },
        therapy_pressure_pa: 1.0e6,
        therapy_azimuth_rad: -0.72,
        therapy_sector_width_rad: 0.50,
        pressure_attenuation_length_m: 3.2e-3,
        delivery_radial_center_m: 1.75e-3,
        delivery_radial_width_m: 1.2e-3,
    })
    .unwrap();

    let pressure = ivus_therapy_pressure_field(IvusTherapyPressureFieldInput {
        radius_m: &radius,
        theta_rad: &theta,
        catheter_radius_m: 0.55e-3,
        peak_pressure_pa: 1.0e6,
        therapy_azimuth_rad: -0.72,
        sector_width_rad: 0.50,
        attenuation_length_m: 3.2e-3,
    })
    .unwrap();
    let response = ivus_therapy_response(IvusTherapyResponseInput {
        pressure_pa: &pressure,
        radius_m: &radius,
        attenuation_db_cm_mhz: &attenuation,
        masks: IvusTissueMasks {
            eel_mask: &eel,
            lumen_mask: &lumen,
            fibrous_cap_mask: &cap,
            lipid_mask: &lipid,
            plaque_mask: &plaque,
        },
        dose: IvusTherapyDose {
            catheter_radius_m: 0.55e-3,
            therapy_frequency_hz: 2.0e6,
            therapy_duty_cycle: 0.25,
            therapy_sonication_s: 0.50,
            density_kg_m3: 1000.0,
            sound_speed_m_s: 1500.0,
            specific_heat_j_kg_k: 4000.0,
        },
        delivery_radial_center_m: 1.75e-3,
        delivery_radial_width_m: 1.2e-3,
    })
    .unwrap();

    assert_eq!(fields.pressure_pa, pressure);
    assert_eq!(fields.intensity_w_m2, response.intensity_w_m2);
    assert_eq!(fields.temperature_rise_k, response.temperature_rise_k);
    assert_eq!(fields.deposition, response.deposition);
    assert_eq!(fields.mechanical_index, response.mechanical_index);
    assert_eq!(
        fields.target_to_offtarget_ratio,
        response.target_to_offtarget_ratio
    );
    assert_eq!(fields.peak_delta_t_k, response.peak_delta_t_k);
}

#[test]
fn ivus_therapy_fields_rejects_missing_target() {
    let err = ivus_therapy_fields(IvusTherapyFieldsInput {
        radius_m: &[1.0e-3],
        theta_rad: &[0.0],
        attenuation_db_cm_mhz: &[1.0],
        masks: IvusTissueMasks {
            eel_mask: &[true],
            lumen_mask: &[false],
            fibrous_cap_mask: &[false],
            lipid_mask: &[false],
            plaque_mask: &[false],
        },
        dose: IvusTherapyDose {
            catheter_radius_m: 0.5e-3,
            therapy_frequency_hz: 2.0e6,
            therapy_duty_cycle: 0.25,
            therapy_sonication_s: 0.5,
            density_kg_m3: 1000.0,
            sound_speed_m_s: 1500.0,
            specific_heat_j_kg_k: 4000.0,
        },
        therapy_pressure_pa: 1.0e6,
        therapy_azimuth_rad: 0.0,
        therapy_sector_width_rad: 0.5,
        pressure_attenuation_length_m: 1.0e-3,
        delivery_radial_center_m: 1.0e-3,
        delivery_radial_width_m: 1.0e-3,
    })
    .unwrap_err();
    assert!(err.contains("target mask"));
}
