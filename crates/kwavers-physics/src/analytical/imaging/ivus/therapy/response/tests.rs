use super::ivus_therapy_response;
use super::IvusTherapyDose;
use super::IvusTherapyResponseInput;
use super::IvusTissueMasks;

#[test]
fn ivus_therapy_response_matches_closed_forms() {
    let pressure = [1.0e6, 1.0e6];
    let radius = [2.30e-3, 2.30e-3];
    let attenuation = [1.0, 1.0];
    let eel = [true, true];
    let lumen = [false, false];
    let cap = [false, true];
    let lipid = [false, false];
    let plaque = [false, true];

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

    let intensity = 1.0e12 / (2.0 * 1000.0 * 1500.0);
    let alpha_eff = 1.0 * 100.0 / 8.686 * 2.0;
    let expected_delta_t = 2.0 * alpha_eff * intensity * 0.25 * 0.50 / (1000.0 * 4000.0);
    let off_deposition = 1.0 - (-0.6_f64).exp();
    let target_deposition = 1.0 - (-3.0_f64).exp();

    assert!((response.intensity_w_m2[0] - intensity).abs() < 1.0e-9);
    assert!((response.temperature_rise_k[0] - expected_delta_t).abs() < 1.0e-12);
    assert!((response.peak_delta_t_k - expected_delta_t).abs() < 1.0e-12);
    assert!((response.deposition[0] - off_deposition).abs() < 1.0e-12);
    assert!((response.deposition[1] - target_deposition).abs() < 1.0e-12);
    assert!((response.mechanical_index - 1.0 / 2.0_f64.sqrt()).abs() < 1.0e-12);
    assert!(
        (response.target_to_offtarget_ratio - target_deposition / off_deposition).abs() < 1.0e-12
    );
}

#[test]
fn ivus_therapy_response_rejects_missing_target() {
    let err = ivus_therapy_response(IvusTherapyResponseInput {
        pressure_pa: &[1.0],
        radius_m: &[1.0e-3],
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
        delivery_radial_center_m: 1.0e-3,
        delivery_radial_width_m: 1.0e-3,
    })
    .unwrap_err();
    assert!(err.contains("target mask"));
}
