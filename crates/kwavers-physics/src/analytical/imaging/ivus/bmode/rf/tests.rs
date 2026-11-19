use super::ivus_polar_bmode_rf;
use super::IvusPolarBmodeRfInput;

#[test]
fn ivus_polar_bmode_rf_matches_two_way_attenuation_and_ring() {
    let x = vec![-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
    let y = vec![-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
    let backscatter = vec![1.0, 2.0, 3.0, 4.0];
    let attenuation = vec![0.0, 1.0, 2.0, 3.0];
    let radius = [1.0e-3, 1.55e-3];
    let theta = [0.0];

    let rf = ivus_polar_bmode_rf(IvusPolarBmodeRfInput {
        x_m: &x,
        y_m: &y,
        backscatter: &backscatter,
        attenuation_db_cm_mhz: &attenuation,
        r_axis_m: &radius,
        theta_axis_rad: &theta,
        catheter_radius_m: 1.0e-3,
        frequency_hz: 20.0e6,
        ring_amplitude: 0.10,
        ring_width_m: 0.22e-3,
    })
    .unwrap();

    assert_eq!(rf[0], 4.0 + 0.10);
    let alpha = 3.0 * 100.0 / 8.686;
    let expected = 4.0 * (-2.0_f64 * alpha * 20.0 * 0.55e-3).exp()
        + 0.10 * (-(0.55e-3_f64 / 0.22e-3_f64).powi(2)).exp();
    assert!((rf[1] - expected).abs() < 1.0e-12);
}

#[test]
fn ivus_polar_bmode_rf_rejects_invalid_inputs() {
    let err = ivus_polar_bmode_rf(IvusPolarBmodeRfInput {
        x_m: &[0.0, 1.0],
        y_m: &[0.0, 1.0],
        backscatter: &[1.0, 1.0],
        attenuation_db_cm_mhz: &[0.0, 0.0],
        r_axis_m: &[1.0],
        theta_axis_rad: &[0.0],
        catheter_radius_m: 0.0,
        frequency_hz: 1.0,
        ring_amplitude: 0.0,
        ring_width_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("square grid"));

    let err = ivus_polar_bmode_rf(IvusPolarBmodeRfInput {
        x_m: &[0.0, 0.0, 1.0, 1.0],
        y_m: &[0.0, 1.0, 0.0, 1.0],
        backscatter: &[1.0, f64::NAN, 1.0, 1.0],
        attenuation_db_cm_mhz: &[0.0, 0.0, 0.0, 0.0],
        r_axis_m: &[1.0],
        theta_axis_rad: &[0.0],
        catheter_radius_m: 0.0,
        frequency_hz: 1.0,
        ring_amplitude: 0.0,
        ring_width_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("backscatter[1]"));
}
