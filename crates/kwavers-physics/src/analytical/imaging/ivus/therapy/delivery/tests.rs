use super::ivus_microbubble_delivery_fraction;
use super::IvusMicrobubbleDeliveryInput;

#[test]
fn ivus_microbubble_delivery_fraction_matches_weighted_force_model() {
    let range = [0.0, 1.75e-3, 1.75e-3];
    let attenuation = [10.0, 10.0, 10.0];
    let intensity = [1.0, 1.0, 1.0];
    let wall = [false, true, true];
    let target = [false, false, true];

    let delivered = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
        range_m: &range,
        attenuation_np_m: &attenuation,
        intensity_w_m2: &intensity,
        wall_mask: &wall,
        target_mask: &target,
        sound_speed_m_s: 1540.0,
        radial_center_m: 1.75e-3,
        radial_width_m: 1.2e-3,
    })
    .unwrap();

    assert_eq!(delivered[0], 0.0);
    let expected_wall = 1.0 - (-3.0_f64 * 0.2).exp();
    let expected_target = 1.0 - (-3.0_f64).exp();
    assert!((delivered[1] - expected_wall).abs() < 1.0e-12);
    assert!((delivered[2] - expected_target).abs() < 1.0e-12);
}

#[test]
fn ivus_microbubble_delivery_fraction_rejects_invalid_inputs() {
    let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
        range_m: &[0.0],
        attenuation_np_m: &[],
        intensity_w_m2: &[1.0],
        wall_mask: &[true],
        target_mask: &[false],
        sound_speed_m_s: 1540.0,
        radial_center_m: 1.75e-3,
        radial_width_m: 1.2e-3,
    })
    .unwrap_err();
    assert!(err.contains("all arrays must have length"));

    let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
        range_m: &[0.0],
        attenuation_np_m: &[1.0],
        intensity_w_m2: &[f64::NAN],
        wall_mask: &[true],
        target_mask: &[false],
        sound_speed_m_s: 1540.0,
        radial_center_m: 1.75e-3,
        radial_width_m: 1.2e-3,
    })
    .unwrap_err();
    assert!(err.contains("intensity_w_m2[0]"));

    let err = ivus_microbubble_delivery_fraction(IvusMicrobubbleDeliveryInput {
        range_m: &[0.0],
        attenuation_np_m: &[1.0],
        intensity_w_m2: &[1.0],
        wall_mask: &[true],
        target_mask: &[false],
        sound_speed_m_s: 0.0,
        radial_center_m: 1.75e-3,
        radial_width_m: 1.2e-3,
    })
    .unwrap_err();
    assert!(err.contains("sound_speed_m_s"));
}
