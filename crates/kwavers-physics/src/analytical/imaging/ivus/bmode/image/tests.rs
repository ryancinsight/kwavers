use super::ivus_bmode_image;
use super::IvusBmodeImageInput;
use crate::analytical::imaging::ivus::bmode::scan::ivus_scan_convert;
use std::f64::consts::PI;

#[test]
fn ivus_bmode_image_returns_consistent_polar_and_cartesian_images() {
    let x = vec![-1.0e-3, -1.0e-3, 1.0e-3, 1.0e-3];
    let y = vec![-1.0e-3, 1.0e-3, -1.0e-3, 1.0e-3];
    let backscatter = vec![1.0, 2.0, 3.0, 4.0];
    let attenuation = vec![0.0, 1.0, 2.0, 3.0];
    let r_axis = [1.0e-3, 1.55e-3, 2.0e-3, 2.45e-3];
    let theta_axis = [-PI, -0.5 * PI, 0.0, 0.5 * PI];
    let radius = [0.5e-3, 1.0e-3, 2.0e-3, 2.5e-3];
    let theta = [-PI, -PI, 0.0, 0.0];

    let image = ivus_bmode_image(IvusBmodeImageInput {
        x_m: &x,
        y_m: &y,
        backscatter: &backscatter,
        attenuation_db_cm_mhz: &attenuation,
        r_axis_m: &r_axis,
        theta_axis_rad: &theta_axis,
        radius_m: &radius,
        theta_m_rad: &theta,
        catheter_radius_m: 1.0e-3,
        frequency_hz: 20.0e6,
        floor_db: -60.0,
        ring_amplitude: 0.10,
        ring_width_m: 0.22e-3,
    })
    .unwrap();

    assert_eq!(image.rf.len(), r_axis.len() * theta_axis.len());
    assert_eq!(image.envelope.len(), image.rf.len());
    assert_eq!(image.db.len(), image.rf.len());
    assert_eq!(image.polar.len(), image.rf.len());
    assert_eq!(image.cartesian.len(), radius.len());
    assert!(image
        .polar
        .iter()
        .all(|&value| (0.0..=1.0).contains(&value)));
    assert!(image.db.iter().all(|&value| (-60.0..=0.0).contains(&value)));
    let expected_cartesian =
        ivus_scan_convert(&image.polar, &r_axis, &theta_axis, &radius, &theta).unwrap();
    assert_eq!(image.cartesian, expected_cartesian);
}

#[test]
fn ivus_bmode_image_rejects_invalid_floor() {
    let err = ivus_bmode_image(IvusBmodeImageInput {
        x_m: &[0.0, 0.0, 1.0, 1.0],
        y_m: &[0.0, 1.0, 0.0, 1.0],
        backscatter: &[1.0, 1.0, 1.0, 1.0],
        attenuation_db_cm_mhz: &[0.0, 0.0, 0.0, 0.0],
        r_axis_m: &[1.0, 2.0],
        theta_axis_rad: &[0.0, 1.0],
        radius_m: &[1.0],
        theta_m_rad: &[0.0],
        catheter_radius_m: 0.0,
        frequency_hz: 1.0,
        floor_db: 0.0,
        ring_amplitude: 0.0,
        ring_width_m: 1.0,
    })
    .unwrap_err();
    assert!(err.contains("floor_db"));
}
