use super::ivus_scan_convert;
use std::f64::consts::PI;

#[test]
fn ivus_scan_convert_maps_nearest_bins_and_radial_bounds() {
    let polar = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let r_axis = [1.0e-3, 2.0e-3];
    let theta_axis = [-PI, -0.5 * PI, 0.0, 0.5 * PI];
    let radius = [0.5e-3, 1.0e-3, 2.0e-3, 2.0e-3, 2.5e-3];
    let theta = [-PI, -PI, 0.0, PI, 0.0];

    let image = ivus_scan_convert(&polar, &r_axis, &theta_axis, &radius, &theta).unwrap();

    assert_eq!(image, vec![0.0, 1.0, 7.0, 5.0, 0.0]);
}

#[test]
fn ivus_scan_convert_rejects_invalid_inputs() {
    let err = ivus_scan_convert(&[1.0], &[1.0], &[-PI, 0.0], &[1.0], &[0.0]).unwrap_err();
    assert!(err.contains("at least two"));

    let err = ivus_scan_convert(
        &[1.0, f64::NAN, 3.0, 4.0],
        &[1.0, 2.0],
        &[-PI, 0.0],
        &[1.0],
        &[0.0],
    )
    .unwrap_err();
    assert!(err.contains("polar[1]"));
}
