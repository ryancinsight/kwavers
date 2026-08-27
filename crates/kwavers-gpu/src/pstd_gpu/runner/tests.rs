use super::{
    collect_sensor_indices, cpml_thickness_limits, has_nonlinear_coefficient,
    validate_sensor_mask_shape, LetoArray3,
};
use kwavers_grid::Grid;

#[test]
fn collect_sensor_indices_preserves_row_major_positions() {
    let mask = LetoArray3::from_shape_vec(
        [2, 2, 2],
        vec![false, true, false, false, true, false, false, true],
    )
    .expect("test mask shape matches storage");

    let indices = collect_sensor_indices(&mask).expect("dense Leto mask is valid");

    assert_eq!(indices, vec![1, 4, 7]);
}

#[test]
fn nonlinear_selection_scans_every_packed_medium_cell() {
    assert!(!has_nonlinear_coefficient(&[0.0, 0.0, 0.0]));
    assert!(has_nonlinear_coefficient(&[0.0, 0.0, 2.6]));
}

#[test]
fn automatic_cpml_is_zero_when_the_grid_cannot_hold_a_complete_face() {
    assert_eq!(cpml_thickness_limits(7, 4, 3), (0, 0));
    assert_eq!(cpml_thickness_limits(8, 8, 8), (3, 3));
}

#[test]
fn sensor_mask_shape_must_match_the_grid() {
    let grid = Grid::new(4, 3, 2, 1.0e-3, 1.0e-3, 1.0e-3).expect("valid test grid");
    let mask = LetoArray3::from_elem([4, 2, 2], false);

    let error = validate_sensor_mask_shape(&grid, &mask)
        .expect_err("mismatched sensor geometry must fail before GPU acquisition");

    assert_eq!(
        error.to_string(),
        "Dimension mismatch: sensor_mask shape [4, 2, 2]; expected [4, 3, 2]"
    );
}
