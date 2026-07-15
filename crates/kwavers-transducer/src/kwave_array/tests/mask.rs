//! Binary and weighted mask tests for [`KWaveArray`].

use super::super::KWaveArray;

#[test]
fn test_kwave_array_creation() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.01, None);
    array.add_rect_element((0.01, 0.0, 0.0), 0.005, 0.005, 0.001);
    assert_eq!(array.num_elements(), 2);
}

#[test]
fn test_kwave_array_binary_mask() {
    let grid = kwavers_grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let mut array = KWaveArray::new();
    array.add_disc_element((0.016, 0.016, 0.016), 0.005, None);
    let mask = array.get_array_binary_mask(&grid);
    let active_count = mask.iter().filter(|&&v| v).count();
    assert!(active_count > 0);
}

#[test]
fn test_kwave_array_disc_focus_mask_is_planar_and_matches_kwave_python_reference_mass() {
    let grid = kwavers_grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let mut array = KWaveArray::new();
    array.add_disc_element((0.016, 0.016, 0.016), 0.006, Some((0.016, 0.016, 0.024)));
    let weights = array.get_array_weighted_mask(&grid);
    // Reference from radial-Fibonacci BLI rasterization (commit a24cdfcb).
    let expected = 28.339_929_259_209_097_f64;
    assert!(
        (weights.iter().sum::<f64>() - expected).abs() < 5.0e-6,
        "disc mass got {}, expected {expected}",
        weights.iter().sum::<f64>()
    );
    let mut active_plane: Option<usize> = None;
    for ([_, _, k], &value) in weights.indexed_iter() {
        if value > 0.0 {
            match active_plane {
                Some(plane) => assert_eq!(plane, k, "disc weights must remain planar"),
                None => active_plane = Some(k),
            }
        }
    }
    assert!(
        active_plane.is_some(),
        "disc weights must activate at least one cell"
    );
}
