//! Unit tests for elastography algorithm primitives.

use leto::Array3;

use kwavers_grid::Grid;
use kwavers_physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::{
    directional_smoothing, fill_boundaries, find_push_locations, spatial_smoothing,
    volumetric_smoothing,
};

#[test]
fn test_spatial_smoothing() {
    let mut field = Array3::from_elem((10, 10, 10), 3.0);
    field[[5, 5, 5]] = 10.0;

    spatial_smoothing(&mut field);

    // Spike reduced but not below background.
    assert!(field[[5, 5, 5]] < 10.0);
    assert!(field[[5, 5, 5]] > 3.0);
}

#[test]
fn test_volumetric_smoothing_preserves_uniform() {
    let mut field = Array3::from_elem((10, 10, 10), 5.0);
    let original = field.clone();

    volumetric_smoothing(&mut field);

    for i in 1..9 {
        for j in 1..9 {
            for k in 1..9 {
                assert!((field[[i, j, k]] - original[[i, j, k]]).abs() < 0.1);
            }
        }
    }
}

#[test]
fn test_directional_smoothing() {
    let mut field = Array3::from_elem((10, 10, 10), 3.0);
    field[[5, 5, 5]] = 8.0;

    directional_smoothing(&mut field);

    assert!(field[[5, 5, 5]] >= 0.5);
    assert!(field[[5, 5, 5]] <= 10.0);
}

#[test]
fn test_fill_boundaries() {
    let mut array = Array3::zeros((10, 10, 10));
    array[[1, 5, 5]] = 1.0;
    array[[8, 5, 5]] = 2.0;
    array[[5, 1, 5]] = 3.0;
    array[[5, 8, 5]] = 4.0;
    array[[5, 5, 1]] = 5.0;
    array[[5, 5, 8]] = 6.0;

    fill_boundaries(&mut array);

    assert_eq!(array[[0, 5, 5]], 1.0);
    assert_eq!(array[[9, 5, 5]], 2.0);
    assert_eq!(array[[5, 0, 5]], 3.0);
    assert_eq!(array[[5, 9, 5]], 4.0);
    assert_eq!(array[[5, 5, 0]], 5.0);
    assert_eq!(array[[5, 5, 9]], 6.0);
}

#[test]
fn test_find_push_locations_empty() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let displacement = DisplacementField::zeros(20, 20, 20);

    let locations = find_push_locations(&displacement, &grid);

    assert!(locations.is_empty());
}

#[test]
fn test_find_push_locations_single_peak() {
    let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
    let mut displacement = DisplacementField::zeros(20, 20, 20);
    displacement.uz[[10, 10, 10]] = 10.0;

    let locations = find_push_locations(&displacement, &grid);

    assert_eq!((locations.len()), 1);
    assert!((locations[0][0] - 10.0 * grid.dx).abs() < 1e-9);
    assert!((locations[0][1] - 10.0 * grid.dy).abs() < 1e-9);
    assert!((locations[0][2] - 10.0 * grid.dz).abs() < 1e-9);
}

#[test]
fn test_smoothing_preserves_positivity() {
    let mut field = Array3::from_elem((10, 10, 10), 2.0);
    field[[5, 5, 5]] = 5.0;

    spatial_smoothing(&mut field);

    for &val in field.iter() {
        assert!(val > 0.0);
    }
}
