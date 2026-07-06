//! Unit tests for `BrainAtlas`.

use leto::Array3 as LetoArray3;
use ndarray::Array3;

use super::BrainAtlas;

#[test]
fn test_brain_atlas_creation() {
    let image = LetoArray3::ones([100, 100, 100]);
    let atlas = BrainAtlas::new(image, [0.01, 0.01, 0.01], [5.0, 5.0, 5.0]).unwrap();
    assert_eq!(atlas.voxel_size(), [0.01, 0.01, 0.01]);
    assert_eq!(atlas.brain_center(), [5.0, 5.0, 5.0]);
}

#[test]
fn test_brain_atlas_default() {
    let atlas = BrainAtlas::load_default().unwrap();
    assert_eq!(atlas.voxel_size(), [0.1, 0.1, 0.1]);
    assert_eq!(atlas.reference_image_ref().shape(), [80, 100, 80]);
    assert!(atlas.reference_image_ref().iter().any(|v| *v == 0.0));
    assert!(atlas.reference_image_ref().iter().any(|v| *v > 0.0));
    assert_ne!(atlas.get_region(&[4.0, 5.0, 3.8]).unwrap(), 0);
}

#[test]
fn test_voxel_to_mm_conversion() {
    let atlas = BrainAtlas::load_default().unwrap();
    let mm = atlas.voxel_to_mm(&[40, 50, 20]);
    assert_eq!(mm, [4.0, 5.0, 2.0]);
}

#[test]
fn test_mm_to_voxel_conversion() {
    let atlas = BrainAtlas::load_default().unwrap();
    // [5.0, 5.0, 5.0] maps to DV = (4.0 − 5.0)/0.1 = −10 → out of bounds.
    assert!(atlas.mm_to_voxel(&[5.0, 5.0, 5.0]).is_err());

    let voxel = atlas.mm_to_voxel(&[4.0, 5.0, 2.0]).unwrap();
    assert_eq!(voxel, [40, 50, 20]);
}

#[test]
fn test_invalid_annotation_shape_is_rejected() {
    let image = LetoArray3::zeros([4, 4, 4]);
    let annotation = Array3::zeros((4, 4, 3));
    let result = BrainAtlas::with_annotation(image, annotation, [0.1, 0.1, 0.1], [0.0; 3]);
    assert!(result.is_err());
}

#[test]
fn test_region_name_lookup() {
    let atlas = BrainAtlas::load_default().unwrap();
    assert_eq!(atlas.get_region_name(1), "Prefrontal Cortex");
    assert_eq!(atlas.get_region_name(6), "Hippocampus");
    assert_eq!(atlas.get_region_name(999), "Unknown Region");
}
