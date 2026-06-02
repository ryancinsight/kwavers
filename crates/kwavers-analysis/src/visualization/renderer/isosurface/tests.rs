use super::IsosurfaceExtractor;
use crate::visualization::VisualizationConfig;
use ndarray::Array3;

#[test]
fn extracts_triangles_for_cube_index_48() {
    let mut field = Array3::zeros((2, 2, 2));
    field[[0, 0, 1]] = 1.0;
    field[[1, 0, 1]] = 1.0;

    let extractor =
        IsosurfaceExtractor::new(&VisualizationConfig::default()).expect("extractor creates");
    let vertices = extractor
        .extract(&field, 0.5)
        .expect("isosurface extraction succeeds");

    assert_eq!(vertices.len(), 6);
    for v in vertices {
        assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        assert!((0.0..=1.0).contains(&v[0]));
        assert!((0.0..=1.0).contains(&v[1]));
        assert!((0.0..=1.0).contains(&v[2]));
    }
}

#[test]
fn extracts_triangles_for_cube_index_1() {
    let mut field = Array3::zeros((2, 2, 2));
    field[[0, 0, 0]] = 1.0;

    let extractor =
        IsosurfaceExtractor::new(&VisualizationConfig::default()).expect("extractor creates");
    let vertices = extractor
        .extract(&field, 0.5)
        .expect("isosurface extraction succeeds");

    assert_eq!(vertices.len(), 3);
    for v in vertices {
        assert!(v[0].is_finite() && v[1].is_finite() && v[2].is_finite());
        assert!((0.0..=1.0).contains(&v[0]));
        assert!((0.0..=1.0).contains(&v[1]));
        assert!((0.0..=1.0).contains(&v[2]));
    }
}
