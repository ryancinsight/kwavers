use super::classify::classify_vessels;
use super::frangi::{compute_frangi_response, symmetric_3x3_eigenvalues};
use super::*;

#[test]
fn test_vessel_segmentation_creation() {
    let image = Array3::ones((10, 10, 10));
    let result = VesselSegmentation::segment(&image);
    assert!(result.is_ok());
}

#[test]
fn test_vessel_segmentation_rejects_small_image() {
    let image = Array3::ones((2, 2, 2));
    let result = VesselSegmentation::segment(&image);
    assert!(result.is_err());
}

#[test]
fn test_vessel_classification_uses_static_contrast_and_geometry() {
    let mut image = Array3::zeros((10, 10, 10));
    let mut mask = Array3::zeros((10, 10, 10));
    // A thin axial vessel along x at (j=5, k=5)
    for i in 2..8 {
        image[[i, 5, 5]] = 5.0;
        mask[[i, 5, 5]] = 1.0;
    }

    let classification = classify_vessels(&image, &mask).unwrap();
    assert_eq!(classification.vessel_type, VesselType::Artery);
    assert!(
        classification.confidence > 0.9,
        "confidence = {}",
        classification.confidence
    );
    assert!(
        classification.diameter > 1.0,
        "diameter = {}",
        classification.diameter
    );
    // Principal axis must be predominantly along x
    assert!(
        classification.orientation[0].abs() > 0.99,
        "orientation = {:?}",
        classification.orientation
    );
    // Flow direction matches orientation for arteries
    assert_eq!(
        classification.flow_direction.unwrap()[0].abs(),
        classification.orientation[0].abs()
    );
}

#[test]
fn test_frangi_response_shape_and_nonnegativity() {
    let image = Array3::ones((10, 10, 10));
    let result = compute_frangi_response(&image);
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.dim(), (10, 10, 10));
    assert!(
        response.iter().all(|&v| v >= 0.0),
        "Frangi response must be non-negative"
    );
}

#[test]
fn test_centerline_extracts_thin_vessel_axis() {
    let mut mask = Array3::zeros((10, 10, 10));
    for i in 2..8 {
        mask[[i, 5, 5]] = 1.0;
    }
    let segmentation = VesselSegmentation {
        mask,
        response: Array3::zeros((10, 10, 10)),
        classification: VesselClassification {
            vessel_type: VesselType::Artery,
            confidence: 1.0,
            diameter: 1.0,
            orientation: [1.0, 0.0, 0.0],
            flow_direction: Some([1.0, 0.0, 0.0]),
        },
        num_segments: 1,
        total_length: 6.0,
    };

    let centerline = segmentation.extract_centerline().unwrap();
    assert_eq!(
        centerline.len(),
        6,
        "thin vessel should be its own centerline, got {:?}",
        centerline
    );
    // Sorted by flood-fill order; all points lie on the axial line
    for pt in &centerline {
        assert_eq!(pt[1], 5.0);
        assert_eq!(pt[2], 5.0);
    }
}

#[test]
fn test_doppler_velocity_formula() {
    // v = f_d · c / (2 f₀ · cos θ) = 2000 * 1540 / (2 * 5_000_000 * 1.0)
    //   = 3_080_000 / 10_000_000 = 0.308 m/s
    let v =
        VesselSegmentation::estimate_flow_velocity_from_doppler(2_000.0, 5_000_000.0, 1540.0, 0.0)
            .unwrap();
    assert!((v - 0.308).abs() < 1e-12, "expected 0.308 m/s, got {v}");
}

#[test]
fn test_doppler_velocity_invalid_inputs() {
    // Perpendicular beam
    assert!(VesselSegmentation::estimate_flow_velocity_from_doppler(
        2000.0,
        5e6,
        1540.0,
        std::f64::consts::FRAC_PI_2
    )
    .is_err());
    // Negative frequency
    assert!(
        VesselSegmentation::estimate_flow_velocity_from_doppler(2000.0, -5e6, 1540.0, 0.0).is_err()
    );
}

#[test]
fn test_static_flow_velocity_is_error() {
    let seg = VesselSegmentation {
        mask: Array3::zeros((3, 3, 3)),
        response: Array3::zeros((3, 3, 3)),
        classification: VesselClassification {
            vessel_type: VesselType::Unknown,
            confidence: 0.0,
            diameter: 0.0,
            orientation: [0.0, 0.0, 0.0],
            flow_direction: None,
        },
        num_segments: 0,
        total_length: 0.0,
    };
    assert!(seg.estimate_flow_velocity().is_err());
}

#[test]
fn test_cardano_eigenvalues_diagonal_matrix() {
    // Diagonal matrix: eigenvalues are the diagonal entries.
    let h = [3.0_f64, 1.0, 2.0, 0.0, 0.0, 0.0];
    let (e1, e2, e3) = symmetric_3x3_eigenvalues(h);
    let mut eigs = [e1, e2, e3];
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((eigs[0] - 1.0).abs() < 1e-10);
    assert!((eigs[1] - 2.0).abs() < 1e-10);
    assert!((eigs[2] - 3.0).abs() < 1e-10);
}
