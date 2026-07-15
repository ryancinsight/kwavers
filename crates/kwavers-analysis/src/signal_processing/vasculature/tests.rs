use super::analysis::count_connected_components;
use super::classify::{classify_vessels, principal_axis, vessel_neighbor_count};
use super::frangi::{compute_frangi_response, symmetric_3x3_eigenvalues};
use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;

#[test]
fn test_vessel_segmentation_creation() {
    let image = Array3::ones((10, 10, 10));
    let seg = VesselSegmentation::segment(&image).unwrap();
    assert_eq!(seg.mask.shape(), [10, 10, 10]);
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
    assert_eq!(classification.vessel_type, VascularVesselType::Artery);
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
    let response = compute_frangi_response(&image).unwrap();
    assert_eq!(response.shape(), [10, 10, 10]);
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
            vessel_type: VascularVesselType::Artery,
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
    // v = f_d · c / (2 f₀ · cos θ) = 2000 * SOUND_SPEED_TISSUE / (2 * 5e6 * 1.0)
    //   = 3_080_000 / 10_000_000 = 0.308 m/s
    let expected_v = 2_000.0 * SOUND_SPEED_TISSUE / (2.0 * 5.0 * MHZ_TO_HZ);
    let v = VesselSegmentation::estimate_flow_velocity_from_doppler(
        2_000.0,
        5.0 * MHZ_TO_HZ,
        SOUND_SPEED_TISSUE,
        0.0,
    )
    .unwrap();
    assert!(
        (v - expected_v).abs() < 1e-12,
        "expected {expected_v} m/s, got {v}"
    );
}

#[test]
fn test_doppler_velocity_invalid_inputs() {
    // Perpendicular beam
    assert!(VesselSegmentation::estimate_flow_velocity_from_doppler(
        2000.0,
        5.0 * MHZ_TO_HZ,
        SOUND_SPEED_TISSUE,
        std::f64::consts::FRAC_PI_2
    )
    .is_err());
    // Negative frequency
    assert!(VesselSegmentation::estimate_flow_velocity_from_doppler(
        2000.0,
        -5.0 * MHZ_TO_HZ,
        SOUND_SPEED_TISSUE,
        0.0
    )
    .is_err());
}

#[test]
fn test_static_flow_velocity_is_error() {
    let seg = VesselSegmentation {
        mask: Array3::zeros((3, 3, 3)),
        response: Array3::zeros((3, 3, 3)),
        classification: VesselClassification {
            vessel_type: VascularVesselType::Unknown,
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
    eigs.sort_by(|a, b| a.total_cmp(b));
    assert!((eigs[0] - 1.0).abs() < 1e-10);
    assert!((eigs[1] - 2.0).abs() < 1e-10);
    assert!((eigs[2] - 3.0).abs() < 1e-10);
}

// ─── count_connected_components: exact flood-fill semantics ──────────────────

/// All-zero mask has no vessel voxels → 0 components, 0 voxels.
///
/// No seed passes the `mask > 0.0` predicate; the outer loop
/// increments neither counter.
#[test]
fn count_connected_components_empty_mask_returns_zero_zero() {
    let mask = Array3::<f64>::zeros((4, 4, 4));
    let (n_comp, n_vox) = count_connected_components(&mask);
    assert_eq!(n_comp, 0, "empty mask: expected 0 components, got {n_comp}");
    assert_eq!(n_vox, 0, "empty mask: expected 0 voxels, got {n_vox}");
}

/// A single non-zero voxel is exactly one 6-connected component with 1 voxel.
///
/// Flood-fill seed: `[1,1,1]`; all 6 face-neighbours are zero → no expansion;
/// result: `(1, 1)`.
#[test]
fn count_connected_components_single_voxel_returns_one_one() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[1, 1, 1]] = 1.0;
    let (n_comp, n_vox) = count_connected_components(&mask);
    assert_eq!(
        n_comp, 1,
        "single voxel: expected 1 component, got {n_comp}"
    );
    assert_eq!(n_vox, 1, "single voxel: expected 1 voxel, got {n_vox}");
}

/// Two voxels separated by a zero gap are 2 components (not 6-adjacent).
///
/// `[0,1,1]` and `[2,1,1]` in a 3×3×3 grid: the only candidate
/// face-neighbour in +x from `[0,1,1]` is `[1,1,1]` which is 0,
/// so the flood-fill terminates after 1 voxel each.
/// Result: `(2, 2)`.
#[test]
fn count_connected_components_two_disconnected_voxels_returns_two_two() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[0, 1, 1]] = 1.0;
    mask[[2, 1, 1]] = 1.0; // gap at [1,1,1] = 0
    let (n_comp, n_vox) = count_connected_components(&mask);
    assert_eq!(
        n_comp, 2,
        "two disconnected: expected 2 components, got {n_comp}"
    );
    assert_eq!(
        n_vox, 2,
        "two disconnected: expected 2 total voxels, got {n_vox}"
    );
}

/// Two face-adjacent voxels form one 6-connected component with 2 voxels.
///
/// `[1,1,1]` and `[1,1,2]` share a z-face: the flood-fill from the first
/// seed enqueues and visits the second before the outer loop can seed it.
/// Result: `(1, 2)`.
#[test]
fn count_connected_components_two_adjacent_voxels_returns_one_two() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[1, 1, 1]] = 1.0;
    mask[[1, 1, 2]] = 1.0; // z-face adjacent
    let (n_comp, n_vox) = count_connected_components(&mask);
    assert_eq!(
        n_comp, 1,
        "face-adjacent pair: expected 1 component, got {n_comp}"
    );
    assert_eq!(
        n_vox, 2,
        "face-adjacent pair: expected 2 voxels, got {n_vox}"
    );
}

// ─── vessel_neighbor_count: 6-adjacency counting ─────────────────────────────

/// An isolated vessel voxel has no positive 6-face neighbours → count = 0.
///
/// Voxel `[1,1,1]` in a 3×3×3 grid where every other entry is 0:
/// all six face checks (`mask[[0,1,1]]`, `mask[[2,1,1]]`, …) return 0.0.
#[test]
fn vessel_neighbor_count_isolated_voxel_is_zero() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[1, 1, 1]] = 1.0;
    let count = vessel_neighbor_count(&mask, &[1, 1, 1]);
    assert_eq!(
        count, 0,
        "isolated voxel: expected 0 neighbours, got {count}"
    );
}

/// The center voxel of a full 3×3×3 solid has exactly 6 vessel neighbours.
///
/// Every entry is 1.0; `[1,1,1]` has six face-adjacent voxels all positive.
/// The six Boolean terms all evaluate to `true` → sum = 6.
#[test]
fn vessel_neighbor_count_center_of_solid_cube_is_six() {
    let mask = Array3::<f64>::ones((3, 3, 3));
    let count = vessel_neighbor_count(&mask, &[1, 1, 1]);
    assert_eq!(
        count, 6,
        "solid-cube center: expected 6 neighbours, got {count}"
    );
}

/// The endpoint of a 3-voxel x-chain has exactly 1 vessel neighbour.
///
/// Chain: `[0,1,1]`, `[1,1,1]`, `[2,1,1]`.
/// For endpoint `[0,1,1]`: `i>0` is false (−x blocked by boundary),
/// `i+1=1` → `mask[[1,1,1]] = 1.0` (the only positive face-neighbour).
/// All y/z face-neighbours are 0.  Count = 1.
#[test]
fn vessel_neighbor_count_chain_endpoint_is_one() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[0, 1, 1]] = 1.0;
    mask[[1, 1, 1]] = 1.0;
    mask[[2, 1, 1]] = 1.0;
    let count = vessel_neighbor_count(&mask, &[0, 1, 1]);
    assert_eq!(
        count, 1,
        "chain endpoint: expected 1 neighbour, got {count}"
    );
}

/// The middle voxel of a 3-voxel x-chain has exactly 2 vessel neighbours.
///
/// Chain: `[0,1,1]`, `[1,1,1]`, `[2,1,1]`.
/// For middle `[1,1,1]`: `mask[[0,1,1]] = 1.0` and `mask[[2,1,1]] = 1.0`;
/// all y/z face-neighbours are 0.  Count = 2.
#[test]
fn vessel_neighbor_count_chain_middle_is_two() {
    let mut mask = Array3::<f64>::zeros((3, 3, 3));
    mask[[0, 1, 1]] = 1.0;
    mask[[1, 1, 1]] = 1.0;
    mask[[2, 1, 1]] = 1.0;
    let count = vessel_neighbor_count(&mask, &[1, 1, 1]);
    assert_eq!(count, 2, "chain middle: expected 2 neighbours, got {count}");
}

// ─── principal_axis: power-iteration convergence ─────────────────────────────

/// An x-aligned collinear point set converges to the x unit vector.
///
/// Points: `[k, 1, 1]` for k ∈ 0..5.  Centred coordinates have non-zero
/// variance only along x.  With initial axis `[1, 0, 0]`:
///   proj = d[0]·1 + d[1]·0 + d[2]·0 = d[0]
///   next = [Σd[0]², 0, 0] = [10, 0, 0],  norm = 10
///   → axis = [1, 0, 0]  (converges on first step and stays).
/// Remaining 11 iterations preserve the fixed point.
/// Assert: |axis[0]| > 0.999, |axis[1]| < 1e-10, |axis[2]| < 1e-10.
#[test]
fn principal_axis_x_aligned_points_returns_x_unit_vector() {
    let points: Vec<[usize; 3]> = (0..5).map(|k| [k, 1, 1]).collect();
    let axis = principal_axis(&points);
    assert!(
        axis[0].abs() > 0.999,
        "x-aligned chain: expected |axis[0]| ≈ 1, got {axis:?}"
    );
    assert!(
        axis[1].abs() < 1e-10,
        "x-aligned chain: expected axis[1] ≈ 0, got {axis:?}"
    );
    assert!(
        axis[2].abs() < 1e-10,
        "x-aligned chain: expected axis[2] ≈ 0, got {axis:?}"
    );
}
