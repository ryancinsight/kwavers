//! Vessel classification, geometry, and medial-axis extraction.
//!
//! # Classification algorithm
//!
//! For a connected tubular mask sampled on a Cartesian grid, the first
//! principal component of masked voxel coordinates is the least-squares
//! vessel axis (power-iteration approximation, 12 iterations).
//!
//! With voxel spacing `(s_x, s_y, s_z)`, the equivalent circular diameter
//! follows from `V = N_voxels s_x s_y s_z`, `A = V / L`, and:
//!
//! ```text
//!   A = V / L,   d = sqrt(4A / π)
//! ```
//!
//! The artery/vein label is derived from static intensity contrast:
//!
//! ```text
//!   contrast = |μ_vessel − μ_background| / max(|μ_vessel|, |μ_background|)
//! ```
//!
//! Confidence is clamped to [0, 0.95] because static fUS cannot prove
//! pulsatility; the label is `Artery` when `μ_vessel ≥ μ_background` and
//! `contrast ≥ 0.1`, otherwise `Vein`, and `Unknown` below 0.1.
//!
//! # Centerline approximation
//!
//! Interior voxels with ≤ 2 vessel-class 6-neighbours are classified as
//! centerline candidates (thin medial axis).  This is exact for 6-connected
//! linear segments and a conservative approximation for branching vessels.

use aequitas::systems::si::quantities::Length;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

use super::{VascularVesselType, VesselClassification, VoxelSpacing};

/// Classify vessels from a static fUS image and its binary mask.
///
/// See module-level docs for the mathematical specification.
///
/// # Errors
/// `InvalidInput` when `image` and `mask` shapes differ.
pub(super) fn classify_vessels(
    image: &Array3<f64>,
    mask: &Array3<f64>,
    spacing: VoxelSpacing,
) -> KwaversResult<VesselClassification> {
    if image.shape() != mask.shape() {
        return Err(KwaversError::InvalidInput(
            "vessel image and mask shapes must match".to_owned(),
        ));
    }

    let points = masked_points(mask);
    if points.is_empty() {
        return Ok(VesselClassification {
            vessel_type: VascularVesselType::Unknown,
            confidence: 0.0,
            diameter: Length::from_base(0.0),
            orientation: [0.0, 0.0, 0.0],
            flow_direction: None,
        });
    }

    let [sx, sy, sz] = spacing.base_values();
    let physical_points = points
        .iter()
        .map(|&[i, j, k]| [i as f64 * sx, j as f64 * sy, k as f64 * sz])
        .collect::<Vec<_>>();
    let orientation = principal_axis_values(&physical_points);
    let centerline = centerline_from_points(mask, &points);
    let length = centerline_length(&centerline, orientation, spacing)
        .into_base()
        .max(f64::MIN_POSITIVE);
    let voxel_volume = sx * sy * sz;
    let diameter = Length::from_base(
        (4.0 * points.len() as f64 * voxel_volume / (std::f64::consts::PI * length)).sqrt(),
    );

    // Compute vessel vs background intensity means.
    let mut vessel_sum = 0.0_f64;
    let mut background_sum = 0.0_f64;
    let mut vessel_count = 0_usize;
    let mut background_count = 0_usize;

    for ([i, j, k], &value) in image.indexed_iter() {
        if mask[[i, j, k]] > 0.0 {
            vessel_sum += value;
            vessel_count += 1;
        } else {
            background_sum += value;
            background_count += 1;
        }
    }

    let vessel_mean = vessel_sum / vessel_count.max(1) as f64;
    let background_mean = background_sum / background_count.max(1) as f64;
    let contrast = (vessel_mean - background_mean).abs()
        / vessel_mean.abs().max(background_mean.abs()).max(1e-12);

    let confidence = contrast.clamp(0.0, 0.95);
    let vessel_type = if confidence < 0.1 {
        VascularVesselType::Unknown
    } else if vessel_mean >= background_mean {
        VascularVesselType::Artery
    } else {
        VascularVesselType::Vein
    };

    Ok(VesselClassification {
        vessel_type,
        confidence,
        diameter,
        orientation,
        flow_direction: (vessel_type == VascularVesselType::Artery).then_some(orientation),
    })
}

/// Return indices of all voxels where `mask > 0`.
pub(super) fn masked_points(mask: &Array3<f64>) -> Vec<[usize; 3]> {
    mask.indexed_iter()
        .filter_map(|([i, j, k], &v)| (v > 0.0).then_some([i, j, k]))
        .collect()
}

/// Return centerline voxels: those with ≤ 2 vessel-class 6-neighbours.
pub(super) fn centerline_from_points(mask: &Array3<f64>, points: &[[usize; 3]]) -> Vec<[usize; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    points
        .iter()
        .copied()
        .filter(|p| vessel_neighbor_count(mask, p) <= 2)
        .collect()
}

/// Estimate the physical length of the centerline using the principal-axis
/// voxel step. The segmentation algorithm returns an unordered medial-axis
/// point set, so this preserves its existing point-count estimator while
/// applying anisotropic spacing in metres.
pub(super) fn centerline_length(
    points: &[[usize; 3]],
    orientation: [f64; 3],
    spacing: VoxelSpacing,
) -> Length<f64> {
    if points.is_empty() {
        return Length::from_base(0.0);
    }
    Length::from_base(points.len() as f64 * spacing.step_along(orientation))
}

/// Count the number of 6-adjacent vessel-class neighbours of `point`.
pub(super) fn vessel_neighbor_count(mask: &Array3<f64>, point: &[usize; 3]) -> usize {
    let [nx, ny, nz] = mask.shape();
    let [i, j, k] = *point;
    usize::from(i > 0 && mask[[i - 1, j, k]] > 0.0)
        + usize::from(i + 1 < nx && mask[[i + 1, j, k]] > 0.0)
        + usize::from(j > 0 && mask[[i, j - 1, k]] > 0.0)
        + usize::from(j + 1 < ny && mask[[i, j + 1, k]] > 0.0)
        + usize::from(k > 0 && mask[[i, j, k - 1]] > 0.0)
        + usize::from(k + 1 < nz && mask[[i, j, k + 1]] > 0.0)
}

/// Estimate the vessel principal axis by power iteration (12 steps).
///
/// Returns a unit vector aligned with the direction of largest variance of the
/// masked voxel coordinates.
pub(super) fn principal_axis(points: &[[usize; 3]]) -> [f64; 3] {
    let values = points
        .iter()
        .map(|&[i, j, k]| [i as f64, j as f64, k as f64])
        .collect::<Vec<_>>();
    principal_axis_values(&values)
}

fn principal_axis_values(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len() as f64;
    let mean = {
        let sum = points.iter().fold([0.0; 3], |mut acc, p| {
            acc[0] += p[0];
            acc[1] += p[1];
            acc[2] += p[2];
            acc
        });
        [sum[0] / n, sum[1] / n, sum[2] / n]
    };

    let mut axis = [1.0, 0.0, 0.0];
    for _ in 0..12 {
        let mut next = [0.0; 3];
        for point in points {
            let d = [point[0] - mean[0], point[1] - mean[1], point[2] - mean[2]];
            let proj = d[2].mul_add(axis[2], d[0].mul_add(axis[0], d[1] * axis[1]));
            next[0] += proj * d[0];
            next[1] += proj * d[1];
            next[2] += proj * d[2];
        }
        let norm = next[2]
            .mul_add(next[2], next[0].mul_add(next[0], next[1] * next[1]))
            .sqrt();
        if norm <= 1e-12 {
            return axis;
        }
        axis = [next[0] / norm, next[1] / norm, next[2] / norm];
    }
    axis
}
