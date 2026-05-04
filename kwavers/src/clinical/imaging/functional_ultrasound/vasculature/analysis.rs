//! Image analysis utilities: threshold delegation and connected-component labelling.
//!
//! # Threshold boundary
//!
//! Otsu threshold selection is delegated to `ritk_core`, the authoritative
//! image-thresholding crate in this workspace. This module owns only the
//! vasculature-specific adaptation from Frangi response volume to RITK's
//! flat slice API and the connected-component measurements used by fUS.
//!
//! # Connected components
//!
//! 6-connectivity (face-adjacent neighbours only) is used for the flood-fill
//! labelling, matching the standard convention for 3-D binary images in
//! medical imaging (Rosenfeld & Pfaltz 1966).
//!
//! # References
//! - Otsu, N. (1979). IEEE Trans. Syst. Man Cybern. 9(1), pp. 62-66.
//! - Rosenfeld & Pfaltz (1966). J. ACM 13(4), pp. 471-494.

use ndarray::Array3;
use ritk_core::segmentation::threshold::otsu::compute_otsu_threshold_from_slice;

/// Compute RITK's Otsu global threshold for a 3-D Frangi response volume.
///
/// The `f64` Frangi response is converted to RITK's `f32` thresholding input
/// because `ritk_core` owns the Otsu implementation and its validated
/// mathematical specification.
pub(super) fn otsu_threshold(image: &Array3<f64>) -> f64 {
    let values: Vec<f32> = image.iter().map(|value| *value as f32).collect();
    f64::from(compute_otsu_threshold_from_slice(&values, 256))
}

/// Count 6-connected components in a binary mask (values > 0).
///
/// Uses an iterative flood-fill to avoid stack overflow on large volumes.
///
/// Returns `(n_components, total_vessel_voxels)`.
pub(super) fn count_connected_components(mask: &Array3<f64>) -> (usize, usize) {
    let (nx, ny, nz) = mask.dim();
    let mut visited = Array3::<bool>::default((nx, ny, nz));
    let mut n_components = 0_usize;
    let mut total_voxels = 0_usize;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if mask[[i, j, k]] > 0.0 && !visited[[i, j, k]] {
                    n_components += 1;
                    let mut stack = vec![(i, j, k)];
                    while let Some((ci, cj, ck)) = stack.pop() {
                        if visited[[ci, cj, ck]] {
                            continue;
                        }
                        visited[[ci, cj, ck]] = true;
                        total_voxels += 1;

                        // 6-connected neighbours
                        if ci > 0 && mask[[ci - 1, cj, ck]] > 0.0 {
                            stack.push((ci - 1, cj, ck));
                        }
                        if ci + 1 < nx && mask[[ci + 1, cj, ck]] > 0.0 {
                            stack.push((ci + 1, cj, ck));
                        }
                        if cj > 0 && mask[[ci, cj - 1, ck]] > 0.0 {
                            stack.push((ci, cj - 1, ck));
                        }
                        if cj + 1 < ny && mask[[ci, cj + 1, ck]] > 0.0 {
                            stack.push((ci, cj + 1, ck));
                        }
                        if ck > 0 && mask[[ci, cj, ck - 1]] > 0.0 {
                            stack.push((ci, cj, ck - 1));
                        }
                        if ck + 1 < nz && mask[[ci, cj, ck + 1]] > 0.0 {
                            stack.push((ci, cj, ck + 1));
                        }
                    }
                }
            }
        }
    }

    (n_components, total_voxels)
}
