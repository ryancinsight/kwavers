use crate::math::numerics::operators::interpolation::trilinear_index_space;
use ndarray::Array3;

use super::transforms::apply_inverse_transform;

/// Resample image to target grid using trilinear interpolation
///
/// Transforms an image from source coordinates to target coordinates using
/// a homogeneous transformation matrix and trilinear interpolation.
///
/// # Arguments
///
/// * `source_image` - Source image data
/// * `transform` - 4x4 homogeneous transformation matrix (column-major)
/// * `target_dims` - Target grid dimensions
///
/// # Returns
///
/// Resampled image on the target grid
#[must_use]
pub fn resample_to_target_grid(
    source_image: &Array3<f64>,
    transform: &[f64; 16],
    target_dims: (usize, usize, usize),
) -> Array3<f64> {
    let mut resampled = Array3::<f64>::zeros(target_dims);

    // Target voxel spacing (assume isotropic for simplicity)
    let target_spacing = 1.0; // 1mm spacing

    for i in 0..target_dims.0 {
        for j in 0..target_dims.1 {
            for k in 0..target_dims.2 {
                // Convert voxel indices to physical coordinates
                let target_coords = [
                    i as f64 * target_spacing,
                    j as f64 * target_spacing,
                    k as f64 * target_spacing,
                ];

                // Apply inverse transform to find source coordinates
                let source_coords = apply_inverse_transform(transform, target_coords);

                // Trilinear interpolation
                let value = trilinear_index_space(
                    source_image,
                    source_coords[0],
                    source_coords[1],
                    source_coords[2],
                );
                resampled[[i, j, k]] = value;
            }
        }
    }

    resampled
}

/// Trilinear interpolation at a physical coordinate expressed as an array.
///
/// `coords` are fractional index-space coordinates `[x, y, z]` into `input`.
/// The `_shape` argument is accepted for API symmetry with call sites that carry
/// explicit shape metadata; `input.dim()` provides the authoritative bound.
///
/// Delegates to [`trilinear_index_space`].
#[cfg(test)]
pub(super) fn trilinear_interpolate(
    input: &Array3<f64>,
    coords: [f64; 3],
    _shape: &[usize],
) -> f64 {
    trilinear_index_space(input, coords[0], coords[1], coords[2])
}
