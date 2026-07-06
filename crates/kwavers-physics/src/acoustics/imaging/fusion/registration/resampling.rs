use leto::Array3;

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
    target_dims: [usize; 3],
) -> Array3<f64> {
    let mut resampled = Array3::<f64>::zeros(target_dims);

    // Target voxel spacing (assume isotropic for simplicity)
    let target_spacing = 1.0; // 1mm spacing

    for i in 0..target_dims[0] {
        for j in 0..target_dims[1] {
            for k in 0..target_dims[2] {
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
/// explicit shape metadata; `input.shape()` provides the authoritative bound.
///
#[cfg(test)]
pub(super) fn trilinear_interpolate(
    input: &Array3<f64>,
    coords: [f64; 3],
    _shape: [usize; 3],
) -> f64 {
    trilinear_index_space(input, coords[0], coords[1], coords[2])
}

fn trilinear_index_space(input: &Array3<f64>, x: f64, y: f64, z: f64) -> f64 {
    let [nx, ny, nz] = input.shape();
    let x0 = x.floor().clamp(0.0, (nx - 1) as f64) as usize;
    let y0 = y.floor().clamp(0.0, (ny - 1) as f64) as usize;
    let z0 = z.floor().clamp(0.0, (nz - 1) as f64) as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z1 = (z0 + 1).min(nz - 1);
    let tx = (x - x0 as f64).clamp(0.0, 1.0);
    let ty = (y - y0 as f64).clamp(0.0, 1.0);
    let tz = (z - z0 as f64).clamp(0.0, 1.0);
    let c00 = input[[x0, y0, z0]].mul_add(1.0 - tx, input[[x1, y0, z0]] * tx);
    let c10 = input[[x0, y1, z0]].mul_add(1.0 - tx, input[[x1, y1, z0]] * tx);
    let c01 = input[[x0, y0, z1]].mul_add(1.0 - tx, input[[x1, y0, z1]] * tx);
    let c11 = input[[x0, y1, z1]].mul_add(1.0 - tx, input[[x1, y1, z1]] * tx);
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    c0 * (1.0 - tz) + c1 * tz
}
