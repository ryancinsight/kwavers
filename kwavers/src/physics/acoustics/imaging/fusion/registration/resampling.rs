use ndarray::Array3;

use super::transforms::apply_inverse_transform;

/// Trilinear interpolation for 3D image resampling
///
/// Interpolates a value at arbitrary coordinates within a 3D grid using
/// trilinear interpolation (linear interpolation in 3D).
///
/// # Arguments
///
/// * `image` - 3D image data
/// * `coords` - Continuous coordinates [x, y, z]
/// * `dims` - Image dimensions
///
/// # Returns
///
/// Interpolated value at the specified coordinates
pub(crate) fn trilinear_interpolate(image: &Array3<f64>, coords: [f64; 3], dims: &[usize]) -> f64 {
    // Clamp coordinates to valid range
    let x = coords[0].max(0.0).min((dims[0] - 1) as f64);
    let y = coords[1].max(0.0).min((dims[1] - 1) as f64);
    let z = coords[2].max(0.0).min((dims[2] - 1) as f64);

    // Find surrounding grid points
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let z0 = z.floor() as usize;

    let x1 = (x0 + 1).min(dims[0] - 1);
    let y1 = (y0 + 1).min(dims[1] - 1);
    let z1 = (z0 + 1).min(dims[2] - 1);

    // Interpolation weights
    let xd = x - x0 as f64;
    let yd = y - y0 as f64;
    let zd = z - z0 as f64;

    // Get values at 8 corners of the interpolation cube
    let c000 = image[[x0, y0, z0]];
    let c001 = image[[x0, y0, z1]];
    let c010 = image[[x0, y1, z0]];
    let c011 = image[[x0, y1, z1]];
    let c100 = image[[x1, y0, z0]];
    let c101 = image[[x1, y0, z1]];
    let c110 = image[[x1, y1, z0]];
    let c111 = image[[x1, y1, z1]];

    // Interpolate along x
    let c00 = c000 * (1.0 - xd) + c100 * xd;
    let c01 = c001 * (1.0 - xd) + c101 * xd;
    let c10 = c010 * (1.0 - xd) + c110 * xd;
    let c11 = c011 * (1.0 - xd) + c111 * xd;

    // Interpolate along y
    let c0 = c00 * (1.0 - yd) + c10 * yd;
    let c1 = c01 * (1.0 - yd) + c11 * yd;

    // Interpolate along z
    c0 * (1.0 - zd) + c1 * zd
}

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
pub fn resample_to_target_grid(
    source_image: &Array3<f64>,
    transform: &[f64; 16],
    target_dims: (usize, usize, usize),
) -> Array3<f64> {
    let mut resampled = Array3::<f64>::zeros(target_dims);
    let source_dims = source_image.shape();

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
                let value = trilinear_interpolate(source_image, source_coords, source_dims);
                resampled[[i, j, k]] = value;
            }
        }
    }

    resampled
}
