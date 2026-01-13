//! Image registration and resampling for multi-modal fusion.
//!
//! This module provides functionality for aligning images from different modalities
//! into a common coordinate system, including transformation application, resampling,
//! and interpolation methods.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Generate coordinate arrays for a given grid dimension
///
/// Creates uniformly spaced coordinate arrays for each spatial dimension.
///
/// # Arguments
///
/// * `dims` - Grid dimensions (nx, ny, nz)
/// * `resolution` - Spatial resolution in each dimension [dx, dy, dz]
///
/// # Returns
///
/// Array of coordinate vectors [x_coords, y_coords, z_coords]
pub fn generate_coordinate_arrays(
    dims: (usize, usize, usize),
    resolution: [f64; 3],
) -> [Vec<f64>; 3] {
    let x_coords: Vec<f64> = (0..dims.0).map(|i| i as f64 * resolution[0]).collect();
    let y_coords: Vec<f64> = (0..dims.1).map(|j| j as f64 * resolution[1]).collect();
    let z_coords: Vec<f64> = (0..dims.2).map(|k| k as f64 * resolution[2]).collect();

    [x_coords, y_coords, z_coords]
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

/// Apply inverse transformation to find source coordinates
///
/// Simplified inverse transform for rigid body transformations.
/// For full affine transforms, proper matrix inversion would be required.
///
/// # Arguments
///
/// * `transform` - 4x4 homogeneous transformation matrix (column-major)
/// * `point` - Target point coordinates
///
/// # Returns
///
/// Source coordinates after applying inverse transformation
fn apply_inverse_transform(transform: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
    // Extract rotation matrix (upper-left 3x3)
    let rot = [
        [transform[0], transform[1], transform[2]],
        [transform[4], transform[5], transform[6]],
        [transform[8], transform[9], transform[10]],
    ];

    // Extract translation vector
    let trans = [transform[3], transform[7], transform[11]];

    // For rigid body inverse: R^T * (p - t)
    let shifted = [
        point[0] - trans[0],
        point[1] - trans[1],
        point[2] - trans[2],
    ];

    [
        rot[0][0] * shifted[0] + rot[1][0] * shifted[1] + rot[2][0] * shifted[2],
        rot[0][1] * shifted[0] + rot[1][1] * shifted[1] + rot[2][1] * shifted[2],
        rot[0][2] * shifted[0] + rot[1][2] * shifted[1] + rot[2][2] * shifted[2],
    ]
}

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
fn trilinear_interpolate(image: &Array3<f64>, coords: [f64; 3], dims: &[usize]) -> f64 {
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

/// Validate that image dimensions are compatible for registration
///
/// Checks that source and target dimensions are within acceptable ratios
/// to avoid excessive resampling artifacts.
pub fn validate_registration_compatibility(
    source_dims: (usize, usize, usize),
    target_dims: (usize, usize, usize),
) -> KwaversResult<()> {
    const MAX_RATIO: f64 = 10.0;

    for dim in 0..3 {
        let source = [source_dims.0, source_dims.1, source_dims.2][dim] as f64;
        let target = [target_dims.0, target_dims.1, target_dims.2][dim] as f64;
        let ratio = (source / target).max(target / source);

        if ratio > MAX_RATIO {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Incompatible dimensions for registration: ratio {} exceeds maximum {}",
                        ratio, MAX_RATIO
                    ),
                },
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_coordinate_arrays() {
        let dims = (16, 8, 4);
        let resolution = [1e-4, 2e-4, 3e-4];
        let coords = generate_coordinate_arrays(dims, resolution);

        assert_eq!(coords[0].len(), dims.0);
        assert_eq!(coords[1].len(), dims.1);
        assert_eq!(coords[2].len(), dims.2);

        assert_eq!(coords[0][0], 0.0);
        assert_eq!(coords[1][0], 0.0);
        assert_eq!(coords[2][0], 0.0);

        assert!((coords[0][1] - resolution[0]).abs() < f64::EPSILON);
        assert!((coords[1][1] - resolution[1]).abs() < f64::EPSILON);
        assert!((coords[2][1] - resolution[2]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trilinear_interpolation_at_grid_point() {
        let image = Array3::<f64>::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k) as f64);

        // Test at exact grid point
        let value = trilinear_interpolate(&image, [1.0, 1.0, 1.0], image.shape());
        assert!((value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trilinear_interpolation_midpoint() {
        let image = Array3::<f64>::from_shape_fn((4, 4, 4), |(i, j, k)| (i + j + k) as f64);

        // Test at midpoint between grid points
        let value = trilinear_interpolate(&image, [1.5, 1.5, 1.5], image.shape());
        let expected = (3.0 + 4.0 + 4.0 + 5.0 + 4.0 + 5.0 + 5.0 + 6.0) / 8.0;
        assert!((value - expected).abs() < 1e-10);
    }

    #[test]
    fn test_trilinear_interpolation_clamping() {
        let image = Array3::<f64>::ones((4, 4, 4));

        // Test coordinates outside valid range
        let value = trilinear_interpolate(&image, [-1.0, -1.0, -1.0], image.shape());
        assert!((value - 1.0).abs() < 1e-10);

        let value = trilinear_interpolate(&image, [10.0, 10.0, 10.0], image.shape());
        assert!((value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_inverse_transform_identity() {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let point = [1.0, 2.0, 3.0];
        let transformed = apply_inverse_transform(&identity, point);

        assert!((transformed[0] - point[0]).abs() < 1e-10);
        assert!((transformed[1] - point[1]).abs() < 1e-10);
        assert!((transformed[2] - point[2]).abs() < 1e-10);
    }

    #[test]
    fn test_apply_inverse_transform_translation() {
        let translation = [
            1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let point = [5.0, 10.0, 15.0];
        let transformed = apply_inverse_transform(&translation, point);

        assert!((transformed[0] - 0.0).abs() < 1e-10);
        assert!((transformed[1] - 0.0).abs() < 1e-10);
        assert!((transformed[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_identity_transform() {
        let source = Array3::<f64>::from_elem((4, 4, 4), 1.0);
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let resampled = resample_to_target_grid(&source, &identity, (4, 4, 4));

        for value in resampled.iter() {
            assert!((value - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_validate_registration_compatibility_valid() {
        let result = validate_registration_compatibility((10, 10, 10), (20, 20, 20));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_registration_compatibility_invalid() {
        let result = validate_registration_compatibility((10, 10, 10), (200, 200, 200));
        assert!(result.is_err());
    }
}
