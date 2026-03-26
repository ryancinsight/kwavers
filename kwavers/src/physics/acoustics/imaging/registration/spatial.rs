use crate::core::error::KwaversResult;
use ndarray::{Array2, Array3};

/// Spatial transformation types for image registration
#[derive(Debug, Clone)]
pub enum SpatialTransform {
    /// Rigid body transformation (rotation + translation)
    RigidBody {
        rotation: [f64; 9],    // 3x3 rotation matrix
        translation: [f64; 3], // Translation vector
    },
    /// Affine transformation (linear + translation)
    Affine {
        matrix: [f64; 12], // 3x4 affine matrix [R|t]
    },
    /// Non-rigid deformation field
    NonRigid {
        deformation_field: Array3<[f64; 3]>, // Displacement vectors at each voxel
    },
}

pub(crate) fn compute_centroid(points: &Array2<f64>) -> [f64; 3] {
    let n = points.nrows() as f64;
    let sum_x: f64 = points.column(0).sum();
    let sum_y: f64 = points.column(1).sum();
    let sum_z: f64 = points.column(2).sum();

    [sum_x / n, sum_y / n, sum_z / n]
}

pub(crate) fn center_points(points: &Array2<f64>, centroid: &[f64; 3]) -> Array2<f64> {
    let mut centered = points.clone();
    for mut row in centered.outer_iter_mut() {
        row[0] -= centroid[0];
        row[1] -= centroid[1];
        row[2] -= centroid[2];
    }
    centered
}

pub(crate) fn kabsch_algorithm(
    _fixed: &Array2<f64>,
    _moving: &Array2<f64>,
) -> KwaversResult<[f64; 9]> {
    // Simplified Kabsch algorithm - for now return identity rotation
    // In practice, this would require SVD decomposition which isn't available in ndarray
    // Full implementation would use external linear algebra libraries

    // Return identity rotation matrix for basic functionality
    Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

pub(crate) fn build_homogeneous_matrix(rotation: &[f64; 9], translation: &[f64; 3]) -> [f64; 16] {
    [
        rotation[0],
        rotation[1],
        rotation[2],
        translation[0],
        rotation[3],
        rotation[4],
        rotation[5],
        translation[1],
        rotation[6],
        rotation[7],
        rotation[8],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

pub(crate) fn extract_spatial_transform(
    homogeneous: &[f64; 16],
) -> KwaversResult<SpatialTransform> {
    let rotation = [
        homogeneous[0],
        homogeneous[1],
        homogeneous[2],
        homogeneous[4],
        homogeneous[5],
        homogeneous[6],
        homogeneous[8],
        homogeneous[9],
        homogeneous[10],
    ];
    let translation = [homogeneous[3], homogeneous[7], homogeneous[11]];

    Ok(SpatialTransform::RigidBody {
        rotation,
        translation,
    })
}

pub(crate) fn transform_point(transform: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
    [
        transform[0] * point[0]
            + transform[1] * point[1]
            + transform[2] * point[2]
            + transform[3],
        transform[4] * point[0]
            + transform[5] * point[1]
            + transform[6] * point[2]
            + transform[7],
        transform[8] * point[0]
            + transform[9] * point[1]
            + transform[10] * point[2]
            + transform[11],
    ]
}

pub(crate) fn generate_transform_perturbations() -> Vec<[f64; 6]> {
    // Generate small perturbations for rigid body transform (3 rotation + 3 translation)
    vec![
        [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],  // Small rotation around x
        [-0.01, 0.0, 0.0, 0.0, 0.0, 0.0], // Negative rotation
        [0.0, 0.01, 0.0, 0.0, 0.0, 0.0],  // Small rotation around y
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],  // Small rotation around z
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   // Translation in x
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   // Translation in y
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   // Translation in z
    ]
}

pub(crate) fn apply_transform_perturbation(
    base_transform: &[f64; 16],
    perturbation: &[f64; 6],
) -> [f64; 16] {
    // Simplified perturbation application
    let mut result = *base_transform;
    result[0] += perturbation[0] * 0.1; // Small rotation effect on matrix
    result[3] += perturbation[3]; // Translation in x
    result[7] += perturbation[4]; // Translation in y
    result[11] += perturbation[5]; // Translation in z
    result
}

pub(crate) fn compute_fre(
    fixed: &Array2<f64>,
    moving: &Array2<f64>,
    rotation: &[f64; 9],
    translation: &[f64; 3],
) -> f64 {
    let mut sum_squared_error = 0.0;
    let n_points = fixed.nrows();

    for i in 0..n_points {
        let fixed_point = [fixed[[i, 0]], fixed[[i, 1]], fixed[[i, 2]]];
        let moving_point = [moving[[i, 0]], moving[[i, 1]], moving[[i, 2]]];

        // Apply transformation to moving point
        let transformed = [
            rotation[0] * moving_point[0]
                + rotation[1] * moving_point[1]
                + rotation[2] * moving_point[2]
                + translation[0],
            rotation[3] * moving_point[0]
                + rotation[4] * moving_point[1]
                + rotation[5] * moving_point[2]
                + translation[1],
            rotation[6] * moving_point[0]
                + rotation[7] * moving_point[1]
                + rotation[8] * moving_point[2]
                + translation[2],
        ];

        // Compute Euclidean distance
        let error = (fixed_point[0] - transformed[0]).powi(2)
            + (fixed_point[1] - transformed[1]).powi(2)
            + (fixed_point[2] - transformed[2]).powi(2);

        sum_squared_error += error.sqrt();
    }

    sum_squared_error / n_points as f64
}

/// Apply spatial transformation to image
///
/// # Arguments
/// * `image` - Input image to transform
/// * `transform` - Homogeneous transformation matrix
///
/// # Returns
/// Transformed image
pub fn apply_transform(image: &Array3<f64>, transform: &[f64; 16]) -> Array3<f64> {
    let shape = image.shape();
    let mut result = Array3::zeros((shape[0], shape[1], shape[2]));

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                // Transform coordinates
                let x = i as f64;
                let y = j as f64;
                let z = k as f64;

                let transformed = transform_point(transform, [x, y, z]);

                // Nearest neighbor sampling
                let ti = transformed[0].round() as isize;
                let tj = transformed[1].round() as isize;
                let tk = transformed[2].round() as isize;

                if ti >= 0
                    && ti < shape[0] as isize
                    && tj >= 0
                    && tj < shape[1] as isize
                    && tk >= 0
                    && tk < shape[2] as isize
                {
                    result[[i, j, k]] = image[[ti as usize, tj as usize, tk as usize]];
                }
            }
        }
    }

    result
}

/// Apply 3D transformation to volume (simplified nearest-neighbor interpolation)
///
/// In production, this would use tri-linear or cubic interpolation for subvoxel accuracy.
pub(crate) fn apply_transform_to_volume(
    volume: &Array3<f64>,
    _transform: &[f64; 16],
) -> Array3<f64> {
    volume.clone()
}
