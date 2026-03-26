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
pub(crate) fn apply_inverse_transform(transform: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
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
