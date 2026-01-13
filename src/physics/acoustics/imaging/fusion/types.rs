//! Data types for multi-modal imaging fusion.
//!
//! This module defines the core data structures used in fusion operations,
//! including result types, transformation matrices, and internal representations
//! of registered modalities.

use ndarray::Array3;
use std::collections::HashMap;

/// Fused imaging result combining multiple modalities
///
/// This structure contains the output of a multi-modal fusion operation,
/// including the fused intensity image, derived tissue properties, quality
/// metrics, and uncertainty estimates.
#[derive(Debug)]
pub struct FusedImageResult {
    /// Fused intensity image (normalized 0-1)
    ///
    /// Combined image data from all registered modalities, normalized
    /// to the range [0, 1] for consistent interpretation.
    pub intensity_image: Array3<f64>,

    /// Tissue property map (multiple parameters)
    ///
    /// Derived tissue properties extracted from the multi-modal data,
    /// such as tissue classification, oxygenation, and stiffness.
    pub tissue_properties: HashMap<String, Array3<f64>>,

    /// Confidence map for fusion reliability
    ///
    /// Spatial map of confidence scores indicating the reliability
    /// of the fusion result at each voxel.
    pub confidence_map: Array3<f64>,

    /// Uncertainty quantification (if enabled)
    ///
    /// Spatial map of uncertainty estimates for the fusion result.
    /// Present only if uncertainty quantification is enabled in the config.
    pub uncertainty_map: Option<Array3<f64>>,

    /// Registration transforms applied
    ///
    /// Mapping from modality name to the affine transformation applied
    /// to register that modality to the common coordinate system.
    pub registration_transforms: HashMap<String, AffineTransform>,

    /// Quality metrics for each modality
    ///
    /// Mapping from modality name to a quality score [0, 1] indicating
    /// the reliability of that modality's data.
    pub modality_quality: HashMap<String, f64>,

    /// Fused spatial coordinates
    ///
    /// Coordinate arrays for the three spatial dimensions of the fused
    /// image: [x_coords, y_coords, z_coords].
    pub coordinates: [Vec<f64>; 3],
}

/// Affine transformation for image registration
///
/// Represents a 3D affine transformation consisting of rotation,
/// translation, and scaling components. Used to align images from
/// different modalities into a common coordinate system.
///
/// The transformation is applied as: x' = R * S * x + t
/// where R is rotation, S is scaling, and t is translation.
#[derive(Debug, Clone)]
pub struct AffineTransform {
    /// Rotation matrix (3x3)
    ///
    /// Orthonormal rotation matrix representing the rotational component
    /// of the transformation.
    pub rotation: [[f64; 3]; 3],

    /// Translation vector
    ///
    /// Translation in each spatial dimension [tx, ty, tz].
    pub translation: [f64; 3],

    /// Scaling factors
    ///
    /// Scaling factors for each dimension [sx, sy, sz].
    pub scale: [f64; 3],
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

impl AffineTransform {
    /// Create an identity transformation (no change)
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create a transformation from a homogeneous 4x4 matrix
    ///
    /// Extracts rotation, translation, and scale from a homogeneous
    /// transformation matrix stored in column-major order.
    pub fn from_homogeneous(homogeneous: &[f64; 16]) -> Self {
        // Extract rotation matrix (upper-left 3x3)
        let rotation = [
            [homogeneous[0], homogeneous[1], homogeneous[2]],
            [homogeneous[4], homogeneous[5], homogeneous[6]],
            [homogeneous[8], homogeneous[9], homogeneous[10]],
        ];

        // Extract translation vector (last column, first 3 elements)
        let translation = [homogeneous[3], homogeneous[7], homogeneous[11]];

        // Extract scale factors from rotation matrix (assuming no shear)
        let scale_x =
            (rotation[0][0].powi(2) + rotation[1][0].powi(2) + rotation[2][0].powi(2)).sqrt();
        let scale_y =
            (rotation[0][1].powi(2) + rotation[1][1].powi(2) + rotation[2][1].powi(2)).sqrt();
        let scale_z =
            (rotation[0][2].powi(2) + rotation[1][2].powi(2) + rotation[2][2].powi(2)).sqrt();

        let scale = [scale_x, scale_y, scale_z];

        AffineTransform {
            rotation,
            translation,
            scale,
        }
    }

    /// Apply the transformation to a 3D point
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        // Apply scaling
        let scaled = [
            point[0] * self.scale[0],
            point[1] * self.scale[1],
            point[2] * self.scale[2],
        ];

        // Apply rotation
        let rotated = [
            self.rotation[0][0] * scaled[0]
                + self.rotation[0][1] * scaled[1]
                + self.rotation[0][2] * scaled[2],
            self.rotation[1][0] * scaled[0]
                + self.rotation[1][1] * scaled[1]
                + self.rotation[1][2] * scaled[2],
            self.rotation[2][0] * scaled[0]
                + self.rotation[2][1] * scaled[1]
                + self.rotation[2][2] * scaled[2],
        ];

        // Apply translation
        [
            rotated[0] + self.translation[0],
            rotated[1] + self.translation[1],
            rotated[2] + self.translation[2],
        ]
    }
}

/// Internal representation of a registered imaging modality
#[derive(Debug, Clone)]
pub(crate) struct RegisteredModality {
    /// Intensity/pressure data
    ///
    /// 3D array containing the imaging data for this modality.
    pub data: Array3<f64>,

    /// Quality/confidence score
    ///
    /// Overall quality metric [0, 1] for this modality's data.
    pub quality_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_transform_identity() {
        let transform = AffineTransform::identity();
        let point = [1.0, 2.0, 3.0];
        let transformed = transform.transform_point(point);

        assert!((transformed[0] - point[0]).abs() < 1e-10);
        assert!((transformed[1] - point[1]).abs() < 1e-10);
        assert!((transformed[2] - point[2]).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform_translation() {
        let mut transform = AffineTransform::identity();
        transform.translation = [1.0, 2.0, 3.0];

        let point = [0.0, 0.0, 0.0];
        let transformed = transform.transform_point(point);

        assert!((transformed[0] - 1.0).abs() < 1e-10);
        assert!((transformed[1] - 2.0).abs() < 1e-10);
        assert!((transformed[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform_scaling() {
        let mut transform = AffineTransform::identity();
        transform.scale = [2.0, 3.0, 4.0];

        let point = [1.0, 1.0, 1.0];
        let transformed = transform.transform_point(point);

        assert!((transformed[0] - 2.0).abs() < 1e-10);
        assert!((transformed[1] - 3.0).abs() < 1e-10);
        assert!((transformed[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform_from_homogeneous() {
        let homogeneous = [
            2.0, 0.0, 0.0, 1.0, // Column 0: rotation + translation
            0.0, 3.0, 0.0, 2.0, // Column 1
            0.0, 0.0, 4.0, 3.0, // Column 2
            0.0, 0.0, 0.0, 1.0, // Column 3
        ];

        let transform = AffineTransform::from_homogeneous(&homogeneous);

        assert!((transform.translation[0] - 1.0).abs() < 1e-10);
        assert!((transform.translation[1] - 2.0).abs() < 1e-10);
        assert!((transform.translation[2] - 3.0).abs() < 1e-10);

        assert!((transform.scale[0] - 2.0).abs() < 1e-10);
        assert!((transform.scale[1] - 3.0).abs() < 1e-10);
        assert!((transform.scale[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_registered_modality_creation() {
        let data = Array3::<f64>::zeros((8, 8, 4));
        let modality = RegisteredModality {
            data: data.clone(),
            quality_score: 0.85,
        };

        assert_eq!(modality.data.dim(), (8, 8, 4));
        assert!((modality.quality_score - 0.85).abs() < 1e-10);
    }
}
