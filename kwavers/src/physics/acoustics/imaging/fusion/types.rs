//! Data types for multi-modal imaging fusion.
//!
//! Note: FusedImageResult and AffineTransform have been moved to domain::imaging::fusion
//! for clean architecture compliance. This module now only contains physics-specific
//! internal representations.

use ndarray::Array3;

// Re-export domain types for backwards compatibility
pub use crate::domain::imaging::fusion::{AffineTransform, FusedImageResult};

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
