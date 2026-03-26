use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};

use super::metrics::RegistrationQualityMetrics;
use super::spatial::{
    build_homogeneous_matrix, center_points, compute_centroid, compute_fre,
    extract_spatial_transform, kabsch_algorithm, SpatialTransform,
};
use super::intensity::{
    compute_correlation, compute_mutual_information, compute_ncc,
};
use super::temporal::{temporal_synchronization, TemporalSync};
use super::spatial::{apply_transform_perturbation, generate_transform_perturbations};

/// Registration result containing transformation and quality metrics
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Spatial transformation
    pub spatial_transform: Option<SpatialTransform>,
    /// Temporal synchronization
    pub temporal_sync: Option<TemporalSync>,
    /// Registration quality metrics
    pub quality_metrics: RegistrationQualityMetrics,
    /// Transformation matrix (4x4 homogeneous)
    pub transform_matrix: [f64; 16],
    /// Registration confidence [0-1]
    pub confidence: f64,
}

/// Image registration engine
#[derive(Debug)]
pub struct ImageRegistration {
    /// Maximum iterations for optimization
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Regularization parameter for non-rigid registration
    #[allow(dead_code)]
    regularization_weight: f64,
}

impl Default for ImageRegistration {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            regularization_weight: 0.1,
        }
    }
}

impl ImageRegistration {
    /// Create new registration engine with custom parameters
    #[must_use]
    pub fn new(max_iterations: usize, tolerance: f64, regularization_weight: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            regularization_weight,
        }
    }

    /// Perform rigid body registration using landmark points
    ///
    /// # Arguments
    /// * `fixed_landmarks` - Landmark points in fixed image [N, 3]
    /// * `moving_landmarks` - Corresponding landmark points in moving image [N, 3]
    ///
    /// # Returns
    /// Registration result with rigid body transformation
    pub fn rigid_registration_landmarks(
        &self,
        fixed_landmarks: &Array2<f64>,
        moving_landmarks: &Array2<f64>,
    ) -> KwaversResult<RegistrationResult> {
        if fixed_landmarks.nrows() != moving_landmarks.nrows() {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "Fixed and moving landmark arrays must have same number of points"
                        .to_string(),
                },
            ));
        }

        if fixed_landmarks.ncols() != 3 || moving_landmarks.ncols() != 3 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::ConstraintViolation {
                    message: "Landmark arrays must have 3 columns (x, y, z)".to_string(),
                },
            ));
        }

        let _n_points = fixed_landmarks.nrows();

        // Compute centroids
        let fixed_centroid = compute_centroid(fixed_landmarks);
        let moving_centroid = compute_centroid(moving_landmarks);

        // Center the points
        let fixed_centered = center_points(fixed_landmarks, &fixed_centroid);
        let moving_centered = center_points(moving_landmarks, &moving_centroid);

        // Compute rotation matrix using Kabsch algorithm
        let rotation = kabsch_algorithm(&fixed_centered, &moving_centered)?;

        // Compute translation
        let translation = [
            fixed_centroid[0]
                - (rotation[0] * moving_centroid[0]
                    + rotation[1] * moving_centroid[1]
                    + rotation[2] * moving_centroid[2]),
            fixed_centroid[1]
                - (rotation[3] * moving_centroid[0]
                    + rotation[4] * moving_centroid[1]
                    + rotation[5] * moving_centroid[2]),
            fixed_centroid[2]
                - (rotation[6] * moving_centroid[0]
                    + rotation[7] * moving_centroid[1]
                    + rotation[8] * moving_centroid[2]),
        ];

        // Build homogeneous transformation matrix
        let transform_matrix = build_homogeneous_matrix(&rotation, &translation);

        // Compute quality metrics
        let fre = compute_fre(fixed_landmarks, moving_landmarks, &rotation, &translation);
        let quality_metrics = RegistrationQualityMetrics {
            fre: Some(fre),
            tre: None,               // Would need anatomical landmarks for TRE
            mutual_information: 0.0, // Not computed for landmark-based registration
            correlation_coefficient: 0.0,
            normalized_cross_correlation: 0.0,
            converged: true,
            iterations: 1,
            final_cost: fre,
        };

        let spatial_transform = SpatialTransform::RigidBody {
            rotation,
            translation,
        };

        Ok(RegistrationResult {
            spatial_transform: Some(spatial_transform),
            temporal_sync: None,
            quality_metrics,
            transform_matrix,
            confidence: (1.0 / (1.0 + fre)).min(1.0), // Higher FRE = lower confidence
        })
    }

    /// Perform intensity-based registration using mutual information
    ///
    /// # Arguments
    /// * `fixed_image` - Reference image
    /// * `moving_image` - Image to be registered
    /// * `initial_transform` - Initial transformation guess
    ///
    /// # Returns
    /// Registration result with optimized transformation
    pub fn intensity_registration_mutual_info(
        &self,
        fixed_image: &Array3<f64>,
        moving_image: &Array3<f64>,
        initial_transform: &[f64; 16],
    ) -> KwaversResult<RegistrationResult> {
        // Simplified mutual information registration
        // In practice, this would use optimization algorithms like Powell's method
        // or gradient descent to maximize mutual information

        let mut current_transform = *initial_transform;
        let mut best_mi =
            compute_mutual_information(fixed_image, moving_image, &current_transform);
        let mut converged = false;

        for _iteration in 0..self.max_iterations {
            // Try small perturbations to the transformation
            let perturbations = generate_transform_perturbations();

            let mut best_perturbation = None;
            let mut best_perturbation_mi = best_mi;

            for perturbation in &perturbations {
                let test_transform =
                    apply_transform_perturbation(&current_transform, perturbation);
                let test_mi =
                    compute_mutual_information(fixed_image, moving_image, &test_transform);

                if test_mi > best_perturbation_mi {
                    best_perturbation_mi = test_mi;
                    best_perturbation = Some(*perturbation);
                }
            }

            if let Some(perturbation) = best_perturbation {
                if (best_perturbation_mi - best_mi).abs() < self.tolerance {
                    converged = true;
                    break;
                }
                current_transform =
                    apply_transform_perturbation(&current_transform, &perturbation);
                best_mi = best_perturbation_mi;
            } else {
                break;
            }
        }

        // Extract spatial transform from homogeneous matrix
        let spatial_transform = extract_spatial_transform(&current_transform)?;

        let quality_metrics = RegistrationQualityMetrics {
            fre: None,
            tre: None,
            mutual_information: best_mi,
            correlation_coefficient: compute_correlation(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            normalized_cross_correlation: compute_ncc(
                fixed_image,
                moving_image,
                &current_transform,
            ),
            converged,
            iterations: self.max_iterations,
            final_cost: -best_mi, // Negative because we maximize MI but cost should be minimized
        };

        Ok(RegistrationResult {
            spatial_transform: Some(spatial_transform),
            temporal_sync: None,
            quality_metrics,
            transform_matrix: current_transform,
            confidence: best_mi.min(1.0),
        })
    }

    /// Perform temporal synchronization for multi-modal acquisition
    ///
    /// # Arguments
    /// * `reference_signal` - Reference modality timing signal
    /// * `target_signal` - Target modality timing signal
    /// * `sampling_rate` - Sampling frequency \[Hz\]
    ///
    /// # Returns
    /// Temporal synchronization result
    pub fn temporal_synchronization(
        &self,
        reference_signal: &Array1<f64>,
        target_signal: &Array1<f64>,
        sampling_rate: f64,
    ) -> KwaversResult<TemporalSync> {
        temporal_synchronization(reference_signal, target_signal, sampling_rate)
    }

    /// Apply spatial transformation to image
    ///
    /// # Arguments
    /// * `image` - Input image to transform
    /// * `transform` - Homogeneous transformation matrix
    ///
    /// # Returns
    /// Transformed image
    pub fn apply_transform(&self, image: &Array3<f64>, transform: &[f64; 16]) -> Array3<f64> {
        super::spatial::apply_transform(image, transform)
    }
}
