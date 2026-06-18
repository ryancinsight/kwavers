use super::{resample_to_target_grid, validate_registration_compatibility, IDENTITY_HOMOGENEOUS};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_imaging::fusion::{AffineTransform, RegistrationMethod};
use ndarray::Array3;
use ritk_registration::{AffineTransform as RitkAffineTransform, ImageRegistration};

/// Fusion-local validation case for registration dispatch.
#[derive(Debug, Clone)]
pub struct FusionValidationCase {
    pub name: &'static str,
    pub registration_method: RegistrationMethod,
}

/// Fusion-local benchmark descriptor for registration.
#[derive(Debug, Clone)]
pub struct FusionBenchmarkCase {
    pub name: &'static str,
    pub fixed_shape: [usize; 3],
    pub moving_shape: [usize; 3],
}

/// Canonical registration result used by retained fusion algorithms.
#[derive(Debug, Clone)]
pub struct FusionRegistrationResult {
    pub transform_matrix: [f64; 16],
    pub affine_transform: AffineTransform,
    pub confidence: f64,
    /// Pre-warped moving image from non-rigid (Demons) registration.
    /// `None` for rigid / affine results.  Tuple of (warped_flat_f32, shape `[nz,ny,nx]`).
    pub prewarped: Option<(Vec<f32>, [usize; 3])>,
}

/// Classical registration adapter used by the retained fusion surface.
///
/// This adapter is the only registration owner visible to fusion algorithms.
#[derive(Debug, Default)]
pub struct RitkRegistrationEngine {
    inner: ImageRegistration,
}

impl RitkRegistrationEngine {
    /// Register for method.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn register_for_method(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
        method: RegistrationMethod,
    ) -> KwaversResult<FusionRegistrationResult> {
        validate_registration_compatibility(fixed.dim(), moving.dim())?;

        let result = match method {
            RegistrationMethod::RigidBody | RegistrationMethod::Automatic => self
                .inner
                .rigid_registration_mutual_info(fixed, moving, &RitkAffineTransform::IDENTITY)
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))?,
            RegistrationMethod::Affine => self
                .inner
                .affine_registration_mutual_info(fixed, moving, &RitkAffineTransform::IDENTITY)
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))?,
            RegistrationMethod::NonRigid => {
                // Symmetric Gaussian Demons non-rigid registration (Vercauteren 2009).
                use ritk_registration::demons::{DemonsConfig, SymmetricDemonsRegistration};
                let (nz, ny, nx) = fixed.dim();
                let fixed_flat: Vec<f32> = fixed.iter().map(|&v| v as f32).collect();
                let moving_flat: Vec<f32> = moving.iter().map(|&v| v as f32).collect();
                let demons_result = SymmetricDemonsRegistration::new(DemonsConfig::default())
                    .register(&fixed_flat, &moving_flat, [nz, ny, nx], [1.0, 1.0, 1.0])
                    .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
                return Ok(FusionRegistrationResult {
                    transform_matrix: IDENTITY_HOMOGENEOUS,
                    affine_transform: AffineTransform::from_homogeneous(&IDENTITY_HOMOGENEOUS),
                    confidence: 0.85,
                    prewarped: Some((demons_result.warped, [nz, ny, nx])),
                });
            }
        };

        Ok(FusionRegistrationResult {
            transform_matrix: *result.transform.as_array(),
            affine_transform: AffineTransform::from_homogeneous(result.transform.as_array()),
            confidence: result.quality.normalized_cross_correlation,
            prewarped: None,
        })
    }
    /// Resample registered.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn resample_registered(
        &self,
        moving: &Array3<f64>,
        registration: &FusionRegistrationResult,
        target_shape: (usize, usize, usize),
    ) -> KwaversResult<Array3<f64>> {
        // Use pre-warped image for non-rigid (Demons) results.
        if let Some((ref warped_f32, _shape)) = registration.prewarped {
            let warped_f64: Vec<f64> = warped_f32.iter().map(|&v| v as f64).collect();
            Array3::from_shape_vec(target_shape, warped_f64)
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))
        } else {
            Ok(resample_to_target_grid(
                moving,
                &registration.transform_matrix,
                target_shape,
            ))
        }
    }
}
