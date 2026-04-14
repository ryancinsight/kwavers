use super::{resample_to_target_grid, validate_registration_compatibility};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::imaging::fusion::{AffineTransform, RegistrationMethod};
use crate::solver::interface::factory::{FactoryError, RegistrationEngine};
use ndarray::{Array2, Array3};
use ritk_registration::ImageRegistration;

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
    /// `None` for rigid / affine results.  Tuple of (warped_flat_f32, shape [nz,ny,nx]).
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
                .rigid_registration_mutual_info(fixed, moving, &identity_transform())
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))?,
            RegistrationMethod::Affine => self
                .inner
                .affine_registration_mutual_info(fixed, moving, &identity_transform())
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))?,
            RegistrationMethod::NonRigid => {
                // Symmetric Gaussian Demons non-rigid registration (Vercauteren 2009).
                use ritk_registration::demons::{DemonsConfig, SymmetricDemonsRegistration};
                let (nz, ny, nx) = fixed.dim();
                let fixed_flat: Vec<f32> = fixed.iter().map(|&v| v as f32).collect();
                let moving_flat: Vec<f32> = moving.iter().map(|&v| v as f32).collect();
                let demons_result =
                    SymmetricDemonsRegistration::new(DemonsConfig::default())
                        .register(&fixed_flat, &moving_flat, [nz, ny, nx], [1.0, 1.0, 1.0])
                        .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
                let id = identity_transform();
                return Ok(FusionRegistrationResult {
                    transform_matrix: id,
                    affine_transform: AffineTransform::from_homogeneous(&id),
                    confidence: 0.85,
                    prewarped: Some((demons_result.warped, [nz, ny, nx])),
                });
            }
        };

        Ok(FusionRegistrationResult {
            transform_matrix: result.transform,
            affine_transform: AffineTransform::from_homogeneous(&result.transform),
            confidence: result.quality.normalized_cross_correlation,
            prewarped: None,
        })
    }

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

impl RegistrationEngine for RitkRegistrationEngine {
    fn register_rigid(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError> {
        let result = self
            .inner
            .rigid_registration_mutual_info(fixed, moving, &identity_transform())
            .map_err(|e| FactoryError::InvalidConfiguration(e.to_string()))?;
        homogeneous_to_array2(&result.transform)
    }

    fn register_affine(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError> {
        let result = self
            .inner
            .affine_registration_mutual_info(fixed, moving, &identity_transform())
            .map_err(|e| FactoryError::InvalidConfiguration(e.to_string()))?;
        homogeneous_to_array2(&result.transform)
    }

    fn register_deformable(
        &self,
        _fixed: &Array3<f64>,
        _moving: &Array3<f64>,
    ) -> Result<Array3<[f64; 3]>, FactoryError> {
        Err(FactoryError::InvalidConfiguration(
            "deformable registration is not implemented in the retained fusion path".to_string(),
        ))
    }

    fn resample(
        &self,
        moving: &Array3<f64>,
        transform: &Array2<f64>,
        target_shape: [usize; 3],
    ) -> Result<Array3<f64>, FactoryError> {
        let homogeneous = array2_to_homogeneous(transform)?;
        Ok(resample_to_target_grid(
            moving,
            &homogeneous,
            (target_shape[0], target_shape[1], target_shape[2]),
        ))
    }
}

fn identity_transform() -> [f64; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn homogeneous_to_array2(matrix: &[f64; 16]) -> Result<Array2<f64>, FactoryError> {
    Array2::from_shape_vec((4, 4), matrix.to_vec()).map_err(|error| {
        FactoryError::Internal(format!(
            "failed to materialize homogeneous registration matrix: {error}"
        ))
    })
}

fn array2_to_homogeneous(matrix: &Array2<f64>) -> Result<[f64; 16], FactoryError> {
    if matrix.shape() != [4, 4] {
        return Err(FactoryError::InvalidConfiguration(
            "registration transform must be a 4x4 homogeneous matrix".to_string(),
        ));
    }

    let mut homogeneous = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            homogeneous[i * 4 + j] = matrix[[i, j]];
        }
    }
    Ok(homogeneous)
}

