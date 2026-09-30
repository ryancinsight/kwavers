use super::{resample_to_target_grid, validate_registration_compatibility, IDENTITY_HOMOGENEOUS};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_imaging::fusion::{AffineTransform, RegistrationMethod};
use leto::Array3;
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
///
/// The ritk engine is built per registration call with a metric whose intensity
/// ranges derive from the volumes being registered: fusion data is normalised
/// (or physically scaled), so the metric's `[0, 255]` default range would bin
/// every sample into one or two bins, degenerating mutual information and
/// making the hill-climb directionless. A bounded iteration budget keeps the
/// per-examination latency inside the workflow's real-time envelope.
#[derive(Debug, Default)]
pub struct RitkRegistrationEngine;

/// Histogram bins per metric axis.
const MI_BINS: usize = 32;
/// Hill-climb iteration budget for fusion pre-alignment.
const MI_MAX_ITERATIONS: usize = 8;
/// Optimiser convergence tolerance in mutual-information units.
const MI_TOLERANCE: f64 = 1e-3;

impl RitkRegistrationEngine {
    /// Build a mutual-information metric with data-derived intensity ranges.
    fn build_metric(
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> KwaversResult<ritk_registration::classical::MutualInformationMetric> {
        let fixed_range = Self::robust_intensity_range(fixed)?;
        let moving_range = Self::robust_intensity_range(moving)?;
        ritk_registration::classical::MutualInformationMetric::with_ranges(
            MI_BINS,
            fixed_range,
            moving_range,
            ritk_registration::classical::NmiNormalization::MeanEntropy,
            ritk_registration::classical::HistogramEstimator::Discrete,
        )
        .map_err(|e| KwaversError::InvalidInput(e.to_string()))
    }

    /// Finite, padded intensity interval covering the volume's samples.
    fn robust_intensity_range(
        volume: &Array3<f64>,
    ) -> KwaversResult<ritk_registration::IntensityRange> {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in volume.iter() {
            if v.is_finite() {
                min = min.min(v);
                max = max.max(v);
            }
        }
        if !min.is_finite() || !max.is_finite() {
            return Err(KwaversError::InvalidInput(
                "registration volume has no finite samples".to_owned(),
            ));
        }
        // Pad so boundary samples are not clamped into the outermost bins.
        let span = max - min;
        if span < 1e-12 {
            // Near-constant volume: a degenerate but valid symmetric interval.
            return ritk_registration::IntensityRange::try_new(min - 0.5, max + 0.5)
                .map_err(|e| KwaversError::InvalidInput(e.to_string()));
        }
        let pad = 0.01 * span;
        ritk_registration::IntensityRange::try_new(min - pad, max + pad)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))
    }

    /// Register for method.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn register_for_method(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
        method: RegistrationMethod,
    ) -> KwaversResult<FusionRegistrationResult> {
        validate_registration_compatibility(fixed.shape(), moving.shape())?;

        // Registering a volume with itself is the identity transform by
        // definition; skip the hill-climb entirely (every fusion algorithm
        // hands the reference modality in as both fixed and moving).
        if std::ptr::eq(fixed, moving) {
            return Ok(FusionRegistrationResult {
                transform_matrix: IDENTITY_HOMOGENEOUS,
                affine_transform: AffineTransform::from_homogeneous(&IDENTITY_HOMOGENEOUS),
                confidence: 1.0,
                prewarped: None,
            });
        }

        // The ritk classical engines require equally shaped volumes. Modalities
        // may legitimately be acquired on different grids, so pre-align the
        // moving image onto the fixed grid with the identity transform before
        // the metric runs; a modality that already shares the fixed grid is
        // untouched. The estimated transform then lives in the fixed grid's
        // frame — exact whenever the estimated transform is the identity, and
        // the standard approximation otherwise.
        let aligned_owned;
        let moving: &Array3<f64> = if fixed.shape() != moving.shape() {
            aligned_owned = resample_to_target_grid(moving, &IDENTITY_HOMOGENEOUS, fixed.shape());
            &aligned_owned
        } else {
            moving
        };

        let metric = Self::build_metric(fixed, moving)?;

        // Already-aligned fast path. MI(fixed, fixed) is the entropy ceiling of
        // the fixed volume under this metric; when the pair already attains ≥98%
        // of that ceiling at the identity transform, the hill-climb has nothing
        // material to recover — its perturbation search would run 13 full-volume
        // metric evaluations to return (at best) a sub-voxel correction. This
        // also covers the reference-modality-against-its-own-clone case that
        // every fusion algorithm evaluates. Misaligned pairs fall through to
        // the full optimisation unchanged.
        let mi_ceiling = metric
            .compute(fixed, fixed)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        let mi_identity = metric
            .compute(fixed, moving)
            .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
        if mi_identity >= 0.98 * mi_ceiling {
            return Ok(FusionRegistrationResult {
                transform_matrix: IDENTITY_HOMOGENEOUS,
                affine_transform: AffineTransform::from_homogeneous(&IDENTITY_HOMOGENEOUS),
                confidence: 1.0,
                prewarped: None,
            });
        }

        let config = ritk_registration::classical::engine::config::ClassicalConfig {
            max_iterations: MI_MAX_ITERATIONS,
            tolerance: MI_TOLERANCE,
            ..ritk_registration::classical::engine::config::ClassicalConfig::default()
        };
        let result = match method {
            RegistrationMethod::RigidBody | RegistrationMethod::Automatic => {
                let engine = ImageRegistration::with_config(config, metric);
                engine
                    .rigid_registration_mutual_info(fixed, moving, &RitkAffineTransform::IDENTITY)
                    .map_err(|e| KwaversError::InvalidInput(e.to_string()))?
            }
            RegistrationMethod::Affine => {
                let engine = ImageRegistration::with_config(config, metric);
                engine
                    .affine_registration_mutual_info(fixed, moving, &RitkAffineTransform::IDENTITY)
                    .map_err(|e| KwaversError::InvalidInput(e.to_string()))?
            }
            RegistrationMethod::NonRigid => {
                // Symmetric Gaussian Demons non-rigid registration (Vercauteren 2009).
                use ritk_registration::demons::{DemonsConfig, SymmetricDemonsRegistration};
                let [nz, ny, nx] = fixed.shape();
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
        target_shape: [usize; 3],
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
