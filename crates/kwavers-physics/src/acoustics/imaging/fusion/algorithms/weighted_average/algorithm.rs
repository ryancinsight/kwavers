use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_domain::imaging::fusion::RegistrationMethod;
use crate::acoustics::imaging::fusion::algorithms::MultiModalFusion;
use crate::acoustics::imaging::fusion::registration::{
    generate_coordinate_arrays, FusionBenchmarkCase, FusionValidationCase, RitkRegistrationEngine,
};
use crate::acoustics::imaging::fusion::types::FusedImageResult;
use ndarray::{Array3, CowArray};
use std::collections::HashMap;

use super::benchmarks::WeightedAverageBenchmarkCase;
use super::references::WEIGHTED_AVERAGE_FUSION_REFERENCES;
use super::validation::{validate_weighted_average_inputs, WeightedAverageValidationCase};

/// Weighted average fusion with explicit registration and resampling ownership.
///
/// # Mathematical Specification
/// Let `I_k(x)` be modality `k` sampled in the fused coordinate system and let
/// `w_k` be the configured modality weight. The retained estimator is
///
/// `F(x) = (sum_k w_k I_k(x)) / (sum_k w_k)`.
///
/// Registration is performed prior to accumulation; unsupported registration
/// requests fail explicitly instead of silently changing algorithms.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn fuse_weighted_average(fusion: &MultiModalFusion) -> KwaversResult<FusedImageResult> {
    let _references = WEIGHTED_AVERAGE_FUSION_REFERENCES;
    let _validation_case = FusionValidationCase {
        name: "weighted_average_fusion",
        registration_method: fusion.config.registration_method,
    };
    let _algorithm_validation_case = WeightedAverageValidationCase {
        name: "weighted_average_fusion_algorithm",
        expected_modalities: fusion.registered_data.len(),
    };
    let _ = (
        _algorithm_validation_case.name,
        _algorithm_validation_case.expected_modalities,
    );
    validate_weighted_average_inputs(fusion)?;

    let registration_engine = RitkRegistrationEngine::default();
    let mut modality_names: Vec<&String> = fusion.registered_data.keys().collect();
    modality_names.sort();

    let reference_name = modality_names.first().ok_or_else(|| {
        KwaversError::Validation(kwavers_core::error::ValidationError::ConstraintViolation {
            message: "No modalities available for fusion".to_owned(),
        })
    })?;

    let reference_modality = fusion.registered_data.get(*reference_name).ok_or_else(|| {
        KwaversError::Validation(kwavers_core::error::ValidationError::ConstraintViolation {
            message: "Reference modality missing".to_owned(),
        })
    })?;

    let target_dims = reference_modality.data.dim();
    let _benchmark_case = FusionBenchmarkCase {
        name: "weighted_average_fusion",
        fixed_shape: [target_dims.0, target_dims.1, target_dims.2],
        moving_shape: [target_dims.0, target_dims.1, target_dims.2],
    };
    let _algorithm_benchmark_case = WeightedAverageBenchmarkCase {
        name: "weighted_average_fusion_algorithm",
        target_dims: [target_dims.0, target_dims.1, target_dims.2],
        modality_count: fusion.registered_data.len(),
    };
    let _ = (
        _algorithm_benchmark_case.name,
        _algorithm_benchmark_case.target_dims,
        _algorithm_benchmark_case.modality_count,
    );

    let mut fused_intensity = Array3::<f64>::zeros(target_dims);
    let mut confidence_map = Array3::<f64>::zeros(target_dims);
    let mut uncertainty_map = if fusion.config.uncertainty_quantification {
        Some(Array3::<f64>::zeros(target_dims))
    } else {
        None
    };

    let mut registration_transforms = HashMap::new();
    let mut modality_quality = HashMap::new();

    for modality_name in modality_names {
        let modality = fusion
            .registered_data
            .get(modality_name.as_str())
            .ok_or_else(|| KwaversError::InvalidInput("Modality missing".to_owned()))?;

        let weight = fusion
            .config
            .modality_weights
            .get(modality_name.as_str())
            .copied()
            .unwrap_or(1.0);

        modality_quality.insert(modality_name.clone(), modality.quality_score);

        let registration_result = registration_engine.register_for_method(
            &reference_modality.data,
            &modality.data,
            normalize_registration_method(fusion.config.registration_method),
        )?;

        registration_transforms.insert(
            modality_name.clone(),
            registration_result.affine_transform.clone(),
        );

        let resampled_data = if modality.data.dim() == target_dims
            && registration_result.transform_matrix == identity_transform()
        {
            CowArray::from(modality.data.view())
        } else {
            CowArray::from(registration_engine.resample_registered(
                &modality.data,
                &registration_result,
                target_dims,
            )?)
        };

        for i in 0..target_dims.0 {
            for j in 0..target_dims.1 {
                for k in 0..target_dims.2 {
                    let value = resampled_data[[i, j, k]];
                    fused_intensity[[i, j, k]] += value * weight;
                    confidence_map[[i, j, k]] += weight;
                    if let Some(ref mut uncertainty) = uncertainty_map {
                        uncertainty[[i, j, k]] += (1.0 - modality.quality_score) * weight;
                    }
                }
            }
        }
    }

    for i in 0..target_dims.0 {
        for j in 0..target_dims.1 {
            for k in 0..target_dims.2 {
                let conf = confidence_map[[i, j, k]];
                if conf > 0.0 {
                    fused_intensity[[i, j, k]] /= conf;
                    if let Some(ref mut uncertainty) = uncertainty_map {
                        uncertainty[[i, j, k]] /= conf;
                    }
                }
            }
        }
    }

    Ok(FusedImageResult {
        intensity_image: fused_intensity,
        tissue_properties: HashMap::new(),
        confidence_map,
        uncertainty_map,
        registration_transforms,
        modality_quality,
        coordinates: generate_coordinate_arrays(target_dims, fusion.config.output_resolution),
    })
}

fn normalize_registration_method(method: RegistrationMethod) -> RegistrationMethod {
    match method {
        RegistrationMethod::Automatic => RegistrationMethod::RigidBody,
        other => other,
    }
}

fn identity_transform() -> [f64; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}
