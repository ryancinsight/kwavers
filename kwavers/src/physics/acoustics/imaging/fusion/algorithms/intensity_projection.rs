//! Voxelwise intensity-projection fusion operators.
//!
//! # Theorem: Order-statistic projection bounds
//!
//! For registered modalities `{I_m(x)}`, maximum-intensity fusion
//! `F_max(x)=max_m I_m(x)` and minimum-intensity fusion
//! `F_min(x)=min_m I_m(x)` are pointwise order statistics. Therefore
//! `min_m I_m(x) <= F_min(x) <= F_max(x) <= max_m I_m(x)` and the selected
//! voxel value is always one of the registered input measurements.

use super::MultiModalFusion;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::imaging::fusion::registration::generate_coordinate_arrays;
use crate::physics::acoustics::imaging::fusion::types::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(crate) enum ProjectionKind {
    Maximum,
    Minimum,
}
/// Fuse intensity projection.
/// # Errors
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub(crate) fn fuse_intensity_projection(
    fusion: &MultiModalFusion,
    kind: ProjectionKind,
) -> KwaversResult<FusedImageResult> {
    let modalities = super::utils::sorted_modalities(fusion)?;
    let dims = super::utils::common_registered_dims(&modalities, "Intensity projection")?;

    let mut intensity_image = Array3::<f64>::zeros(dims);
    let mut confidence_map = Array3::<f64>::zeros(dims);
    let mut uncertainty_map = fusion
        .config
        .uncertainty_quantification
        .then(|| Array3::<f64>::zeros(dims));

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            for k in 0..dims.2 {
                let mut selected_value = match kind {
                    ProjectionKind::Maximum => f64::NEG_INFINITY,
                    ProjectionKind::Minimum => f64::INFINITY,
                };
                let mut selected_quality = 0.0;

                for (_, current) in &modalities {
                    let value = current.data[[i, j, k]];
                    let select = match kind {
                        ProjectionKind::Maximum => value > selected_value,
                        ProjectionKind::Minimum => value < selected_value,
                    };
                    if select {
                        selected_value = value;
                        selected_quality = current.quality_score;
                    }
                }

                intensity_image[[i, j, k]] = selected_value;
                confidence_map[[i, j, k]] = selected_quality;
                if let Some(ref mut uncertainty) = uncertainty_map {
                    uncertainty[[i, j, k]] = 1.0 - selected_quality;
                }
            }
        }
    }

    Ok(FusedImageResult {
        intensity_image,
        tissue_properties: HashMap::new(),
        confidence_map,
        uncertainty_map,
        registration_transforms: super::utils::identity_registration_transforms(&modalities),
        modality_quality: super::utils::modality_quality_map(&modalities),
        coordinates: generate_coordinate_arrays(dims, fusion.config.output_resolution),
    })
}
