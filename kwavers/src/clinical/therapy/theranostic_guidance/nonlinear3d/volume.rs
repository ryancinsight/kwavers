//! CT/NIfTI volume preparation for nonlinear 3-D propagation.

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::AnatomyKind;
use super::types::{Nonlinear3dConfig, Nonlinear3dVolume};

mod attenuation;
mod bbox;
mod centroid;
mod mask;
mod material;
mod resample;
mod validation;

pub(crate) use centroid::{centroid_float, centroid_index};

pub(crate) fn prepare_volume(
    anatomy: AnatomyKind,
    ct_hu: &Array3<f64>,
    label_volume: Option<&Array3<i16>>,
    spacing_mm: [f64; 3],
    config: &Nonlinear3dConfig,
) -> KwaversResult<Nonlinear3dVolume> {
    validation::validate_inputs(ct_hu, label_volume, spacing_mm)?;
    let body = mask::body_mask_full(anatomy, ct_hu, label_volume);
    let target = match anatomy {
        AnatomyKind::Brain => None,
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            Some(mask::target_mask_full(label_volume.ok_or_else(|| {
                KwaversError::InvalidInput(
                    "nonlinear abdominal 3-D simulation requires a segmentation volume".to_owned(),
                )
            })?)?)
        }
    };
    let bbox = bbox::crop_bbox(anatomy, &body, target.as_ref(), spacing_mm)?;
    let n = config.grid_size;
    let ct = resample::resample_scalar(ct_hu, bbox, n);
    let label = if let Some(labels) = label_volume {
        resample::resample_labels(labels, bbox, n)
    } else {
        Array3::<i16>::zeros((n, n, n))
    };
    let spacing_m = resample::isotropic_spacing_m(bbox, spacing_mm, n);
    let (body_mask, target_mask) = mask::masks(anatomy, &ct, &label, spacing_m)?;
    let inversion_mask = mask::inversion_mask(&target_mask, &body_mask, spacing_m);
    let focus = centroid_index(&target_mask).ok_or_else(|| {
        KwaversError::InvalidInput("nonlinear 3-D target support is empty".to_owned())
    })?;
    let (background, density, background_beta, attenuation_np_per_m_mhz, attenuation_power_law_y) =
        material::material_maps(anatomy, &ct, &label, &body_mask);
    let true_sound_speed_m_s = Array3::from_shape_fn((n, n, n), |idx| {
        if target_mask[idx] {
            (background[idx] + config.lesion_delta_c_m_s).max(343.0)
        } else {
            background[idx]
        }
    });
    let true_beta = Array3::from_shape_fn((n, n, n), |idx| {
        if target_mask[idx] {
            (background_beta[idx] + config.lesion_delta_beta).clamp(1.0, 12.0)
        } else {
            background_beta[idx]
        }
    });
    Ok(Nonlinear3dVolume {
        anatomy,
        ct_hu: ct,
        label,
        body_mask,
        target_mask,
        inversion_mask,
        density_kg_m3: density,
        background_beta,
        true_beta,
        background_sound_speed_m_s: background,
        true_sound_speed_m_s,
        attenuation_np_per_m_mhz,
        attenuation_power_law_y,
        spacing_m,
        focus,
    })
}
