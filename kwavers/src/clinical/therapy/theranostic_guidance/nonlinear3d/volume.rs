//! CT/NIfTI volume preparation for nonlinear 3-D propagation.

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::scene::target_index_from_mask_fraction_3d;
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
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<Nonlinear3dVolume> {
    validation::validate_inputs(ct_hu, label_volume, spacing_mm)?;
    let source_dimensions = {
        let (nx, ny, nz) = ct_hu.dim();
        [nx, ny, nz]
    };
    let source_spacing_m = [
        spacing_mm[0] * 1.0e-3,
        spacing_mm[1] * 1.0e-3,
        spacing_mm[2] * 1.0e-3,
    ];
    let body = mask::body_mask_full(anatomy, ct_hu, label_volume);
    let brain_target_center_index = match anatomy {
        AnatomyKind::Brain => Some(brain_target_center_index(
            &body,
            ct_hu,
            target_fraction_xyz,
        )?),
        AnatomyKind::Liver | AnatomyKind::Kidney => None,
    };
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
    let aperture_skin_index = match (anatomy, target.as_ref()) {
        (AnatomyKind::Liver | AnatomyKind::Kidney, Some(target)) => Some(
            bbox::planned_abdominal_skin_index(&body, target, spacing_mm)?,
        ),
        _ => None,
    };
    let aperture_direction = match (target.as_ref(), aperture_skin_index) {
        (Some(target), Some(skin)) => Some(focus_to_skin_direction(target, skin, spacing_mm)?),
        _ => None,
    };
    let bbox = bbox::crop_bbox(
        anatomy,
        &body,
        target.as_ref(),
        aperture_skin_index,
        brain_target_center_index,
        spacing_mm,
        config.treatment_window_radius_m,
    )?;
    let crop_bounds_index = [bbox.x0, bbox.x1, bbox.y0, bbox.y1, bbox.z0, bbox.z1];
    let n = config.grid_size;
    let ct = resample::resample_scalar(ct_hu, bbox, n);
    let single_target_labels = match (anatomy, label_volume, target.as_ref()) {
        (AnatomyKind::Liver | AnatomyKind::Kidney, Some(labels), Some(target)) => {
            Some(mask::single_target_label_volume(labels, target))
        }
        _ => None,
    };
    let label = if let Some(labels) = single_target_labels.as_ref() {
        resample::resample_labels(labels, bbox, n)
    } else if let Some(labels) = label_volume {
        resample::resample_labels(labels, bbox, n)
    } else {
        Array3::<i16>::zeros((n, n, n))
    };
    let spacing_m = resample::isotropic_spacing_m(bbox, spacing_mm, n);
    let resampled_brain_target_center = match (anatomy, brain_target_center_index) {
        (AnatomyKind::Brain, Some(center)) => Some(map_index_to_resampled_grid(center, bbox, n)),
        _ => None,
    };
    let aperture_skin = aperture_skin_index.map(|skin| map_index_to_grid_index(skin, bbox, n));
    let (body_mask, target_mask) = mask::masks(
        anatomy,
        &ct,
        &label,
        spacing_m,
        target_fraction_xyz,
        resampled_brain_target_center,
    )?;
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
        source_dimensions,
        source_spacing_m,
        crop_bounds_index,
        aperture_direction,
        aperture_skin,
        focus,
    })
}

fn brain_target_center_index(
    body: &Array3<bool>,
    ct_hu: &Array3<f64>,
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<[f64; 3]> {
    let brain_support = Array3::from_shape_fn(body.dim(), |idx| body[idx] && ct_hu[idx] < 300.0);
    let support = if brain_support.iter().any(|active| *active) {
        &brain_support
    } else {
        body
    };
    if let Some(fraction) = target_fraction_xyz {
        let index = target_index_from_mask_fraction_3d(support, fraction)?;
        Ok([index[0] as f64, index[1] as f64, index[2] as f64])
    } else {
        centroid_float(support, None)
            .or_else(|| centroid_float(body, None))
            .ok_or_else(|| KwaversError::InvalidInput("brain body support is empty".to_owned()))
    }
}

fn map_index_to_resampled_grid(
    index: [f64; 3],
    bbox: crate::clinical::therapy::theranostic_guidance::geometry::IndexBounds3,
    n: usize,
) -> [f64; 3] {
    [
        map_axis_to_resampled_grid(index[0], bbox.x0, bbox.x1, n),
        map_axis_to_resampled_grid(index[1], bbox.y0, bbox.y1, n),
        map_axis_to_resampled_grid(index[2], bbox.z0, bbox.z1, n),
    ]
}

fn map_axis_to_resampled_grid(index: f64, min: usize, max: usize, n: usize) -> f64 {
    if n <= 1 || max == min {
        return 0.0;
    }
    ((index - min as f64) * (n - 1) as f64 / (max - min) as f64).clamp(0.0, (n - 1) as f64)
}

fn map_index_to_grid_index(
    index: [f64; 3],
    bbox: crate::clinical::therapy::theranostic_guidance::geometry::IndexBounds3,
    n: usize,
) -> super::types::GridIndex {
    let mapped = map_index_to_resampled_grid(index, bbox, n);
    let to_index = |value: f64| value.round().clamp(0.0, (n - 1) as f64) as usize;
    super::types::GridIndex {
        x: to_index(mapped[0]),
        y: to_index(mapped[1]),
        z: to_index(mapped[2]),
    }
}

fn focus_to_skin_direction(
    target: &Array3<bool>,
    skin: [f64; 3],
    spacing_mm: [f64; 3],
) -> KwaversResult<[f64; 3]> {
    let focus = centroid_float(target, None).ok_or_else(|| {
        KwaversError::InvalidInput("abdominal nonlinear target mask is empty".to_owned())
    })?;
    let raw = [
        (skin[0] - focus[0]) * spacing_mm[0],
        (skin[1] - focus[1]) * spacing_mm[1],
        (skin[2] - focus[2]) * spacing_mm[2],
    ];
    let norm = raw[0].hypot(raw[1]).hypot(raw[2]);
    if norm <= 1.0e-12 || !norm.is_finite() {
        return Err(KwaversError::InvalidInput(
            "abdominal aperture direction is degenerate".to_owned(),
        ));
    }
    Ok([raw[0] / norm, raw[1] / norm, raw[2] / norm])
}
