//! Body, target, and inversion mask construction for nonlinear 3-D volumes.

use ndarray::Array3;

use crate::core::error::{KwaversError, KwaversResult};

use super::super::super::scene::target_index_from_mask_fraction_3d;
use super::super::super::AnatomyKind;
use super::centroid::centroid_float;

pub(super) fn body_mask_full(
    anatomy: AnatomyKind,
    ct_hu: &Array3<f64>,
    label_volume: Option<&Array3<i16>>,
) -> Array3<bool> {
    Array3::from_shape_fn(ct_hu.dim(), |idx| {
        let label = label_volume.map_or(0, |labels| labels[idx]);
        match anatomy {
            AnatomyKind::Brain => ct_hu[idx] > -300.0,
            AnatomyKind::Liver | AnatomyKind::Kidney => ct_hu[idx] > -450.0 || label > 0,
        }
    })
}

pub(super) fn target_mask_full(label: &Array3<i16>) -> KwaversResult<Array3<bool>> {
    let target = label.mapv(|value| value == 2);
    let count = target.iter().filter(|active| **active).count();
    if count == 0 {
        return Err(KwaversError::InvalidInput(
            "segmentation contains no label-2 nonlinear 3-D target".to_owned(),
        ));
    }
    Ok(target)
}

pub(super) fn masks(
    anatomy: AnatomyKind,
    ct: &Array3<f64>,
    label: &Array3<i16>,
    spacing_m: f64,
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<(Array3<bool>, Array3<bool>)> {
    let body = body_mask_full(anatomy, ct, Some(label));
    let target = match anatomy {
        AnatomyKind::Brain => synthetic_brain_target(ct, &body, spacing_m, target_fraction_xyz)?,
        AnatomyKind::Liver | AnatomyKind::Kidney => label.mapv(|value| value == 2),
    };
    let body_count = body.iter().filter(|active| **active).count();
    let target_count = target.iter().filter(|active| **active).count();
    if body_count < 32 || target_count < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "nonlinear 3-D support too small: body={body_count}, target={target_count}"
        )));
    }
    Ok((body, target))
}

fn synthetic_brain_target(
    ct: &Array3<f64>,
    body: &Array3<bool>,
    spacing_m: f64,
    target_fraction_xyz: Option<[f64; 3]>,
) -> KwaversResult<Array3<bool>> {
    let n = body.dim().0;
    let brain_support = Array3::from_shape_fn(body.dim(), |idx| body[idx] && ct[idx] < 300.0);
    let support = if brain_support.iter().any(|active| *active) {
        &brain_support
    } else {
        body
    };
    let center = if let Some(fraction) = target_fraction_xyz {
        let index = target_index_from_mask_fraction_3d(support, fraction)?;
        [index[0] as f64, index[1] as f64, index[2] as f64]
    } else {
        centroid_float(body, None).unwrap_or([
            0.5 * (n - 1) as f64,
            0.5 * (n - 1) as f64,
            0.5 * (n - 1) as f64,
        ])
    };
    let rx = (6.0e-3 / spacing_m).max(1.3);
    let ry = (8.0e-3 / spacing_m).max(1.3);
    let rz = (6.0e-3 / spacing_m).max(1.3);
    Ok(Array3::from_shape_fn(body.dim(), |(ix, iy, iz)| {
        body[[ix, iy, iz]]
            && ((ix as f64 - center[0]) / rx).powi(2)
                + ((iy as f64 - center[1]) / ry).powi(2)
                + ((iz as f64 - center[2]) / rz).powi(2)
                <= 1.0
    }))
}

pub(super) fn inversion_mask(
    target: &Array3<bool>,
    body: &Array3<bool>,
    spacing_m: f64,
) -> Array3<bool> {
    let radius = (8.0e-3 / spacing_m).ceil().max(1.0) as isize;
    Array3::from_shape_fn(target.dim(), |(ix, iy, iz)| {
        if !body[[ix, iy, iz]] {
            return false;
        }
        let ix = ix as isize;
        let iy = iy as isize;
        let iz = iz as isize;
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if dx * dx + dy * dy + dz * dz > radius * radius {
                        continue;
                    }
                    let x = ix + dx;
                    let y = iy + dy;
                    let z = iz + dz;
                    if x < 0 || y < 0 || z < 0 {
                        continue;
                    }
                    let idx = (x as usize, y as usize, z as usize);
                    if idx.0 < target.dim().0
                        && idx.1 < target.dim().1
                        && idx.2 < target.dim().2
                        && target[idx]
                    {
                        return true;
                    }
                }
            }
        }
        false
    })
}
