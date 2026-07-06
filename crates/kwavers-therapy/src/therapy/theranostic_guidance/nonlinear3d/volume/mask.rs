//! Body, target, and inversion mask construction for nonlinear 3-D volumes.

use std::collections::VecDeque;

use ndarray::{s, Array3};

use kwavers_core::constants::ct_acoustics::{
    HU_ABDOMEN_BODY_THRESHOLD, HU_BONE_THRESHOLD, HU_BRAIN_BODY_THRESHOLD,
};
use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::super::abdominal3d::helpers::exterior_air_mask;
use super::super::super::medium::{largest_connected_target_component, largest_target_slice};
use super::super::super::scene::target_index_from_mask_fraction_3d;
use super::super::super::AnatomyKind;
use super::centroid::centroid_float;
use super::INTERNAL_GAS_HU_THRESHOLD;

pub(super) fn body_mask_full(
    anatomy: AnatomyKind,
    ct_hu: &Array3<f64>,
    label_volume: Option<&Array3<i16>>,
) -> Array3<bool> {
    let anatomical_body = Array3::from_shape_fn(ct_hu.dim(), |idx| {
        let label = label_volume.map_or(0, |labels| labels[idx]);
        match anatomy {
            AnatomyKind::Brain => ct_hu[idx] > HU_BRAIN_BODY_THRESHOLD,
            AnatomyKind::Liver | AnatomyKind::Kidney => {
                ct_hu[idx] > HU_ABDOMEN_BODY_THRESHOLD || label > 0
            }
        }
    });
    let exterior_air = exterior_air_mask(&anatomical_body);
    Array3::from_shape_fn(ct_hu.dim(), |idx| {
        let label = label_volume.map_or(0, |labels| labels[idx]);
        anatomical_body[idx]
            || (ct_hu[idx] < INTERNAL_GAS_HU_THRESHOLD && label == 0 && !exterior_air[idx])
    })
}

pub(super) fn target_mask_full(label: &Array3<i16>) -> KwaversResult<Array3<bool>> {
    let z = largest_target_slice(label)?;
    let label_slice = label.slice(s![.., .., z]).to_owned();
    let slice_target = largest_connected_target_component(&label_slice)?;
    let seed = slice_target
        .indexed_iter()
        .find_map(|((x, y), active)| active.then_some((x, y, z)))
        .ok_or_else(|| {
            KwaversError::InvalidInput(
                "segmentation contains no connected label-2 nonlinear 3-D target".to_owned(),
            )
        })?;
    let target = connected_target_component_3d(label, seed);
    let count = target.iter().filter(|active| **active).count();
    if count < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "nonlinear 3-D connected target component is too small: {count}"
        )));
    }
    Ok(target)
}

pub(super) fn single_target_label_volume(
    label: &Array3<i16>,
    target: &Array3<bool>,
) -> Array3<i16> {
    Array3::from_shape_fn(label.dim(), |idx| {
        if label[idx] == 2 && !target[idx] {
            1
        } else {
            label[idx]
        }
    })
}

pub(super) fn masks(
    anatomy: AnatomyKind,
    ct: &Array3<f64>,
    label: &Array3<i16>,
    spacing_m: f64,
    target_fraction_xyz: Option<[f64; 3]>,
    target_center_index: Option<[f64; 3]>,
) -> KwaversResult<(Array3<bool>, Array3<bool>)> {
    let body = body_mask_full(anatomy, ct, Some(label));
    let target = match anatomy {
        AnatomyKind::Brain => synthetic_brain_target(
            ct,
            &body,
            spacing_m,
            target_fraction_xyz,
            target_center_index,
        )?,
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

fn connected_target_component_3d(label: &Array3<i16>, seed: (usize, usize, usize)) -> Array3<bool> {
    let (nx, ny, nz) = label.dim();
    let mut component = Array3::<bool>::from_elem(label.dim(), false);
    let mut queue = VecDeque::from([seed]);
    component[seed] = true;
    while let Some((x, y, z)) = queue.pop_front() {
        if label[[x, y, z]] != 2 {
            continue;
        }
        for next in target_neighbors_3d(x, y, z, nx, ny, nz) {
            if component[next] || label[next] != 2 {
                continue;
            }
            component[next] = true;
            queue.push_back(next);
        }
    }
    component
}

fn target_neighbors_3d(
    x: usize,
    y: usize,
    z: usize,
    nx: usize,
    ny: usize,
    nz: usize,
) -> impl Iterator<Item = (usize, usize, usize)> {
    let mut neighbors = [(x, y, z); 6];
    let mut count = 0;
    if x > 0 {
        neighbors[count] = (x - 1, y, z);
        count += 1;
    }
    if x + 1 < nx {
        neighbors[count] = (x + 1, y, z);
        count += 1;
    }
    if y > 0 {
        neighbors[count] = (x, y - 1, z);
        count += 1;
    }
    if y + 1 < ny {
        neighbors[count] = (x, y + 1, z);
        count += 1;
    }
    if z > 0 {
        neighbors[count] = (x, y, z - 1);
        count += 1;
    }
    if z + 1 < nz {
        neighbors[count] = (x, y, z + 1);
        count += 1;
    }
    neighbors.into_iter().take(count)
}

fn synthetic_brain_target(
    ct: &Array3<f64>,
    body: &Array3<bool>,
    spacing_m: f64,
    target_fraction_xyz: Option<[f64; 3]>,
    target_center_index: Option<[f64; 3]>,
) -> KwaversResult<Array3<bool>> {
    let n = body.dim().0;
    let brain_support =
        Array3::from_shape_fn(body.dim(), |idx| body[idx] && ct[idx] < HU_BONE_THRESHOLD);
    let support = if brain_support.iter().any(|active| *active) {
        &brain_support
    } else {
        body
    };
    let center = if let Some(center) = target_center_index {
        center
    } else if let Some(fraction) = target_fraction_xyz {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abdominal_target_mask_uses_one_connected_treatment_component() {
        let mut label = Array3::<i16>::zeros((8, 8, 4));
        for idx in [
            (1, 1, 1),
            (1, 2, 1),
            (2, 1, 1),
            (2, 1, 2),
            (6, 6, 1),
            (6, 5, 1),
            (6, 6, 2),
        ] {
            label[idx] = 2;
        }

        let target = target_mask_full(&label).unwrap();

        assert!(target[[1, 1, 1]]);
        assert!(target[[1, 2, 1]]);
        assert!(target[[2, 1, 1]]);
        assert!(target[[2, 1, 2]]);
        assert!(!target[[6, 6, 1]]);
        assert!(!target[[6, 5, 1]]);
        assert!(!target[[6, 6, 2]]);
        assert_eq!(target.iter().filter(|active| **active).count(), 4);
    }

    #[test]
    fn single_target_label_volume_demotes_nonselected_tumours_to_organ() {
        let mut label = Array3::<i16>::zeros((4, 4, 2));
        label[[1, 1, 0]] = 2;
        label[[3, 3, 0]] = 2;
        let mut target = Array3::<bool>::from_elem(label.dim(), false);
        target[[1, 1, 0]] = true;

        let filtered = single_target_label_volume(&label, &target);

        assert_eq!(filtered[[1, 1, 0]], 2);
        assert_eq!(filtered[[3, 3, 0]], 1);
    }

    #[test]
    fn body_mask_preserves_enclosed_internal_gas_and_excludes_exterior_air() {
        let mut ct = Array3::<f64>::from_elem((7, 7, 7), -1000.0);
        let label = Array3::<i16>::zeros((7, 7, 7));
        for ix in 1..6 {
            for iy in 1..6 {
                for iz in 1..6 {
                    ct[[ix, iy, iz]] = 40.0;
                }
            }
        }
        ct[[3, 3, 3]] = -900.0;

        let body = body_mask_full(AnatomyKind::Liver, &ct, Some(&label));

        assert!(
            body[[3, 3, 3]],
            "enclosed gas must remain in the patient support"
        );
        assert!(
            !body[[0, 0, 0]],
            "boundary-connected CT air must remain exterior"
        );
    }
}
