//! Abdominal CT preprocessing for theranostic inverse (liver, kidney).
//!
//! Accepts a full 3-D CT volume and segmentation, selects the axial slice with
//! the largest tumour cross-section, crops to a square bounding box, resamples
//! to the solver grid, and derives acoustic property maps from HU and label.
//!
//! # Acoustic property table (per label)
//!
//! | Label | Tissue    | c [m/s]                | α [Np/m/MHz] |
//! |-------|-----------|------------------------|--------------|
//! | 0     | Air/other | 343.0                  | 0.05         |
//! | body  | Soft tissue| 1480 + 0.18·HU        | 0.55         |
//! | 1     | Organ     | organ_c + 0.10·HU      | 0.80         |
//! | 2     | Tumour    | organ_c − 22 + 0.12·HU | 1.05        |
//! | bone  | HU > 250  | 2450 + 0.42·(HU − 250) | 18.0        |
//!
//! Reference organ sound speeds: liver 1595 m/s, kidney 1567 m/s (Duck 1990).

use crate::core::constants::fundamental::{SOUND_SPEED_AIR, SOUND_SPEED_TISSUE};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2, Array3, Axis, Zip};
use std::collections::VecDeque;

use super::{resample, resample_labels_max, validate_masks, AnatomyKind, PreparedTheranosticSlice};

pub fn prepare_abdominal_slice(
    anatomy: AnatomyKind,
    ct_volume_hu: &Array3<f64>,
    label_volume: &Array3<i16>,
    spacing_mm: [f64; 3],
    grid_size: usize,
) -> KwaversResult<PreparedTheranosticSlice> {
    if grid_size < 2 {
        return Err(KwaversError::InvalidInput(format!(
            "grid_size must be at least 2, got {grid_size}"
        )));
    }
    if ct_volume_hu.dim() != label_volume.dim() {
        return Err(KwaversError::InvalidInput(format!(
            "CT shape {:?} does not match segmentation shape {:?}",
            ct_volume_hu.dim(),
            label_volume.dim()
        )));
    }
    let slice_index = largest_target_slice(label_volume)?;
    let ct_slice = ct_volume_hu.slice(s![.., .., slice_index]).to_owned();
    let label_slice = label_volume.slice(s![.., .., slice_index]).to_owned();
    let treatment_target = largest_connected_target_component(&label_slice)?;
    let target_seed = target_seed_index(&treatment_target)?;
    let body_component = connected_body_component(&ct_slice, &label_slice, target_seed)?;
    let bbox = square_bbox_from_mask(&body_component, 6)?;
    let ct_crop = ct_slice
        .slice(s![bbox.0..=bbox.1, bbox.2..=bbox.3])
        .to_owned();
    let mut label_crop = label_slice
        .slice(s![bbox.0..=bbox.1, bbox.2..=bbox.3])
        .to_owned();
    let target_crop = treatment_target
        .slice(s![bbox.0..=bbox.1, bbox.2..=bbox.3])
        .to_owned();
    for ((ix, iy), label) in label_crop.indexed_iter_mut() {
        if *label == 2 && !target_crop[[ix, iy]] {
            *label = 1;
        }
    }
    let ct = resample(&ct_crop, grid_size);
    let label = resample_labels_max(&label_crop, grid_size);
    let spacing_m = ((bbox.1 - bbox.0 + 1) as f64 * spacing_mm[0] * 1.0e-3)
        .max((bbox.3 - bbox.2 + 1) as f64 * spacing_mm[1] * 1.0e-3)
        / grid_size as f64;
    let (sound_speed, attenuation, body, organ, target) =
        abdominal_properties(anatomy, &ct, &label);
    validate_masks(&body, &target)?;
    Ok(PreparedTheranosticSlice {
        anatomy,
        ct_hu: ct,
        label,
        sound_speed_m_s: sound_speed,
        attenuation_np_per_m_mhz: attenuation,
        body_mask: body,
        organ_mask: organ,
        target_mask: target,
        spacing_m,
        source_slice_index: slice_index,
        source_dimensions: [ct_volume_hu.dim().0, ct_volume_hu.dim().1],
        source_spacing_m: [spacing_mm[0] * 1.0e-3, spacing_mm[1] * 1.0e-3],
        crop_bounds_index: [bbox.0, bbox.1, bbox.2, bbox.3],
    })
}

/// Index of the axial (z) slice that contains the most label-2 (tumour) voxels.
///
/// Iterates z-slices via [`ndarray::Axis`] view and returns the index of the
/// slice with the maximum label-2 count.  Returns [`KwaversError::InvalidInput`]
/// when no label-2 cell exists in the volume.
pub(crate) fn largest_target_slice(label: &Array3<i16>) -> KwaversResult<usize> {
    label
        .axis_iter(Axis(2))
        .enumerate()
        .map(|(z, slice)| (z, slice.iter().filter(|&&v| v == 2).count()))
        .filter(|&(_, count)| count > 0)
        .max_by_key(|&(_, count)| count)
        .map(|(z, _)| z)
        .ok_or_else(|| {
            KwaversError::InvalidInput("segmentation contains no label-2 target".to_owned())
        })
}

/// Largest 4-connected component of label-2 cells in a 2-D slice.
pub(crate) fn largest_connected_target_component(
    label: &Array2<i16>,
) -> KwaversResult<Array2<bool>> {
    let (nx, ny) = label.dim();
    let mut visited = Array2::<bool>::from_elem((nx, ny), false);
    let mut best = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            if visited[[ix, iy]] || label[[ix, iy]] != 2 {
                continue;
            }
            let component = target_component(label, &mut visited, ix, iy);
            if component.len() > best.len() {
                best = component;
            }
        }
    }
    if best.is_empty() {
        return Err(KwaversError::InvalidInput(
            "segmentation contains no connected label-2 target component".to_owned(),
        ));
    }
    let mut target = Array2::<bool>::from_elem((nx, ny), false);
    for (ix, iy) in best {
        target[[ix, iy]] = true;
    }
    Ok(target)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Reference sound speed [m/s] for the dominant organ per anatomy.
///
/// Sources: Duck FA, *Physical Properties of Tissue*, Academic Press, 1990.
/// - Liver: 1595 m/s
/// - Kidney: 1567 m/s
/// - Brain (soft-tissue fallback used when anatomy is [`AnatomyKind::Brain`]): 1540 m/s
#[inline]
fn organ_reference_speed(anatomy: AnatomyKind) -> f64 {
    match anatomy {
        AnatomyKind::Liver => 1595.0,
        AnatomyKind::Kidney => 1567.0,
        AnatomyKind::Brain => SOUND_SPEED_TISSUE,
    }
}

/// Derive voxel-wise acoustic property maps from HU values and segmentation labels.
///
/// All output arrays share the shape of `ct`.  The computation uses two
/// [`ndarray::Zip`] passes over the output and input arrays simultaneously,
/// eliminating per-element random indexing and enabling LLVM auto-vectorisation.
///
/// # Tissue classification hierarchy (evaluated top-to-bottom; later rules override)
///
/// | Priority | Condition            | c [m/s]                              | α [Np/m/MHz] |
/// |----------|----------------------|--------------------------------------|--------------|
/// | 1 (base) | air / background     | 343.0                                | 0.05         |
/// | 2        | body (HU > −450 or label > 0) | 1480 + 0.18·clamp(HU, −150, 250)  | 0.55  |
/// | 3        | organ (label 1 or 2) | c_organ + 0.10·clamp(HU, −100, 200) | 0.80         |
/// | 4        | tumour (label 2)     | c_organ − 22 + 0.12·clamp(HU, −50, 220) | 1.05     |
/// | 5 (top)  | bone (HU > 250)      | 2450 + 0.42·clamp(HU − 250, 0, 1400) | 18.0        |
fn abdominal_properties(
    anatomy: AnatomyKind,
    ct: &Array2<f64>,
    label: &Array2<i16>,
) -> (
    Array2<f64>,
    Array2<f64>,
    Array2<bool>,
    Array2<bool>,
    Array2<bool>,
) {
    let (nx, ny) = ct.dim();
    let c_organ = organ_reference_speed(anatomy);
    let mut speed = Array2::<f64>::from_elem((nx, ny), SOUND_SPEED_AIR);
    let mut attenuation = Array2::<f64>::from_elem((nx, ny), 0.05);
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut organ = Array2::<bool>::from_elem((nx, ny), false);
    let mut target = Array2::<bool>::from_elem((nx, ny), false);
    // Pass 1: classify each voxel into anatomical masks.
    Zip::from(&mut body)
        .and(&mut organ)
        .and(&mut target)
        .and(ct)
        .and(label)
        .for_each(|bod, org, tgt, &hu, &lab| {
            *bod = hu > -450.0 || lab > 0;
            *org = lab == 1 || lab == 2;
            *tgt = lab == 2;
        });
    // Pass 2: map masks + HU to acoustic properties.
    Zip::from(&mut speed)
        .and(&mut attenuation)
        .and(&body)
        .and(&organ)
        .and(&target)
        .and(ct)
        .for_each(|spd, att, &bod, &org, &tgt, &hu| {
            if bod {
                *spd = 1480.0 + 0.18 * hu.clamp(-150.0, 250.0);
                *att = 0.55;
            }
            if org {
                *spd = c_organ + 0.10 * hu.clamp(-100.0, 200.0);
                *att = 0.8;
            }
            if tgt {
                *spd = c_organ - 22.0 + 0.12 * hu.clamp(-50.0, 220.0);
                *att = 1.05;
            }
            if hu > 250.0 {
                *spd = 2450.0 + 0.42 * (hu - 250.0).clamp(0.0, 1400.0);
                *att = 18.0;
            }
        });
    (speed, attenuation, body, organ, target)
}

/// BFS flood from `(seed_x, seed_y)` collecting all 4-connected label-2 cells.
///
/// Only label-2 cells are ever enqueued: the outer
/// [`largest_connected_target_component`] loop guarantees the seed has
/// `label == 2`, and the push predicate `label[[next]] == 2` ensures only
/// label-2 neighbours enter the queue.  `visited` is shared across all
/// component seeds to avoid re-entering already-processed cells.
fn target_component(
    label: &Array2<i16>,
    visited: &mut Array2<bool>,
    seed_x: usize,
    seed_y: usize,
) -> Vec<(usize, usize)> {
    let (nx, ny) = label.dim();
    let mut component = Vec::new();
    let mut queue = VecDeque::from([(seed_x, seed_y)]);
    visited[[seed_x, seed_y]] = true;
    while let Some((ix, iy)) = queue.pop_front() {
        component.push((ix, iy));
        for (next_x, next_y) in body_neighbors(ix, iy, nx, ny) {
            if !visited[[next_x, next_y]] && label[[next_x, next_y]] == 2 {
                visited[[next_x, next_y]] = true;
                queue.push_back((next_x, next_y));
            }
        }
    }
    component
}

fn target_seed_index(target: &Array2<bool>) -> KwaversResult<(usize, usize)> {
    for ((ix, iy), active) in target.indexed_iter() {
        if *active {
            return Ok((ix, iy));
        }
    }
    Err(KwaversError::InvalidInput(
        "target seed is empty".to_owned(),
    ))
}

fn connected_body_component(
    ct: &Array2<f64>,
    label: &Array2<i16>,
    seed: (usize, usize),
) -> KwaversResult<Array2<bool>> {
    let (nx, ny) = ct.dim();
    let mut component = Array2::<bool>::from_elem((nx, ny), false);
    if !is_abdominal_body_candidate(ct[[seed.0, seed.1]], label[[seed.0, seed.1]]) {
        return Err(KwaversError::InvalidInput(
            "target seed is not inside abdominal body support".to_owned(),
        ));
    }
    let mut queue = VecDeque::from([seed]);
    component[[seed.0, seed.1]] = true;
    while let Some((ix, iy)) = queue.pop_front() {
        for (nx_i, ny_i) in body_neighbors(ix, iy, nx, ny) {
            if component[[nx_i, ny_i]]
                || !is_abdominal_body_candidate(ct[[nx_i, ny_i]], label[[nx_i, ny_i]])
            {
                continue;
            }
            component[[nx_i, ny_i]] = true;
            queue.push_back((nx_i, ny_i));
        }
    }
    let count = component.iter().filter(|active| **active).count();
    if count < 16 {
        return Err(KwaversError::InvalidInput(format!(
            "abdominal body component is too small: {count}"
        )));
    }
    Ok(component)
}

fn is_abdominal_body_candidate(hu: f64, label: i16) -> bool {
    hu > -450.0 || label > 0
}

/// Von-Neumann 4-neighbourhood within `(nx, ny)` bounds.
fn body_neighbors(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut neighbors = [(ix, iy); 4];
    let mut count = 0;
    if ix > 0 {
        neighbors[count] = (ix - 1, iy);
        count += 1;
    }
    if ix + 1 < nx {
        neighbors[count] = (ix + 1, iy);
        count += 1;
    }
    if iy > 0 {
        neighbors[count] = (ix, iy - 1);
        count += 1;
    }
    if iy + 1 < ny {
        neighbors[count] = (ix, iy + 1);
        count += 1;
    }
    neighbors.into_iter().take(count)
}

/// Tightest square bounding box around `mask` with a `margin`-cell border,
/// expanded symmetrically and clamped to the grid.
///
/// # Algorithm
///
/// 1. Compute the tight axis-aligned bounding rectangle from all active cells
///    in a single `indexed_iter` pass.
/// 2. Apply the margin, clamped to the grid extent.
/// 3. Set `side = max(width, height)` to guarantee a square output.
/// 4. Re-centre the square window on the original bounding-rectangle centre
///    and clamp the origin so the window stays within the grid.
///
/// Returns `(x0, x1, y0, y1)` inclusive row/column indices.
fn square_bbox_from_mask(
    mask: &Array2<bool>,
    margin: usize,
) -> KwaversResult<(usize, usize, usize, usize)> {
    let (nx, ny) = mask.dim();
    let bbox = mask
        .indexed_iter()
        .filter_map(|((ix, iy), &active)| active.then_some((ix, iy)))
        .fold(None::<(usize, usize, usize, usize)>, |acc, (ix, iy)| {
            Some(match acc {
                None => (ix, ix, iy, iy),
                Some((x0, x1, y0, y1)) => (x0.min(ix), x1.max(ix), y0.min(iy), y1.max(iy)),
            })
        });
    let (x0, x1, y0, y1) =
        bbox.ok_or_else(|| KwaversError::InvalidInput("body bbox is empty".to_owned()))?;
    let x0 = x0.saturating_sub(margin);
    let x1 = (x1 + margin).min(nx - 1);
    let y0 = y0.saturating_sub(margin);
    let y1 = (y1 + margin).min(ny - 1);
    let side = (x1 - x0 + 1).max(y1 - y0 + 1).min(nx).min(ny);
    let cx2 = x0 + x1;
    let cy2 = y0 + y1;
    let mut sx = ((cx2 + 1).saturating_sub(side)) / 2;
    let mut sy = ((cy2 + 1).saturating_sub(side)) / 2;
    sx = sx.min(nx - side);
    sy = sy.min(ny - side);
    Ok((sx, sx + side - 1, sy, sy + side - 1))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    // ── body_neighbors ────────────────────────────────────────────────────────

    #[test]
    fn body_neighbors_corner_yields_two() {
        let ns: Vec<_> = body_neighbors(0, 0, 5, 5).collect();
        assert_eq!(ns.len(), 2);
        assert!(ns.contains(&(1, 0)));
        assert!(ns.contains(&(0, 1)));
    }

    #[test]
    fn body_neighbors_interior_yields_four() {
        let ns: Vec<_> = body_neighbors(2, 3, 5, 5).collect();
        assert_eq!(ns.len(), 4);
    }

    #[test]
    fn body_neighbors_edge_row_zero_yields_three() {
        // ix=0, iy interior: no (ix-1) neighbour
        let ns: Vec<_> = body_neighbors(0, 2, 5, 5).collect();
        assert_eq!(ns.len(), 3);
        assert!(!ns.iter().any(|&(ix, _)| ix == usize::MAX));
    }

    // ── largest_target_slice ──────────────────────────────────────────────────

    #[test]
    fn largest_target_slice_selects_max_label2_z() {
        // 3×3×4 volume: z=2 has three label-2 voxels, z=0 and z=3 have one each.
        let mut label = Array3::<i16>::zeros((3, 3, 4));
        label[[1, 1, 0]] = 2;
        label[[0, 0, 2]] = 2;
        label[[1, 1, 2]] = 2;
        label[[2, 2, 2]] = 2;
        label[[0, 1, 3]] = 2;
        assert_eq!(largest_target_slice(&label).unwrap(), 2);
    }

    #[test]
    fn largest_target_slice_errors_when_no_target() {
        let label = Array3::<i16>::zeros((3, 3, 3));
        assert!(largest_target_slice(&label).is_err());
    }

    // ── largest_connected_target_component ───────────────────────────────────

    #[test]
    fn largest_connected_selects_bigger_blob() {
        // 10×10 grid: 2×2 blob at (1,1) (4 cells) and isolated cell at (8,8).
        let mut label = Array2::<i16>::zeros((10, 10));
        for x in 1..=2 {
            for y in 1..=2 {
                label[[x, y]] = 2;
            }
        }
        label[[8, 8]] = 2;
        let target = largest_connected_target_component(&label).unwrap();
        assert_eq!(target.iter().filter(|&&v| v).count(), 4);
        assert!(target[[1, 1]]);
        assert!(!target[[8, 8]]);
    }

    #[test]
    fn largest_connected_errors_when_no_target() {
        let label = Array2::<i16>::zeros((5, 5));
        assert!(largest_connected_target_component(&label).is_err());
    }

    // ── square_bbox_from_mask ─────────────────────────────────────────────────

    #[test]
    fn square_bbox_single_active_cell_no_margin() {
        let mut mask = Array2::<bool>::from_elem((10, 10), false);
        mask[[5, 5]] = true;
        let (x0, x1, y0, y1) = square_bbox_from_mask(&mask, 0).unwrap();
        assert_eq!(x1 - x0, 0); // side == 1
        assert_eq!(y1 - y0, 0);
        assert_eq!(x0, 5);
        assert_eq!(y0, 5);
    }

    #[test]
    fn square_bbox_margin_expands_symmetrically() {
        let mut mask = Array2::<bool>::from_elem((20, 20), false);
        mask[[10, 10]] = true;
        let (x0, x1, y0, y1) = square_bbox_from_mask(&mask, 3).unwrap();
        assert_eq!(x0, 7);
        assert_eq!(x1, 13);
        assert_eq!(y0, 7);
        assert_eq!(y1, 13);
    }

    #[test]
    fn square_bbox_output_is_always_square() {
        // Asymmetric mask: 16-cell horizontal line in x, single cell in y.
        let mut mask = Array2::<bool>::from_elem((30, 30), false);
        for x in 5..=20 {
            mask[[x, 10]] = true;
        }
        let (x0, x1, y0, y1) = square_bbox_from_mask(&mask, 2).unwrap();
        assert_eq!(x1 - x0, y1 - y0, "bounding box must be square");
    }

    #[test]
    fn square_bbox_empty_mask_errors() {
        let mask = Array2::<bool>::from_elem((10, 10), false);
        assert!(square_bbox_from_mask(&mask, 0).is_err());
    }

    // ── abdominal_properties ─────────────────────────────────────────────────

    #[test]
    fn properties_air_cell_has_default_values() {
        let ct = Array2::<f64>::from_elem((1, 1), -1000.0);
        let label = Array2::<i16>::zeros((1, 1));
        let (speed, att, body, organ, target) =
            abdominal_properties(AnatomyKind::Liver, &ct, &label);
        assert!(!body[[0, 0]]);
        assert!(!organ[[0, 0]]);
        assert!(!target[[0, 0]]);
        assert_eq!(speed[[0, 0]], SOUND_SPEED_AIR);
        assert_eq!(att[[0, 0]], 0.05);
    }

    #[test]
    fn properties_soft_tissue_speed_matches_formula() {
        // HU = 0, no label → body tissue.
        let ct = Array2::<f64>::from_elem((1, 1), 0.0);
        let label = Array2::<i16>::zeros((1, 1));
        let (speed, att, body, _, _) = abdominal_properties(AnatomyKind::Liver, &ct, &label);
        assert!(body[[0, 0]]);
        // c = 1480 + 0.18 × clamp(0, −150, 250) = 1480
        assert!((speed[[0, 0]] - 1480.0).abs() < 1.0e-10);
        assert_eq!(att[[0, 0]], 0.55);
    }

    #[test]
    fn properties_liver_organ_speed_at_hu_100() {
        let ct = Array2::<f64>::from_elem((1, 1), 100.0);
        let mut label = Array2::<i16>::zeros((1, 1));
        label[[0, 0]] = 1;
        let (speed, att, _, organ, target) = abdominal_properties(AnatomyKind::Liver, &ct, &label);
        assert!(organ[[0, 0]]);
        assert!(!target[[0, 0]]);
        // c = 1595 + 0.10 × clamp(100, −100, 200) = 1595 + 10 = 1605
        assert!((speed[[0, 0]] - 1605.0).abs() < 1.0e-10);
        assert_eq!(att[[0, 0]], 0.8);
    }

    #[test]
    fn properties_tumour_speed_offset_below_organ() {
        // label=2, HU=50, liver anatomy.
        let ct = Array2::<f64>::from_elem((1, 1), 50.0);
        let mut label = Array2::<i16>::zeros((1, 1));
        label[[0, 0]] = 2;
        let (speed, att, _, _, target) = abdominal_properties(AnatomyKind::Liver, &ct, &label);
        assert!(target[[0, 0]]);
        // c = (1595 − 22) + 0.12 × clamp(50, −50, 220) = 1573 + 6 = 1579
        assert!((speed[[0, 0]] - 1579.0).abs() < 1.0e-10);
        assert_eq!(att[[0, 0]], 1.05);
    }

    #[test]
    fn properties_bone_overrides_organ_label() {
        // HU=700, label=1: bone rule (HU > 250) takes final precedence.
        let ct = Array2::<f64>::from_elem((1, 1), 700.0);
        let mut label = Array2::<i16>::zeros((1, 1));
        label[[0, 0]] = 1;
        let (speed, att, _, _, _) = abdominal_properties(AnatomyKind::Liver, &ct, &label);
        // c = 2450 + 0.42 × clamp(700 − 250, 0, 1400) = 2450 + 0.42 × 450 = 2639
        assert!((speed[[0, 0]] - 2639.0).abs() < 1.0e-10);
        assert_eq!(att[[0, 0]], 18.0);
    }

    #[test]
    fn properties_kidney_organ_speed_differs_from_liver() {
        let ct = Array2::<f64>::from_elem((1, 1), 0.0);
        let mut label = Array2::<i16>::zeros((1, 1));
        label[[0, 0]] = 1;
        let (speed_liver, _, _, _, _) = abdominal_properties(AnatomyKind::Liver, &ct, &label);
        let (speed_kidney, _, _, _, _) = abdominal_properties(AnatomyKind::Kidney, &ct, &label);
        // Liver organ baseline: 1595; kidney: 1567 (both at HU=0 so offset=0).
        assert!((speed_liver[[0, 0]] - 1595.0).abs() < 1.0e-10);
        assert!((speed_kidney[[0, 0]] - 1567.0).abs() < 1.0e-10);
    }

    // ── connected_body_component ──────────────────────────────────────────────

    #[test]
    fn connected_body_fails_for_air_seed() {
        let ct = Array2::<f64>::from_elem((5, 5), -1000.0);
        let label = Array2::<i16>::zeros((5, 5));
        assert!(connected_body_component(&ct, &label, (2, 2)).is_err());
    }

    #[test]
    fn connected_body_floods_contiguous_tissue_block() {
        let mut ct = Array2::<f64>::from_elem((10, 10), -1000.0);
        for x in 2..7 {
            for y in 2..7 {
                ct[[x, y]] = 50.0; // 5×5 = 25 tissue cells
            }
        }
        let label = Array2::<i16>::zeros((10, 10));
        let comp = connected_body_component(&ct, &label, (3, 3)).unwrap();
        let active: usize = comp.iter().filter(|&&v| v).count();
        assert_eq!(active, 25);
    }

    // ── prepare_abdominal_slice guard ─────────────────────────────────────────

    #[test]
    fn prepare_abdominal_slice_rejects_grid_size_one() {
        let ct = Array3::<f64>::from_elem((5, 5, 1), 0.0);
        let label = Array3::<i16>::zeros((5, 5, 1));
        assert!(
            prepare_abdominal_slice(AnatomyKind::Liver, &ct, &label, [1.0, 1.0, 1.0], 1).is_err()
        );
    }
}
