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
//! Reference organ sound speeds: liver 1578 m/s, kidney 1560 m/s (Duck 1990 Table 4.6).

use kwavers_core::constants::acoustic_parameters::SOFT_TISSUE_HU_BASE_SPEED_M_S;
use kwavers_core::constants::ct_acoustics::HU_ABDOMEN_BODY_THRESHOLD;
use kwavers_core::constants::fundamental::{SOUND_SPEED_AIR, SOUND_SPEED_TISSUE};
use kwavers_core::constants::tissue_acoustics::{SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array2, Array3, Axis, Zip};

// ── Abdominal CT acoustic property model calibration constants ────────────────
//
// Calibrated against Duck (1990) Table 4.6 (organ sound speeds) and
// Marsac et al. (2017) soft-tissue HU model (slope parameters).

/// Background soft-tissue HU-to-speed slope [m/s/HU].
///
/// Matches `CT_SPEED_SLOPE_BACKGROUND_M_S_PER_HU` in the nonlinear 3-D volume model.
const ABDOM_BG_SPEED_SLOPE_M_S_PER_HU: f64 = 0.18;
/// HU clamp lower bound for background soft-tissue speed [HU].
const ABDOM_BG_HU_LOWER: f64 = -150.0;
/// HU clamp upper bound for background soft-tissue speed [HU].
const ABDOM_BG_HU_UPPER: f64 = 250.0;
/// Background soft-tissue attenuation coefficient [dB/(cm·MHz)].
///
/// Slightly above `ACOUSTIC_ABSORPTION_TISSUE` (0.5) to account for fibrous
/// stroma and connective tissue in the abdominal wall (Duck 1990 Table 4.2).
const ABDOM_BG_ATTENUATION_DB_CM_MHZ: f64 = 0.55;

/// Organ HU-to-speed slope [m/s/HU].
const ABDOM_ORGAN_SPEED_SLOPE_M_S_PER_HU: f64 = 0.10;
/// HU clamp lower bound for organ speed correction [HU].
const ABDOM_ORGAN_HU_LOWER: f64 = -100.0;
/// HU clamp upper bound for organ speed correction [HU].
const ABDOM_ORGAN_HU_UPPER: f64 = 200.0;
/// Organ attenuation coefficient [dB/(cm·MHz)] (Duck 1990 liver/kidney range 0.4–0.9).
const ABDOM_ORGAN_ATTENUATION_DB_CM_MHZ: f64 = 0.8;

/// Target tumour speed offset from organ baseline [m/s].
///
/// Tumour tissue has slightly lower speed than surrounding parenchyma due to
/// increased water content (Duck 1990 Chapter 4).
const ABDOM_TARGET_SPEED_OFFSET_M_S: f64 = -22.0;
/// Target HU-to-speed slope [m/s/HU].
const ABDOM_TARGET_SPEED_SLOPE_M_S_PER_HU: f64 = 0.12;
/// HU clamp lower bound for target speed correction [HU].
const ABDOM_TARGET_HU_LOWER: f64 = -50.0;
/// HU clamp upper bound for target speed correction [HU].
const ABDOM_TARGET_HU_UPPER: f64 = 220.0;
/// Target (tumour) attenuation coefficient [dB/(cm·MHz)] (hypervascular tumour).
const ABDOM_TARGET_ATTENUATION_DB_CM_MHZ: f64 = 1.05;

/// HU threshold for abdominal calcification/bone classification [HU].
const ABDOM_CALCIFICATION_HU_THRESHOLD: f64 = 250.0;
/// Calcified-tissue sound speed base at the classification threshold [m/s].
///
/// At HU = 250 (onset of calcification), c ≈ 2450 m/s — between cortical bone
/// (2900 m/s) and highly mineralised soft tissue.
const ABDOM_CALC_SPEED_BASE_M_S: f64 = 2450.0;
/// Calcified-tissue speed slope above `ABDOM_CALCIFICATION_HU_THRESHOLD` [m/s/HU].
const ABDOM_CALC_SPEED_SLOPE_M_S_PER_HU: f64 = 0.42;
/// HU range over which calcification speed saturates above threshold [HU].
const ABDOM_CALC_HU_RANGE: f64 = 1400.0;
/// Calcified-tissue attenuation coefficient [dB/(cm·MHz)].
///
/// Intermediate between soft tissue and cortical bone (Connor & Hynynen 2002).
const ABDOM_CALC_ATTENUATION_DB_CM_MHZ: f64 = 18.0;

/// Background (air/unclassified) attenuation coefficient [dB/(cm·MHz)].
const ABDOM_DEFAULT_ATTENUATION_DB_CM_MHZ: f64 = 0.05;
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
    let body_crop = body_component
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
    let body_support = resample_mask_any(&body_crop, grid_size);
    let spacing_m = ((bbox.1 - bbox.0 + 1) as f64 * spacing_mm[0] * 1.0e-3)
        .max((bbox.3 - bbox.2 + 1) as f64 * spacing_mm[1] * 1.0e-3)
        / grid_size as f64;
    let (sound_speed, attenuation, body, organ, target) =
        abdominal_properties(anatomy, &ct, &label, &body_support);
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
/// Sources: Duck FA, *Physical Properties of Tissue*, Academic Press, 1990, Table 4.6.
/// - Liver: `SOUND_SPEED_LIVER` = 1578 m/s
/// - Kidney: `SOUND_SPEED_KIDNEY` = 1560 m/s
/// - Brain (soft-tissue fallback used when anatomy is [`AnatomyKind::Brain`]): 1540 m/s
#[inline]
fn organ_reference_speed(anatomy: AnatomyKind) -> f64 {
    match anatomy {
        AnatomyKind::Liver => SOUND_SPEED_LIVER,
        AnatomyKind::Kidney => SOUND_SPEED_KIDNEY,
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
    body_support: &Array2<bool>,
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
    let mut attenuation = Array2::<f64>::from_elem((nx, ny), ABDOM_DEFAULT_ATTENUATION_DB_CM_MHZ);
    let mut body = Array2::<bool>::from_elem((nx, ny), false);
    let mut organ = Array2::<bool>::from_elem((nx, ny), false);
    let mut target = Array2::<bool>::from_elem((nx, ny), false);
    // Pass 1: classify each voxel into anatomical masks.
    Zip::from(&mut body)
        .and(&mut organ)
        .and(&mut target)
        .and(body_support)
        .and(label)
        .for_each(|bod, org, tgt, &support, &lab| {
            *org = lab == 1 || lab == 2;
            *tgt = lab == 2;
            *bod = support || *org || *tgt;
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
                *spd = SOFT_TISSUE_HU_BASE_SPEED_M_S
                    + ABDOM_BG_SPEED_SLOPE_M_S_PER_HU
                        * hu.clamp(ABDOM_BG_HU_LOWER, ABDOM_BG_HU_UPPER);
                *att = ABDOM_BG_ATTENUATION_DB_CM_MHZ;
            }
            if org {
                *spd = c_organ
                    + ABDOM_ORGAN_SPEED_SLOPE_M_S_PER_HU
                        * hu.clamp(ABDOM_ORGAN_HU_LOWER, ABDOM_ORGAN_HU_UPPER);
                *att = ABDOM_ORGAN_ATTENUATION_DB_CM_MHZ;
            }
            if tgt {
                *spd = c_organ
                    + ABDOM_TARGET_SPEED_OFFSET_M_S
                    + ABDOM_TARGET_SPEED_SLOPE_M_S_PER_HU
                        * hu.clamp(ABDOM_TARGET_HU_LOWER, ABDOM_TARGET_HU_UPPER);
                *att = ABDOM_TARGET_ATTENUATION_DB_CM_MHZ;
            }
            if hu > ABDOM_CALCIFICATION_HU_THRESHOLD {
                *spd = ABDOM_CALC_SPEED_BASE_M_S
                    + ABDOM_CALC_SPEED_SLOPE_M_S_PER_HU
                        * (hu - ABDOM_CALCIFICATION_HU_THRESHOLD).clamp(0.0, ABDOM_CALC_HU_RANGE);
                *att = ABDOM_CALC_ATTENUATION_DB_CM_MHZ;
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
    hu > HU_ABDOMEN_BODY_THRESHOLD || label > 0
}

fn resample_mask_any(input: &Array2<bool>, size: usize) -> Array2<bool> {
    let (nx, ny) = input.dim();
    Array2::from_shape_fn((size, size), |(ix, iy)| {
        let x0 = (ix * nx) / size;
        let x1 = (((ix + 1) * nx).saturating_sub(1)) / size;
        let y0 = (iy * ny) / size;
        let y1 = (((iy + 1) * ny).saturating_sub(1)) / size;
        for x in x0..=x1.min(nx - 1) {
            for y in y0..=y1.min(ny - 1) {
                if input[[x, y]] {
                    return true;
                }
            }
        }
        false
    })
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

#[cfg(test)]
mod tests;
