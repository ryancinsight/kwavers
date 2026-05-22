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
    // Liver organ baseline: SOUND_SPEED_LIVER = 1578 m/s; kidney: SOUND_SPEED_KIDNEY = 1560 m/s
    // Both at HU=0 so the HU-offset contribution is zero.
    use crate::core::constants::fundamental::{SOUND_SPEED_KIDNEY, SOUND_SPEED_LIVER};
    assert!((speed_liver[[0, 0]] - SOUND_SPEED_LIVER).abs() < 1.0e-10);
    assert!((speed_kidney[[0, 0]] - SOUND_SPEED_KIDNEY).abs() < 1.0e-10);
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
