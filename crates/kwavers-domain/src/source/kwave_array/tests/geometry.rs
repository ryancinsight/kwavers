//! Spatial position, rotation, and weighted-mask tests for [`KWaveArray`].

use super::super::KWaveArray;

#[test]
fn test_set_array_position_matches_manual_position_rotation() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use crate::grid::Grid;

    let grid = Grid::new(41, 41, 11, 5.0e-4, 5.0e-4, 5.0e-4).expect("grid");
    let translation = (5.0e-3, 0.0, 2.0e-3);
    let global_euler = (0.0, 20.0, 0.0);
    let per_element_euler = (0.0, 5.0, 0.0);
    let dims = (1.0e-3, 1.0e-3, 5.0e-4);

    let grid_center = (
        grid.nx as f64 * grid.dx / 2.0,
        grid.ny as f64 * grid.dy / 2.0,
        grid.nz as f64 * grid.dz / 2.0,
    );
    let world_translation = (
        translation.0 + grid_center.0,
        translation.1 + grid_center.1,
        translation.2 + grid_center.2,
    );

    let mut manual = KWaveArray::new();
    let mut native = KWaveArray::new();
    for kx in -2..=2 {
        let local = (1.0e-3 * kx as f64, 0.0, 0.0);
        let r_global = euler_xyz_rotation_matrix(global_euler);
        let rotated_local = apply_matrix(&r_global, local);
        let world = (
            rotated_local.0 + world_translation.0,
            rotated_local.1 + world_translation.1,
            rotated_local.2 + world_translation.2,
        );
        manual.add_rect_rot_element(world, dims.0, dims.1, dims.2, per_element_euler);
        native.add_rect_rot_element(local, dims.0, dims.1, dims.2, per_element_euler);
    }
    native.set_array_position(world_translation, global_euler);

    let m_manual = manual.get_array_binary_mask(&grid);
    let m_native = native.get_array_binary_mask(&grid);

    let manual_count = m_manual.iter().filter(|&&b| b).count();
    let native_count = m_native.iter().filter(|&&b| b).count();
    let inter = ndarray::Zip::from(&m_manual)
        .and(&m_native)
        .fold(0usize, |acc, &a, &b| acc + usize::from(a && b));

    assert!(
        manual_count > 0 && native_count > 0,
        "both masks must be non-empty: manual={manual_count}, native={native_count}",
    );
    let iou = inter as f64 / (manual_count + native_count - inter).max(1) as f64;
    assert!(
        iou >= 0.90,
        "set_array_position must match manual translation/rotation: IoU={iou}, \
         manual={manual_count}, native={native_count}, inter={inter}",
    );
}

#[test]
fn test_rect_rotation_90_swaps_width_and_height() {
    use crate::grid::Grid;

    let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
    let mut unrot = KWaveArray::new();
    unrot.add_rect_element(
        (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
        8.0e-4,
        2.0e-4,
        1.0e-4,
    );
    let unrot_mask = unrot.get_array_binary_mask(&grid);

    let mut rot = KWaveArray::new();
    rot.add_rect_rot_element(
        (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
        8.0e-4,
        2.0e-4,
        1.0e-4,
        (0.0, 0.0, 90.0),
    );
    let rot_mask = rot.get_array_binary_mask(&grid);

    let unrot_count: usize = unrot_mask.iter().filter(|&&b| b).count();
    let rot_count: usize = rot_mask.iter().filter(|&&b| b).count();
    assert!(
        unrot_count > 0 && rot_count > 0,
        "both masks must be non-empty: unrot={unrot_count}, rot={rot_count}",
    );

    let (nx, ny, _nz) = (grid.nx, grid.ny, grid.nz);
    let mut swapped_hits = 0usize;
    for i in 0..nx {
        for j in 0..ny {
            if unrot_mask[[i, j, 2]] {
                let mirror_i = j;
                let mirror_j = nx - 1 - i;
                if mirror_i < nx && mirror_j < ny && rot_mask[[mirror_i, mirror_j, 2]] {
                    swapped_hits += 1;
                }
            }
        }
    }
    assert!(
        swapped_hits >= unrot_count / 2,
        "90-deg Z rotation must overlap after axis swap ({swapped_hits}/{unrot_count})",
    );
}

#[test]
fn test_rect_weighted_mask_matches_kwave_python_reference_mass() {
    use crate::grid::Grid;

    let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
    let mut array = KWaveArray::new();
    array.add_rect_rot_element(
        (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
        8.0e-4,
        2.0e-4,
        1.0e-4,
        (0.0, 0.0, 90.0),
    );
    let weights = array.get_array_weighted_mask(&grid);
    let expected = 16.036_130_608_724_637_f64;
    assert!(
        (weights.sum() - expected).abs() < 5.0e-6,
        "rect weighted mass got {}, expected {expected}",
        weights.sum()
    );
}
