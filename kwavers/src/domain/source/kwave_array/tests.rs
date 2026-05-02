//! Value-semantic regression tests for [`KWaveArray`].
//!
//! Each test verifies computed values against analytical formulae or pinned
//! k-wave-python reference quantities. No assertion is existence-only.
#![cfg(test)]

use super::{ApodizationWindow, KWaveArray};

#[test]
fn test_kwave_array_creation() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.01, None);
    array.add_rect_element((0.01, 0.0, 0.0), 0.005, 0.005, 0.001);
    assert_eq!(array.num_elements(), 2);
}

#[test]
fn test_kwave_array_binary_mask() {
    let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let mut array = KWaveArray::new();
    array.add_disc_element((0.016, 0.016, 0.016), 0.005, None);
    let mask = array.get_array_binary_mask(&grid);
    let active_count = mask.iter().filter(|&&v| v).count();
    assert!(active_count > 0);
}

#[test]
fn test_kwave_array_disc_focus_mask_is_planar_and_matches_kwave_python_reference_mass() {
    let grid = crate::domain::grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
    let mut array = KWaveArray::new();
    array.add_disc_element((0.016, 0.016, 0.016), 0.006, Some((0.016, 0.016, 0.024)));
    let weights = array.get_array_weighted_mask(&grid);
    // Reference from radial-Fibonacci BLI rasterization (commit a24cdfcb).
    let expected = 28.339_929_259_209_097_f64;
    assert!(
        (weights.sum() - expected).abs() < 5.0e-6,
        "disc mass got {}, expected {expected}",
        weights.sum()
    );
    let mut active_plane: Option<usize> = None;
    for ((_, _, k), &value) in weights.indexed_iter() {
        if value > 0.0 {
            match active_plane {
                Some(plane) => assert_eq!(plane, k, "disc weights must remain planar"),
                None => active_plane = Some(k),
            }
        }
    }
    assert!(
        active_plane.is_some(),
        "disc weights must activate at least one cell"
    );
}

#[test]
fn test_set_array_position_matches_manual_position_rotation() {
    use super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use crate::domain::grid::Grid;

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
    use crate::domain::grid::Grid;

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
    use crate::domain::grid::Grid;

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

/// Annulus has strictly fewer active cells than the full bowl of the same
/// outer diameter, and the surface-area formula satisfies the closed form:
/// `annulus(0, D) == bowl(D)`.
#[test]
fn test_annulus_is_subset_of_bowl_same_outer_diameter() {
    use crate::domain::grid::Grid;

    let dx = 2.0e-4;
    let grid = Grid::new(81, 81, 81, dx, dx, dx).expect("grid");
    let radius = 8.0e-3;
    let cx = 10.0e-3;
    let cy = 40.0 * dx;
    let cz = 40.0 * dx;
    let outer_d = 6.0e-3;
    let inner_d = 3.0e-3;

    let mut bowl = KWaveArray::new();
    bowl.add_bowl_element((cx, cy, cz), radius, outer_d);
    let bowl_mask = bowl.get_array_binary_mask(&grid);

    let mut annulus = KWaveArray::new();
    annulus.add_annular_element((cx, cy, cz), radius, inner_d, outer_d);
    let annulus_mask = annulus.get_array_binary_mask(&grid);

    let bowl_count: usize = bowl_mask.iter().filter(|&&b| b).count();
    let annulus_count: usize = annulus_mask.iter().filter(|&&b| b).count();
    assert!(bowl_count > 0 && annulus_count > 0);
    assert!(
        annulus_count < bowl_count,
        "annulus (inner_d>0) must have fewer cells than full bowl: {annulus_count} vs {bowl_count}",
    );

    let a_bowl = KWaveArray::bowl_surface_area(radius, outer_d);
    let a_ann_full = KWaveArray::annulus_surface_area(radius, 0.0, outer_d);
    assert!(
        (a_bowl - a_ann_full).abs() / a_bowl < 1.0e-12,
        "annulus(0, D) must equal bowl(D): {a_bowl} vs {a_ann_full}",
    );

    let a_ann = KWaveArray::annulus_surface_area(radius, inner_d, outer_d);
    assert!(
        a_ann > 0.0 && a_ann < a_bowl,
        "annulus area must be positive and less than full bowl: {a_ann} vs {a_bowl}",
    );
}

#[test]
fn test_build_per_element_source_superposition() {
    // Theorem: for two elements with per-element signals s1, s2 and a
    // per-cell signal built as Σᵢ Wᵢ[c] · sᵢ[t], setting s1 = s2 = s
    // must reproduce the shared-signal result (W_sum[c] · s[t]).
    use crate::domain::grid::Grid;
    use ndarray::Array2;

    let dx = 5.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let cx = 30.0 * dx;
    let cy = 30.0 * dx;
    let cz = 30.0 * dx;

    let mut arr = KWaveArray::new();
    arr.add_annular_element((cx, cy, cz), 15.0e-3, 0.0, 4.0e-3);
    arr.add_annular_element((cx, cy, cz), 15.0e-3, 6.0e-3, 10.0e-3);
    assert_eq!(arr.num_elements(), 2);

    let n_times = 4;
    let s: Vec<f64> = (0..n_times).map(|t| (t as f64).sin()).collect();
    let mut shared = Array2::<f64>::zeros((2, n_times));
    for t in 0..n_times {
        shared[[0, t]] = s[t];
        shared[[1, t]] = s[t];
    }

    let (mask_unit, per_cell) = arr
        .build_per_element_source(&grid, &shared)
        .expect("build per-element source");
    let w_sum = arr.get_array_weighted_mask(&grid);

    let (nx, ny, nz) = mask_unit.dim();
    let mut active_cells = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let m = mask_unit[[i, j, k]];
                if m != 0.0 {
                    assert!((m - 1.0).abs() < 1.0e-12);
                    assert!(w_sum[[i, j, k]] != 0.0);
                    active_cells.push((i, j, k));
                }
            }
        }
    }
    assert!(!active_cells.is_empty());
    assert_eq!(active_cells.len(), per_cell.shape()[0]);

    for (idx, &(i, j, k)) in active_cells.iter().enumerate() {
        for t in 0..n_times {
            let expected = w_sum[[i, j, k]] * s[t];
            let got = per_cell[[idx, t]];
            assert!(
                (got - expected).abs() < 1.0e-10 * expected.abs().max(1.0),
                "cell ({i},{j},{k}) t={t}: got {got}, expected {expected}",
            );
        }
    }
}

#[test]
fn test_kwave_array_setters_preserve_elements() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.set_frequency(2.0e6);
    array.set_sound_speed(1600.0);
    assert_eq!(array.num_elements(), 1);
    assert!((array.frequency() - 2.0e6).abs() < 1.0e-12);
    let delays = array.get_focus_delays((0.0, 0.0, 1.0));
    assert_eq!(delays.len(), 1);
    assert!((delays[0] - 1.0 / 1600.0).abs() < 1.0e-12);
}

#[test]
fn test_focus_delays() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);
    let delays = array.get_focus_delays((0.005, 0.0, 0.02));
    assert_eq!(delays.len(), 2);
    assert!(delays[0] > 0.0);
    assert!(delays[1] > 0.0);
}

/// `get_element_delays` returns zero for both elements of a symmetric two-element array.
#[test]
fn test_get_element_delays_symmetric_array() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((-0.005, 0.0, 0.0), 0.002, None);
    array.add_disc_element((0.005, 0.0, 0.0), 0.002, None);
    let delays = array.get_element_delays((0.0, 0.0, 0.02));
    assert_eq!(delays.len(), 2);
    assert!(
        delays[0].abs() < 1e-12 && delays[1].abs() < 1e-12,
        "symmetric elements should have equal (zero) delays: {delays:?}"
    );
}

/// All delays are non-negative and the minimum delay is exactly 0.
#[test]
fn test_get_element_delays_non_negative_min_zero() {
    let mut array = KWaveArray::with_params(1e6, 1500.0);
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.01, 0.0, 0.0), 0.005, None);
    array.add_disc_element((0.02, 0.0, 0.0), 0.005, None);
    let delays = array.get_element_delays((0.01, 0.0, 0.03));
    assert_eq!(delays.len(), 3);
    for &d in &delays {
        assert!(d >= 0.0, "all delays must be non-negative");
    }
    let min_delay = delays.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(min_delay.abs() < 1e-15, "minimum delay must be 0");
}

/// Rectangular apodization returns all-ones.
#[test]
fn test_apodization_rectangular_all_ones() {
    let mut array = KWaveArray::new();
    for i in 0..8 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Rectangular);
    assert_eq!(weights.len(), 8);
    for w in &weights {
        assert!((w - 1.0).abs() < 1e-15, "rectangular weight must be 1.0");
    }
}

/// Hann window: endpoints ≈ 0, center = 1.
#[test]
fn test_apodization_hann_endpoints_near_zero() {
    let mut array = KWaveArray::new();
    for i in 0..9 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Hann);
    assert_eq!(weights.len(), 9);
    assert!(
        weights[0].abs() < 1e-12,
        "Hann first weight must be ~0, got {}",
        weights[0]
    );
    assert!(
        weights[8].abs() < 1e-12,
        "Hann last weight must be ~0, got {}",
        weights[8]
    );
    assert!(
        (weights[4] - 1.0).abs() < 1e-12,
        "Hann center weight must be 1.0, got {}",
        weights[4]
    );
}

/// Hamming window: all weights in [0.08, 1.0] and symmetric.
#[test]
fn test_apodization_hamming_range_and_symmetry() {
    let mut array = KWaveArray::new();
    for i in 0..7 {
        array.add_disc_element((i as f64 * 0.001, 0.0, 0.0), 0.001, None);
    }
    let weights = array.get_apodization(ApodizationWindow::Hamming);
    assert_eq!(weights.len(), 7);
    for &w in &weights {
        assert!(
            (0.07..=1.01).contains(&w),
            "Hamming weight out of range: {w}"
        );
    }
    for i in 0..7 {
        assert!(
            (weights[i] - weights[6 - i]).abs() < 1e-12,
            "Hamming not symmetric at i={i}: w[{i}]={} w[{}]={}",
            weights[i],
            6 - i,
            weights[6 - i]
        );
    }
}

/// Single-element array: all windows return `[1.0]`.
#[test]
fn test_apodization_single_element() {
    let mut array = KWaveArray::new();
    array.add_disc_element((0.0, 0.0, 0.0), 0.005, None);
    for window in [
        ApodizationWindow::Rectangular,
        ApodizationWindow::Hann,
        ApodizationWindow::Hamming,
    ] {
        let weights = array.get_apodization(window);
        assert_eq!(weights.len(), 1);
        assert!(
            (weights[0] - 1.0).abs() < 1e-12,
            "{window:?}: single element weight must be 1.0"
        );
    }
}

/// SSOT: `KWaveArray::new()` uses `SOUND_SPEED_TISSUE`.
#[test]
fn test_default_sound_speed_is_ssot_constant() {
    use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
    let mut arr = KWaveArray::new();
    arr.add_disc_element((0.0, 0.0, 0.0), 0.001, None);
    let delays = arr.get_focus_delays((0.0, 0.0, 1.0));
    assert!(
        (delays[0] - 1.0 / SOUND_SPEED_TISSUE).abs() < 1e-10,
        "default sound speed must equal SOUND_SPEED_TISSUE"
    );
}
