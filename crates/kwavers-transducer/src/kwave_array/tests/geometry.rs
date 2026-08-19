//! Spatial position, rotation, and weighted-mask tests for [`KWaveArray`].

use super::super::KWaveArray;

#[test]
fn test_set_array_position_matches_manual_position_rotation() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use kwavers_grid::Grid;

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
    let inter = m_manual
        .iter()
        .zip(m_native.iter())
        .filter(|(&a, &b)| a && b)
        .count();

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
    use kwavers_grid::Grid;

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
    use kwavers_grid::Grid;

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
        (weights.iter().sum::<f64>() - expected).abs() < 5.0e-6,
        "rect weighted mass got {}, expected {expected}",
        weights.iter().sum::<f64>()
    );
}

/// ## Theorem
/// For every element of a convex array, the orientation `add_convex_array`
/// stores rotates the rectangle's local `+z` onto that element's outward
/// normal, and the stored centre is the element's layout position.
///
/// ## Why this is the oracle (ADR 112)
/// The wiring maps each element onto a rotated rectangle at Euler
/// `(0, θᵢ°, 0)`, derived from the rect's `+z` local normal and the `Rz·Ry·Rx`
/// composition. A wrong sign, a swapped Euler slot, or a radian/degree slip
/// would still produce a full, plausible-looking element mask — just facing the
/// wrong way. This drives the stored orientation through the same
/// `euler_xyz_rotation_matrix` the rasterizer uses and compares the result
/// against the layout's own `element_normal`, so the derivation is asserted
/// rather than trusted.
///
/// θ is non-trivial and asymmetric (7 elements, 9° pitch) so a sign error
/// cannot cancel, and the centre element (θ = 0) is included so a mapping that
/// only works at the apex is still caught by its neighbours.
#[test]
fn convex_array_elements_face_along_their_layout_normals() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use super::super::{ElementShape, KWaveElement};
    use crate::curvilinear::ConvexArrayGeometry;
    use aequitas::systems::si::quantities::Length;
    use aequitas::systems::si::units::Meter;

    let radius = 40.0e-3;
    let elements = 7;
    let pitch = 9.0_f64.to_radians();
    let geometry =
        ConvexArrayGeometry::from_angular_pitch(radius, elements, pitch).expect("geometry");

    let mut array = KWaveArray::new();
    array.add_convex_array(
        &geometry,
        Length::from_unit::<Meter>(4.0e-3),
        Length::from_unit::<Meter>(3.0e-3),
    );
    assert_eq!(
        array.num_elements(),
        elements,
        "one element stored per array element"
    );

    for i in 0..elements {
        let KWaveElement::Shape(ElementShape::Rect {
            position,
            euler_xyz_deg,
            ..
        }) = &array.elements[i]
        else {
            panic!("element {i} is not a rotated rect; the convex wiring must use that primitive");
        };

        let expected_centre = geometry.element_position(i);
        for (axis, (got, want)) in [position.0, position.1, position.2]
            .iter()
            .zip(expected_centre.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() <= 1e-15,
                "element {i} centre axis {axis}: {got:.6e} against layout {want:.6e}"
            );
        }

        let rotation = euler_xyz_rotation_matrix(*euler_xyz_deg);
        let (nx, ny, nz) = apply_matrix(&rotation, (0.0, 0.0, 1.0));
        let want = geometry.element_normal(i);
        let worst = (nx - want[0])
            .abs()
            .max((ny - want[1]).abs())
            .max((nz - want[2]).abs());
        assert!(
            worst <= 1e-12,
            "element {i} (θ = {:.3}°) faces [{nx:.6}, {ny:.6}, {nz:.6}] but its layout normal is \
             [{:.6}, {:.6}, {:.6}]; a sign, axis, or degree/radian error shows here",
            geometry.element_angle(i).to_degrees(),
            want[0],
            want[1],
            want[2]
        );
    }
}
