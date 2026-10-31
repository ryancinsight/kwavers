//! Spatial position, rotation, and weighted-mask tests for [`KWaveArray`].

use super::super::KWaveArray;
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape, SurfaceConvention};

#[test]
fn test_set_array_position_matches_manual_position_rotation() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use kwavers_grid::Grid;

    let grid = Grid::new(41, 41, 11, 5.0e-4, 5.0e-4, 5.0e-4).expect("grid");
    let translation = (5.0e-3, 0.0, 2.0e-3);
    let global_euler = (0.0, 20.0, 0.0);
    let per_element_euler = (0.0, 5.0, 0.0);
    let dims = (1.0e-3, 1.0e-3);

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
        manual
            .add_rect_rot_element(world, dims.0, dims.1, per_element_euler)
            .expect("valid element");
        native
            .add_rect_rot_element(local, dims.0, dims.1, per_element_euler)
            .expect("valid element");
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
    unrot
        .add_rect_element((20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4), 8.0e-4, 2.0e-4)
        .expect("valid element");
    let unrot_mask = unrot.get_array_binary_mask(&grid);

    let mut rot = KWaveArray::new();
    rot.add_rect_rot_element(
        (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
        8.0e-4,
        2.0e-4,
        (0.0, 0.0, 90.0),
    )
    .expect("valid element");
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
    array
        .add_rect_rot_element(
            (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
            8.0e-4,
            2.0e-4,
            (0.0, 0.0, 90.0),
        )
        .expect("valid element");
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
/// ## Why this is the oracle (ADR 112, ADR 136)
/// The wiring maps each element onto a `Rectangle` planar aperture whose
/// stored `(first_axis, normal)` frame derives from the Euler `(0, θᵢ°, 0)`
/// rotation. A wrong sign, a swapped Euler slot, or a radian/degree slip
/// would still produce a full, plausible-looking element mask — just facing the
/// wrong way. This compares the stored frame directly against the layout's
/// own `element_normal`, so the derivation is asserted rather than trusted.
///
/// θ is non-trivial and asymmetric (7 elements, 9° pitch) so a sign error
/// cannot cancel, and the centre element (θ = 0) is included so a mapping that
/// only works at the apex is still caught by its neighbours.
#[test]
fn convex_array_elements_face_along_their_layout_normals() {
    use super::super::KWaveElement;
    use crate::curvilinear::ConvexArrayGeometry;
    use aequitas::systems::si::quantities::Length;
    use aequitas::systems::si::units::Meter;

    let radius = 40.0e-3;
    let elements = 7;
    let pitch = 9.0_f64.to_radians();
    let geometry =
        ConvexArrayGeometry::from_angular_pitch(radius, elements, pitch).expect("geometry");

    let mut array = KWaveArray::new();
    array
        .add_convex_array(
            &geometry,
            Length::from_unit::<Meter>(4.0e-3),
            Length::from_unit::<Meter>(3.0e-3),
        )
        .expect("valid element");
    assert_eq!(
        array.num_elements(),
        elements,
        "one element stored per array element"
    );

    for i in 0..elements {
        let KWaveElement::PlanarAperture(aperture) = &array.elements[i] else {
            panic!(
                "element {i} is not a planar aperture; the convex wiring must use that primitive"
            );
        };

        let expected_centre = geometry.element_position(i);
        for (axis, (got, want)) in [
            aperture.center_m()[0],
            aperture.center_m()[1],
            aperture.center_m()[2],
        ]
        .iter()
        .zip(expected_centre.iter())
        .enumerate()
        {
            assert!(
                (got - want).abs() <= 1e-15,
                "element {i} centre axis {axis}: {got:.6e} against layout {want:.6e}"
            );
        }

        let [nx, ny, nz] = aperture.normal();
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

/// Largest component error of a rotated, re-validated unit frame vector
/// (salvaged from the superseded one-rectangle-element attempt, PR #967).
///
/// The Euler matrix `R` is orthogonal to within 8 ε (each entry multiplies
/// at most three rounded trigonometric factors). Re-validating `R·n` moves
/// each component by the norm defect, at most 8 ε. The first axis adds its
/// projection off the normal and a second renormalisation, so 24 ε bounds
/// every component; 32 ε leaves rounding headroom.
const FRAME_COMPONENT_BOUND: f64 = 32.0 * f64::EPSILON;

/// Under `AreaConserving`, `set_array_position(t, euler)` is the rigid
/// transform of the element frame: the centre goes to `R·c + t`, and the
/// normal and first axis go to `R·n` and `R·f` (salvaged from PR #967).
///
/// The Euler angles tilt about every axis and the rectangle has `w ≠ h`, so
/// a missing normal or first-axis rotation changes the mask. The first axis
/// is the stronger check: rotating the normal alone and re-projecting the
/// old axis into the new plane leaves it rolled about the normal.
///
/// # Panics
/// - Panics if `grid` or the rectangle is invalid.
#[test]
fn area_conserving_rectangle_follows_the_array_transform_rigidly() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};
    use kwavers_grid::Grid;

    let dx = 1.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let shape = PlanarApertureShape::Rectangle {
        width_m: 1.2e-3,
        height_m: 4.0e-4,
    };
    let local_centre = [5.0e-4, -3.0e-4, 2.0e-4];
    let translation = (3.3e-3, 3.1e-3, 2.9e-3);
    let global_euler = (10.0, 20.0, 30.0);
    let geometry =
        PlanarApertureGeometry::oriented(local_centre, [0.2, -0.3, 0.9], [1.0, 0.1, 0.0], shape)
            .expect("valid rectangle");

    let mut native = KWaveArray::new();
    native.add_planar_aperture_element(geometry);
    native.set_array_position(translation, global_euler);

    let rotation = euler_xyz_rotation_matrix(global_euler);
    let rotate = |axis: [f64; 3]| {
        let (x, y, z) = apply_matrix(&rotation, (axis[0], axis[1], axis[2]));
        [x, y, z]
    };
    let [cx, cy, cz] = rotate(local_centre);
    let world_centre = [cx + translation.0, cy + translation.1, cz + translation.2];

    // Frame: the library's transformed geometry against the public validating
    // constructor fed the manually transformed centre, normal and first axis.
    let transformed = native.apply_transform_planar(geometry, SurfaceConvention::AreaConserving);
    let manual = PlanarApertureGeometry::oriented(
        world_centre,
        rotate(geometry.normal()),
        rotate(geometry.first_axis()),
        shape,
    )
    .expect("valid transformed rectangle");
    assert_eq!(transformed.center_m(), manual.center_m());
    assert_eq!(transformed.shape(), shape);
    for (name, got, want) in [
        ("normal", transformed.normal(), manual.normal()),
        ("first axis", transformed.first_axis(), manual.first_axis()),
    ] {
        for axis in 0..3 {
            assert!(
                (got[axis] - want[axis]).abs() <= FRAME_COMPONENT_BOUND,
                "{name} component {axis}: {} against {}",
                got[axis],
                want[axis]
            );
        }
    }

    // Mask: the transformed element must land inside the grid.
    let weights = native.get_array_weighted_mask(&grid);
    assert!(
        weights.iter().any(|&w| w != 0.0),
        "the transformed rectangle must land inside the grid"
    );
}

/// The per-element source paths apply the array transform like the masks do
/// (salvaged from PR #967).
///
/// `for_each_element_weighted_cell` and `build_per_element_source` rasterize
/// through their own dispatch. For one `KWave` and one `AreaConserving`
/// rectangle under a rotating, translating `set_array_position`, each path
/// reproduces `get_array_weighted_mask` of the same single-element array.
/// The transformed mask must also differ from the untransformed one, so a
/// path that skipped the transform cannot pass.
///
/// # Panics
/// - Panics if `grid` or a rectangle is invalid.
#[test]
fn per_element_source_paths_honour_the_array_transform() {
    use kwavers_grid::Grid;
    use leto::{Array2, Array3};

    let dx = 1.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let shape = PlanarApertureShape::Rectangle {
        width_m: 1.2e-3,
        height_m: 4.0e-4,
    };
    let local_centre = (5.0e-4, -3.0e-4, 2.0e-4);
    let geometry = PlanarApertureGeometry::oriented(
        [local_centre.0, local_centre.1, local_centre.2],
        [0.2, -0.3, 0.9],
        [1.0, 0.1, 0.0],
        shape,
    )
    .expect("valid rectangle");

    for convention in [SurfaceConvention::KWave, SurfaceConvention::AreaConserving] {
        let mut array = KWaveArray::new();
        match convention {
            SurfaceConvention::KWave => {
                array.add_rect_rot_element(local_centre, 1.2e-3, 4.0e-4, (0.0, 5.0, 40.0));
            }
            SurfaceConvention::AreaConserving => {
                array.add_planar_aperture_element(geometry);
            }
        }
        let untransformed: Vec<f64> = array
            .get_array_weighted_mask(&grid)
            .iter()
            .copied()
            .collect();
        array.set_array_position((3.3e-3, 3.1e-3, 2.9e-3), (10.0, 20.0, 30.0));
        let transformed: Vec<f64> = array
            .get_array_weighted_mask(&grid)
            .iter()
            .copied()
            .collect();
        assert_ne!(
            untransformed, transformed,
            "{convention:?}: the transform moves it"
        );
        assert!(
            transformed.iter().any(|&w| w != 0.0),
            "{convention:?}: the transformed rectangle must land inside the grid"
        );

        let mut cells: Array3<f64> = grid.array_zeros();
        array.for_each_element_weighted_cell(&grid, |element, i, j, k, weight| {
            assert_eq!(element, 0);
            cells[[i, j, k]] += weight;
        });
        assert_eq!(
            cells.iter().copied().collect::<Vec<f64>>(),
            transformed,
            "{convention:?}: for_each_element_weighted_cell"
        );

        let mask: Array3<f64> = array.get_array_weighted_mask(&grid);
        let (_, per_cell) = array
            .build_per_element_source(&grid, &Array2::<f64>::ones((1, 1)))
            .expect("per-element source");
        let active = KWaveArray::active_cells_fortran_order(&mask);
        assert_eq!(
            per_cell.shape()[0],
            active.len(),
            "{convention:?}: active cells"
        );
        for (row, &(i, j, k)) in active.iter().enumerate() {
            assert_eq!(
                per_cell[[row, 0]],
                mask[[i, j, k]],
                "{convention:?}: build_per_element_source cell ({i},{j},{k})"
            );
        }
    }
}
