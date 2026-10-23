//! Spatial position, rotation, and weighted-mask tests for [`KWaveArray`].

use super::super::{KWaveArray, KWaveElement, SurfaceConvention};
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape};
use eunomia::FloatElement;
use kwavers_grid::Grid;

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
            .expect("valid rectangle element");
        native
            .add_rect_rot_element(local, dims.0, dims.1, per_element_euler)
            .expect("valid rectangle element");
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
        .expect("valid rectangle element");
    let unrot_mask = unrot.get_array_binary_mask(&grid);

    let mut rot = KWaveArray::new();
    rot.add_rect_rot_element(
        (20.0 * 1.0e-4, 20.0 * 1.0e-4, 2.0 * 1.0e-4),
        8.0e-4,
        2.0e-4,
        (0.0, 0.0, 90.0),
    )
    .expect("valid rectangle element");
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
        .expect("valid rectangle element");
    let weights = array.get_array_weighted_mask(&grid);
    let expected = 16.036_130_608_724_637_f64;
    assert!(
        (weights.iter().sum::<f64>() - expected).abs() < 5.0e-6,
        "rect weighted mass got {}, expected {expected}",
        weights.iter().sum::<f64>()
    );
}

/// Weighted mass of the 8×2-cell reference rectangle under k-wave-python's
/// unnormalised deposition (commit `aeff4fb1a`); the exact area is 16 cells.
const KWAVE_RECTANGLE_REFERENCE_MASS: f64 = 16.036_130_608_724_637;
/// Absolute tolerance on the reference mass, inherited from the reference
/// test: the mass was recorded to 15 significant digits, and a k-wave-python
/// summation order differs from this one by far less than 5e-6.
const KWAVE_REFERENCE_TOLERANCE: f64 = 5.0e-6;
/// Largest component error of a rotated, re-normalised unit frame vector.
///
/// The Euler matrix `R` is orthogonal to within 8 ε (each entry multiplies at
/// most three rounded trigonometric factors). Renormalising `R·n` moves each
/// component by the norm defect, at most 8 ε. The first axis adds its
/// projection off the normal (`n·f` ≤ 8 ε) and a second renormalisation
/// (8 ε), so 24 ε bounds every component; 32 ε leaves rounding headroom.
const FRAME_COMPONENT_BOUND: f64 = 32.0 * f64::EPSILON;

fn planar_element(array: &KWaveArray, index: usize) -> (PlanarApertureGeometry, SurfaceConvention) {
    let KWaveElement::PlanarAperture {
        geometry,
        convention,
    } = array.elements[index]
    else {
        panic!("element {index} is not a planar aperture");
    };
    (geometry, convention)
}

fn weighted_values(array: &KWaveArray, grid: &Grid) -> Vec<f64> {
    array
        .get_array_weighted_mask(grid)
        .iter()
        .copied()
        .collect()
}

/// A rectangle one sample wide puts that sample on the centre line.
///
/// With `w = 10 µm`, `h = 2.05 mm` and `dx = 100 µm` the target count is
/// `ceil(10·w·h/dx²) = ceil(20.5) = 21`, so `ceil(sqrt(21·w/h)) = ceil(0.32) = 1` sample
/// across the width. The midpoint lattice puts it at `w·(0.5/1 − 0.5) = 0`,
/// so its coordinate along the width axis equals the centre exactly. The
/// pre-ADR-136 rectangle generator put it at the edge, `−w/2`.
#[test]
fn single_sample_across_a_rectangle_sits_at_its_centre() {
    let grid = Grid::new(41, 41, 5, 1.0e-4, 1.0e-4, 1.0e-4).expect("grid");
    let centre = [20.0e-4, 20.0e-4, 2.0e-4];
    let (width, height) = (1.0e-5, 2.05e-3);
    let mut array = KWaveArray::new();
    array
        .add_rect_element((centre[0], centre[1], centre[2]), width, height)
        .expect("valid rectangle element");
    let (geometry, _) = planar_element(&array, 0);

    let mut samples = Vec::new();
    KWaveArray::rasterize_planar_aperture_points(&grid, geometry, |point, _| {
        samples.push(point);
    });
    assert_eq!(samples.len(), 21, "one sample across, 21 along the height");
    for point in &samples {
        assert_eq!(point[0], centre[0], "width coordinate is the centre");
        assert_eq!(point[2], centre[2], "the rectangle is planar");
    }
    // Height midpoints are symmetric about the centre, so their mean is the
    // centre up to one rounding per term: 21 terms of magnitude <= h/2.
    let sample_count = f64::from_count(samples.len());
    let mean_y = samples.iter().map(|point| point[1]).sum::<f64>() / sample_count;
    let mean_bound = 2.0 * sample_count * f64::EPSILON * height;
    assert!(
        (mean_y - centre[1]).abs() <= mean_bound,
        "mean height coordinate {mean_y}, centre {}",
        centre[1]
    );
}

/// Under `AreaConserving`, `set_array_position(t, euler)` is the rigid
/// transform of the element frame: the centre goes to `R·c + t`, and the
/// normal and first axis go to `R·n` and `R·f`.
///
/// The Euler angles tilt about every axis and the rectangle has `w ≠ h`, so
/// a missing normal or first-axis rotation changes the mask. The first axis is
/// the stronger check: rotating the normal alone and re-projecting the old
/// axis into the new plane leaves it rolled about the normal.
#[test]
fn area_conserving_rectangle_follows_the_array_transform_rigidly() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};

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

    // Masks: the dispatch applies that frame to the binary and weighted masks.
    // The reference array rebuilds the same frame without renormalising, so
    // the sample points and deposits are bitwise identical.
    let mut reference = KWaveArray::new();
    reference.add_planar_aperture_element(geometry.with_frame(
        world_centre,
        rotate(geometry.normal()),
        rotate(geometry.first_axis()),
    ));
    let weights = weighted_values(&native, &grid);
    assert_eq!(weights, weighted_values(&reference, &grid));
    assert_eq!(
        native.get_array_binary_mask(&grid),
        reference.get_array_binary_mask(&grid)
    );
    assert!(
        weights.iter().any(|&w| w != 0.0),
        "the transformed rectangle must land inside the grid"
    );
}

/// Under `KWave`, `set_array_position` moves the centre only: the element
/// keeps its own Euler orientation, which is k-wave-python's rectangle
/// contract. The reference rectangle is built directly at the transformed
/// centre with the same per-element Euler, so the same sample points and
/// deposits arise bitwise.
#[test]
fn kwave_rectangle_follows_the_array_transform_by_centre_only() {
    use super::super::math::{apply_matrix, euler_xyz_rotation_matrix};

    let dx = 1.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let (width, height) = (1.2e-3, 4.0e-4);
    let local_centre = (5.0e-4, -3.0e-4, 2.0e-4);
    let translation = (3.3e-3, 3.1e-3, 2.9e-3);
    let global_euler = (10.0, 20.0, 30.0);
    let element_euler = (0.0, 5.0, 40.0);

    let mut native = KWaveArray::new();
    native
        .add_rect_rot_element(local_centre, width, height, element_euler)
        .expect("valid rectangle element");
    native.set_array_position(translation, global_euler);

    let rotation = euler_xyz_rotation_matrix(global_euler);
    let rotated = apply_matrix(&rotation, local_centre);
    let world_centre = (
        rotated.0 + translation.0,
        rotated.1 + translation.1,
        rotated.2 + translation.2,
    );
    let mut reference = KWaveArray::new();
    reference
        .add_rect_rot_element(world_centre, width, height, element_euler)
        .expect("valid rectangle element");

    let weights = weighted_values(&native, &grid);
    assert_eq!(weights, weighted_values(&reference, &grid));
    assert_eq!(
        native.get_array_binary_mask(&grid),
        reference.get_array_binary_mask(&grid)
    );
    assert!(
        weights.iter().any(|&w| w != 0.0),
        "the transformed rectangle must land inside the grid"
    );
}

/// One rectangle, two conventions (ADR 136): the same 8×2-cell rectangle
/// carries the k-wave-python mass under `KWave` and the exact area under
/// `AreaConserving`.
///
/// Each of the `N` samples deposits `scale / Σ K` times a stencil of at most
/// `(2·7 + 1)³` weights, where 7 is `ceil(1 / (π · 0.05))`. A sum of `m` terms
/// carries relative error `m · u` of the absolute sum, `u = ε / 2`, and the
/// per-axis absolute weight sum is at most
/// `3 + (2 / π) · Σₖ₌₂⁷ 1 / (k − 1/2) = 4.25` (three central terms of at most
/// 1, then `|sinc(πx)| ≤ 1 / (π |x|)`), so the absolute mass is at most
/// `4.25³` times the exact mass. Per-sample normalisation, accumulation into
/// cells and the final sum contribute `2·N·S + 3·N` terms in total.
#[test]
fn rectangle_mass_follows_its_surface_convention() {
    const STENCIL_CELLS: f64 = 15.0 * 15.0 * 15.0;
    const AXIS_ABSOLUTE_SUM: f64 = 4.25;

    let dx = 1.0e-4;
    let grid = Grid::new(41, 41, 5, dx, dx, dx).expect("grid");
    let centre = (20.0 * dx, 20.0 * dx, 2.0 * dx);
    let (width, height) = (8.0e-4, 2.0e-4);
    let exact_mass = width * height / (dx * dx);

    let mut kwave = KWaveArray::new();
    kwave
        .add_rect_rot_element(centre, width, height, (0.0, 0.0, 90.0))
        .expect("valid rectangle element");
    let (geometry, convention) = planar_element(&kwave, 0);
    assert_eq!(convention, SurfaceConvention::KWave);

    let mut samples = 0_usize;
    KWaveArray::rasterize_planar_aperture_points(&grid, geometry, |_, _| samples += 1);
    let sample_count = f64::from_count(samples);
    let terms = 2.0 * sample_count * STENCIL_CELLS + 3.0 * sample_count;
    let rounding_bound = terms * 0.5 * f64::EPSILON * AXIS_ABSOLUTE_SUM.powi(3) * exact_mass;

    let mut conserving = KWaveArray::new();
    conserving.add_planar_aperture_element(geometry);
    assert_eq!(
        planar_element(&conserving, 0).1,
        SurfaceConvention::AreaConserving
    );

    let area_mass: f64 = weighted_values(&conserving, &grid).iter().sum();
    let kwave_mass: f64 = weighted_values(&kwave, &grid).iter().sum();
    assert!(
        (area_mass - exact_mass).abs() <= rounding_bound,
        "area-conserving mass {area_mass}, exact {exact_mass}, bound {rounding_bound}"
    );
    assert!(
        (kwave_mass - KWAVE_RECTANGLE_REFERENCE_MASS).abs() < KWAVE_REFERENCE_TOLERANCE,
        "k-wave mass {kwave_mass}"
    );
    assert_eq!(
        kwave.get_array_binary_mask(&grid),
        conserving.get_array_binary_mask(&grid),
        "the binary mask is the same under both conventions"
    );
}

/// A rectangle with a non-positive or non-finite side is rejected with an
/// invalid-value error that names the offending side and its value, and the
/// array keeps its elements. The unit-valued side passes `validate_positive`
/// (`finite && > 0`), so only the side under test can fail.
#[test]
fn rectangle_rejects_invalid_sides_by_name() {
    use kwavers_core::error::{ConfigError, KwaversError};

    const VALID_SIDE_M: f64 = 1.0e-3;
    let bad_sides = [0.0, -1.0e-3, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for bad in bad_sides {
        for (parameter, width, height) in [
            ("width_m", bad, VALID_SIDE_M),
            ("height_m", VALID_SIDE_M, bad),
        ] {
            let mut array = KWaveArray::new();
            let outcome = array.add_rect_element((0.0, 0.0, 0.0), width, height);
            let Err(error) = outcome else {
                panic!("{parameter} = {bad} must be rejected");
            };
            assert!(
                matches!(
                    &error,
                    KwaversError::Config(ConfigError::InvalidValue { parameter: name, value, .. })
                        if name == parameter && *value == bad.to_string()
                ),
                "{parameter} = {bad}: got {error:?}"
            );
            assert_eq!(array.num_elements(), 0, "a rejected element is not added");
        }
    }
}

/// A non-finite centre or Euler angle is rejected, and a convex array whose
/// elements cannot be built leaves the array as it was.
#[test]
fn rectangle_and_convex_array_reject_non_finite_pose_atomically() {
    use crate::curvilinear::ConvexArrayGeometry;
    use aequitas::systems::si::quantities::Length;
    use aequitas::systems::si::units::Meter;
    use kwavers_core::error::{ConfigError, KwaversError};

    let mut array = KWaveArray::new();
    let centre = array.add_rect_element((f64::NAN, 0.0, 0.0), 1.0e-3, 1.0e-3);
    assert!(
        matches!(
            centre,
            Err(KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }))
                if parameter == "center_m"
        ),
        "non-finite centre"
    );
    let angle = array.add_rect_rot_element((0.0, 0.0, 0.0), 1.0e-3, 1.0e-3, (0.0, f64::NAN, 0.0));
    assert!(
        matches!(
            angle,
            Err(KwaversError::Config(ConfigError::InvalidValue { .. }))
        ),
        "non-finite Euler angle"
    );

    let geometry = ConvexArrayGeometry::from_angular_pitch(40.0e-3, 5, 9.0_f64.to_radians())
        .expect("geometry");
    array
        .add_rect_element((0.0, 0.0, 0.0), 1.0e-3, 1.0e-3)
        .expect("valid rectangle element");
    let convex = array.add_convex_array(
        &geometry,
        Length::from_unit::<Meter>(0.0),
        Length::from_unit::<Meter>(3.0e-3),
    );
    assert!(
        matches!(
            convex,
            Err(KwaversError::Config(ConfigError::InvalidValue { ref parameter, .. }))
                if parameter == "width_m"
        ),
        "zero element width"
    );
    assert_eq!(
        array.num_elements(),
        1,
        "only the earlier valid rectangle remains"
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
/// wrong way. This reads the stored aperture normal, which the constructor
/// builds as `R·ẑ` from `euler_xyz_rotation_matrix`, and compares it against
/// the layout's own `element_normal`, so the derivation is asserted rather
/// than trusted.
///
/// θ is non-trivial and asymmetric (7 elements, 9° pitch) so a sign error
/// cannot cancel, and the centre element (θ = 0) is included so a mapping that
/// only works at the apex is still caught by its neighbours.
#[test]
fn convex_array_elements_face_along_their_layout_normals() {
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
        .expect("valid rectangle element");
    assert_eq!(
        array.num_elements(),
        elements,
        "one element stored per array element"
    );

    for i in 0..elements {
        let (aperture, convention) = planar_element(&array, i);
        assert_eq!(
            convention,
            SurfaceConvention::KWave,
            "element {i}: the convex wiring must use the k-Wave rectangle"
        );

        let expected_centre = geometry.element_position(i);
        for (axis, (got, want)) in aperture
            .center_m()
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
