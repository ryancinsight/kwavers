//! Annulus, bowl, and per-element source tests for [`KWaveArray`].

use super::super::{DiscSourceProfile, KWaveArray};
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape};
use eunomia::FloatElement;

/// Annulus has strictly fewer active cells than the full bowl of the same
/// outer diameter, and the surface-area formula satisfies the closed form:
/// `annulus(0, D) == bowl(D)`.
/// # Panics
/// - Panics if `grid`.
///
#[test]
fn test_annulus_is_subset_of_bowl_same_outer_diameter() {
    use kwavers_grid::Grid;

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
    annulus
        .add_annular_element((cx, cy, cz), radius, inner_d, outer_d)
        .expect("valid annulus");
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
    use kwavers_grid::Grid;
    use leto::Array2;

    let dx = 5.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let cx = 30.0 * dx;
    let cy = 30.0 * dx;
    let cz = 30.0 * dx;

    let mut arr = KWaveArray::new();
    arr.add_annular_element((cx, cy, cz), 15.0e-3, 0.0, 4.0e-3)
        .expect("valid annulus");
    arr.add_annular_element((cx, cy, cz), 15.0e-3, 6.0e-3, 10.0e-3)
        .expect("valid annulus");
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

    let [nx, ny, nz] = mask_unit.shape();
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
fn test_disc_source_profile_radial_power_contract() {
    let profile = DiscSourceProfile::radial_power(2.0).expect("profile");
    assert_eq!(profile.weight_at_normalized_radius(0.0), 0.0);
    assert!((profile.weight_at_normalized_radius(0.5) - 0.5).abs() < f64::EPSILON);
    assert!((profile.weight_at_normalized_radius(1.0) - 2.0).abs() < f64::EPSILON);
    assert!(DiscSourceProfile::radial_power(-1.0).is_err());
    assert!(DiscSourceProfile::radial_power(f64::NAN).is_err());
}

#[test]
fn test_profiled_disc_enters_per_element_source_weights() {
    use kwavers_grid::Grid;
    use leto::Array2;

    let dx = 5.0e-4;
    let grid = Grid::new(61, 61, 61, dx, dx, dx).expect("grid");
    let center = (30.0 * dx, 30.0 * dx, 30.0 * dx);
    let diameter = 6.0e-3;

    let mut uniform = KWaveArray::new();
    uniform
        .add_disc_element(center, diameter, None)
        .expect("valid element");
    let uniform_weights = uniform.get_array_weighted_mask(&grid);

    let mut profiled = KWaveArray::new();
    profiled.add_profiled_disc_element(
        center,
        diameter,
        None,
        DiscSourceProfile::radial_power(2.0).expect("profile"),
    );
    let profiled_weights = profiled.get_array_weighted_mask(&grid);

    let weight_delta: f64 = uniform_weights
        .iter()
        .zip(profiled_weights.iter())
        .map(|(&uniform, &profiled)| (uniform - profiled).abs())
        .sum();
    assert!(
        weight_delta > 1.0,
        "profiled disc must change the finite-source weight map"
    );
    assert!(profiled_weights.iter().sum::<f64>() > 0.0);

    let signal = Array2::<f64>::ones((1, 1));
    let (mask, per_cell) = profiled
        .build_per_element_source(&grid, &signal)
        .expect("profiled source");
    let active_cells = KWaveArray::active_cells_fortran_order(&mask);
    assert_eq!(active_cells.len(), per_cell.shape()[0]);
    assert!(!active_cells.is_empty());
    for (row, &(i, j, k)) in active_cells.iter().enumerate() {
        assert!(
            (per_cell[[row, 0]] - profiled_weights[[i, j, k]]).abs()
                <= 1.0e-12 * profiled_weights[[i, j, k]].abs().max(1.0),
            "profiled source row {row} did not preserve BLI weight"
        );
    }
}

#[test]
fn profiled_disc_clips_at_the_domain_boundary_without_distant_tail_sources() {
    // A clipped aperture keeps finite BLI support; an aperture beyond that BLI
    // window must not be projected onto a boundary cell by the sinc tail.
    use kwavers_grid::Grid;

    let spacing = 0.25e-3;
    let grid = Grid::new(17, 17, 1, spacing, spacing, spacing).expect("grid");
    let mut array = KWaveArray::new();
    array.add_profiled_disc_element(
        (-0.10e-3, 1.0e-3, 0.0),
        0.80e-3,
        Some((-0.10e-3, 1.0e-3, 1.0)),
        DiscSourceProfile::uniform(),
    );
    array.add_profiled_disc_element(
        (100.0e-3, 1.0e-3, 0.0),
        0.20e-3,
        Some((100.0e-3, 1.0e-3, 1.0)),
        DiscSourceProfile::uniform(),
    );

    let mut has_support = [false; 2];
    array.for_each_element_weighted_cell(&grid, |element, _, _, _, weight| {
        has_support[element] |= weight != 0.0;
    });

    assert_eq!(has_support, [true, false]);
}

#[test]
fn test_many_profiled_discs_per_element_source_matches_weighted_mask() {
    // Regression: per-element source assembly must preserve the weighted-mask
    // superposition contract without requiring one dense 3-D mask per element.
    use kwavers_grid::Grid;
    use leto::Array2;

    let dx = 5.0e-4;
    let grid = Grid::new(41, 41, 9, dx, dx, dx).expect("grid");
    let mut arr = KWaveArray::new();
    let center_z = 4.0 * dx;
    for ix in 0..8 {
        for iy in 0..8 {
            let center = ((8 + ix * 3) as f64 * dx, (8 + iy * 3) as f64 * dx, center_z);
            let profile = DiscSourceProfile::radial_power(((ix + iy) % 3) as f64)
                .expect("finite radial profile");
            arr.add_profiled_disc_element(center, 7.5e-4, None, profile);
        }
    }

    assert_eq!(arr.num_elements(), 64);
    let signal = Array2::<f64>::ones((arr.num_elements(), 1));
    let (mask, per_cell) = arr
        .build_per_element_source(&grid, &signal)
        .expect("many-element source");
    let weighted = arr.get_array_weighted_mask(&grid);
    let active_cells = KWaveArray::active_cells_fortran_order(&mask);

    assert_eq!(active_cells.len(), per_cell.shape()[0]);
    assert!(!active_cells.is_empty());
    for (row, &(i, j, k)) in active_cells.iter().enumerate() {
        let expected = weighted[[i, j, k]];
        let got = per_cell[[row, 0]];
        assert!(
            (got - expected).abs() <= 1.0e-10 * expected.abs().max(1.0),
            "row {row} cell ({i},{j},{k}): got {got}, expected {expected}"
        );
    }
}

#[test]
fn planar_quadrants_conserve_area_and_keep_signals_independent() {
    use kwavers_grid::Grid;
    use leto::Array2;

    let dx = 2.5e-4;
    let grid = Grid::new(81, 81, 9, dx, dx, dx).expect("grid");
    let center = [40.0 * dx, 40.0 * dx, 4.0 * dx];
    let inner = 0.75e-3;
    let outer = 2.0e-3;
    let span = std::f64::consts::FRAC_PI_2;
    let mut array = KWaveArray::new();
    for quadrant in 0..4 {
        let geometry = PlanarApertureGeometry::oriented(
            center,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            PlanarApertureShape::AnnularSector {
                inner_radius_m: inner,
                outer_radius_m: outer,
                start_angle_rad: quadrant as f64 * span,
                span_angle_rad: span,
            },
        )
        .expect("valid quadrant");
        array.add_planar_aperture_element(geometry);
    }

    let signals = Array2::<f64>::from_shape_fn(
        (4, 4),
        |[row, column]| {
            if row == column {
                1.0
            } else {
                0.0
            }
        },
    );
    let (_, per_cell) = array
        .build_per_element_source(&grid, &signals)
        .expect("sector source");
    let expected_quadrant_mass = 0.5 * (outer * outer - inner * inner) * span / (dx * dx);
    let operation_bound = 64.0 * f64::EPSILON * f64::from_count(per_cell.shape()[0]);
    for quadrant in 0..4 {
        let realized: f64 = (0..per_cell.shape()[0])
            .map(|row| per_cell[[row, quadrant]])
            .sum();
        assert!(
            (realized - expected_quadrant_mass).abs()
                <= operation_bound * expected_quadrant_mass.max(1.0),
            "quadrant {quadrant}: realized {realized}, expected {expected_quadrant_mass}"
        );
    }
    assert!(
        (array.compute_total_surface_area() - 4.0 * expected_quadrant_mass * dx * dx).abs()
            <= 16.0 * f64::EPSILON * (outer * outer).max(1.0)
    );
}

#[test]
fn planar_rectangle_conserves_area_and_stays_in_its_footprint() {
    use kwavers_grid::Grid;
    use leto::Array2;

    let dx = 2.5e-4;
    let grid = Grid::new(41, 41, 9, dx, dx, dx).expect("grid");
    let center = [20.0 * dx, 20.0 * dx, 4.0 * dx];
    let (width, height) = (0.46e-3, 1.5e-3);
    let geometry = PlanarApertureGeometry::oriented(
        center,
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        PlanarApertureShape::Rectangle {
            width_m: width,
            height_m: height,
        },
    )
    .expect("valid rectangle");

    // Every sample carries mass / N for N samples, one rounding each; summing
    // the N values adds N − 1 more, and the rasterizer's mass w·h/(dx·dy) and
    // this test's w·h/dx² carry three roundings apiece. To first order the
    // total differs from w·h/dx² by at most (N + 6)·u·mass, with u = ε/2.
    let expected_mass = width * height / (dx * dx);
    let (mut samples, mut total) = (0_usize, 0.0);
    let mut inside = true;
    let lattice_center = (center[0], center[1], center[2]);
    KWaveArray::rasterize_planar_aperture_points(
        &grid,
        geometry,
        lattice_center,
        |point, scale| {
            samples += 1;
            total += scale;
            // Width along the first axis (y), height along normal × first (−x).
            inside &= (point[1] - center[1]).abs() < 0.5 * width
                && (point[0] - center[0]).abs() < 0.5 * height
                && point[2] == center[2];
        },
    );
    assert!(inside, "a sample left the rectangle");
    let rounding_bound = f64::from_count(samples + 6) * 0.5 * f64::EPSILON * expected_mass;
    assert!(
        (total - expected_mass).abs() <= rounding_bound,
        "sample mass {total}, expected {expected_mass}"
    );

    let mut array = KWaveArray::new();
    array.add_planar_aperture_element(geometry);
    let (mask, _) = array
        .build_per_element_source(&grid, &Array2::<f64>::ones((1, 1)))
        .expect("rectangle source");
    // The first axis, and so the 0.46 mm width, runs along y: the footprint in
    // the source plane must be longer in x, along the 1.5 mm height.
    let (mut x_extent, mut y_extent) = (0_usize, 0_usize);
    for i in 0..41 {
        for j in 0..41 {
            if mask[[i, j, 4]] != 0.0 {
                x_extent = x_extent.max(i.abs_diff(20));
                y_extent = y_extent.max(j.abs_diff(20));
            }
        }
    }
    assert!(
        x_extent > y_extent,
        "x extent {x_extent}, y extent {y_extent}"
    );
    assert_eq!(array.compute_total_surface_area(), width * height,);
}

/// A one-sample disc sits at its centre (ADR 136 disc acceptance): a disc
/// smaller than one radial step emits exactly the centre sample carrying
/// the full per-sample scale, so a degenerate disc never drifts off-centre.
///
/// # Panics
/// - Panics if `grid`.
#[test]
fn test_single_sample_disc_lattice_sits_at_centre() {
    use kwavers_grid::Grid;

    let dx = 1.0e-3;
    let grid = Grid::new(16, 16, 16, dx, dx, dx).expect("grid");
    let center = (8.0e-3, 8.0e-3, 8.0e-3);
    let radius = dx / 4.0;
    let mut points = Vec::new();
    let mut scales = Vec::new();
    KWaveArray::rasterize_framed_disc_points(
        &grid,
        center,
        radius,
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        DiscSourceProfile::uniform(),
        |point, scale| {
            points.push(point);
            scales.push(scale);
        },
    );
    assert_eq!(
        points.len(),
        1,
        "a sub-cell disc must emit exactly one sample"
    );
    for (axis, (got, want)) in points[0]
        .iter()
        .zip([center.0, center.1, center.2].iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() == 0.0,
            "single disc sample axis {axis}: {got:.6e} must equal centre {want:.6e} exactly"
        );
    }
    let expected_scale = std::f64::consts::PI * radius * radius / (dx * dx);
    assert!(
        (scales[0] - expected_scale).abs() == 0.0,
        "single disc sample must carry the full per-sample scale"
    );
}
