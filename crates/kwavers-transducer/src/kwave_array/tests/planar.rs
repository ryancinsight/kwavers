//! Planar aperture source tests for [`KWaveArray`].

use super::super::KWaveArray;
use crate::transducers::physics::{PlanarApertureGeometry, PlanarApertureShape};
use eunomia::FloatElement;

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
