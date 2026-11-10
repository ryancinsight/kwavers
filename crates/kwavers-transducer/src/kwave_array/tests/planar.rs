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
    let quadrants: Vec<PlanarApertureGeometry> = (0..4_u32)
        .map(|quadrant| {
            PlanarApertureGeometry::oriented(
                center,
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                PlanarApertureShape::AnnularSector {
                    inner_radius_m: inner,
                    outer_radius_m: outer,
                    start_angle_rad: f64::from(quadrant) * span,
                    span_angle_rad: span,
                },
            )
            .expect("valid quadrant")
        })
        .collect();

    // Sample mass. Each sample carries mass / N (one rounding) and summing N
    // of them adds N − 1. The area ½(o² − i²)·span rounds o² and i²
    // (together at most κ·u·(o² − i²), κ = (o² + i²)/(o² − i²)), the
    // difference and the product; dividing by dx·dy adds two more. The
    // rasterizer and this test each evaluate that mass, so to first order
    // the total is within (N + 2κ + 8)·u·mass of it, u = ε/2.
    let u = 0.5 * f64::EPSILON;
    let expected_mass = 0.5 * (outer * outer - inner * inner) * span / (dx * dx);
    let kappa = (outer * outer + inner * inner) / (outer * outer - inner * inner);
    let (x_vec, y_vec, z_vec) = (
        grid.x_coordinates(),
        grid.y_coordinates(),
        grid.z_coordinates(),
    );
    let lattice_center = (center[0], center[1], center[2]);
    let mut cell_bounds = [0.0; 4];
    for (quadrant, &geometry) in quadrants.iter().enumerate() {
        let (mut samples, mut total) = (0_usize, 0.0);
        let (mut emitted, mut widest, mut spread_mass) = (0_usize, 0_usize, 0.0);
        let mut inside = true;
        let start = span * f64::from_count(quadrant);
        KWaveArray::rasterize_planar_aperture_points(
            &grid,
            geometry,
            lattice_center,
            |point, scale| {
                samples += 1;
                total += scale;
                let (x, y) = (point[0] - center[0], point[1] - center[1]);
                let angle = (y.atan2(x) - start).rem_euclid(std::f64::consts::TAU);
                inside &= angle < span && (inner..outer).contains(&x.hypot(y));
                let (mut stencil, mut sum, mut magnitude) = (0_usize, 0.0, 0.0);
                KWaveArray::map_surface_sample(
                    &grid,
                    &x_vec,
                    &y_vec,
                    &z_vec,
                    point,
                    1.0,
                    false,
                    |_, _, _, weight| {
                        stencil += 1;
                        sum += weight;
                        magnitude += weight.abs();
                    },
                );
                assert!(sum != 0.0, "a sample's stencil sums to zero");
                emitted += stencil;
                widest = widest.max(stencil);
                spread_mass += scale * magnitude / sum.abs();
            },
        );
        assert!(inside, "a sample left quadrant {quadrant}");
        let sample_bound = (f64::from_count(samples) + 2.0 * kappa + 8.0) * u * expected_mass;
        assert!(
            (total - expected_mass).abs() <= sample_bound,
            "quadrant {quadrant}: sample mass {total}, expected {expected_mass}"
        );
        // Cell mass. A sample's m stencil weights w are normalized by their
        // sum S: S rounds by (m − 1)·u·Σ|w|, scale / S once, and each emitted
        // weight carries three roundings beyond the two of w, so the sample
        // reaches the cells within (m + 5)·u·scale·Σ|w|/|S|. Summing all T
        // emitted weights into the column adds (T − 1)·u of their magnitude,
        // so the column is within (T + m_max + 5)·u·A of the sample mass,
        // A = Σ scale·Σ|w|/|S|. Dropping the normalization moves it by the
        // stencil's truncation, orders of magnitude more.
        cell_bounds[quadrant] = sample_bound
            + (f64::from_count(emitted) + f64::from_count(widest) + 5.0) * u * spread_mass;
    }

    // Each element's signal reaches only its own weights: with signal q
    // nonzero only at time q, the source's column q is quadrant q's own
    // weighted mask, bit for bit, since every other element contributes
    // an exact zero.
    let mut array = KWaveArray::new();
    for &geometry in &quadrants {
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
    let (mask, per_cell) = array
        .build_per_element_source(&grid, &signals)
        .expect("sector source");
    let active_cells = KWaveArray::active_cells_fortran_order(&mask);
    assert_eq!(active_cells.len(), per_cell.shape()[0]);
    for (quadrant, &geometry) in quadrants.iter().enumerate() {
        let column: f64 = (0..per_cell.shape()[0])
            .map(|row| per_cell[[row, quadrant]])
            .sum();
        assert!(
            (column - expected_mass).abs() <= cell_bounds[quadrant],
            "quadrant {quadrant}: cell mass {column}, expected {expected_mass}"
        );
        let mut single = KWaveArray::new();
        single.add_planar_aperture_element(geometry);
        let own = single.get_array_weighted_mask(&grid);
        for (row, &(i, j, k)) in active_cells.iter().enumerate() {
            assert_eq!(
                per_cell[[row, quadrant]],
                own[[i, j, k]],
                "quadrant {quadrant} cell {:?}",
                (i, j, k)
            );
        }
    }

    // The area sums four quadrant areas of (κ + 2)·u relative error each,
    // and the sum adds three roundings; the reference has (κ + 2)·u.
    let total_area = 4.0 * (0.5 * (outer * outer - inner * inner) * span);
    assert!(
        (array.compute_total_surface_area() - total_area).abs()
            <= (2.0 * kappa + 7.0) * 0.5 * f64::EPSILON * total_area
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
