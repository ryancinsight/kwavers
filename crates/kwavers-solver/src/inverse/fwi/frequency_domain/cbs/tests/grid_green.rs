use super::*;

#[test]
fn bli_weights_collapse_to_single_on_grid_voxel() {
    let grid = GridSpec::new((3, 3, 3), 0.01).unwrap();
    let point = grid.center_at(1, 1, 1);
    let weights = bli_weights(grid, point, BliConfig::default()).unwrap();

    assert_eq!((weights.shape()[0] * weights.shape()[1] * weights.shape()[2]), 1);
    assert_eq!(weights[0].linear_index, grid.linear_index(1, 1, 1));
    assert_eq!(weights[0].weight, 1.0);
}

#[test]
fn dense_green_matches_shifted_outgoing_green_for_unit_point_source() {
    let grid = GridSpec::new((2, 1, 1), 0.01).unwrap();
    let source_index = grid.linear_index(0, 0, 0);
    let receiver_index = grid.linear_index(1, 0, 0);
    let source_density = [
        Complex64::new(1.0 / grid.cell_volume_m3(), 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let potential = [0.0, 0.0];
    let solution = solve_volume_field(
        grid,
        3.0,
        &potential,
        &source_density,
        CbsConfig {
            max_iterations: 4,
            relative_tolerance: 1.0e-12,
        },
    )
    .unwrap();
    let shifted = super::super::green::shifted_wavenumber(3.0, solution.epsilon);
    let expected = shifted_outgoing_green(
        grid.center_at(0, 0, 0),
        grid.center_at(1, 0, 0),
        shifted,
        grid.min_distance_m(),
    );

    assert_eq!(source_index, 0);
    assert_eq!(receiver_index, 1);
    assert!((solution.field[receiver_index] - expected).norm() <= 1.0e-10);
}

#[test]
fn shifted_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((2, 2, 1), 0.01).unwrap();
    let x = [
        Complex64::new(0.5, -0.25),
        Complex64::new(-1.0, 0.75),
        Complex64::new(0.125, 0.5),
        Complex64::new(0.25, -0.875),
    ];
    let y = [
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, 0.125),
        Complex64::new(1.0, -0.5),
        Complex64::new(-0.75, -0.25),
    ];
    let gx = apply_shifted_green(grid, 3.0, 0.2, &x);
    let ghy = apply_shifted_green_adjoint(grid, 3.0, 0.2, &y);
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}
