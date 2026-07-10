use super::*;

#[test]
fn polynomial_absorbing_boundary_has_unit_interior_and_edge_decay() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let boundary = AbsorbingBoundary::polynomial(1, 2.0, 2).unwrap();
    let weights = absorbing_weights(grid, boundary).unwrap();

    assert_eq!(weights[grid.linear_index(2, 2, 2)], 1.0);
    assert!((weights[grid.linear_index(0, 2, 2)] - (-2.0_f64).exp()).abs() <= f64::EPSILON);
    assert!((weights[grid.linear_index(0, 0, 0)] - (-6.0_f64).exp()).abs() <= f64::EPSILON);
}

#[test]
fn spectral_absorbing_boundary_damps_edge_source_response() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let mut source = vec![Complex64::new(0.0, 0.0); grid.len() ];
    source[grid.linear_index(0, 2, 2)] = Complex64::new(1.0, 0.0);
    let periodic = apply_shifted_green_spectral(grid, 11.0, 0.25, &source);
    let absorbed = apply_shifted_green_spectral_with_boundary(
        grid,
        11.0,
        0.25,
        &source,
        AbsorbingBoundary::polynomial(1, 2.0, 2).unwrap(),
    );

    assert!(
        norm(&absorbed) < norm(&periodic),
        "absorbed_norm={}, periodic_norm={}",
        norm(&absorbed),
        norm(&periodic)
    );
}

#[test]
fn spectral_absorbing_green_adjoint_satisfies_inner_product_identity() {
    let grid = GridSpec::new((5, 5, 5), 0.01).unwrap();
    let boundary = AbsorbingBoundary::polynomial(1, 1.5, 2).unwrap();
    let x = (0..(grid.len()))
        .map(|index| Complex64::new(index as f64 * 0.125, -0.03125 * index as f64))
        .collect::<Vec<_>>();
    let y = (0..(grid.len()))
        .map(|index| Complex64::new(0.25 - index as f64 * 0.0625, 0.125 * index as f64))
        .collect::<Vec<_>>();
    let gx = apply_shifted_green_spectral_with_boundary(grid, 11.0, 0.25, &x, boundary);
    let ghy = apply_shifted_green_spectral_adjoint_with_boundary(grid, 11.0, 0.25, &y, boundary);
    let lhs = inner_product(&gx, &y);
    let rhs = inner_product(&x, &ghy);

    assert!(
        (lhs - rhs).norm() <= 1.0e-12 * lhs.norm().max(rhs.norm()).max(1.0),
        "lhs={lhs}, rhs={rhs}"
    );
}
