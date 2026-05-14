use super::*;
use crate::solver::multiphysics::monolithic::state_vector::field_block_view;
use ndarray::{Array3, ArrayBase, Data, Ix3};

fn laplacian_3d<S>(
    field: &ArrayBase<S, Ix3>,
    grid_dims: (usize, usize, usize),
    dx: f64,
    dy: f64,
    dz: f64,
) -> Array3<f64>
where
    S: Data<Elem = f64>,
{
    let mut lap = Array3::zeros(field.dim());
    laplacian_3d_into(field, grid_dims, dx, dy, dz, &mut lap);
    lap
}

#[test]
fn test_laplacian_uniform_field() {
    let field = Array3::from_elem((8, 8, 8), 5.0);
    let dx = 1e-3;
    let lap = laplacian_3d(&field, (8, 8, 8), dx, dx, dx);

    for i in 1..7 {
        for j in 1..7 {
            for k in 1..7 {
                assert!(
                    lap[[i, j, k]].abs() < 1e-15,
                    "Laplacian of constant should be 0 at interior [{i},{j},{k}], got {}",
                    lap[[i, j, k]]
                );
            }
        }
    }
}

/// Spacing of 1 m vs 1 mm on identical fields gives a 10^6 Laplacian ratio.
#[test]
fn test_laplacian_unit_vs_nonunit_spacing() {
    let mut field = Array3::from_elem((5, 5, 5), 1.0);
    field[[2, 2, 2]] = 2.0;

    let lap_1m = laplacian_3d(&field, (5, 5, 5), 1.0, 1.0, 1.0);
    let lap_1mm = laplacian_3d(&field, (5, 5, 5), 1e-3, 1e-3, 1e-3);

    let ratio = lap_1mm[[2, 2, 2]] / lap_1m[[2, 2, 2]];
    assert!(
        (ratio - 1e6).abs() < 1.0,
        "1 mm vs 1 m Laplacian ratio should be 1e6, got {ratio}"
    );
}

/// ∇²(x²) = 2 exactly for second-order central differences.
#[test]
fn test_laplacian_quadratic_field_exact() {
    let n = 10;
    let dx = 0.5e-3;
    let field = Array3::from_shape_fn((n, n, n), |(i, _j, _k)| {
        let x = i as f64 * dx;
        x * x
    });
    let lap = laplacian_3d(&field, (n, n, n), dx, dx, dx);

    let i = n / 2;
    let j = n / 2;
    let k = n / 2;
    assert!((lap[[i, j, k]] - 2.0).abs() < 1e-8);
}

#[test]
fn test_laplacian_view_matches_owned_input() {
    let n = 7;
    let dx = 0.25;
    let stacked = Array3::from_shape_fn((2 * n, n, n), |(i, j, k)| {
        let local_i = if i < n { i } else { i - n };
        let x = local_i as f64 * dx;
        let y = j as f64 * dx;
        let z = k as f64 * dx;
        x * x + 2.0 * y * y + 3.0 * z * z
    });
    let block = field_block_view(&stacked, n, 1);
    let owned = block.to_owned();

    let lap_view = laplacian_3d(&block, (n, n, n), dx, dx, dx);
    let lap_owned = laplacian_3d(&owned, (n, n, n), dx, dx, dx);

    assert_eq!(lap_view.dim(), (n, n, n));
    assert_eq!(lap_view, lap_owned);
    assert!((lap_view[[3, 3, 3]] - 12.0).abs() < 1e-12);
}

#[test]
fn test_laplacian_zero_field() {
    let field = Array3::zeros((6, 6, 6));
    for &dx in &[1.0, 1e-3, 1e-6] {
        let lap = laplacian_3d(&field, (6, 6, 6), dx, dx, dx);
        assert!(
            lap.iter().all(|&v| v.abs() < 1e-15),
            "Laplacian of zero field must be zero for dx={dx}"
        );
    }
}
