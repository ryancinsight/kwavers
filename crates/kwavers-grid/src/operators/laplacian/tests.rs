use super::*;
use crate::Grid;
use eunomia::assert_relative_eq;

/// Every interior cell gets the stencil at its own index, across several
/// plane tasks.
///
/// The constant and linear cases below are blind to a plane that a task
/// dropped, doubled or shifted — their Laplacian is zero everywhere, so a
/// wrong plane still reads zero — and at 10 cubed they sit under the
/// policy's parallel floor, so no task split ran at all. This case is 16
/// cubed, gives every cell a distinct value, and compares against the same
/// expression the operator applies, so the assertion is exact and a
/// mis-mapped plane fails it.
#[test]
fn every_interior_cell_takes_the_stencil_at_its_own_index() {
    const N: usize = 16;
    let grid = Grid::new(N, N, N, 0.1, 0.2, 0.05).unwrap();

    let mut field = Array3::zeros([N, N, N]);
    let mut value = 0.0_f64;
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                value += 0.25;
                field[[i, j, k]] = value;
            }
        }
    }

    let operator = LaplacianOperator::second_order(&grid);
    let result = operator.apply(field.view()).unwrap();

    let (dx2_inv, dy2_inv, dz2_inv) = (1.0 / (0.1 * 0.1), 1.0 / (0.2 * 0.2), 1.0 / (0.05 * 0.05));
    for i in 1..N - 1 {
        for j in 1..N - 1 {
            for k in 1..N - 1 {
                let centre = field[[i, j, k]];
                let expected = (2.0_f64.mul_add(-centre, field[[i + 1, j, k]])
                    + field[[i - 1, j, k]])
                    * dx2_inv
                    + (2.0_f64.mul_add(-centre, field[[i, j + 1, k]]) + field[[i, j - 1, k]])
                        * dy2_inv
                    + (2.0_f64.mul_add(-centre, field[[i, j, k + 1]]) + field[[i, j, k - 1]])
                        * dz2_inv;
                assert_eq!(
                    result[[i, j, k]].to_bits(),
                    expected.to_bits(),
                    "laplacian[{i}, {j}, {k}]"
                );
            }
        }
    }
}

#[test]
fn test_laplacian_constant_field() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let field = Array3::from_elem([10, 10, 10], 5.0);

    let operator = LaplacianOperator::second_order(&grid);
    let result = operator.apply(field.view()).unwrap();

    // Laplacian of constant field should be zero (except boundaries)
    for k in 1..9 {
        for j in 1..9 {
            for i in 1..9 {
                assert_relative_eq!(result[[i, j, k]], 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_laplacian_linear_field() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut field = Array3::zeros([10, 10, 10]);

    // Create linear field: f(x,y,z) = x + y + z
    for k in 0..10 {
        for j in 0..10 {
            for i in 0..10 {
                field[[i, j, k]] = i as f64 + j as f64 + k as f64;
            }
        }
    }

    let operator = LaplacianOperator::second_order(&grid);
    let result = operator.apply(field.view()).unwrap();

    // Laplacian of linear field should be zero
    for k in 1..9 {
        for j in 1..9 {
            for i in 1..9 {
                assert_relative_eq!(result[[i, j, k]], 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_laplacian_nonstandard_output_view() {
    let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1).unwrap();
    let mut field = Array3::zeros([10, 10, 10]);
    for k in 0..10 {
        for j in 0..10 {
            for i in 0..10 {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                field[[i, j, k]] = x * x + y * y + z * z;
            }
        }
    }

    let operator = LaplacianOperator::second_order(&grid);
    let expected = operator.apply(field.view()).unwrap();
    let mut storage = Array3::zeros([12, 10, 10]);
    {
        let mut output = storage
            .slice_mut(&[(1, 11, 1), (0, 10, 1), (0, 10, 1)])
            .unwrap();
        operator.apply_mut(field.view(), output.reborrow()).unwrap();
    }

    for k in 1..9 {
        for j in 1..9 {
            for i in 1..9 {
                assert_relative_eq!(storage[[i + 1, j, k]], expected[[i, j, k]], epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_different_orders() {
    let grid = Grid::new(20, 20, 20, 0.1, 0.1, 0.1).unwrap();
    let mut field = Array3::zeros([20, 20, 20]);

    // Create a smooth test field
    for k in 0..20 {
        for j in 0..20 {
            for i in 0..20 {
                let x = i as f64 * 0.1;
                let y = j as f64 * 0.1;
                let z = k as f64 * 0.1;
                field[[i, j, k]] = (x * std::f64::consts::PI).sin()
                    * (y * std::f64::consts::PI).sin()
                    * (z * std::f64::consts::PI).sin();
            }
        }
    }

    let op2 = LaplacianOperator::with_order(&grid, FdAccuracyOrder::Second);
    let op4 = LaplacianOperator::with_order(&grid, FdAccuracyOrder::Fourth);

    let result2 = op2.apply(field.view()).unwrap();
    let result4 = op4.apply(field.view()).unwrap();

    // Higher order should be more accurate (different from second order)
    let diff: f64 = result4
        .iter()
        .zip(result2.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    assert!(diff > 1e-6, "Fourth order should differ from second order");
}
