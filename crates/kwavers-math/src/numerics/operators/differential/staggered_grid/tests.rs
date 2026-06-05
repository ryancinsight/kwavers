use super::operator::StaggeredGridOperator;
use approx::assert_abs_diff_eq;
use kwavers_core::error::{KwaversError, KwaversResult, NumericalError};
use ndarray::Array3;

fn assert_invalid_grid_spacing(
    result: KwaversResult<StaggeredGridOperator>,
    dx: f64,
    dy: f64,
    dz: f64,
) {
    match result {
        Err(KwaversError::Numerical(NumericalError::InvalidGridSpacing {
            dx: actual_dx,
            dy: actual_dy,
            dz: actual_dz,
        })) => {
            assert_eq!(actual_dx.to_bits(), dx.to_bits());
            assert_eq!(actual_dy.to_bits(), dy.to_bits());
            assert_eq!(actual_dz.to_bits(), dz.to_bits());
        }
        Err(error) => panic!("expected invalid grid spacing, got {error:?}"),
        Ok(_) => panic!("expected invalid grid spacing for ({dx}, {dy}, {dz})"),
    }
}

fn assert_insufficient_grid_points<T>(
    result: KwaversResult<T>,
    required: usize,
    actual: usize,
    direction: &str,
) {
    match result {
        Err(KwaversError::Numerical(NumericalError::InsufficientGridPoints {
            required: actual_required,
            actual: actual_points,
            direction: actual_direction,
        })) => {
            assert_eq!(actual_required, required);
            assert_eq!(actual_points, actual);
            assert_eq!(actual_direction, direction);
        }
        Err(error) => panic!("expected insufficient grid points, got {error:?}"),
        Ok(_) => panic!("expected insufficient grid points for {direction} direction"),
    }
}

#[test]
fn test_constructor_valid() {
    use crate::numerics::operators::differential::DifferentialOperator;

    let op = StaggeredGridOperator::new(0.1, 0.2, 0.4).unwrap();
    assert_eq!(op.order(), 2);
    assert_eq!(op.stencil_width(), 2);
    assert!(op.is_conservative());
    assert!(op.is_adjoint_consistent());
}

#[test]
fn test_constructor_invalid_spacing() {
    assert_invalid_grid_spacing(StaggeredGridOperator::new(0.0, 0.1, 0.1), 0.0, 0.1, 0.1);
    assert_invalid_grid_spacing(StaggeredGridOperator::new(-0.1, 0.1, 0.1), -0.1, 0.1, 0.1);
    assert_invalid_grid_spacing(StaggeredGridOperator::new(0.1, 0.0, 0.1), 0.1, 0.0, 0.1);
    assert_invalid_grid_spacing(StaggeredGridOperator::new(0.1, 0.1, -0.1), 0.1, 0.1, -0.1);
}

#[test]
fn test_forward_difference_linear_function() {
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();
    let mut field = Array3::zeros((10, 5, 5));
    for i in 0..10 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 2.0 * (i as f64) * dx;
            }
        }
    }
    let grad = op.apply_forward_x(field.view()).unwrap();
    assert_eq!(grad.dim(), (9, 5, 5));
    for i in 0..9 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(grad[[i, j, k]], 2.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_backward_difference_linear_function() {
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();
    let mut field = Array3::zeros((10, 5, 5));
    for i in 0..10 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 2.0 * (i as f64) * dx;
            }
        }
    }
    let grad = op.apply_backward_x(field.view()).unwrap();
    assert_eq!(grad.dim(), (10, 5, 5));
    for i in 1..10 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(grad[[i, j, k]], 2.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_constant_field_has_zero_derivative() {
    let op = StaggeredGridOperator::new(0.01, 0.01, 0.01).unwrap();
    let field = Array3::from_elem((10, 10, 10), 5.0);
    let grad_forward = op.apply_forward_x(field.view()).unwrap();
    let grad_backward = op.apply_backward_x(field.view()).unwrap();
    for i in 0..9 {
        for j in 0..10 {
            for k in 0..10 {
                assert_abs_diff_eq!(grad_forward[[i, j, k]], 0.0, epsilon = 1e-15);
            }
        }
    }
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                assert_abs_diff_eq!(grad_backward[[i, j, k]], 0.0, epsilon = 1e-15);
            }
        }
    }
}

#[test]
fn test_insufficient_grid_points() {
    let op = StaggeredGridOperator::new(0.1, 0.1, 0.1).unwrap();
    let field_x = Array3::zeros((1, 10, 10));
    let field_y = Array3::zeros((10, 1, 10));
    let field_z = Array3::zeros((10, 10, 1));
    assert_insufficient_grid_points(op.apply_forward_x(field_x.view()), 2, 1, "X");
    assert_insufficient_grid_points(op.apply_backward_x(field_x.view()), 2, 1, "X");
    assert_insufficient_grid_points(op.apply_forward_y(field_y.view()), 2, 1, "Y");
    assert_insufficient_grid_points(op.apply_backward_y(field_y.view()), 2, 1, "Y");
    assert_insufficient_grid_points(op.apply_forward_z(field_z.view()), 2, 1, "Z");
    assert_insufficient_grid_points(op.apply_backward_z(field_z.view()), 2, 1, "Z");
}

#[test]
fn test_properties() {
    use crate::numerics::operators::differential::DifferentialOperator;
    let op = StaggeredGridOperator::new(0.1, 0.1, 0.1).unwrap();
    assert_eq!(op.order(), 2);
    assert_eq!(op.stencil_width(), 2);
    assert!(op.is_conservative());
    assert!(op.is_adjoint_consistent());
}

#[test]
fn test_forward_backward_complementarity() {
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();
    let mut field = Array3::zeros((10, 5, 5));
    for i in 0..10 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x;
            }
        }
    }
    let grad_forward = op.apply_forward_x(field.view()).unwrap();
    let grad_backward = op.apply_backward_x(field.view()).unwrap();
    for i in 0..9 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(
                    grad_forward[[i, j, k]],
                    grad_backward[[i + 1, j, k]],
                    epsilon = 1e-10
                );
            }
        }
    }
}

#[test]
fn test_all_directions() {
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();
    let mut field = Array3::zeros((10, 10, 10));
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                field[[i, j, k]] =
                    2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
            }
        }
    }
    let grad_y = op.apply_backward_y(field.view()).unwrap();
    let grad_z = op.apply_backward_z(field.view()).unwrap();
    for i in 0..10 {
        for j in 1..10 {
            for k in 1..10 {
                assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_forward_into_x_matches_allocating() {
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();
    let mut field = Array3::zeros((10, 5, 5));
    for i in 0..10 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 3.7 * (i as f64) * dx - 1.2 * (j as f64) * dx;
            }
        }
    }
    let expected = op.apply_forward_x(field.view()).unwrap();
    let mut result = Array3::zeros((9, 5, 5));
    op.apply_forward_x_into(field.view(), &mut result).unwrap();
    assert_eq!(result.dim(), expected.dim());
    for i in 0..9 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(result[[i, j, k]], expected[[i, j, k]], epsilon = 0.0);
            }
        }
    }
}

#[test]
fn test_forward_into_y_matches_allocating() {
    let dy = 0.05;
    let op = StaggeredGridOperator::new(dy, dy, dy).unwrap();
    let mut field = Array3::zeros((5, 10, 5));
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..5 {
                field[[i, j, k]] = 2.5 * (j as f64) * dy + 0.1 * (i as f64);
            }
        }
    }
    let expected = op.apply_forward_y(field.view()).unwrap();
    let mut result = Array3::zeros((5, 9, 5));
    op.apply_forward_y_into(field.view(), &mut result).unwrap();
    assert_eq!(result.dim(), expected.dim());
    for i in 0..5 {
        for j in 0..9 {
            for k in 0..5 {
                assert_abs_diff_eq!(result[[i, j, k]], expected[[i, j, k]], epsilon = 0.0);
            }
        }
    }
}

#[test]
fn test_forward_into_z_matches_allocating() {
    let dz = 0.02;
    let op = StaggeredGridOperator::new(dz, dz, dz).unwrap();
    let mut field = Array3::zeros((4, 4, 8));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..8 {
                field[[i, j, k]] = -1.5 * (k as f64) * dz + 0.3 * (j as f64);
            }
        }
    }
    let expected = op.apply_forward_z(field.view()).unwrap();
    let mut result = Array3::zeros((4, 4, 7));
    op.apply_forward_z_into(field.view(), &mut result).unwrap();
    assert_eq!(result.dim(), expected.dim());
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..7 {
                assert_abs_diff_eq!(result[[i, j, k]], expected[[i, j, k]], epsilon = 0.0);
            }
        }
    }
}

#[test]
fn test_forward_into_x_insufficient_points() {
    let op = StaggeredGridOperator::new(0.1, 0.1, 0.1).unwrap();
    let field = Array3::<f64>::zeros((1, 4, 4));
    let mut dst = Array3::<f64>::zeros((0, 4, 4));
    assert_insufficient_grid_points(op.apply_forward_x_into(field.view(), &mut dst), 2, 1, "X");
}
