use super::*;
use crate::core::error::{KwaversError, NumericalError};
use approx::assert_abs_diff_eq;

fn assert_invalid_grid_spacing(
    result: KwaversResult<CentralDifference2>,
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
    let op = CentralDifference2::new(0.1, 0.2, 0.4).unwrap();
    assert_eq!(op.order(), 2);
    assert_eq!(op.stencil_width(), 3);
    assert!(op.is_adjoint_consistent());
    assert!(!op.is_conservative());
}

#[test]
fn test_constructor_invalid_spacing() {
    assert_invalid_grid_spacing(CentralDifference2::new(0.0, 0.1, 0.1), 0.0, 0.1, 0.1);
    assert_invalid_grid_spacing(CentralDifference2::new(-0.1, 0.1, 0.1), -0.1, 0.1, 0.1);
    assert_invalid_grid_spacing(CentralDifference2::new(0.1, 0.0, 0.1), 0.1, 0.0, 0.1);
    assert_invalid_grid_spacing(CentralDifference2::new(0.1, 0.1, -0.1), 0.1, 0.1, -0.1);
}

#[test]
fn test_apply_x_linear_function() {
    // Test on linear function: u(x,y,z) = 2x
    // Exact derivative: du/dx = 2
    let dx = 0.1;
    let op = CentralDifference2::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((10, 5, 5));
    for i in 0..10 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 2.0 * (i as f64) * dx;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();

    assert_abs_diff_eq!(grad_x[[0, 0, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(grad_x[[9, 0, 0]], 2.0, epsilon = 1e-10);

    // Check interior points (exact for linear functions)
    for i in 1..9 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_apply_y_linear_function() {
    let dy = 0.1;
    let op = CentralDifference2::new(dy, dy, dy).unwrap();

    let mut field = Array3::zeros((5, 10, 5));
    for i in 0..5 {
        for j in 0..10 {
            for k in 0..5 {
                field[[i, j, k]] = 3.0 * (j as f64) * dy;
            }
        }
    }

    let grad_y = op.apply_y(field.view()).unwrap();

    assert_abs_diff_eq!(grad_y[[0, 0, 0]], 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(grad_y[[0, 9, 0]], 3.0, epsilon = 1e-10);

    for i in 0..5 {
        for j in 1..9 {
            for k in 0..5 {
                assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_apply_z_linear_function() {
    let dz = 0.1;
    let op = CentralDifference2::new(dz, dz, dz).unwrap();

    let mut field = Array3::zeros((5, 5, 10));
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..10 {
                field[[i, j, k]] = 4.0 * (k as f64) * dz;
            }
        }
    }

    let grad_z = op.apply_z(field.view()).unwrap();

    assert_abs_diff_eq!(grad_z[[0, 0, 0]], 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(grad_z[[0, 0, 9]], 4.0, epsilon = 1e-10);

    for i in 0..5 {
        for j in 0..5 {
            for k in 1..9 {
                assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_constant_field_has_zero_derivative() {
    let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
    let field = Array3::from_elem((10, 10, 10), 5.0);

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(grad_y[[i, j, k]], 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(grad_z[[i, j, k]], 0.0, epsilon = 1e-15);
            }
        }
    }
}

#[test]
fn test_insufficient_grid_points() {
    let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();

    let field_x = Array3::zeros((2, 10, 10));
    let field_y = Array3::zeros((10, 2, 10));
    let field_z = Array3::zeros((10, 10, 2));

    assert_insufficient_grid_points(op.apply_x(field_x.view()), 3, 2, "X");
    assert_insufficient_grid_points(op.apply_y(field_y.view()), 3, 2, "Y");
    assert_insufficient_grid_points(op.apply_z(field_z.view()), 3, 2, "Z");
}

#[test]
fn test_properties() {
    let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
    assert_eq!(op.order(), 2);
    assert_eq!(op.stencil_width(), 3);
    assert!(op.is_adjoint_consistent());
    assert!(!op.is_conservative());
}
