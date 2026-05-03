use super::*;
use approx::assert_abs_diff_eq;

#[test]
fn test_constructor_valid() {
    let op = CentralDifference2::new(0.1, 0.1, 0.1);
    assert!(op.is_ok());
}

#[test]
fn test_constructor_invalid_spacing() {
    assert!(CentralDifference2::new(0.0, 0.1, 0.1).is_err());
    assert!(CentralDifference2::new(-0.1, 0.1, 0.1).is_err());
    assert!(CentralDifference2::new(0.1, 0.0, 0.1).is_err());
    assert!(CentralDifference2::new(0.1, 0.1, -0.1).is_err());
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

    assert!(op.apply_x(field_x.view()).is_err());
    assert!(op.apply_y(field_y.view()).is_err());
    assert!(op.apply_z(field_z.view()).is_err());
}

#[test]
fn test_properties() {
    let op = CentralDifference2::new(0.1, 0.1, 0.1).unwrap();
    assert_eq!(op.order(), 2);
    assert_eq!(op.stencil_width(), 3);
    assert!(op.is_adjoint_consistent());
    assert!(!op.is_conservative());
}
