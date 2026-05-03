use super::super::DifferentialOperator;
use super::core::CentralDifference6;
use approx::assert_abs_diff_eq;
use ndarray::Array3;

#[test]
fn test_constructor_valid() {
    let op = CentralDifference6::new(0.1, 0.1, 0.1);
    assert!(op.is_ok());
    let op = op.unwrap();
    assert_eq!(op.order(), 6);
}

#[test]
fn test_constructor_invalid_spacing() {
    assert!(CentralDifference6::new(0.0, 0.1, 0.1).is_err());
    assert!(CentralDifference6::new(-0.1, 0.1, 0.1).is_err());
}

#[test]
fn test_apply_x_linear_function() {
    // Sixth-order scheme is exact for linear functions
    let dx = 0.1;
    let op = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((12, 5, 5));
    for i in 0..12 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = 2.0 * (i as f64) * dx;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();

    // Check interior points (sixth-order stencil)
    for i in 3..9 {
        for j in 0..5 {
            for k in 0..5 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_all_directions_linear_function() {
    let dx = 0.1;
    let op = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((12, 12, 12));
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                field[[i, j, k]] =
                    2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    for i in 3..9 {
        for j in 3..9 {
            for k in 3..9 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_constant_field_has_zero_derivative() {
    let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();
    let field = Array3::from_elem((12, 12, 12), 5.0);

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(grad_y[[i, j, k]], 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(grad_z[[i, j, k]], 0.0, epsilon = 1e-15);
            }
        }
    }
}

#[test]
fn test_insufficient_grid_points() {
    let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();

    let field_x = Array3::zeros((6, 10, 10));
    let field_y = Array3::zeros((10, 6, 10));
    let field_z = Array3::zeros((10, 10, 6));

    assert!(op.apply_x(field_x.view()).is_err());
    assert!(op.apply_y(field_y.view()).is_err());
    assert!(op.apply_z(field_z.view()).is_err());
}

#[test]
fn test_properties() {
    let op = CentralDifference6::new(0.1, 0.1, 0.1).unwrap();
    assert_eq!(op.order(), 6);
    assert_eq!(op.stencil_width(), 7);
    assert!(op.is_adjoint_consistent());
    assert!(!op.is_conservative());
}

#[test]
fn test_cubic_polynomial() {
    // u(x) = x³ → du/dx = 3x²
    let dx = 0.1;
    let op = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((20, 5, 5));
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x * x;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();

    for i in 5..15 {
        let x = (i as f64) * dx;
        let expected = 3.0 * x * x;
        assert_abs_diff_eq!(grad_x[[i, 2, 2]], expected, epsilon = 1e-8);
    }
}
