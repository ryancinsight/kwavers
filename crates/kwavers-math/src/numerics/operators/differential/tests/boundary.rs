//! Boundary accuracy and anisotropic grid tests.

use super::super::*;
use approx::assert_abs_diff_eq;
use ndarray::Array3;

#[test]
fn test_boundary_accuracy_degradation() {
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

    let grad = op.apply_x(field.view()).unwrap();

    let interior_i = 10;
    let near_boundary_i = 2;
    let boundary_i = 0;

    let x_interior = (interior_i as f64) * dx;
    let x_near = (near_boundary_i as f64) * dx;
    let x_boundary = (boundary_i as f64) * dx;

    let exact_interior = 3.0 * x_interior * x_interior;
    let exact_near = 3.0 * x_near * x_near;
    let exact_boundary = 3.0 * x_boundary * x_boundary;

    let error_interior = (grad[[interior_i, 2, 2]] - exact_interior).abs();
    let error_near = (grad[[near_boundary_i, 2, 2]] - exact_near).abs();
    let error_boundary = (grad[[boundary_i, 2, 2]] - exact_boundary).abs();

    assert!(
        error_interior < 1e-8,
        "Interior error should be small: {}",
        error_interior
    );
    assert!(
        error_boundary > error_interior * 10.0,
        "Boundary error should be much larger: boundary={}, near={}, interior={}",
        error_boundary,
        error_near,
        error_interior
    );
}

#[test]
fn test_anisotropic_grid() {
    let dx = 0.1;
    let dy = 0.05;
    let dz = 0.2;

    let op = CentralDifference4::new(dx, dy, dz).unwrap();

    let mut field = Array3::zeros((20, 40, 10));
    for i in 0..20 {
        for j in 0..40 {
            for k in 0..10 {
                let x = (i as f64) * dx;
                let y = (j as f64) * dy;
                let z = (k as f64) * dz;
                field[[i, j, k]] = 2.0 * x + 3.0 * y + 4.0 * z;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    for i in 5..15 {
        for j in 5..35 {
            for k in 2..8 {
                assert_abs_diff_eq!(grad_x[[i, j, k]], 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_y[[i, j, k]], 3.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_z[[i, j, k]], 4.0, epsilon = 1e-10);
            }
        }
    }
}
