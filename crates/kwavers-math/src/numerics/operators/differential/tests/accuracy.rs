//! Convergence order and linear accuracy tests.

use super::super::*;
use eunomia::assert_abs_diff_eq;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

#[test]
fn test_all_operators_linear_function() {
    let dx = 0.1;

    let op2 = CentralDifference2::new(dx, dx, dx).unwrap();
    let op4 = CentralDifference4::new(dx, dx, dx).unwrap();
    let op6 = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros([20, 20, 20]);
    for i in 0..20 {
        for j in 0..20 {
            for k in 0..20 {
                field[[i, j, k]] =
                    2.0 * (i as f64) * dx + 3.0 * (j as f64) * dx + 4.0 * (k as f64) * dx;
            }
        }
    }

    let grad_x_2 = op2.apply_x(field.view()).unwrap();
    let grad_x_4 = op4.apply_x(field.view()).unwrap();
    let grad_x_6 = op6.apply_x(field.view()).unwrap();

    for i in 5..15 {
        for j in 5..15 {
            for k in 5..15 {
                assert_abs_diff_eq!(grad_x_2[[i, j, k]], 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_x_4[[i, j, k]], 2.0, epsilon = 1e-10);
                assert_abs_diff_eq!(grad_x_6[[i, j, k]], 2.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_convergence_order_second_order() {
    // u(x) = sin(2πx), du/dx = 2π·cos(2πx)
    let test_function = |x: f64| (TWO_PI * x).sin();
    let test_derivative = |x: f64| TWO_PI * (TWO_PI * x).cos();

    let grid_sizes = vec![20, 40, 80];
    let mut errors = Vec::new();

    for nx in grid_sizes {
        let dx = 1.0 / (nx as f64);
        let op = CentralDifference2::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros([nx, 5, 5]);
        for i in 0..nx {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = test_function(x);
                }
            }
        }

        let grad = op.apply_x(field.view()).unwrap();

        let mut error_sum = 0.0;
        let mut count = 0;
        for i in 5..nx - 5 {
            let x = (i as f64) * dx;
            let exact = test_derivative(x);
            let numerical = grad[[i, 2, 2]];
            error_sum += (exact - numerical).powi(2);
            count += 1;
        }
        let l2_error = (error_sum / count as f64).sqrt();
        errors.push(l2_error);
    }

    // Error decreases by factor of 4 (2²) per grid refinement.
    let rate_1 = errors[0] / errors[1];
    let rate_2 = errors[1] / errors[2];

    assert!(
        rate_1 > 3.5 && rate_1 < 4.5,
        "Convergence rate 1: {}",
        rate_1
    );
    assert!(
        rate_2 > 3.5 && rate_2 < 4.5,
        "Convergence rate 2: {}",
        rate_2
    );
}

#[test]
fn test_convergence_order_fourth_order() {
    let test_function = |x: f64| (TWO_PI * x).sin();
    let test_derivative = |x: f64| TWO_PI * (TWO_PI * x).cos();

    let grid_sizes = vec![20, 40, 80];
    let mut errors = Vec::new();

    for nx in grid_sizes {
        let dx = 1.0 / (nx as f64);
        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros([nx, 5, 5]);
        for i in 0..nx {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = test_function(x);
                }
            }
        }

        let grad = op.apply_x(field.view()).unwrap();

        let mut error_sum = 0.0;
        let mut count = 0;
        for i in 5..nx - 5 {
            let x = (i as f64) * dx;
            let exact = test_derivative(x);
            let numerical = grad[[i, 2, 2]];
            error_sum += (exact - numerical).powi(2);
            count += 1;
        }
        let l2_error = (error_sum / count as f64).sqrt();
        errors.push(l2_error);
    }

    // Error decreases by factor of 16 (2⁴) per grid refinement.
    let rate_1 = errors[0] / errors[1];
    let rate_2 = errors[1] / errors[2];

    assert!(
        rate_1 > 12.0 && rate_1 < 20.0,
        "Convergence rate 1: {}",
        rate_1
    );
    assert!(
        rate_2 > 12.0 && rate_2 < 20.0,
        "Convergence rate 2: {}",
        rate_2
    );
}
