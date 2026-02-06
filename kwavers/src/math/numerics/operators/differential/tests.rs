//! # Integration Tests for Differential Operators
//!
//! This module contains integration tests that verify the correctness of all
//! differential operator implementations against analytical solutions and
//! mathematical specifications.
//!
//! ## Test Categories
//!
//! 1. **Accuracy Tests**: Verify order of accuracy on polynomial test functions
//! 2. **Consistency Tests**: Compare operators against each other
//! 3. **Conservation Tests**: Verify conservation properties
//! 4. **Boundary Tests**: Validate boundary treatment
//!
//! ## Test Functions
//!
//! Standard test functions include:
//! - Constant: u(x) = c → du/dx = 0
//! - Linear: u(x) = ax + b → du/dx = a
//! - Quadratic: u(x) = ax² + bx + c → du/dx = 2ax + b
//! - Cubic: u(x) = ax³ + bx² + cx + d → du/dx = 3ax² + 2bx + c
//! - Sinusoidal: u(x) = sin(kx) → du/dx = k·cos(kx)

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::Array3;
use std::f64::consts::PI;

#[test]
fn test_all_operators_linear_function() {
    // All operators should be exact for linear functions
    let dx = 0.1;

    let op2 = CentralDifference2::new(dx, dx, dx).unwrap();
    let op4 = CentralDifference4::new(dx, dx, dx).unwrap();
    let op6 = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((20, 20, 20));
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

    // Check interior points for all operators
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
    // Verify second-order convergence on smooth function
    // u(x) = sin(2πx), du/dx = 2π·cos(2πx)

    let test_function = |x: f64| (2.0 * PI * x).sin();
    let test_derivative = |x: f64| 2.0 * PI * (2.0 * PI * x).cos();

    let grid_sizes = vec![20, 40, 80];
    let mut errors = Vec::new();

    for nx in grid_sizes {
        let dx = 1.0 / (nx as f64);
        let op = CentralDifference2::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((nx, 5, 5));
        for i in 0..nx {
            let x = (i as f64) * dx;
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = test_function(x);
                }
            }
        }

        let grad = op.apply_x(field.view()).unwrap();

        // Compute L2 error in interior
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

    // Check convergence rate: error should decrease by factor of 4 (2²)
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
    // Verify fourth-order convergence on smooth function
    let test_function = |x: f64| (2.0 * PI * x).sin();
    let test_derivative = |x: f64| 2.0 * PI * (2.0 * PI * x).cos();

    let grid_sizes = vec![20, 40, 80];
    let mut errors = Vec::new();

    for nx in grid_sizes {
        let dx = 1.0 / (nx as f64);
        let op = CentralDifference4::new(dx, dx, dx).unwrap();

        let mut field = Array3::zeros((nx, 5, 5));
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

    // Check convergence rate: error should decrease by factor of 16 (2⁴)
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

#[test]
fn test_staggered_conservation() {
    // Test that staggered grid conserves discrete sums
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

    // Create a field with non-uniform values
    let mut field = Array3::zeros((20, 10, 10));
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..10 {
            for k in 0..10 {
                field[[i, j, k]] = (x * 2.0 * PI).sin();
            }
        }
    }

    let grad_forward = op.apply_forward_x(field.view()).unwrap();
    let _grad_backward = op.apply_backward_x(field.view()).unwrap();

    // Sum of forward differences should relate to boundary values
    let mut sum_forward = 0.0;
    for i in 0..19 {
        for j in 0..10 {
            for k in 0..10 {
                sum_forward += grad_forward[[i, j, k]] * dx;
            }
        }
    }

    // This sum should equal field[19] - field[0] (telescoping sum)
    let expected_sum = field[[19, 5, 5]] - field[[0, 5, 5]];
    assert_abs_diff_eq!(sum_forward / 100.0, expected_sum, epsilon = 1e-10);
}

#[test]
fn test_operator_consistency_on_quadratic() {
    // All operators should agree on smooth functions in interior
    let dx = 0.1;

    let op2 = CentralDifference2::new(dx, dx, dx).unwrap();
    let op4 = CentralDifference4::new(dx, dx, dx).unwrap();
    let op6 = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((30, 5, 5));
    for i in 0..30 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x; // Quadratic
            }
        }
    }

    let grad_2 = op2.apply_x(field.view()).unwrap();
    let grad_4 = op4.apply_x(field.view()).unwrap();
    let grad_6 = op6.apply_x(field.view()).unwrap();

    // Check that all operators agree in deep interior
    for i in 10..20 {
        let x = (i as f64) * dx;
        let expected = 2.0 * x;

        // All should be very close to exact
        assert_abs_diff_eq!(grad_2[[i, 2, 2]], expected, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_4[[i, 2, 2]], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_6[[i, 2, 2]], expected, epsilon = 1e-12);
    }
}

#[test]
fn test_all_directions_symmetry() {
    // Test that operators are symmetric across spatial directions
    let dx = 0.1;
    let op = CentralDifference4::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((20, 20, 20));
    for i in 0..20 {
        for j in 0..20 {
            for k in 0..20 {
                let x = (i as f64) * dx;
                let y = (j as f64) * dx;
                let z = (k as f64) * dx;
                field[[i, j, k]] = x * x + y * y + z * z; // Spherically symmetric
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    // Check that gradient points radially outward
    let center = 10;
    for offset in 1..5 {
        // Check symmetry: grad should have same magnitude in each direction
        let gx = grad_x[[center + offset, center, center]];
        let gy = grad_y[[center, center + offset, center]];
        let gz = grad_z[[center, center, center + offset]];

        assert_abs_diff_eq!(gx.abs(), gy.abs(), epsilon = 1e-10);
        assert_abs_diff_eq!(gy.abs(), gz.abs(), epsilon = 1e-10);
    }
}

#[test]
fn test_high_frequency_dispersion() {
    // Test numerical dispersion characteristics
    // High-order methods should have less dispersion error
    let dx = 0.1;
    let wavelength = 2.0; // 20 points per wavelength (well-resolved)
    let wave_number = 2.0 * PI / wavelength;

    let op2 = CentralDifference2::new(dx, dx, dx).unwrap();
    let op4 = CentralDifference4::new(dx, dx, dx).unwrap();
    let op6 = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((80, 5, 5));
    for i in 0..80 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = (wave_number * x).sin();
            }
        }
    }

    let grad_2 = op2.apply_x(field.view()).unwrap();
    let grad_4 = op4.apply_x(field.view()).unwrap();
    let grad_6 = op6.apply_x(field.view()).unwrap();

    // Compute absolute error in interior
    let mut error_2 = 0.0;
    let mut error_4 = 0.0;
    let mut error_6 = 0.0;
    let mut count = 0;

    for i in 10..70 {
        let x = (i as f64) * dx;
        let exact = wave_number * (wave_number * x).cos();

        error_2 += (grad_2[[i, 2, 2]] - exact).abs();
        error_4 += (grad_4[[i, 2, 2]] - exact).abs();
        error_6 += (grad_6[[i, 2, 2]] - exact).abs();
        count += 1;
    }

    error_2 /= count as f64;
    error_4 /= count as f64;
    error_6 /= count as f64;

    // Higher-order methods should have smaller absolute error on smooth functions
    assert!(
        error_4 < error_2,
        "Fourth-order should be more accurate: error_2={}, error_4={}",
        error_2,
        error_4
    );
    assert!(
        error_6 < error_4,
        "Sixth-order should be more accurate: error_4={}, error_6={}",
        error_4,
        error_6
    );
}

#[test]
fn test_boundary_accuracy_degradation() {
    // Verify that accuracy degrades gracefully at boundaries
    let dx = 0.1;
    let op = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((20, 5, 5));
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x * x; // Cubic
            }
        }
    }

    let grad = op.apply_x(field.view()).unwrap();

    // Check that interior is more accurate than near-boundary
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

    // Errors should increase toward boundary (with small tolerance)
    // Interior should be significantly more accurate than boundary
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
    // Test operators on anisotropic grids (different dx, dy, dz)
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

    // Check interior points
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
