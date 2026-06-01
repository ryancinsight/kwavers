//! Operator consistency, symmetry, and dispersion tests.

use super::super::*;
use kwavers_core::constants::numerical::TWO_PI;
use approx::assert_abs_diff_eq;
use ndarray::Array3;

#[test]
fn test_operator_consistency_on_quadratic() {
    let dx = 0.1;

    let op2 = CentralDifference2::new(dx, dx, dx).unwrap();
    let op4 = CentralDifference4::new(dx, dx, dx).unwrap();
    let op6 = CentralDifference6::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((30, 5, 5));
    for i in 0..30 {
        let x = (i as f64) * dx;
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = x * x;
            }
        }
    }

    let grad_2 = op2.apply_x(field.view()).unwrap();
    let grad_4 = op4.apply_x(field.view()).unwrap();
    let grad_6 = op6.apply_x(field.view()).unwrap();

    for i in 10..20 {
        let x = (i as f64) * dx;
        let expected = 2.0 * x;

        assert_abs_diff_eq!(grad_2[[i, 2, 2]], expected, epsilon = 1e-8);
        assert_abs_diff_eq!(grad_4[[i, 2, 2]], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(grad_6[[i, 2, 2]], expected, epsilon = 1e-12);
    }
}

#[test]
fn test_all_directions_symmetry() {
    let dx = 0.1;
    let op = CentralDifference4::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros((20, 20, 20));
    for i in 0..20 {
        for j in 0..20 {
            for k in 0..20 {
                let x = (i as f64) * dx;
                let y = (j as f64) * dx;
                let z = (k as f64) * dx;
                field[[i, j, k]] = x * x + y * y + z * z;
            }
        }
    }

    let grad_x = op.apply_x(field.view()).unwrap();
    let grad_y = op.apply_y(field.view()).unwrap();
    let grad_z = op.apply_z(field.view()).unwrap();

    let center = 10;
    for offset in 1..5 {
        let gx = grad_x[[center + offset, center, center]];
        let gy = grad_y[[center, center + offset, center]];
        let gz = grad_z[[center, center, center + offset]];

        assert_abs_diff_eq!(gx.abs(), gy.abs(), epsilon = 1e-10);
        assert_abs_diff_eq!(gy.abs(), gz.abs(), epsilon = 1e-10);
    }
}

#[test]
fn test_high_frequency_dispersion() {
    let dx = 0.1;
    let wavelength = 2.0; // 20 points per wavelength
    let wave_number = TWO_PI / wavelength;

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
