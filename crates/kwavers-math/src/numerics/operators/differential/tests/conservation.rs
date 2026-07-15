//! Conservation and staggered grid tests.

use super::super::*;
use approx::assert_abs_diff_eq;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

#[test]
fn test_staggered_conservation() {
    // Sum of forward differences = field[last] − field[first] (telescoping sum).
    let dx = 0.1;
    let op = StaggeredGridOperator::new(dx, dx, dx).unwrap();

    let mut field = Array3::zeros([20, 10, 10]);
    for i in 0..20 {
        let x = (i as f64) * dx;
        for j in 0..10 {
            for k in 0..10 {
                field[[i, j, k]] = (x * TWO_PI).sin();
            }
        }
    }

    let grad_forward = op.apply_forward_x(field.view()).unwrap();
    let _grad_backward = op.apply_backward_x(field.view()).unwrap();

    let mut sum_forward = 0.0;
    for i in 0..19 {
        for j in 0..10 {
            for k in 0..10 {
                sum_forward += grad_forward[[i, j, k]] * dx;
            }
        }
    }

    let expected_sum = field[[19, 5, 5]] - field[[0, 5, 5]];
    assert_abs_diff_eq!(sum_forward / 100.0, expected_sum, epsilon = 1e-10);
}
