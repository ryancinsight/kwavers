use super::linear::LinearInterpolator;
use super::traits::Interpolator;
use super::trilinear::NumericsTrilinearInterpolator;
use eunomia::assert_abs_diff_eq;
use leto::{Array1, Array3};

#[test]
fn test_linear_interpolator_simple() {
    let dx = 0.1;
    let interp = LinearInterpolator::new(dx);

    let data = Array1::from_vec(5, vec![0.0, 0.2, 0.4, 0.6, 0.8]).unwrap();
    let target = Array1::from_vec(1, vec![0.05]).unwrap();
    let result = interp.interpolate_1d(data.view(), target.view()).unwrap();

    assert_abs_diff_eq!(result[0], 0.1, epsilon = 1e-10);
}

#[test]
fn test_linear_interpolator_exact_at_grid_points() {
    let dx = 1.0;
    let interp = LinearInterpolator::new(dx);

    let data = Array1::from_vec(4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let target = Array1::from_vec(1, vec![2.0]).unwrap();
    let result = interp.interpolate_1d(data.view(), target.view()).unwrap();

    assert_abs_diff_eq!(result[0], 3.0, epsilon = 1e-10);
}

#[test]
fn test_trilinear_constant_field() {
    let dx = 0.1;
    let interp = NumericsTrilinearInterpolator::new(dx, dx, dx);

    let data = Array3::from_elem((5, 5, 5), 10.0);
    let result = interp
        .interpolate_point(data.view(), 0.25, 0.15, 0.35)
        .unwrap();

    assert_abs_diff_eq!(result, 10.0, epsilon = 1e-10);
}

#[test]
fn test_trilinear_linear_function() {
    let dx = 1.0;
    let interp = NumericsTrilinearInterpolator::new(dx, dx, dx);

    let mut data = Array3::zeros([4, 4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                data[[i, j, k]] = (i as f64) + 2.0 * (j as f64) + 3.0 * (k as f64);
            }
        }
    }

    let x = 1.5;
    let y = 2.3;
    let z = 1.7;
    let result = interp.interpolate_point(data.view(), x, y, z).unwrap();
    let expected = x + 2.0 * y + 3.0 * z;

    assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_trilinear_at_corner() {
    let dx = 0.1;
    let interp = NumericsTrilinearInterpolator::new(dx, dx, dx);

    let mut data = Array3::zeros([3, 3, 3]);
    data[[1, 1, 1]] = 5.0;

    let result = interp
        .interpolate_point(data.view(), 0.1, 0.1, 0.1)
        .unwrap();

    assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
}

#[test]
fn test_interpolation_out_of_bounds() {
    let dx = 0.1;
    let interp = NumericsTrilinearInterpolator::new(dx, dx, dx);

    let data = Array3::zeros([5, 5, 5]);
    let result = interp.interpolate_point(data.view(), 1.0, 0.0, 0.0);

    assert!(result.is_err());
}

#[test]
fn test_trilinear_3d_batch() {
    let dx = 1.0;
    let interp = NumericsTrilinearInterpolator::new(dx, dx, dx);

    let data = Array3::from_vec([2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

    let target_x = Array1::from_vec(1, vec![0.5]).unwrap();
    let target_y = Array1::from_vec(1, vec![0.5]).unwrap();
    let target_z = Array1::from_vec(1, vec![0.5]).unwrap();

    let result = interp
        .interpolate_3d(
            data.view(),
            target_x.view(),
            target_y.view(),
            target_z.view(),
        )
        .unwrap();

    let expected = (0.0 + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0) / 8.0;
    assert_abs_diff_eq!(result[[0, 0, 0]], expected, epsilon = 1e-10);
}
