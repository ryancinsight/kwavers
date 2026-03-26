use super::field_ops::FieldOps;
use super::kernel::{apply_kernel, windowed_operation};
use super::reduction::FieldReduction;
use approx::assert_abs_diff_eq;
use ndarray::Array3;

#[test]
fn test_filter_indices_lazy() {
    let field = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| i + j + k);
    let indices: Vec<_> = field.filter_indices(|&x| x > 5).collect();

    assert!(!indices.is_empty());
    assert!(indices.contains(&(2, 2, 2)));
}

#[test]
fn test_sparse_kernel() {
    let field = Array3::from_elem((5, 5, 5), 1.0);
    let mut kernel = Array3::zeros((3, 3, 3));
    kernel[[1, 1, 1]] = 2.0;

    let result = apply_kernel(&field, &kernel, |f: &f64, k: &f64| f * k);
    assert_abs_diff_eq!(result[[2, 2, 2]], 2.0);
}

#[test]
fn test_parallel_map() {
    let field = Array3::from_shape_fn((10, 10, 10), |(i, j, k)| (i + j + k) as f64);
    let result = field.par_map_field(|&x| x * 2.0);

    assert_abs_diff_eq!(field[[5, 5, 5]], 5.0 + 5.0 + 5.0);
    assert_abs_diff_eq!(result[[5, 5, 5]], (5.0 + 5.0 + 5.0) * 2.0);
    assert_abs_diff_eq!(result[[0, 0, 0]], 0.0);
    assert_abs_diff_eq!(result[[1, 1, 1]], 6.0);
    assert_abs_diff_eq!(result[[9, 9, 9]], 54.0);
}

#[test]
fn test_field_reduction() {
    let field = Array3::from_shape_fn((2, 2, 2), |_| 3.0);

    assert_abs_diff_eq!(field.sum(), 24.0);
    assert_abs_diff_eq!(field.mean().unwrap(), 3.0);
    let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert_abs_diff_eq!(max_val, 3.0);
}

#[test]
fn test_windowed_operation() {
    let field = Array3::from_elem((5, 5, 5), 1.0);
    let result = windowed_operation(&field, (3, 3, 3), |window| window.iter().sum::<f64>());

    assert_abs_diff_eq!(result[[2, 2, 2]], 27.0);
}
