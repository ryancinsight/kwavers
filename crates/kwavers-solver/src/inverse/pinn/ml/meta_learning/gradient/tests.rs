use super::*;

#[test]
fn test_gradient_norm_empty() {
    let grads: Vec<Option<Vec<f32>>> = Vec::new();
    let norm = utils::gradient_norm(&grads);
    assert_eq!(norm, 0.0);
}

#[test]
fn test_gradient_norm_computes_l2() {
    let grads: Vec<Option<Vec<f32>>> = vec![Some(vec![3.0, 4.0])];
    let norm = utils::gradient_norm(&grads);
    assert!((norm - 5.0).abs() < 1e-6);
}

#[test]
fn test_clip_gradients_by_norm_scales_down() {
    let grads: Vec<Option<Vec<f32>>> = vec![Some(vec![3.0, 4.0])];
    let clipped = utils::clip_gradients_by_norm(grads, 1.0);
    let v = clipped[0].as_ref().unwrap();
    let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_clip_gradients_by_value_clamps() {
    let grads: Vec<Option<Vec<f32>>> = vec![Some(vec![-5.0, 5.0])];
    let clipped = utils::clip_gradients_by_value(grads, 1.0);
    assert_eq!(clipped[0].as_ref().unwrap(), &vec![-1.0, 1.0]);
}

#[test]
fn test_add_gradients_sums_elementwise() {
    let g1: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 2.0])];
    let g2: Vec<Option<Vec<f32>>> = vec![Some(vec![3.0, 4.0])];
    let sum = utils::add_gradients(g1, &g2);
    assert_eq!(sum[0].as_ref().unwrap(), &vec![4.0, 6.0]);
}

#[test]
fn test_scale_gradients_multiplies() {
    let grads: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0, 2.0])];
    let scaled = utils::scale_gradients(grads, 2.0);
    assert_eq!(scaled[0].as_ref().unwrap(), &vec![2.0, 4.0]);
}
