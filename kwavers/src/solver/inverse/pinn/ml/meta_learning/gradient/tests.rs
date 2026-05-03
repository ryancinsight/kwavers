use super::*;

#[test]
fn test_gradient_extractor_new() {
    // This is a compile-time test to ensure the API is correct
    // Actual runtime testing requires a full Burn backend setup
}

#[test]
fn test_gradient_applicator_new() {
    use burn::backend::NdArray;
    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> = Vec::new();
    let applicator = GradientApplicator::<TestBackend>::new(grads, 0.01);
    assert_eq!(applicator.learning_rate(), 0.01);
    assert_eq!(applicator.num_gradients(), 0);
    assert!(applicator.is_complete());
}

#[test]
#[should_panic(expected = "Learning rate must be non-negative")]
fn test_gradient_applicator_negative_lr() {
    use burn::backend::NdArray;
    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> = Vec::new();
    let _ = GradientApplicator::<TestBackend>::new(grads, -0.01);
}

#[test]
fn test_gradient_norm_empty() {
    use burn::backend::NdArray;
    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    let grads: Vec<Option<Tensor<<TestBackend as AutodiffBackend>::InnerBackend, 1>>> = Vec::new();
    let norm = utils::gradient_norm::<TestBackend>(&grads);
    assert_eq!(norm, 0.0);
}
