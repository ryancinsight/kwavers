//! Dynamic Tanh activation tests (Phase C-7).

use coeus_autograd::Var;

use super::super::dynamic_tanh::DynamicTanh;
use super::B;

fn var_col(backend: &B, values: &[f32], cols: usize) -> Var<f32, B> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![(values.shape()[0] * values.shape()[1] * values.shape()[2]) / cols, cols], values, backend),
        false,
    )
}

#[test]
fn test_default_init_recovers_vanilla_tanh() {
    // `DynamicTanh::new()` initialises (α=1, γ=1, β=0), so
    // `DyT(x) = γ·tanh(α·x) + β = tanh(x)` at init time.
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new();
    let xs: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let input = var_col(&backend, &xs, 1);
    let out = dyt.forward(&input);
    let out_vec = out.tensor.as_slice();
    for (i, &x) in xs.iter().enumerate() {
        let expected = x.tanh();
        let diff = (out_vec[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "vanilla-tanh init mismatch at x={x}: dyt={}, tanh(x)={expected}",
            out_vec[i]
        );
    }
}

#[test]
fn test_custom_init_scalars_round_trip() {
    let dyt = DynamicTanh::<B>::with_init(0.7, 1.3, -0.2);
    let (alpha, gamma, beta) = dyt.scalars();
    assert!((alpha - 0.7).abs() < 1e-6);
    assert!((gamma - 1.3).abs() < 1e-6);
    assert!((beta + 0.2).abs() < 1e-6);
}

#[test]
fn test_forward_applies_affine_then_tanh_then_affine() {
    // DyT(x) = γ·tanh(α·x) + β. With α=2, γ=3, β=-1: at x=0 the
    // output is exactly β = -1 (since tanh(0) = 0).
    let backend = B::default();
    let dyt = DynamicTanh::<B>::with_init(2.0, 3.0, -1.0);
    let input = var_col(&backend, &[0.0_f32, 1.0_f32], 1);
    let out_vec = dyt.forward(&input).tensor.as_slice().to_vec();
    // x=0 → tanh(0)=0 → γ·0 + β = β = -1
    assert!(
        (out_vec[0] + 1.0).abs() < 1e-5,
        "x=0 case: got {}, expected -1",
        out_vec[0]
    );
    // x=1 → tanh(2)=0.9640 → 3·0.9640 - 1 = 1.8920
    let expected = 3.0 * 2.0_f32.tanh() - 1.0;
    assert!(
        (out_vec[1] - expected).abs() < 1e-4,
        "x=1 case: got {}, expected {expected}",
        out_vec[1]
    );
}

#[test]
fn test_forward_shape_preserved() {
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new();
    let n = 17;
    let f = 11;
    let data: Vec<f32> = (0..n * f).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let input = var_col(&backend, &data, f);
    let out = dyt.forward(&input);
    assert_eq!(out.tensor.shape(), &[n, f]);
}

#[test]
fn test_autodiff_gradient_flow_through_dyt() {
    // Validate that gradients propagate through DyT — backward
    // through `(x * α).tanh() * γ + β` must reach (and update) both
    // the inputs and the (α, γ, β) parameters. We construct a tiny
    // MSE between a forward output and a zero target, take the
    // backward, and assert no NaN/Inf in the resulting gradients.
    let backend = B::default();
    let dyt = DynamicTanh::<B>::new();
    let input = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![3, 1], &[0.3_f32, 0.6, -0.4], &backend),
        true,
    );

    for p in dyt.parameters() {
        p.zero_grad();
    }
    let out = dyt.forward(&input);
    let loss = coeus_autograd::mean(&coeus_autograd::mul(&out, &out));
    loss.backward();

    for p in dyt.parameters() {
        let grad = p.grad().expect("DyT parameter must receive a gradient");
        assert!(grad.as_slice().iter().all(|g| g.is_finite()));
    }
    let input_grad = input.grad().expect("input must receive a gradient");
    assert!(input_grad.as_slice().iter().all(|g| g.is_finite()));
}
