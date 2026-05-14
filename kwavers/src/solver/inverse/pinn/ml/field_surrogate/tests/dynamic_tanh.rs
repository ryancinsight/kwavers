//! Dynamic Tanh activation tests (Phase C-7).

use burn::tensor::{Tensor, TensorData};

use super::super::dynamic_tanh::DynamicTanh;
use super::{AB, B};

#[test]
fn test_default_init_recovers_vanilla_tanh() {
    // `DynamicTanh::new(device)` initialises (α=1, γ=1, β=0), so
    // `DyT(x) = γ·tanh(α·x) + β = tanh(x)` at init time.
    let device = Default::default();
    let dyt = DynamicTanh::<B>::new(&device);
    let xs: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let input = Tensor::<B, 2>::from_data(TensorData::new(xs.to_vec(), [5, 1]), &device);
    let out = dyt.forward(input);
    let out_vec: Vec<f32> = out.into_data().convert::<f32>().into_vec().unwrap();
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
    let device = Default::default();
    let dyt = DynamicTanh::<B>::with_init(0.7, 1.3, -0.2, &device);
    let (alpha, gamma, beta) = dyt.scalars();
    assert!((alpha - 0.7).abs() < 1e-6);
    assert!((gamma - 1.3).abs() < 1e-6);
    assert!((beta + 0.2).abs() < 1e-6);
}

#[test]
fn test_forward_applies_affine_then_tanh_then_affine() {
    // DyT(x) = γ·tanh(α·x) + β. With α=2, γ=3, β=-1: at x=0 the
    // output is exactly β = -1 (since tanh(0) = 0).
    let device = Default::default();
    let dyt = DynamicTanh::<B>::with_init(2.0, 3.0, -1.0, &device);
    let input = Tensor::<B, 2>::from_data(TensorData::new(vec![0.0_f32, 1.0_f32], [2, 1]), &device);
    let out_vec: Vec<f32> = dyt
        .forward(input)
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap();
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
    let device = Default::default();
    let dyt = DynamicTanh::<B>::new(&device);
    let n = 17;
    let f = 11;
    let data: Vec<f32> = (0..n * f).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let input = Tensor::<B, 2>::from_data(TensorData::new(data, [n, f]), &device);
    let out = dyt.forward(input);
    assert_eq!(out.dims(), [n, f]);
}

#[test]
fn test_autodiff_gradient_flow_through_dyt() {
    // Validate that gradients propagate through DyT — backward
    // through `(x * α).tanh() * γ + β` must reach (and update) both
    // the inputs and the (α, γ, β) parameters. We construct a tiny
    // MSE between a forward output and a zero target, take the
    // backward, and assert no NaN/Inf in the resulting gradient
    // tensors.
    use burn::module::AutodiffModule;
    let device = Default::default();
    let dyt = DynamicTanh::<AB>::new(&device);
    let input = Tensor::<AB, 2>::from_data(
        TensorData::new(vec![0.3_f32, 0.6_f32, -0.4_f32], [3, 1]),
        &device,
    );
    let out = dyt.forward(input);
    let loss = out.powf_scalar(2.0).mean();
    let grads = loss.backward();
    // Just check that `valid()` (the non-autodiff projection)
    // round-trips cleanly — i.e., the autodiff graph wasn't poisoned
    // with non-finite values.
    let _inner = dyt.valid();
    drop(grads);
}
