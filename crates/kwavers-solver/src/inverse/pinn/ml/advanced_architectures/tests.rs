//! Tests for advanced PINN architectures.

use coeus_autograd::Var;

use super::fourier::FourierFeatures;
use super::residual::ResidualBlock;
use super::resnet::{ResNetPINN1D, ResNetPINN2D, ResNetPINNConfig};

type TestBackend = coeus_core::MoiraiBackend;

fn var_row(backend: &TestBackend, values: &[f32]) -> Var<f32, TestBackend> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, (values.shape()[0] * values.shape()[1] * values.shape()[2])], values, backend),
        false,
    )
}

#[test]
fn test_fourier_features() {
    let backend = TestBackend::default();
    let fourier = FourierFeatures::<TestBackend>::new(2, 10, 1.0);

    let input = var_row(&backend, &[0.5, 0.3]);
    let output = fourier.forward(&input);

    // Should have 20 features (10 cos + 10 sin).
    assert_eq!(output.tensor.shape(), &[1, 20]);
}

#[test]
fn test_residual_block() {
    let backend = TestBackend::default();
    let block = ResidualBlock::<TestBackend>::new(10, 20);

    let input = var_row(&backend, &[0.1f32; 10]);
    let output = block.forward(&input);

    // Should maintain input dimension.
    assert_eq!(output.tensor.shape(), &[1, 10]);
}

#[test]
fn test_resnet_pinn_1d() {
    let backend = TestBackend::default();
    let config = ResNetPINNConfig {
        input_dim: 2,
        hidden_layers: vec![32, 64],
        num_blocks: 2,
        use_fourier_features: true,
        fourier_scale: 5.0,
    };

    let pinn = ResNetPINN1D::<TestBackend>::new(&config);

    let x = var_row(&backend, &[0.5]);
    let t = var_row(&backend, &[0.1]);
    let output = pinn.forward(&x, &t);

    assert_eq!(output.tensor.shape(), &[1, 1]);
}

#[test]
fn test_resnet_pinn_2d() {
    let backend = TestBackend::default();
    let config = ResNetPINNConfig {
        input_dim: 3,
        hidden_layers: vec![32, 64],
        num_blocks: 2,
        use_fourier_features: true,
        fourier_scale: 5.0,
    };

    let pinn = ResNetPINN2D::<TestBackend>::new(&config);

    let x = var_row(&backend, &[0.5]);
    let y = var_row(&backend, &[0.3]);
    let t = var_row(&backend, &[0.1]);
    let output = pinn.forward(&x, &y, &t);

    assert_eq!(output.tensor.shape(), &[1, 1]);
}
