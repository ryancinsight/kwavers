//! Tests for advanced PINN architectures.

use burn::backend::NdArray;
use burn::tensor::Tensor;

use super::fourier::FourierFeatures;
use super::residual::ResidualBlock;
use super::resnet::{ResNetPINN1D, ResNetPINN2D, ResNetPINNConfig};

type TestBackend = NdArray<f32>;

#[test]
fn test_fourier_features() {
    let device = Default::default();
    let fourier = FourierFeatures::<TestBackend>::new(2, 10, 1.0, &device);

    let input = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.3]], &device);
    let output = fourier.forward(input);

    // Should have 20 features (10 cos + 10 sin).
    assert_eq!(output.dims(), [1, 20]);
}

#[test]
fn test_residual_block() {
    let device = Default::default();
    let block = ResidualBlock::<TestBackend>::new(10, 20, &device);

    let input = Tensor::<TestBackend, 2>::from_floats([[0.1f32; 10]], &device);
    let output = block.forward(input);

    // Should maintain input dimension.
    assert_eq!(output.dims(), [1, 10]);
}

#[test]
fn test_resnet_pinn_1d() {
    let device = Default::default();
    let config = ResNetPINNConfig {
        input_dim: 2,
        hidden_layers: vec![32, 64],
        num_blocks: 2,
        use_fourier_features: true,
        fourier_scale: 5.0,
    };

    let pinn = ResNetPINN1D::<TestBackend>::new(&config, &device);

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let output = pinn.forward(x, t);

    assert_eq!(output.dims(), [1, 1]);
}

#[test]
fn test_resnet_pinn_2d() {
    let device = Default::default();
    let config = ResNetPINNConfig {
        input_dim: 3,
        hidden_layers: vec![32, 64],
        num_blocks: 2,
        use_fourier_features: true,
        fourier_scale: 5.0,
    };

    let pinn = ResNetPINN2D::<TestBackend>::new(&config, &device);

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let y = Tensor::<TestBackend, 2>::from_floats([[0.3]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let output = pinn.forward(x, y, t);

    assert_eq!(output.dims(), [1, 1]);
}
