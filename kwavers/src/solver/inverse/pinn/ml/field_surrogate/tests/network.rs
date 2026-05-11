//! Architecture construction and forward-pass shape tests.

use burn::tensor::{Tensor, TensorData};

use super::super::config::ParamFieldPINNConfig;
use super::super::network::ParamFieldPINNNetwork;
use super::B;

#[test]
fn test_network_construction_with_default_config() {
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).expect("construct");
    // Default has 3 hidden layers → 2 intermediate Linear layers.
    assert_eq!(net.hidden_layer_count(), cfg.hidden_layers.len() - 1);
}

#[test]
fn test_network_construction_with_minimal_hidden_stack() {
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![32],
        ..ParamFieldPINNConfig::default()
    };
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).expect("construct");
    // Single hidden layer → zero intermediate layers (input → out).
    assert_eq!(net.hidden_layer_count(), 0);
}

#[test]
fn test_forward_output_shape() {
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).unwrap();

    let batch = 17usize;
    let data: Vec<f32> = (0..batch * 5).map(|i| (i as f32) * 0.01).collect();
    let input = Tensor::<B, 2>::from_data(TensorData::new(data, [batch, 5]), &device);
    let output = net.forward(input);
    assert_eq!(output.dims(), [batch, 3]);
}

#[test]
fn test_forward_xyz_params_shape() {
    let cfg = ParamFieldPINNConfig::default();
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg, &device).unwrap();

    let batch = 8usize;
    let mk = |fill: f32| {
        Tensor::<B, 2>::from_data(
            TensorData::new(vec![fill; batch], [batch, 1]),
            &device,
        )
    };
    let out = net.forward_xyz_params(mk(0.1), mk(0.2), mk(0.3), mk(0.4), mk(0.5));
    assert_eq!(out.dims(), [batch, 3]);
}
