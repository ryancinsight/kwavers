//! Architecture construction and forward-pass shape tests.

use coeus_autograd::Var;

use super::super::config::ParamFieldPINNConfig;
use super::super::network::ParamFieldPINNNetwork;
use super::B;

#[test]
fn test_network_construction_with_default_config() {
    let cfg = ParamFieldPINNConfig::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg).expect("construct");
    // Default has 3 hidden layers → 2 intermediate Linear layers.
    assert_eq!(net.hidden_layer_count(), cfg.hidden_layers.len() - 1);
}

#[test]
fn test_network_construction_with_minimal_hidden_stack() {
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![32],
        ..ParamFieldPINNConfig::default()
    };
    let net = ParamFieldPINNNetwork::<B>::new(&cfg).expect("construct");
    // Single hidden layer → zero intermediate layers (input → out).
    assert_eq!(net.hidden_layer_count(), 0);
}

#[test]
fn test_forward_output_shape() {
    let backend = B::default();
    let cfg = ParamFieldPINNConfig::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg).unwrap();

    let batch = 17usize;
    let data: Vec<f32> = (0..batch * 5).map(|i| (i as f32) * 0.01).collect();
    let input = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![batch, 5], &data, &backend),
        false,
    );
    let output = net.forward(&input);
    assert_eq!(output.tensor.shape(), &[batch, 3]);
}

#[test]
fn test_forward_xyz_params_shape() {
    let backend = B::default();
    let cfg = ParamFieldPINNConfig::default();
    let net = ParamFieldPINNNetwork::<B>::new(&cfg).unwrap();

    let batch = 8usize;
    let mk = |fill: f32| {
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch, 1], &vec![fill; batch], &backend),
            false,
        )
    };
    let out = net.forward_xyz_params(&mk(0.1), &mk(0.2), &mk(0.3), &mk(0.4), &mk(0.5));
    assert_eq!(out.tensor.shape(), &[batch, 3]);
}
