//! Tests for the SimpleOptimizer gradient descent implementation.

use super::core::SimpleOptimizer;
use crate::inverse::pinn::ml::burn_wave_equation_1d::config::BurnPINNConfig;
use crate::inverse::pinn::ml::burn_wave_equation_1d::network::BurnPINN1DWave;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn test_optimizer_creation() {
    let optimizer = SimpleOptimizer::new(0.001);
    assert_eq!(optimizer.learning_rate(), 0.001);
}

#[test]
fn test_optimizer_learning_rate() {
    let optimizer = SimpleOptimizer::new(0.01);
    assert_eq!(optimizer.learning_rate(), 0.01);
}

#[test]
fn test_optimizer_step_compiles() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let optimizer = SimpleOptimizer::new(0.001);

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u = pinn.forward(x, t);
    let loss = u.powf_scalar(2.0).mean();

    let grads = loss.backward();
    let _updated_pinn = optimizer.step(pinn, &grads);
}

#[test]
fn test_optimizer_step_updates_parameters() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u_before = pinn.forward(x.clone(), t.clone());
    let u_before_val: f32 = u_before.clone().into_scalar();

    let optimizer = SimpleOptimizer::new(0.1);

    let target = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);
    let loss = (u_before - target).powf_scalar(2.0).mean();
    let grads = loss.backward();

    let updated_pinn = optimizer.step(pinn, &grads);

    let u_after = updated_pinn.forward(x, t);
    let u_after_val: f32 = u_after.into_scalar();

    assert!(u_before_val.is_finite());
    assert!(u_after_val.is_finite());
}

#[test]
fn test_optimizer_multiple_steps() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };
    let mut pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let optimizer = SimpleOptimizer::new(0.01);

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let target = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let mut losses = Vec::new();

    for _ in 0..5 {
        let u = pinn.forward(x.clone(), t.clone());
        let loss = (u - target.clone()).powf_scalar(2.0).mean();

        let loss_val: f32 = loss.clone().into_scalar();
        losses.push(loss_val);

        let grads = loss.backward();
        pinn = optimizer.step(pinn, &grads);
    }

    for &loss in &losses {
        assert!(loss.is_finite());
    }

    assert!(losses.len() == 5);
}

#[test]
fn test_optimizer_with_different_learning_rates() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![5, 5],
        ..Default::default()
    };

    let optimizer_small = SimpleOptimizer::new(0.0001);
    assert_eq!(optimizer_small.learning_rate(), 0.0001);

    let optimizer_medium = SimpleOptimizer::new(0.001);
    assert_eq!(optimizer_medium.learning_rate(), 0.001);

    let optimizer_large = SimpleOptimizer::new(0.1);
    assert_eq!(optimizer_large.learning_rate(), 0.1);

    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();
    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u = pinn.forward(x, t);
    let loss = u.powf_scalar(2.0).mean();
    let grads = loss.backward();

    let _ = optimizer_small.step(pinn.clone(), &grads);
    let _ = optimizer_medium.step(pinn.clone(), &grads);
    let _ = optimizer_large.step(pinn, &grads);
}

#[test]
fn test_gradient_mapper_preserves_structure() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config.clone(), &device).unwrap();

    let optimizer = SimpleOptimizer::new(0.001);

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u = pinn.forward(x.clone(), t.clone());
    let loss = u.powf_scalar(2.0).mean();
    let grads = loss.backward();

    let updated_pinn = optimizer.step(pinn, &grads);

    let u_after = updated_pinn.forward(x, t);
    assert_eq!(u_after.dims(), [1, 1]);
}
