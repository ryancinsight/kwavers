use super::super::config::BurnPINNConfig;
use super::core::BurnPINN1DWave;
use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use ndarray::Array1;

type TestBackend = MoiraiBackend;

#[test]
fn test_pinn_creation() {
    let config = BurnPINNConfig::default();
    let result = BurnPINN1DWave::<TestBackend>::new(config);
    let _pinn = result.unwrap();
}

#[test]
fn test_pinn_invalid_config_empty_layers() {
    let config = BurnPINNConfig {
        hidden_layers: vec![],
        ..Default::default()
    };
    let result = BurnPINN1DWave::<TestBackend>::new(config);
    assert!(result.is_err());
}

#[test]
fn test_pinn_forward_pass() {
    let backend = TestBackend::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config).unwrap();

    let x = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[0.5f32], &backend),
        false,
    );
    let t = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![1, 1], &[0.1f32], &backend),
        false,
    );

    let u = pinn.forward(&x, &t);

    assert_eq!(u.tensor.shape(), &[1, 1]);
    let u_val = u.tensor.as_slice()[0];
    assert!(u_val.is_finite());
}

#[test]
fn test_pinn_forward_pass_batch() {
    let backend = TestBackend::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config).unwrap();

    let x = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![3, 1], &[0.0f32, 0.5, 1.0], &backend),
        false,
    );
    let t = Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![3, 1], &[0.0f32, 0.1, 0.2], &backend),
        false,
    );

    let u = pinn.forward(&x, &t);
    assert_eq!(u.tensor.shape(), &[3, 1]);

    for &val in u.tensor.as_slice() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_pinn_predict() {
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config).unwrap();

    let x = Array1::from_vec(vec![0.0, 0.5, 1.0]);
    let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

    let result = pinn.predict(&x, &t);

    let u = result.unwrap();
    assert_eq!(u.shape(), &[3, 1]);
    for &val in u.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_pinn_predict_mismatched_lengths() {
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config).unwrap();

    let x = Array1::from_vec(vec![0.0, 0.5]);
    let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

    let result = pinn.predict(&x, &t);
    assert!(result.is_err());
}
