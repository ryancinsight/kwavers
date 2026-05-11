use super::super::config::BurnPINNConfig;
use super::core::BurnPINN1DWave;
use burn::backend::NdArray;
use burn::tensor::Tensor;
use ndarray::Array1;

type TestBackend = NdArray<f32>;

#[test]
fn test_pinn_creation() {
    let device = Default::default();
    let config = BurnPINNConfig::default();
    let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
    let _pinn = result.unwrap();
}

#[test]
fn test_pinn_invalid_config_empty_layers() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![],
        ..Default::default()
    };
    let result = BurnPINN1DWave::<TestBackend>::new(config, &device);
    assert!(result.is_err());
}

#[test]
fn test_pinn_forward_pass() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 1>::from_floats([0.5], &device).reshape([1, 1]);
    let t = Tensor::<TestBackend, 1>::from_floats([0.1], &device).reshape([1, 1]);

    let u = pinn.forward(x, t);

    assert_eq!(u.dims(), [1, 1]);
    let u_val = u.to_data().as_slice::<f32>().unwrap()[0];
    assert!(u_val.is_finite());
}

#[test]
fn test_pinn_forward_pass_batch() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, 1.0], &device).reshape([3, 1]);
    let t = Tensor::<TestBackend, 1>::from_floats([0.0, 0.1, 0.2], &device).reshape([3, 1]);

    let u = pinn.forward(x, t);
    assert_eq!(u.dims(), [3, 1]);

    let binding = u.to_data();
    let u_vals = binding.as_slice::<f32>().unwrap();
    for &val in u_vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_pinn_predict() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Array1::from_vec(vec![0.0, 0.5, 1.0]);
    let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

    let result = pinn.predict(&x, &t, &device);

    let u = result.unwrap();
    assert_eq!(u.shape(), &[3, 1]);
    for &val in u.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_pinn_predict_mismatched_lengths() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Array1::from_vec(vec![0.0, 0.5]);
    let t = Array1::from_vec(vec![0.0, 0.1, 0.2]);

    let result = pinn.predict(&x, &t, &device);
    assert!(result.is_err());
}

#[test]
fn test_pinn_device() {
    let device = Default::default();
    let config = BurnPINNConfig::default();
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();
    let _ = pinn.device();
}

#[cfg(feature = "pinn-gpu")]
mod gpu_tests {
    use crate::solver::inverse::pinn::ml::burn_wave_equation_1d::{BurnPINN1DWave, BurnPINNConfig};
    use burn::backend::{Autodiff, Wgpu};
    use burn::tensor::Tensor;

    type GpuBackend = Autodiff<Wgpu<f32>>;

    #[test]
    fn test_pinn_gpu_creation() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![20, 20],
            ..Default::default()
        };
        let result = BurnPINN1DWave::<GpuBackend>::new(config, &device);
        let _ = result;
    }

    #[test]
    fn test_pinn_gpu_forward_pass() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        if let Ok(pinn) = BurnPINN1DWave::<GpuBackend>::new(config, &device) {
            let x = Tensor::<GpuBackend, 1>::from_floats([0.5], &device).reshape([1, 1]);
            let t = Tensor::<GpuBackend, 1>::from_floats([0.1], &device).reshape([1, 1]);

            let u = pinn.forward(x, t);
            assert!(u.to_data().as_slice::<f32>().is_ok());
        }
    }
}
