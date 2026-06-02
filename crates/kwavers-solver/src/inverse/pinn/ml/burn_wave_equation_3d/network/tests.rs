use super::super::config::BurnPINN3DConfig;
use super::core::PINN3DNetwork;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

type TestBackend = NdArray<f32>;

#[test]
fn test_network_creation() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![128, 128],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

    // Verify architecture
    assert_eq!(network.hidden_layer_count(), 1); // 2 hidden dims → 1 connection
    Ok(())
}

#[test]
fn test_forward_pass() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![32, 32],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

    // Create batch of 10 points
    let batch_size = 10;
    let x = Tensor::<TestBackend, 2>::zeros([batch_size, 1], &device);
    let y = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device);
    let z = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device).mul_scalar(0.5);
    let t = Tensor::<TestBackend, 2>::ones([batch_size, 1], &device).mul_scalar(0.1);

    let output = network.forward(x, y, z, t);

    // Verify output shape
    assert_eq!(output.shape().dims, [batch_size, 1]);
    Ok(())
}

#[test]
fn test_pde_residual_shape() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16, 16],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

    // Create collocation points
    let n_points = 5;
    let x = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
    let y = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
    let z = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);
    let t = Tensor::<TestBackend, 2>::zeros([n_points, 1], &device);

    // Constant wave speed
    let wave_speed = |_x: f32, _y: f32, _z: f32| Ok(SOUND_SPEED_WATER_SIM as f32);

    let residual = network.compute_pde_residual(x, y, z, t, wave_speed)?;

    // Verify residual shape matches input
    assert_eq!(residual.shape().dims, [n_points, 1]);
    Ok(())
}

#[test]
fn test_pde_residual_heterogeneous_medium() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

    // Create points in two different regions
    let x_data = vec![0.25, 0.75]; // Left and right regions
    let y_data = vec![0.5, 0.5];
    let z_data = vec![0.5, 0.5];
    let t_data = vec![0.1, 0.1];

    let x = Tensor::<TestBackend, 1>::from_data(TensorData::from(x_data.as_slice()), &device)
        .unsqueeze_dim(1);
    let y = Tensor::<TestBackend, 1>::from_data(TensorData::from(y_data.as_slice()), &device)
        .unsqueeze_dim(1);
    let z = Tensor::<TestBackend, 1>::from_data(TensorData::from(z_data.as_slice()), &device)
        .unsqueeze_dim(1);
    let t = Tensor::<TestBackend, 1>::from_data(TensorData::from(t_data.as_slice()), &device)
        .unsqueeze_dim(1);

    // Layered medium: different speeds in left/right halves
    let wave_speed = |x: f32, _y: f32, _z: f32| {
        Ok(if x < 0.5 {
            SOUND_SPEED_WATER_SIM as f32
        } else {
            3000.0_f32
        })
    };

    let residual = network.compute_pde_residual(x, y, z, t, wave_speed)?;

    // Verify residual is computed (non-trivial)
    assert_eq!(residual.shape().dims, [2, 1]);
    let residual_data = residual.into_data();
    let residual_data = residual_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    assert!(residual_data.iter().all(|&r| r.is_finite()));
    Ok(())
}

#[test]
fn test_network_forward_deterministic() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config, &device)?;

    let x = Tensor::<TestBackend, 2>::ones([3, 1], &device);
    let y = Tensor::<TestBackend, 2>::ones([3, 1], &device);
    let z = Tensor::<TestBackend, 2>::ones([3, 1], &device);
    let t = Tensor::<TestBackend, 2>::ones([3, 1], &device);

    // Two forward passes with same input should give same output
    let output1 = network.forward(x.clone(), y.clone(), z.clone(), t.clone());
    let output2 = network.forward(x, y, z, t);

    let output1_data = output1.into_data();
    let data1 = output1_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let output2_data = output2.into_data();
    let data2 = output2_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;

    assert_eq!(data1, data2);
    Ok(())
}
