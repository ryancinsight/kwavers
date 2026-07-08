use super::super::config::PinnConfig3D;
use super::core::PINN3DNetwork;
use coeus_autograd::Var;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;

type TestBackend = coeus_core::MoiraiBackend;

fn var_col(backend: &TestBackend, values: &[f32]) -> Var<f32, TestBackend> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![values.len(), 1], values, backend),
        false,
    )
}

fn var_const(backend: &TestBackend, n: usize, value: f32) -> Var<f32, TestBackend> {
    var_col(backend, &vec![value; n])
}

#[test]
fn test_network_creation() -> KwaversResult<()> {
    let config = PinnConfig3D {
        hidden_layers: vec![128, 128],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config)?;

    // Verify architecture
    assert_eq!(network.hidden_layer_count(), 1); // 2 hidden dims → 1 connection
    Ok(())
}

#[test]
fn test_forward_pass() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig3D {
        hidden_layers: vec![32, 32],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config)?;

    let batch_size = 10;
    let x = var_const(&backend, batch_size, 0.0);
    let y = var_const(&backend, batch_size, 1.0);
    let z = var_const(&backend, batch_size, 0.5);
    let t = var_const(&backend, batch_size, 0.1);

    let output = network.forward(&x, &y, &z, &t);

    assert_eq!(output.tensor.shape(), &[batch_size, 1]);
    Ok(())
}

#[test]
fn test_pde_residual_shape() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig3D {
        hidden_layers: vec![16, 16],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config)?;

    let n_points = 5;
    let x = var_const(&backend, n_points, 0.0);
    let y = var_const(&backend, n_points, 0.0);
    let z = var_const(&backend, n_points, 0.0);
    let t = var_const(&backend, n_points, 0.0);

    let wave_speed = |_x: f32, _y: f32, _z: f32| Ok(SOUND_SPEED_WATER_SIM as f32);

    let residual = network.compute_pde_residual(&x, &y, &z, &t, wave_speed)?;

    assert_eq!(residual.tensor.shape(), &[n_points, 1]);
    Ok(())
}

#[test]
fn test_pde_residual_heterogeneous_medium() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig3D {
        hidden_layers: vec![16],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config)?;

    let x = var_col(&backend, &[0.25, 0.75]);
    let y = var_col(&backend, &[0.5, 0.5]);
    let z = var_col(&backend, &[0.5, 0.5]);
    let t = var_col(&backend, &[0.1, 0.1]);

    // Layered medium: different speeds in left/right halves
    let wave_speed = |x: f32, _y: f32, _z: f32| {
        Ok(if x < 0.5 {
            SOUND_SPEED_WATER_SIM as f32
        } else {
            3000.0_f32
        })
    };

    let residual = network.compute_pde_residual(&x, &y, &z, &t, wave_speed)?;

    assert_eq!(residual.tensor.shape(), &[2, 1]);
    assert!(residual.tensor.as_slice().iter().all(|&r| r.is_finite()));
    Ok(())
}

#[test]
fn test_network_forward_deterministic() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let config = PinnConfig3D {
        hidden_layers: vec![8],
        ..Default::default()
    };

    let network = PINN3DNetwork::<TestBackend>::new(&config)?;

    let x = var_const(&backend, 3, 1.0);
    let y = var_const(&backend, 3, 1.0);
    let z = var_const(&backend, 3, 1.0);
    let t = var_const(&backend, 3, 1.0);

    // Two forward passes with same input should give same output
    let output1 = network.forward(&x, &y, &z, &t);
    let output2 = network.forward(&x, &y, &z, &t);

    assert_eq!(output1.tensor.as_slice(), output2.tensor.as_slice());
    Ok(())
}
