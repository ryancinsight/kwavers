use super::*;
use crate::inverse::pinn::ml::wave_equation_1d::config::PinnConfig;
use coeus_core::MoiraiBackend;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

type TestBackend = MoiraiBackend;

fn var2(vals: &[f32], backend: &TestBackend) -> Var<f32, TestBackend> {
    let n = (vals.shape()[0] * vals.shape()[1] * vals.shape()[2]);
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![n, 1], vals, backend),
        false,
    )
}

#[test]
fn test_pde_residual_computation() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![20, 20],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x = var2(&[0.5], &backend);
    let t = var2(&[0.1], &backend);

    let residual = pinn.compute_pde_residual(&x, &t, 343.0);

    assert_eq!(residual.tensor.shape(), &[1, 1]);
    assert!(residual.tensor.as_slice()[0].is_finite());
}

#[test]
fn test_pde_residual_batch() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![15, 15],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x = var2(&[0.0, 0.5, 1.0], &backend);
    let t = var2(&[0.0, 0.1, 0.2], &backend);

    let residual = pinn.compute_pde_residual(&x, &t, 343.0);

    assert_eq!(residual.tensor.shape(), &[3, 1]);
    for &val in residual.tensor.as_slice() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_physics_loss_computation() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x_data = var2(&[0.5, 0.6], &backend);
    let t_data = var2(&[0.1, 0.2], &backend);
    let u_data = var2(&[0.3, 0.4], &backend);

    let x_colloc = var2(&[0.0, 0.5, 1.0], &backend);
    let t_colloc = var2(&[0.0, 0.1, 0.2], &backend);

    let x_bc = var2(&[-1.0, 1.0], &backend);
    let t_bc = var2(&[0.0, 0.0], &backend);
    let u_bc = var2(&[0.0, 0.0], &backend);

    let (total, data, pde, bc) = pinn.compute_physics_loss(
        &x_data,
        &t_data,
        &u_data,
        &x_colloc,
        &t_colloc,
        &x_bc,
        &t_bc,
        &u_bc,
        343.0,
        LossWeights::default(),
    );

    assert_eq!(total.tensor.shape(), &[1]);
    assert_eq!(data.tensor.shape(), &[1]);
    assert_eq!(pde.tensor.shape(), &[1]);
    assert_eq!(bc.tensor.shape(), &[1]);

    let total_val = total.tensor.as_slice()[0];
    let data_val = data.tensor.as_slice()[0];
    let pde_val = pde.tensor.as_slice()[0];
    let bc_val = bc.tensor.as_slice()[0];

    assert!(total_val.is_finite());
    assert!(data_val.is_finite());
    assert!(pde_val.is_finite());
    assert!(bc_val.is_finite());

    assert!(total_val >= 0.0);
}

#[test]
fn test_physics_loss_weighting() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x_data = var2(&[0.5], &backend);
    let t_data = var2(&[0.1], &backend);
    let u_data = var2(&[0.3], &backend);

    let x_colloc = var2(&[0.5], &backend);
    let t_colloc = var2(&[0.1], &backend);

    let x_bc = var2(&[-1.0], &backend);
    let t_bc = var2(&[0.0], &backend);
    let u_bc = var2(&[0.0], &backend);

    let weights_balanced = LossWeights {
        data: 1.0,
        pde: 1.0,
        boundary: 1.0,
    };
    let weights_data_heavy = LossWeights {
        data: 10.0,
        pde: 1.0,
        boundary: 1.0,
    };

    let (total_balanced, _, _, _) = pinn.compute_physics_loss(
        &x_data,
        &t_data,
        &u_data,
        &x_colloc,
        &t_colloc,
        &x_bc,
        &t_bc,
        &u_bc,
        343.0,
        weights_balanced,
    );

    let (total_data_heavy, _, _, _) = pinn.compute_physics_loss(
        &x_data,
        &t_data,
        &u_data,
        &x_colloc,
        &t_colloc,
        &x_bc,
        &t_bc,
        &u_bc,
        343.0,
        weights_data_heavy,
    );

    let balanced_val = total_balanced.tensor.as_slice()[0];
    let data_heavy_val = total_data_heavy.tensor.as_slice()[0];

    assert!(balanced_val.is_finite());
    assert!(data_heavy_val.is_finite());
}

#[test]
fn test_pde_residual_different_wave_speeds() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x = var2(&[0.5], &backend);
    let t = var2(&[0.1], &backend);

    let residual_343 = pinn.compute_pde_residual(&x, &t, 343.0);
    let residual_1500 = pinn.compute_pde_residual(&x, &t, SOUND_SPEED_WATER_SIM);

    assert!(residual_343.tensor.as_slice()[0].is_finite());
    assert!(residual_1500.tensor.as_slice()[0].is_finite());
}

#[test]
fn test_loss_components_non_negative() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x_data = var2(&[0.5], &backend);
    let t_data = var2(&[0.1], &backend);
    let u_data = var2(&[0.0], &backend);

    let x_colloc = var2(&[0.5], &backend);
    let t_colloc = var2(&[0.1], &backend);

    let x_bc = var2(&[0.0], &backend);
    let t_bc = var2(&[0.0], &backend);
    let u_bc = var2(&[0.0], &backend);

    let (total, data, pde, bc) = pinn.compute_physics_loss(
        &x_data,
        &t_data,
        &u_data,
        &x_colloc,
        &t_colloc,
        &x_bc,
        &t_bc,
        &u_bc,
        343.0,
        LossWeights::default(),
    );

    assert!(total.tensor.as_slice()[0] >= 0.0);
    assert!(data.tensor.as_slice()[0] >= 0.0);
    assert!(pde.tensor.as_slice()[0] >= 0.0);
    assert!(bc.tensor.as_slice()[0] >= 0.0);
}

#[test]
fn test_backward_compatibility_with_autodiff() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x_data = var2(&[0.5], &backend);
    let t_data = var2(&[0.1], &backend);
    let u_data = var2(&[0.0], &backend);

    let x_colloc = var2(&[0.5], &backend);
    let t_colloc = var2(&[0.1], &backend);

    let x_bc = var2(&[0.0], &backend);
    let t_bc = var2(&[0.0], &backend);
    let u_bc = var2(&[0.0], &backend);

    let (total, _, _, _) = pinn.compute_physics_loss(
        &x_data,
        &t_data,
        &u_data,
        &x_colloc,
        &t_colloc,
        &x_bc,
        &t_bc,
        &u_bc,
        343.0,
        LossWeights::default(),
    );

    total.backward();
}

#[test]
fn test_large_batch_stability() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let n = 100;
    let x_vals: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let t_vals: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32) * 0.5).collect();

    let x = var2(&x_vals, &backend);
    let t = var2(&t_vals, &backend);

    let residual = pinn.compute_pde_residual(&x, &t, 343.0);

    assert_eq!(residual.tensor.shape(), &[n, 1]);
    for &val in residual.tensor.as_slice() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_zero_boundary_conditions() {
    let backend = TestBackend::default();
    let config = PinnConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = PinnWave1D::<TestBackend>::new(config).unwrap();

    let x_bc = var2(&[-1.0, 1.0], &backend);
    let t_bc = var2(&[0.0, 0.0], &backend);
    let u_bc = var2(&[0.0, 0.0], &backend);

    let u_pred = pinn.forward(&x_bc, &t_bc);
    let diff = coeus_autograd::sub(&u_pred, &u_bc);
    let bc_loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));

    let bc_val = bc_loss.tensor.as_slice()[0];
    assert!(bc_val.is_finite());
    assert!(bc_val >= 0.0);
}
