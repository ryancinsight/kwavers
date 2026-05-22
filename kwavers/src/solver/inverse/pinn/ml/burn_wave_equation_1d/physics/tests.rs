use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::solver::inverse::pinn::ml::burn_wave_equation_1d::config::BurnPINNConfig;
use burn::backend::{Autodiff, NdArray};

type TestBackend = Autodiff<NdArray<f32>>;

#[test]
fn test_pde_residual_computation() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![20, 20],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let residual = pinn.compute_pde_residual(x, t, 343.0);

    assert_eq!(residual.dims(), [1, 1]);

    let residual_val: f32 = residual.into_scalar();
    assert!(residual_val.is_finite());
}

#[test]
fn test_pde_residual_batch() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![15, 15],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.5], [1.0]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.1], [0.2]], &device);

    let residual = pinn.compute_pde_residual(x, t, 343.0);

    assert_eq!(residual.dims(), [3, 1]);

    let residual_data = residual.into_data();
    let residual_vals = residual_data.as_slice::<f32>().unwrap();
    for &val in residual_vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_physics_loss_computation() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x_data = Tensor::<TestBackend, 2>::from_floats([[0.5], [0.6]], &device);
    let t_data = Tensor::<TestBackend, 2>::from_floats([[0.1], [0.2]], &device);
    let u_data = Tensor::<TestBackend, 2>::from_floats([[0.3], [0.4]], &device);

    let x_colloc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.5], [1.0]], &device);
    let t_colloc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.1], [0.2]], &device);

    let x_bc = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);
    let t_bc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.0]], &device);
    let u_bc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.0]], &device);

    let (total, data, pde, bc) = pinn.compute_physics_loss(
        x_data,
        t_data,
        u_data,
        x_colloc,
        t_colloc,
        x_bc,
        t_bc,
        u_bc,
        343.0,
        BurnLossWeights::default(),
    );

    assert_eq!(total.dims(), [1]);
    assert_eq!(data.dims(), [1]);
    assert_eq!(pde.dims(), [1]);
    assert_eq!(bc.dims(), [1]);

    let total_val: f32 = total.into_scalar();
    let data_val: f32 = data.into_scalar();
    let pde_val: f32 = pde.into_scalar();
    let bc_val: f32 = bc.into_scalar();

    assert!(total_val.is_finite());
    assert!(data_val.is_finite());
    assert!(pde_val.is_finite());
    assert!(bc_val.is_finite());

    assert!(total_val >= 0.0);
}

#[test]
fn test_physics_loss_weighting() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x_data = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_data = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u_data = Tensor::<TestBackend, 2>::from_floats([[0.3]], &device);

    let x_colloc = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_colloc = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let x_bc = Tensor::<TestBackend, 2>::from_floats([[-1.0]], &device);
    let t_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);
    let u_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let weights_balanced = BurnLossWeights {
        data: 1.0,
        pde: 1.0,
        boundary: 1.0,
    };
    let weights_data_heavy = BurnLossWeights {
        data: 10.0,
        pde: 1.0,
        boundary: 1.0,
    };

    let (total_balanced, _, _, _) = pinn.compute_physics_loss(
        x_data.clone(),
        t_data.clone(),
        u_data.clone(),
        x_colloc.clone(),
        t_colloc.clone(),
        x_bc.clone(),
        t_bc.clone(),
        u_bc.clone(),
        343.0,
        weights_balanced,
    );

    let (total_data_heavy, _, _, _) = pinn.compute_physics_loss(
        x_data,
        t_data,
        u_data,
        x_colloc,
        t_colloc,
        x_bc,
        t_bc,
        u_bc,
        343.0,
        weights_data_heavy,
    );

    let balanced_val: f32 = total_balanced.into_scalar();
    let data_heavy_val: f32 = total_data_heavy.into_scalar();

    assert!(balanced_val.is_finite());
    assert!(data_heavy_val.is_finite());
}

#[test]
fn test_pde_residual_different_wave_speeds() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let residual_343 = pinn.compute_pde_residual(x.clone(), t.clone(), 343.0);
    let residual_1500 = pinn.compute_pde_residual(x, t, SOUND_SPEED_WATER_SIM);

    let val_343: f32 = residual_343.into_scalar();
    let val_1500: f32 = residual_1500.into_scalar();

    assert!(val_343.is_finite());
    assert!(val_1500.is_finite());
}

#[test]
fn test_loss_components_non_negative() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x_data = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_data = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u_data = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let x_colloc = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_colloc = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let x_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);
    let t_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);
    let u_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let (total, data, pde, bc) = pinn.compute_physics_loss(
        x_data,
        t_data,
        u_data,
        x_colloc,
        t_colloc,
        x_bc,
        t_bc,
        u_bc,
        343.0,
        BurnLossWeights::default(),
    );

    let total_val: f32 = total.into_scalar();
    let data_val: f32 = data.into_scalar();
    let pde_val: f32 = pde.into_scalar();
    let bc_val: f32 = bc.into_scalar();

    assert!(total_val >= 0.0);
    assert!(data_val >= 0.0);
    assert!(pde_val >= 0.0);
    assert!(bc_val >= 0.0);
}

#[test]
fn test_backward_compatibility_with_autodiff() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x_data = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_data = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);
    let u_data = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let x_colloc = Tensor::<TestBackend, 2>::from_floats([[0.5]], &device);
    let t_colloc = Tensor::<TestBackend, 2>::from_floats([[0.1]], &device);

    let x_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);
    let t_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);
    let u_bc = Tensor::<TestBackend, 2>::from_floats([[0.0]], &device);

    let (total, _, _, _) = pinn.compute_physics_loss(
        x_data,
        t_data,
        u_data,
        x_colloc,
        t_colloc,
        x_bc,
        t_bc,
        u_bc,
        343.0,
        BurnLossWeights::default(),
    );

    let grads = total.backward();
    let _ = grads;
}

#[test]
fn test_large_batch_stability() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let n = 100;
    let x_vals: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let t_vals: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32) * 0.5).collect();

    let x = Tensor::<TestBackend, 1>::from_floats(x_vals.as_slice(), &device).reshape([n, 1]);
    let t = Tensor::<TestBackend, 1>::from_floats(t_vals.as_slice(), &device).reshape([n, 1]);

    let residual = pinn.compute_pde_residual(x, t, 343.0);

    assert_eq!(residual.dims(), [n, 1]);

    let residual_data = residual.into_data();
    let residual_vals = residual_data.as_slice::<f32>().unwrap();
    for &val in residual_vals {
        assert!(val.is_finite());
    }
}

#[test]
fn test_zero_boundary_conditions() {
    let device = Default::default();
    let config = BurnPINNConfig {
        hidden_layers: vec![10, 10],
        ..Default::default()
    };
    let pinn = BurnPINN1DWave::<TestBackend>::new(config, &device).unwrap();

    let x_bc = Tensor::<TestBackend, 2>::from_floats([[-1.0], [1.0]], &device);
    let t_bc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.0]], &device);
    let u_bc = Tensor::<TestBackend, 2>::from_floats([[0.0], [0.0]], &device);

    let u_pred = pinn.forward(x_bc.clone(), t_bc.clone());
    let bc_loss = (u_pred - u_bc).powf_scalar(2.0).mean();

    let bc_val: f32 = bc_loss.into_scalar();
    assert!(bc_val.is_finite());
    assert!(bc_val >= 0.0);
}
