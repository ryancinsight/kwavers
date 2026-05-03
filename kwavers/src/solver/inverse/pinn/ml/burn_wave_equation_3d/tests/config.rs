//! Config defaults and collocation point generation tests.

use super::super::*;
use crate::core::error::{KwaversError, KwaversResult, SystemError};
use burn::backend::{Autodiff, NdArray};

type TestBackend = Autodiff<NdArray>;

#[test]
fn test_config_defaults() {
    let config = BurnPINN3DConfig::default();

    assert_eq!(config.hidden_layers, vec![100, 100, 100]);
    assert_eq!(config.num_collocation_points, 10000);
    assert_eq!(config.learning_rate, 1e-4);
    assert_eq!(config.batch_size, 1000);
    assert_eq!(config.max_grad_norm, 1.0);

    let weights = config.loss_weights;
    assert_eq!(weights.data_weight, 1.0);
    assert_eq!(weights.pde_weight, 1.0);
    assert_eq!(weights.bc_weight, 1.0);
    assert_eq!(weights.ic_weight, 1.0);
}

#[test]
fn test_collocation_points_rectangular() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 2.0, 0.0, 3.0, 0.0, 4.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

    let (x_colloc, y_colloc, z_colloc, t_colloc) =
        solver.generate_collocation_points(&config, &device);

    let n_generated = x_colloc.dims()[0];
    assert_eq!(n_generated, config.num_collocation_points);
    assert_eq!(y_colloc.dims()[0], n_generated);
    assert_eq!(z_colloc.dims()[0], n_generated);
    assert_eq!(t_colloc.dims()[0], n_generated);

    let x_colloc_data = x_colloc.into_data();
    let x_data = x_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let y_colloc_data = y_colloc.into_data();
    let y_data = y_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let z_colloc_data = z_colloc.into_data();
    let z_data = z_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;
    let t_colloc_data = t_colloc.into_data();
    let t_data = t_colloc_data.as_slice::<f32>().map_err(|e| {
        KwaversError::System(SystemError::InvalidOperation {
            operation: "tensor_to_f32_slice".to_string(),
            reason: format!("{e:?}"),
        })
    })?;

    assert!(x_data.iter().all(|&x| (0.0..=2.0).contains(&x)));
    assert!(y_data.iter().all(|&y| (0.0..=3.0).contains(&y)));
    assert!(z_data.iter().all(|&z| (0.0..=4.0).contains(&z)));
    assert!(t_data.iter().all(|&t| (0.0..=1.0).contains(&t)));
    Ok(())
}

#[test]
fn test_collocation_points_spherical_filtering() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.2);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

    let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
        solver.generate_collocation_points(&config, &device);

    let n_generated = x_colloc.dims()[0];
    assert!(n_generated > 0);
    assert!(n_generated < config.num_collocation_points);
    Ok(())
}
