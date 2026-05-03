//! Training metrics and prediction shape consistency tests.

use super::super::*;
use crate::core::error::KwaversResult;
use burn::backend::{Autodiff, NdArray};

type TestBackend = Autodiff<NdArray>;

#[test]
fn test_training_metrics_completeness() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 10,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    let x_data = vec![0.5];
    let y_data = vec![0.5];
    let z_data = vec![0.5];
    let t_data = vec![0.1];
    let u_data = vec![0.0];

    let epochs = 5;
    let metrics = solver.train(
        &x_data, &y_data, &z_data, &t_data, &u_data, None, &device, epochs,
    )?;

    assert_eq!(metrics.epochs_completed, epochs);
    assert_eq!(metrics.total_loss.len(), epochs);
    assert_eq!(metrics.data_loss.len(), epochs);
    assert_eq!(metrics.pde_loss.len(), epochs);
    assert_eq!(metrics.bc_loss.len(), epochs);
    assert_eq!(metrics.ic_loss.len(), epochs);
    assert!(metrics.training_time_secs > 0.0);

    assert!(metrics.total_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.data_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.pde_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.bc_loss.iter().all(|&l| l.is_finite()));
    assert!(metrics.ic_loss.iter().all(|&l| l.is_finite()));
    Ok(())
}

#[test]
fn test_prediction_shape_consistency() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

    let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    for n in [1, 5, 10, 50] {
        let x_test = vec![0.5; n];
        let y_test = vec![0.5; n];
        let z_test = vec![0.5; n];
        let t_test = vec![0.1; n];

        let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
        assert_eq!(predictions.len(), n);
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }
    Ok(())
}
