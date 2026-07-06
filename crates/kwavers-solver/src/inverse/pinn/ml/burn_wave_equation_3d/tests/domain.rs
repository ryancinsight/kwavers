//! End-to-end domain geometry tests (rectangular, spherical, cylindrical).

use super::super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;

type TestBackend = coeus_core::MoiraiBackend;

#[test]
fn test_end_to_end_rectangular_domain() -> KwaversResult<()> {
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16, 16],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed)?;

    let x_data = vec![0.5, 0.6, 0.7];
    let y_data = vec![0.5, 0.5, 0.5];
    let z_data = vec![0.5, 0.5, 0.5];
    let t_data = vec![0.1, 0.2, 0.3];
    let u_data = vec![0.0, 0.1, 0.0];

    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, None, 10)?;
    assert_eq!(metrics.epochs_completed, 10);
    assert!(metrics.training_time_secs > 0.0);

    let x_test = vec![0.5, 0.6];
    let y_test = vec![0.5, 0.5];
    let z_test = vec![0.5, 0.5];
    let t_test = vec![0.15, 0.25];

    let u_pred = solver.predict(&x_test, &y_test, &z_test, &t_test)?;
    assert_eq!(u_pred.len(), 2);
    assert!(u_pred.iter().all(|&p| p.is_finite()));
    Ok(())
}

#[test]
fn test_end_to_end_spherical_domain() -> KwaversResult<()> {
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 50,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed)?;

    let x_data = vec![0.5, 0.6];
    let y_data = vec![0.5, 0.5];
    let z_data = vec![0.5, 0.5];
    let t_data = vec![0.1, 0.2];
    let u_data = vec![0.0, 0.0];

    let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, None, 5)?;
    assert_eq!(metrics.epochs_completed, 5);
    Ok(())
}

#[test]
fn test_end_to_end_cylindrical_domain() -> KwaversResult<()> {
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 30,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::cylindrical(0.5, 0.5, 0.0, 1.0, 0.3);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed)?;

    let x_data = vec![0.5, 0.6];
    let y_data = vec![0.5, 0.5];
    let z_data = vec![0.5, 0.5];
    let t_data = vec![0.1, 0.2];
    let u_data = vec![0.0, 0.0];

    solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, None, 5)?;
    Ok(())
}
