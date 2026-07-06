//! Config defaults and collocation point generation tests.

use super::super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;

type TestBackend = coeus_core::MoiraiBackend;

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
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 2.0, 0.0, 3.0, 0.0, 4.0);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed)?;

    let (x_colloc, y_colloc, z_colloc, t_colloc) = solver.generate_collocation_points(&config);

    let n_generated = x_colloc.tensor.shape()[0];
    assert_eq!(n_generated, config.num_collocation_points);
    assert_eq!(y_colloc.tensor.shape()[0], n_generated);
    assert_eq!(z_colloc.tensor.shape()[0], n_generated);
    assert_eq!(t_colloc.tensor.shape()[0], n_generated);

    let x_data = x_colloc.tensor.as_slice();
    let y_data = y_colloc.tensor.as_slice();
    let z_data = z_colloc.tensor.as_slice();
    let t_data = t_colloc.tensor.as_slice();

    assert!(x_data.iter().all(|&x| (0.0..=2.0).contains(&x)));
    assert!(y_data.iter().all(|&y| (0.0..=3.0).contains(&y)));
    assert!(z_data.iter().all(|&z| (0.0..=4.0).contains(&z)));
    assert!(t_data.iter().all(|&t| (0.0..=1.0).contains(&t)));
    Ok(())
}

#[test]
fn test_collocation_points_spherical_filtering() -> KwaversResult<()> {
    let config = BurnPINN3DConfig {
        num_collocation_points: 100,
        ..Default::default()
    };

    let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.2);
    let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

    let solver = BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed)?;

    let (x_colloc, _y_colloc, _z_colloc, _t_colloc) = solver.generate_collocation_points(&config);

    let n_generated = x_colloc.tensor.shape()[0];
    assert!(n_generated > 0);
    assert!(n_generated < config.num_collocation_points);
    Ok(())
}
