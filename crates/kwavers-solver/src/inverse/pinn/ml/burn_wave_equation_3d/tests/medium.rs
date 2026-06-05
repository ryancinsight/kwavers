//! Heterogeneous and radially varying medium tests.

use super::super::*;
use burn::backend::{Autodiff, NdArray};
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::KwaversResult;

type TestBackend = Autodiff<NdArray>;

#[test]
fn test_heterogeneous_layered_medium() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![16],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    let wave_speed = |_x: f32, _y: f32, z: f32| {
        if z < 0.5 {
            SOUND_SPEED_WATER_SIM as f32
        } else {
            3000.0
        }
    };

    let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    assert_eq!(
        solver.get_wave_speed(0.5, 0.5, 0.3)?,
        SOUND_SPEED_WATER_SIM as f32
    );
    assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);

    let x_data = vec![0.5, 0.5, 0.5, 0.5];
    let y_data = vec![0.5, 0.5, 0.5, 0.5];
    let z_data = vec![0.3, 0.4, 0.6, 0.7];
    let t_data = vec![0.1, 0.2, 0.1, 0.2];
    let u_data = vec![0.0, 0.0, 0.0, 0.0];

    let metrics = solver.train(
        &x_data, &y_data, &z_data, &t_data, &u_data, None, &device, 10,
    )?;
    assert_eq!(metrics.epochs_completed, 10);
    Ok(())
}

#[test]
fn test_radially_varying_medium() -> KwaversResult<()> {
    let device = Default::default();
    let config = BurnPINN3DConfig {
        hidden_layers: vec![8],
        num_collocation_points: 20,
        learning_rate: 1e-2,
        ..Default::default()
    };

    let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    let wave_speed = |x: f32, y: f32, z: f32| {
        let dx = x - 0.5;
        let dy = y - 0.5;
        let dz = z - 0.5;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r < 0.3 {
            2500.0
        } else {
            SOUND_SPEED_WATER_SIM as f32
        }
    };

    let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

    assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.5)?, 2500.0);
    assert_eq!(
        solver.get_wave_speed(0.9, 0.5, 0.5)?,
        SOUND_SPEED_WATER_SIM as f32
    );
    Ok(())
}
