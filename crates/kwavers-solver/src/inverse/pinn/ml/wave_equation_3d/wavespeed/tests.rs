use super::*;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

type TestBackend = coeus_core::MoiraiBackend;

fn grid_of(
    dims: [usize; 3],
    value: f32,
    backend: &TestBackend,
) -> coeus_tensor::Tensor<f32, TestBackend> {
    let n = dims[0] * dims[1] * dims[2];
    coeus_tensor::Tensor::from_slice_on(dims.to_vec(), &vec![value; n], backend)
}

#[test]
fn test_wavespeed_creation_closure() -> KwaversResult<()> {
    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| SOUND_SPEED_WATER_SIM as f32));

    assert!(!wave_speed.has_grid());
    assert_eq!(wave_speed.get(0.0, 0.0, 0.0), SOUND_SPEED_WATER_SIM as f32);
    Ok(())
}

#[test]
fn test_wavespeed_creation_grid() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let grid = grid_of([10, 10, 10], 3000.0, &backend);

    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;

    assert!(wave_speed.has_grid());
    Ok(())
}

#[test]
fn test_wavespeed_evaluation() -> KwaversResult<()> {
    let constant =
        WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| SOUND_SPEED_WATER_SIM as f32));
    assert_eq!(constant.get(0.5, 0.5, 0.5), SOUND_SPEED_WATER_SIM as f32);
    assert_eq!(constant.get(1.0, 1.0, 1.0), SOUND_SPEED_WATER_SIM as f32);

    let layered = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, z| {
        if z < 0.5 {
            SOUND_SPEED_WATER_SIM as f32
        } else {
            3000.0
        }
    }));
    assert_eq!(layered.get(0.5, 0.5, 0.3), SOUND_SPEED_WATER_SIM as f32);
    assert_eq!(layered.get(0.5, 0.5, 0.7), 3000.0);
    Ok(())
}

#[test]
fn test_wavespeed_radial_variation() -> KwaversResult<()> {
    let radial = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, y, z| {
        let r = (x * x + y * y + z * z).sqrt();
        if r < 0.5 {
            2000.0
        } else {
            SOUND_SPEED_WATER_SIM as f32
        }
    }));

    assert_eq!(radial.get(0.0, 0.0, 0.0), 2000.0);
    assert_eq!(radial.get(1.0, 0.0, 0.0), SOUND_SPEED_WATER_SIM as f32);
    Ok(())
}

#[test]
fn test_wavespeed_clone() -> KwaversResult<()> {
    let original =
        WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| SOUND_SPEED_WATER_SIM as f32));
    let cloned = original.clone();

    assert_eq!(original.get(0.5, 0.5, 0.5), cloned.get(0.5, 0.5, 0.5));
    Ok(())
}

#[test]
fn test_wavespeed_debug_format() -> KwaversResult<()> {
    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| SOUND_SPEED_WATER_SIM as f32));
    let debug_str = format!("{:?}", wave_speed);

    assert!(debug_str.contains("WaveSpeedFn3D"));
    assert!(debug_str.contains("has_grid"));
    Ok(())
}

#[test]
fn test_wavespeed_grid_shape() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let grid = grid_of([32, 64, 128], SOUND_SPEED_WATER_SIM as f32, &backend);
    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
    assert_eq!(wave_speed.grid_dims(), Some([32, 64, 128]));
    Ok(())
}

#[test]
fn test_wavespeed_grid_trilinear_interpolation() -> KwaversResult<()> {
    let backend = TestBackend::default();
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let grid = coeus_tensor::Tensor::from_slice_on(vec![2, 2, 2], &data, &backend);
    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;

    assert!((wave_speed.get(0.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
    assert!((wave_speed.get(1.0, 1.0, 1.0) - 8.0).abs() < 1e-6);
    assert!((wave_speed.get(0.5, 0.5, 0.5) - 4.5).abs() < 1e-6);
    assert!((wave_speed.get(0.5, 0.0, 0.0) - 3.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_wavespeed_grid_invalid_bbox_rejected() -> KwaversResult<()> {
    use kwavers_core::error::{KwaversError, SystemError};
    let backend = TestBackend::default();
    let grid = grid_of([2, 2, 2], SOUND_SPEED_WATER_SIM as f32, &backend);

    let result =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
    let err = match result {
        Ok(_) => {
            return Err(KwaversError::System(SystemError::InvalidOperation {
                operation: "from_grid_with_bbox".to_string(),
                reason: "Expected invalid bbox to be rejected".to_string(),
            }));
        }
        Err(e) => e,
    };

    assert!(matches!(
        err,
        KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
            if parameter == "wave_speed_grid_bbox"
    ));
    Ok(())
}

#[test]
fn test_wavespeed_grid_invalid_values_rejected() -> KwaversResult<()> {
    use kwavers_core::error::{KwaversError, SystemError};
    let backend = TestBackend::default();
    let data: Vec<f32> = vec![
        SOUND_SPEED_WATER_SIM as f32,
        0.0,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
    ];
    let grid = coeus_tensor::Tensor::from_slice_on(vec![2, 2, 2], &data, &backend);

    let result =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let err = match result {
        Ok(_) => {
            return Err(KwaversError::System(SystemError::InvalidOperation {
                operation: "from_grid_with_bbox".to_string(),
                reason: "Expected non-positive wave speed values to be rejected".to_string(),
            }));
        }
        Err(e) => e,
    };

    assert!(matches!(
        err,
        KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
            if parameter == "wave_speed_grid"
    ));
    Ok(())
}

#[test]
fn test_wavespeed_grid_nan_values_rejected() -> KwaversResult<()> {
    use kwavers_core::error::{KwaversError, SystemError};
    let backend = TestBackend::default();
    let data: Vec<f32> = vec![
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        SOUND_SPEED_WATER_SIM as f32,
        f32::NAN,
    ];
    let grid = coeus_tensor::Tensor::from_slice_on(vec![2, 2, 2], &data, &backend);

    let result =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let err = match result {
        Ok(_) => {
            return Err(KwaversError::System(SystemError::InvalidOperation {
                operation: "from_grid_with_bbox".to_string(),
                reason: "Expected NaN wave speed values to be rejected".to_string(),
            }));
        }
        Err(e) => e,
    };

    assert!(matches!(
        err,
        KwaversError::System(SystemError::InvalidConfiguration { parameter, .. })
            if parameter == "wave_speed_grid"
    ));
    Ok(())
}

#[test]
fn test_wavespeed_complex_heterogeneity() -> KwaversResult<()> {
    let complex = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, _y, z| {
        if z < 0.3 {
            SOUND_SPEED_WATER_SIM as f32
        } else if z < 0.7 && x < 0.5 {
            2500.0
        } else if z < 0.7 {
            3500.0
        } else {
            1000.0
        }
    }));

    assert_eq!(complex.get(0.3, 0.5, 0.2), SOUND_SPEED_WATER_SIM as f32);
    assert_eq!(complex.get(0.3, 0.5, 0.5), 2500.0);
    assert_eq!(complex.get(0.7, 0.5, 0.5), 3500.0);
    assert_eq!(complex.get(0.5, 0.5, 0.8), 1000.0);
    Ok(())
}
