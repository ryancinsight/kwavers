use super::*;
use burn::backend::NdArray;

type TestBackend = NdArray<f32>;

#[test]
fn test_wavespeed_creation_closure() -> KwaversResult<()> {
    let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));

    assert!(!wave_speed.has_grid());
    assert_eq!(wave_speed.get(0.0, 0.0, 0.0), 1500.0);
    Ok(())
}

#[test]
fn test_wavespeed_creation_grid() -> KwaversResult<()> {
    let device = Default::default();
    let grid = Tensor::<TestBackend, 3>::ones([10, 10, 10], &device).mul_scalar(3000.0);

    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;

    assert!(wave_speed.has_grid());
    Ok(())
}

#[test]
fn test_wavespeed_evaluation() -> KwaversResult<()> {
    let constant = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
    assert_eq!(constant.get(0.5, 0.5, 0.5), 1500.0);
    assert_eq!(constant.get(1.0, 1.0, 1.0), 1500.0);

    let layered =
        WaveSpeedFn3D::<TestBackend>::new(Arc::new(
            |_x, _y, z| {
                if z < 0.5 {
                    1500.0
                } else {
                    3000.0
                }
            },
        ));
    assert_eq!(layered.get(0.5, 0.5, 0.3), 1500.0);
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
            1500.0
        }
    }));

    assert_eq!(radial.get(0.0, 0.0, 0.0), 2000.0);
    assert_eq!(radial.get(1.0, 0.0, 0.0), 1500.0);
    Ok(())
}

#[test]
fn test_wavespeed_device_migration() -> KwaversResult<()> {
    let device1 = Default::default();
    let grid1 = Tensor::<TestBackend, 3>::ones([5, 5, 5], &device1);
    let wave_speed1 =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid1, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;

    let device2 = Default::default();
    let wave_speed2 = wave_speed1.to_device(&device2);

    assert!(wave_speed2.has_grid());
    Ok(())
}

#[test]
fn test_wavespeed_clone() -> KwaversResult<()> {
    let original = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
    let cloned = original.clone();

    assert_eq!(original.get(0.5, 0.5, 0.5), cloned.get(0.5, 0.5, 0.5));
    Ok(())
}

#[test]
fn test_wavespeed_debug_format() -> KwaversResult<()> {
    let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
    let debug_str = format!("{:?}", wave_speed);

    assert!(debug_str.contains("WaveSpeedFn3D"));
    assert!(debug_str.contains("has_grid"));
    Ok(())
}

#[test]
fn test_wavespeed_grid_shape() -> KwaversResult<()> {
    let device = Default::default();
    let grid = Tensor::<TestBackend, 3>::ones([32, 64, 128], &device).mul_scalar(1500.0);
    let wave_speed =
        WaveSpeedFn3D::<TestBackend>::from_grid_with_bbox(grid, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])?;
    assert_eq!(wave_speed.grid_dims(), Some([32, 64, 128]));
    Ok(())
}

#[test]
fn test_wavespeed_grid_trilinear_interpolation() -> KwaversResult<()> {
    let device = Default::default();
    let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let grid = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data, [2, 2, 2]),
        &device,
    );
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
    use crate::core::error::{KwaversError, SystemError};
    let device = Default::default();
    let grid = Tensor::<TestBackend, 3>::ones([2, 2, 2], &device);

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
    use crate::core::error::{KwaversError, SystemError};
    let device = Default::default();
    let data: Vec<f32> = vec![1500.0, 0.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0];
    let grid = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data, [2, 2, 2]),
        &device,
    );

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
    use crate::core::error::{KwaversError, SystemError};
    let device = Default::default();
    let data: Vec<f32> = vec![
        1500.0,
        1500.0,
        1500.0,
        1500.0,
        1500.0,
        1500.0,
        1500.0,
        f32::NAN,
    ];
    let grid = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data, [2, 2, 2]),
        &device,
    );

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
            1500.0
        } else if z < 0.7 && x < 0.5 {
            2500.0
        } else if z < 0.7 {
            3500.0
        } else {
            1000.0
        }
    }));

    assert_eq!(complex.get(0.3, 0.5, 0.2), 1500.0);
    assert_eq!(complex.get(0.3, 0.5, 0.5), 2500.0);
    assert_eq!(complex.get(0.7, 0.5, 0.5), 3500.0);
    assert_eq!(complex.get(0.5, 0.5, 0.8), 1000.0);
    Ok(())
}
