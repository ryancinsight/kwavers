use burn::module::{Ignored, Module};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};

use super::super::config::BurnPINN3DConfig;
use super::super::geometry::Geometry3D;
use super::super::network::PINN3DNetwork;
use super::super::optimizer::SimpleOptimizer3D;
use super::super::wavespeed::WaveSpeedFn3D;
/// Main solver for 3D wave equation PINN
///
/// Orchestrates training and prediction by coordinating the network, optimizer,
/// geometry, and wave speed function.
///
/// # Type Parameters
///
/// * `B` - Backend type (e.g., NdArray, Autodiff<NdArray>, WGPU)
///
/// # Fields
///
/// * `pinn` - Neural network module
/// * `geometry` - Domain geometry (rectangular, spherical, cylindrical)
/// * `wave_speed_fn` - Wave speed function c(x, y, z)
/// * `optimizer` - Simple SGD optimizer
/// * `config` - Training configuration
#[derive(Module, Debug)]
pub struct BurnPINN3DWave<B: Backend> {
    /// Neural network for wave equation solution
    pub pinn: PINN3DNetwork<B>,
    /// Geometry definition (wrapped in Ignored for Module trait)
    pub geometry: Ignored<Geometry3D>,
    /// Wave speed function c(x,y,z)
    pub wave_speed_fn: Option<WaveSpeedFn3D<B>>,
    /// Simple optimizer for parameter updates
    pub optimizer: Ignored<SimpleOptimizer3D>,
    /// Configuration (wrapped in Ignored)
    pub config: Ignored<BurnPINN3DConfig>,
    /// Backend type marker
    _backend: PhantomData<B>,
}

impl<B: Backend> BurnPINN3DWave<B> {
    /// Create a new 3D PINN solver
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration (hidden layers, learning rate, etc.)
    /// * `geometry` - Domain geometry
    /// * `wave_speed_fn` - Function c(x, y, z) returning wave speed
    /// * `device` - Target device for network parameters
    ///
    /// # Returns
    ///
    /// A new `BurnPINN3DWave` solver instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::{
    ///     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
    /// };
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let config = BurnPINN3DConfig::default();
    /// let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    /// let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;
    ///
    /// let solver = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
    /// ```
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new<F>(
        config: BurnPINN3DConfig,
        geometry: Geometry3D,
        wave_speed_fn: F,
        device: &B::Device,
    ) -> KwaversResult<Self>
    where
        F: Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
    {
        config.validate()?;
        let pinn = PINN3DNetwork::new(&config, device)?;
        let optimizer = SimpleOptimizer3D::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            geometry: Ignored(geometry),
            wave_speed_fn: Some(WaveSpeedFn3D::new(std::sync::Arc::new(wave_speed_fn))),
            optimizer: Ignored(optimizer),
            config: Ignored(config),
            _backend: PhantomData,
        })
    }

    /// Get wave speed at a specific location
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate (meters)
    /// * `y` - Y-coordinate (meters)
    /// * `z` - Z-coordinate (meters)
    ///
    /// # Returns
    ///
    /// Wave speed c(x, y, z) in m/s
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn get_wave_speed(&self, x: f32, y: f32, z: f32) -> KwaversResult<f32> {
        let wave_speed = self
            .wave_speed_fn
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Wave speed function is missing".into()))?
            .get(x, y, z);
        if !wave_speed.is_finite() || wave_speed <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "wave_speed".to_string(),
                value: wave_speed as f64,
                reason: "must be finite and > 0".to_string(),
            }));
        }
        Ok(wave_speed)
    }

    /// Make predictions at new points
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinates for prediction
    /// * `y` - Y-coordinates for prediction
    /// * `z` - Z-coordinates for prediction
    /// * `t` - Time coordinates for prediction
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// Predicted displacement/pressure values u(x, y, z, t)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let x_test = vec![0.5, 0.6];
    /// let y_test = vec![0.5, 0.5];
    /// let z_test = vec![0.5, 0.5];
    /// let t_test = vec![0.5, 0.5];
    ///
    /// let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
    /// ```
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn predict(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        t: &[f32],
        device: &B::Device,
    ) -> KwaversResult<Vec<f32>> {
        let n = x.len();
        if n == 0 {
            return Err(KwaversError::InvalidInput(
                "Prediction inputs must be non-empty".into(),
            ));
        }
        if y.len() != n || z.len() != n || t.len() != n {
            return Err(KwaversError::InvalidInput(
                "x, y, z, and t must have equal length".into(),
            ));
        }

        let x_tensor = Tensor::<B, 2>::from_data(TensorData::new(x.to_vec(), [n, 1]), device);
        let y_tensor = Tensor::<B, 2>::from_data(TensorData::new(y.to_vec(), [n, 1]), device);
        let z_tensor = Tensor::<B, 2>::from_data(TensorData::new(z.to_vec(), [n, 1]), device);
        let t_tensor = Tensor::<B, 2>::from_data(TensorData::new(t.to_vec(), [n, 1]), device);

        let u_pred = self.pinn.forward(x_tensor, y_tensor, z_tensor, t_tensor);
        let u_vec = Self::extract_column_vec(&u_pred)?;

        Ok(u_vec)
    }
    /// Scalar f32.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn extract_scalar(t: &Tensor<B, 1>) -> KwaversResult<f32> {
        let data = t.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            })
        })?;
        if slice.len() != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "len=1".to_string(),
                    actual: format!("len={}", slice.len()),
                },
            ));
        }
        slice.first().copied().ok_or_else(|| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_scalar_extract".to_string(),
                reason: "missing scalar element".to_string(),
            })
        })
    }
    /// Tensor column vec f32.
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn extract_column_vec(t: &Tensor<B, 2>) -> KwaversResult<Vec<f32>> {
        let shape = t.shape();
        let dims = shape.dims.as_slice();
        let [n, m] = dims else {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{dims:?}"),
                },
            ));
        };
        if *m != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{dims:?}"),
                },
            ));
        }
        let data = t.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            })
        })?;
        if slice.len() != *n {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("len={n}"),
                    actual: format!("len={}", slice.len()),
                },
            ));
        }
        Ok(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use burn::backend::{Autodiff, NdArray};
    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_solver_creation() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        assert!(solver.pinn.hidden_layer_count() > 0);
        // Verify the stored function returns the expected constant speed
        assert_eq!(
            solver.wave_speed_fn.as_ref().unwrap().get(0.5, 0.5, 0.5),
            SOUND_SPEED_WATER_SIM as f32
        );
        Ok(())
    }

    #[test]
    fn test_solver_get_wave_speed() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, z: f32| {
            if z < 0.5 {
                SOUND_SPEED_WATER_SIM as f32
            } else {
                3000.0
            }
        };

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        assert_eq!(
            solver.get_wave_speed(0.5, 0.5, 0.3)?,
            SOUND_SPEED_WATER_SIM as f32
        );
        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);
        Ok(())
    }

    #[test]
    fn test_solver_prediction() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_test = vec![0.5, 0.6];
        let y_test = vec![0.5, 0.5];
        let z_test = vec![0.5, 0.5];
        let t_test = vec![0.1, 0.2];

        let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
        Ok(())
    }
}
