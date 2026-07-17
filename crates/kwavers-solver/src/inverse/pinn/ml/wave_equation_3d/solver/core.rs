use coeus_autograd::Var;

use kwavers_core::error::{KwaversError, KwaversResult, SystemError, ValidationError};

use super::super::config::PinnConfig3D;
use super::super::geometry::Geometry3D;
use super::super::network::PINN3DNetwork;
use super::super::optimizer::SimpleOptimizer3D;
use super::super::wavespeed::WaveSpeedFn3D;

/// Main solver for 3D wave equation PINN
///
/// Orchestrates training and prediction by coordinating the network, optimizer,
/// geometry, and wave speed function.
pub struct PinnWave3D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Neural network for wave equation solution
    pub pinn: PINN3DNetwork<B>,
    /// Geometry definition
    pub geometry: Geometry3D,
    /// Wave speed function c(x,y,z)
    pub wave_speed_fn: Option<WaveSpeedFn3D<B>>,
    /// Simple optimizer for parameter updates
    pub optimizer: SimpleOptimizer3D,
    /// Configuration
    pub config: PinnConfig3D,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for PinnWave3D<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnWave3D")
            .field("pinn", &self.pinn)
            .field("optimizer", &self.optimizer)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave3D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new 3D PINN solver
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn new<F>(
        config: PinnConfig3D,
        geometry: Geometry3D,
        wave_speed_fn: F,
    ) -> KwaversResult<Self>
    where
        F: Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
    {
        config.validate()?;
        let pinn = PINN3DNetwork::new(&config)?;
        let optimizer = SimpleOptimizer3D::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            geometry,
            wave_speed_fn: Some(WaveSpeedFn3D::new(std::sync::Arc::new(wave_speed_fn))),
            optimizer,
            config,
        })
    }

    /// Get wave speed at a specific location
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    pub fn predict(&self, x: &[f32], y: &[f32], z: &[f32], t: &[f32]) -> KwaversResult<Vec<f32>> {
        let n = x.len();
        if n == 0 {
            return Err(KwaversError::InvalidInput(
                "Prediction inputs must be non-empty".into(),
            ));
        }
        if (y.len()) != n || (z.len()) != n || (t.len()) != n {
            return Err(KwaversError::InvalidInput(
                "x, y, z, and t must have equal length".into(),
            ));
        }

        let backend = B::default();
        let x_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], x, &backend),
            false,
        );
        let y_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], y, &backend),
            false,
        );
        let z_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], z, &backend),
            false,
        );
        let t_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n, 1], t, &backend),
            false,
        );

        let u_pred = self.pinn.forward(&x_var, &y_var, &z_var, &t_var);
        Self::extract_column_vec(&u_pred)
    }

    /// Scalar f32.
    /// # Errors
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    pub(crate) fn extract_scalar(v: &Var<f32, B>) -> KwaversResult<f32> {
        let slice = v.tensor.as_slice();
        if (slice.len()) != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "len=1".to_string(),
                    actual: format!("len={}", (slice.len())),
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
    /// - Returns [`crate::KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    pub(crate) fn extract_column_vec(v: &Var<f32, B>) -> KwaversResult<Vec<f32>> {
        let shape = v.tensor.shape();
        let [n, m] = shape else {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{shape:?}"),
                },
            ));
        };
        if *m != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{shape:?}"),
                },
            ));
        }
        let slice = v.tensor.as_slice();
        if (slice.len()) != *n {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("len={n}"),
                    actual: format!("len={}", (slice.len())),
                },
            ));
        }
        Ok(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    type TestBackend = coeus_core::MoiraiBackend;

    #[test]
    fn test_solver_creation() -> KwaversResult<()> {
        let config = PinnConfig3D::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

        assert!(solver.pinn.hidden_layer_count() > 0);
        assert_eq!(
            solver.wave_speed_fn.as_ref().unwrap().get(0.5, 0.5, 0.5),
            SOUND_SPEED_WATER_SIM as f32
        );
        Ok(())
    }

    #[test]
    fn test_solver_get_wave_speed() -> KwaversResult<()> {
        let config = PinnConfig3D::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, z: f32| {
            if z < 0.5 {
                SOUND_SPEED_WATER_SIM as f32
            } else {
                3000.0
            }
        };

        let solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

        assert_eq!(
            solver.get_wave_speed(0.5, 0.5, 0.3)?,
            SOUND_SPEED_WATER_SIM as f32
        );
        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);
        Ok(())
    }

    #[test]
    fn test_solver_prediction() -> KwaversResult<()> {
        let config = PinnConfig3D {
            hidden_layers: vec![8],
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

        let x_test = vec![0.5, 0.6];
        let y_test = vec![0.5, 0.5];
        let z_test = vec![0.5, 0.5];
        let t_test = vec![0.1, 0.2];

        let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test)?;
        assert_eq!((predictions.len()), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
        Ok(())
    }
}
