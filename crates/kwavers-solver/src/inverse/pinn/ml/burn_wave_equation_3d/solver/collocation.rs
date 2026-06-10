use super::core::BurnPINN3DWave;
use crate::inverse::pinn::ml::burn_wave_equation_3d::config::BurnPINN3DConfig;
use burn::tensor::{backend::Backend, Tensor, TensorData};
// Used only by the test module's reference wave-speed closure.
#[cfg(test)]
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

impl<B: Backend> BurnPINN3DWave<B> {
    /// Generate collocation points for PDE residual computation
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying number of collocation points
    /// * `device` - Target device for tensors
    ///
    /// # Returns
    ///
    /// Tuple: (x_colloc, y_colloc, z_colloc, t_colloc) as tensors [n_points, 1]
    ///
    /// # Algorithm
    ///
    /// 1. Get bounding box from geometry
    /// 2. Generate random points in bounding box
    /// 3. Filter points to those inside geometry (for complex shapes)
    /// 4. Convert to tensors
    ///
    /// # Notes
    ///
    /// - Time domain: [0, 1] (normalized)
    /// - Spatial domain: From geometry bounding box
    /// - Points may be fewer than requested if geometry is complex
    pub(crate) fn generate_collocation_points(
        &self,
        config: &BurnPINN3DConfig,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n_points = config.num_collocation_points;
        let mut x_points = Vec::with_capacity(n_points);
        let mut y_points = Vec::with_capacity(n_points);
        let mut z_points = Vec::with_capacity(n_points);
        let mut t_points = Vec::with_capacity(n_points);

        let (x_min, x_max, y_min, y_max, z_min, z_max) = self.geometry.0.bounding_box();
        let t_max = 1.0; // Normalized time

        // Generate random points within geometry
        for _ in 0..n_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            let z = z_min + (z_max - z_min) * rand::random::<f64>();
            let t = t_max * rand::random::<f64>();

            // Check if point is inside geometry (for complex shapes)
            if self.geometry.0.contains(x, y, z) {
                x_points.push(x as f32);
                y_points.push(y as f32);
                z_points.push(z as f32);
                t_points.push(t as f32);
            }
        }

        if x_points.is_empty() {
            let (x, y, z) = self.geometry.0.interior_point();
            x_points.push(x as f32);
            y_points.push(y as f32);
            z_points.push(z as f32);
            t_points.push(0.0);
        }

        let x_tensor = Tensor::<B, 1>::from_data(TensorData::from(x_points.as_slice()), device)
            .unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 1>::from_data(TensorData::from(y_points.as_slice()), device)
            .unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 1>::from_data(TensorData::from(z_points.as_slice()), device)
            .unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 1>::from_data(TensorData::from(t_points.as_slice()), device)
            .unsqueeze_dim(1);

        (x_tensor, y_tensor, z_tensor, t_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::pinn::ml::burn_wave_equation_3d::geometry::Geometry3D;
    use burn::backend::{Autodiff, NdArray};
    use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_collocation_points_generation() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Should generate approximately the requested number (some may be filtered)
        let n_generated = match x_colloc.shape().dims.as_slice() {
            [n, _] => *n,
            dims => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "generate_collocation_points".to_string(),
                    reason: format!("Expected 2D tensor shape, got dims {dims:?}"),
                }));
            }
        };
        assert!(n_generated > 0 && n_generated <= config.num_collocation_points);
        for (name, tensor) in [
            ("y_colloc", &y_colloc),
            ("z_colloc", &z_colloc),
            ("t_colloc", &t_colloc),
        ] {
            let n = match tensor.shape().dims.as_slice() {
                [n, _] => *n,
                dims => {
                    return Err(KwaversError::System(SystemError::InvalidOperation {
                        operation: "generate_collocation_points".to_string(),
                        reason: format!("Expected 2D tensor shape for {name}, got dims {dims:?}"),
                    }));
                }
            };
            assert_eq!(n, n_generated);
        }
        Ok(())
    }

    #[test]
    fn test_collocation_points_spherical_geometry() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 100,
            ..Default::default()
        };
        let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

        let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Spherical geometry filters many points, so expect fewer than requested
        let n_generated = match x_colloc.shape().dims.as_slice() {
            [n, _] => *n,
            dims => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "generate_collocation_points".to_string(),
                    reason: format!("Expected 2D tensor shape, got dims {dims:?}"),
                }));
            }
        };
        assert!(n_generated > 0);
        assert!(n_generated < config.num_collocation_points);
        Ok(())
    }
}
