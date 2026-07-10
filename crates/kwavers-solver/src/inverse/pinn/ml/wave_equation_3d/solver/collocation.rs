use super::core::PinnWave3D;
use crate::inverse::pinn::ml::wave_equation_3d::config::PinnConfig3D;
use coeus_autograd::Var;

/// Collocation coordinate tensors `(x, y, z, t)`, each leaf `Var<f32,B>` `[n_points, 1]`.
#[allow(clippy::type_complexity)] // 4 independent coordinate tensors, no cohesive grouping
type CollocationTensors<B> = (Var<f32, B>, Var<f32, B>, Var<f32, B>, Var<f32, B>);

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave3D<B>
where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    /// Generate collocation points for PDE residual computation
    ///
    /// Returns `(x_colloc, y_colloc, z_colloc, t_colloc)` as leaf `Var`s `[n_points, 1]`.
    pub(crate) fn generate_collocation_points(
        &self,
        config: &PinnConfig3D,
    ) -> CollocationTensors<B> {
        let n_points = config.num_collocation_points;
        let mut x_points = Vec::with_capacity(n_points);
        let mut y_points = Vec::with_capacity(n_points);
        let mut z_points = Vec::with_capacity(n_points);
        let mut t_points = Vec::with_capacity(n_points);

        let (x_min, x_max, y_min, y_max, z_min, z_max) = self.geometry.bounding_box();
        let t_max = 1.0; // Normalized time

        for _ in 0..n_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            let z = z_min + (z_max - z_min) * rand::random::<f64>();
            let t = t_max * rand::random::<f64>();

            if self.geometry.contains(x, y, z) {
                x_points.push(x as f32);
                y_points.push(y as f32);
                z_points.push(z as f32);
                t_points.push(t as f32);
            }
        }

        if x_points.is_empty() {
            let (x, y, z) = self.geometry.interior_point();
            x_points.push(x as f32);
            y_points.push(y as f32);
            z_points.push(z as f32);
            t_points.push(0.0);
        }

        let backend = B::default();
        let n = (x_points.len());
        let mk = |v: &[f32]| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], v, &backend),
                false,
            )
        };

        (mk(&x_points), mk(&y_points), mk(&z_points), mk(&t_points))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::pinn::ml::wave_equation_3d::geometry::Geometry3D;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::error::{KwaversError, KwaversResult, SystemError};
    type TestBackend = coeus_core::MoiraiBackend;

    #[test]
    fn test_collocation_points_generation() -> KwaversResult<()> {
        let config = PinnConfig3D {
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = PinnWave3D::<TestBackend>::new(config.clone(), geometry, wave_speed)?;

        let (x_colloc, y_colloc, z_colloc, t_colloc) = solver.generate_collocation_points(&config);

        let n_generated = match x_colloc.tensor.shape() {
            [n, _] => *n,
            dims => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "generate_collocation_points".to_string(),
                    reason: format!("Expected 2D tensor shape, got dims {dims:?}"),
                }));
            }
        };
        assert!(n_generated > 0 && n_generated <= config.num_collocation_points);
        for (name, var) in [
            ("y_colloc", &y_colloc),
            ("z_colloc", &z_colloc),
            ("t_colloc", &t_colloc),
        ] {
            let n = match var.tensor.shape() {
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
        let config = PinnConfig3D {
            num_collocation_points: 100,
            ..Default::default()
        };
        let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        let wave_speed = |_x: f32, _y: f32, _z: f32| SOUND_SPEED_WATER_SIM as f32;

        let solver = PinnWave3D::<TestBackend>::new(config.clone(), geometry, wave_speed)?;

        let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
            solver.generate_collocation_points(&config);

        let n_generated = match x_colloc.tensor.shape() {
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
