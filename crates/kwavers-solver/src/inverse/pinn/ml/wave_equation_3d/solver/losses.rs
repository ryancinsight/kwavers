use super::core::PinnWave3D;
use crate::inverse::pinn::ml::wave_equation_3d::config::LossWeights3D;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

/// Decomposed physics-informed loss components returned by
/// `compute_physics_loss`: `(total, data, pde, boundary, initial)` scalar losses.
#[allow(clippy::type_complexity)] // 5 independent scalar losses, no cohesive grouping
type PhysicsLossComponents3D<B> = (
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
);

/// Adaptive loss scaling for normalization
///
/// Tracks exponential moving averages of loss component magnitudes
/// to prevent any single component from dominating during training.
///
/// # Mathematical Specification
///
/// For each loss component L ∈ {data, pde, bc, ic}:
///   scale_t = α × |L_t| + (1-α) × scale_{t-1}
///
/// Normalized loss: L_norm = L / (scale + ε)
#[derive(Debug, Clone)]
pub struct LossScales {
    pub data_scale: f32,
    pub pde_scale: f32,
    pub bc_scale: f32,
    pub ic_scale: f32,
    pub ema_alpha: f32,
}

impl LossScales {
    fn update(&mut self, data_loss: f32, pde_loss: f32, bc_loss: f32, ic_loss: f32) {
        let alpha = self.ema_alpha;
        self.data_scale = alpha * data_loss.abs() + (1.0 - alpha) * self.data_scale;
        self.pde_scale = alpha * pde_loss.abs() + (1.0 - alpha) * self.pde_scale;
        self.bc_scale = alpha * bc_loss.abs() + (1.0 - alpha) * self.bc_scale;
        self.ic_scale = alpha * ic_loss.abs() + (1.0 - alpha) * self.ic_scale;
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave3D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Compute physics-informed loss with all components
    ///
    /// Returns `(total_loss, data_loss, pde_loss, bc_loss, ic_loss)`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    // Independent data/collocation/initial tensors plus weights and mutable
    // loss-scale state with no cohesive sub-grouping; bundling would not clarify
    // the call site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_physics_loss(
        &self,
        x_data: &Var<f32, B>,
        y_data: &Var<f32, B>,
        z_data: &Var<f32, B>,
        t_data: &Var<f32, B>,
        u_data: &Var<f32, B>,
        x_colloc: &Var<f32, B>,
        y_colloc: &Var<f32, B>,
        z_colloc: &Var<f32, B>,
        t_colloc: &Var<f32, B>,
        x_ic: &Var<f32, B>,
        y_ic: &Var<f32, B>,
        z_ic: &Var<f32, B>,
        t_ic: &Var<f32, B>,
        u_ic: &Var<f32, B>,
        v_ic: Option<&Var<f32, B>>,
        weights: &LossWeights3D,
        loss_scales: &mut LossScales,
    ) -> KwaversResult<PhysicsLossComponents3D<B>> {
        // Data loss: MSE between predictions and training data
        let u_pred = self.pinn.forward(x_data, y_data, z_data, t_data);
        let data_diff = coeus_autograd::sub(&u_pred, u_data);
        let data_loss_raw = coeus_autograd::mean(&coeus_autograd::mul(&data_diff, &data_diff));

        // PDE loss: MSE of PDE residual at collocation points
        let pde_residual =
            self.pinn
                .compute_pde_residual(x_colloc, y_colloc, z_colloc, t_colloc, |x, y, z| {
                    self.get_wave_speed(x, y, z)
                })?;
        let pde_loss_raw = coeus_autograd::mean(&coeus_autograd::mul(&pde_residual, &pde_residual));

        // Boundary condition loss: sample the 6 domain faces and enforce Dirichlet u=0.
        let bc_loss_raw = self.compute_bc_loss_internal();

        // Initial condition loss: displacement + optional velocity at t=0.
        let u_ic_pred = self.pinn.forward(x_ic, y_ic, z_ic, t_ic);
        let ic_disp_diff = coeus_autograd::sub(&u_ic_pred, u_ic);
        let ic_disp_loss = coeus_autograd::mean(&coeus_autograd::mul(&ic_disp_diff, &ic_disp_diff));

        let ic_loss_raw = if let Some(v_ic_var) = v_ic {
            let du_dt = self.compute_temporal_derivative_at_t0(x_ic, y_ic, z_ic, t_ic);
            let ic_vel_diff = coeus_autograd::sub(&du_dt, v_ic_var);
            let ic_vel_loss =
                coeus_autograd::mean(&coeus_autograd::mul(&ic_vel_diff, &ic_vel_diff));

            coeus_autograd::add(
                &coeus_autograd::scalar_mul(&ic_disp_loss, 0.5),
                &coeus_autograd::scalar_mul(&ic_vel_loss, 0.5),
            )
        } else {
            ic_disp_loss
        };

        // Extract scalar values for scale update
        let data_loss_val = Self::extract_scalar(&data_loss_raw).unwrap_or(1.0);
        let pde_loss_val = Self::extract_scalar(&pde_loss_raw).unwrap_or(1.0);
        let bc_loss_val = Self::extract_scalar(&bc_loss_raw).unwrap_or(1.0);
        let ic_loss_val = Self::extract_scalar(&ic_loss_raw).unwrap_or(1.0);

        loss_scales.update(data_loss_val, pde_loss_val, bc_loss_val, ic_loss_val);

        let eps = 1e-8_f32;
        let data_loss_normalized =
            coeus_autograd::scalar_mul(&data_loss_raw, 1.0 / (loss_scales.data_scale + eps));
        let pde_loss_normalized =
            coeus_autograd::scalar_mul(&pde_loss_raw, 1.0 / (loss_scales.pde_scale + eps));
        let bc_loss_normalized =
            coeus_autograd::scalar_mul(&bc_loss_raw, 1.0 / (loss_scales.bc_scale + eps));
        let ic_loss_normalized =
            coeus_autograd::scalar_mul(&ic_loss_raw, 1.0 / (loss_scales.ic_scale + eps));

        let total_loss = coeus_autograd::add(
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&data_loss_normalized, weights.data_weight),
                &coeus_autograd::scalar_mul(&pde_loss_normalized, weights.pde_weight),
            ),
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&bc_loss_normalized, weights.bc_weight),
                &coeus_autograd::scalar_mul(&ic_loss_normalized, weights.ic_weight),
            ),
        );

        Ok((
            total_loss,
            data_loss_raw,
            pde_loss_raw,
            bc_loss_raw,
            ic_loss_raw,
        ))
    }

    /// Compute boundary condition loss for rectangular domains
    ///
    /// Samples points on the 6 boundary faces and evaluates Dirichlet BC (u=0) violations.
    pub(crate) fn compute_bc_loss_internal(&self) -> Var<f32, B> {
        let (x_min, x_max, y_min, y_max, z_min, z_max) = self.geometry.bounding_box();

        let n_bc_per_face = 100;

        let mut bc_points_x = Vec::new();
        let mut bc_points_y = Vec::new();
        let mut bc_points_z = Vec::new();
        let mut bc_points_t = Vec::new();

        let t_samples = 5;

        for t_idx in 0..t_samples {
            let t = (t_idx as f64 / (t_samples - 1) as f64).clamp(0.0, 1.0);

            for _ in 0..n_bc_per_face {
                bc_points_x.push(x_min as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            for _ in 0..n_bc_per_face {
                bc_points_x.push(x_max as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push(y_min as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push(y_max as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push(z_min as f32);
                bc_points_t.push(t as f32);
            }

            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push(z_max as f32);
                bc_points_t.push(t as f32);
            }
        }

        let backend = B::default();
        let n = bc_points_x.len();
        let mk = |v: &[f32]| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], v, &backend),
                false,
            )
        };
        let x_bc = mk(&bc_points_x);
        let y_bc = mk(&bc_points_y);
        let z_bc = mk(&bc_points_z);
        let t_bc = mk(&bc_points_t);

        // Evaluate PINN at boundary points
        let u_bc = self.pinn.forward(&x_bc, &y_bc, &z_bc, &t_bc);

        // Dirichlet BC: u = 0 on boundary
        coeus_autograd::mean(&coeus_autograd::mul(&u_bc, &u_bc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::pinn::ml::wave_equation_3d::{config::PinnConfig3D, geometry::Geometry3D};
    type TestBackend = coeus_core::MoiraiBackend;

    #[test]
    fn test_training_loss_components() -> KwaversResult<()> {
        let config = PinnConfig3D {
            hidden_layers: vec![8],
            num_collocation_points: 10,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| {
            kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM as f32
        };

        let mut solver = PinnWave3D::<TestBackend>::new(config, geometry, wave_speed)?;

        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.1];
        let u_data = vec![0.0];

        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, None, 3)?;

        assert_eq!((metrics.total_loss.len()), 3);
        assert_eq!((metrics.data_loss.len()), 3);
        assert_eq!((metrics.pde_loss.len()), 3);
        assert_eq!((metrics.bc_loss.len()), 3);
        assert_eq!((metrics.ic_loss.len()), 3);

        assert!(metrics.total_loss.iter().all(|&l| l.is_finite()));
        assert!(metrics.data_loss.iter().all(|&l| l.is_finite()));
        assert!(metrics.pde_loss.iter().all(|&l| l.is_finite()));
        Ok(())
    }
}
