use super::core::BurnPINN3DWave;
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnLossWeights3D;
use burn::tensor::{backend::Backend, Tensor, TensorData};

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
/// Where α is the EMA smoothing factor (typically 0.1).
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
    /// Update scales with exponential moving average
    fn update(&mut self, data_loss: f32, pde_loss: f32, bc_loss: f32, ic_loss: f32) {
        let alpha = self.ema_alpha;
        self.data_scale = alpha * data_loss.abs() + (1.0 - alpha) * self.data_scale;
        self.pde_scale = alpha * pde_loss.abs() + (1.0 - alpha) * self.pde_scale;
        self.bc_scale = alpha * bc_loss.abs() + (1.0 - alpha) * self.bc_scale;
        self.ic_scale = alpha * ic_loss.abs() + (1.0 - alpha) * self.ic_scale;
    }
}

impl<B: Backend> BurnPINN3DWave<B> {
    /// Compute physics-informed loss with all components
    ///
    /// # Arguments
    ///
    /// * `x_data`, `y_data`, `z_data`, `t_data` - Training data coordinates
    /// * `u_data` - Training data observations
    /// * `x_colloc`, `y_colloc`, `z_colloc`, `t_colloc` - Collocation points
    /// * `weights` - Loss weighting factors
    ///
    /// # Returns
    ///
    /// Tuple: (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    ///
    /// # Loss Components
    ///
    /// - **data_loss**: MSE(u_pred, u_data)
    /// - **pde_loss**: MSE(R) where R = ∂²u/∂t² - c²∇²u
    /// - **bc_loss**: Boundary condition violations
    /// - **ic_loss**: Initial condition violations
    /// - **total_loss**: Weighted sum of all components
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(crate) fn compute_physics_loss(
        &self,
        x_data: Tensor<B, 2>,
        y_data: Tensor<B, 2>,
        z_data: Tensor<B, 2>,
        t_data: Tensor<B, 2>,
        u_data: Tensor<B, 2>,
        x_colloc: Tensor<B, 2>,
        y_colloc: Tensor<B, 2>,
        z_colloc: Tensor<B, 2>,
        t_colloc: Tensor<B, 2>,
        x_ic: Tensor<B, 2>,
        y_ic: Tensor<B, 2>,
        z_ic: Tensor<B, 2>,
        t_ic: Tensor<B, 2>,
        u_ic: Tensor<B, 2>,
        v_ic: Option<&Tensor<B, 2>>,
        weights: &BurnLossWeights3D,
        loss_scales: &mut LossScales,
    ) -> KwaversResult<(
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    )> {
        // Data loss: MSE between predictions and training data
        let u_pred = self.pinn.forward(x_data, y_data, z_data, t_data);
        let data_loss_raw = (u_pred.clone() - u_data).powf_scalar(2.0).mean();

        // PDE loss: MSE of PDE residual at collocation points
        let pde_residual = self.pinn.compute_pde_residual(
            x_colloc.clone(),
            y_colloc.clone(),
            z_colloc.clone(),
            t_colloc.clone(),
            |x, y, z| self.get_wave_speed(x, y, z),
        )?;
        let pde_loss_raw = pde_residual.powf_scalar(2.0).mean();

        // Boundary condition loss: Enforce BC on all domain boundaries
        // Mathematical specification:
        //   L_BC = (1/N_bc) Σ_{x∈∂Ω} |u(x) - g(x)|² (Dirichlet)
        //        = (1/N_bc) Σ_{x∈∂Ω} |∂u/∂n(x) - h(x)|² (Neumann)
        //
        // Implementation: Sample points on 6 faces of rectangular domain,
        // evaluate PINN predictions, compute BC violations based on type
        let bc_loss_raw = self.compute_bc_loss_internal(&x_colloc, &y_colloc, &z_colloc, &t_colloc);

        // Initial condition loss: Enforce displacement and velocity at t=0
        // Mathematical specification:
        //   L_IC = (1/N_Ω) [Σ ||u(x,0) - u₀(x)||² + Σ ||∂u/∂t(x,0) - v₀(x)||²]
        //
        // Displacement component
        let u_ic_pred = self
            .pinn
            .forward(x_ic.clone(), y_ic.clone(), z_ic.clone(), t_ic.clone());
        let ic_disp_loss = (u_ic_pred - u_ic).powf_scalar(2.0).mean();

        // Velocity component (if provided)
        let ic_loss_raw = if let Some(v_ic_tensor) = v_ic {
            // Compute temporal derivative ∂u/∂t at t=0 via forward finite difference
            let du_dt = self.compute_temporal_derivative_at_t0(
                x_ic.clone(),
                y_ic.clone(),
                z_ic.clone(),
                t_ic.clone(),
            );
            let ic_vel_loss = (du_dt - v_ic_tensor.clone()).powf_scalar(2.0).mean();

            // Combined IC loss: equal weighting of displacement and velocity
            ic_disp_loss
                .clone()
                .mul_scalar(0.5)
                .add(ic_vel_loss.mul_scalar(0.5))
        } else {
            // Displacement only
            ic_disp_loss
        };

        // Extract scalar values for scale update
        let data_loss_val = Self::extract_scalar(&data_loss_raw).unwrap_or(1.0);
        let pde_loss_val = Self::extract_scalar(&pde_loss_raw).unwrap_or(1.0);
        let bc_loss_val = Self::extract_scalar(&bc_loss_raw).unwrap_or(1.0);
        let ic_loss_val = Self::extract_scalar(&ic_loss_raw).unwrap_or(1.0);

        // Update loss scales with exponential moving average
        loss_scales.update(data_loss_val, pde_loss_val, bc_loss_val, ic_loss_val);

        // Normalize losses by their scales to prevent dominance
        let eps = 1e-8_f32;
        let data_loss_normalized = data_loss_raw.clone() / (loss_scales.data_scale + eps);
        let pde_loss_normalized = pde_loss_raw.clone() / (loss_scales.pde_scale + eps);
        let bc_loss_normalized = bc_loss_raw.clone() / (loss_scales.bc_scale + eps);
        let ic_loss_normalized = ic_loss_raw.clone() / (loss_scales.ic_scale + eps);

        // Total weighted loss with normalized components
        let total_loss = weights.data_weight * data_loss_normalized
            + weights.pde_weight * pde_loss_normalized
            + weights.bc_weight * bc_loss_normalized
            + weights.ic_weight * ic_loss_normalized;

        // Return raw (unnormalized) losses for metrics tracking
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
    /// Samples points on the 6 boundary faces and evaluates BC violations.
    /// Currently supports Dirichlet BC (u=0 on boundary).
    ///
    /// # Mathematical Specification
    ///
    /// For Dirichlet BC: L_BC = (1/N_bc) Σ_{x∈∂Ω} |u(x,t)|²
    ///
    /// # Arguments
    ///
    /// * `x_colloc`, `y_colloc`, `z_colloc`, `t_colloc` - Collocation point tensors (unused in current impl)
    ///
    /// # Returns
    ///
    /// Boundary condition loss tensor (scalar)
    pub(crate) fn compute_bc_loss_internal(
        &self,
        _x_colloc: &Tensor<B, 2>,
        _y_colloc: &Tensor<B, 2>,
        _z_colloc: &Tensor<B, 2>,
        _t_colloc: &Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        // Get domain bounds
        let (x_min, x_max, y_min, y_max, z_min, z_max) = self.geometry.0.bounding_box();

        // Number of BC points per face
        let n_bc_per_face = 100;

        // Generate boundary samples on all 6 faces
        let mut bc_points_x = Vec::new();
        let mut bc_points_y = Vec::new();
        let mut bc_points_z = Vec::new();
        let mut bc_points_t = Vec::new();

        let t_samples = 5; // Sample at multiple time points

        for t_idx in 0..t_samples {
            let t = (t_idx as f64 / (t_samples - 1) as f64).clamp(0.0, 1.0);

            // Face 1: x = x_min
            for _ in 0..n_bc_per_face {
                bc_points_x.push(x_min as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            // Face 2: x = x_max
            for _ in 0..n_bc_per_face {
                bc_points_x.push(x_max as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            // Face 3: y = y_min
            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push(y_min as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            // Face 4: y = y_max
            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push(y_max as f32);
                bc_points_z.push((z_min + rand::random::<f64>() * (z_max - z_min)) as f32);
                bc_points_t.push(t as f32);
            }

            // Face 5: z = z_min
            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push(z_min as f32);
                bc_points_t.push(t as f32);
            }

            // Face 6: z = z_max
            for _ in 0..n_bc_per_face {
                bc_points_x.push((x_min + rand::random::<f64>() * (x_max - x_min)) as f32);
                bc_points_y.push((y_min + rand::random::<f64>() * (y_max - y_min)) as f32);
                bc_points_z.push(z_max as f32);
                bc_points_t.push(t as f32);
            }
        }

        // Convert to tensors
        let device = _x_colloc.device();
        let x_bc = Tensor::<B, 1>::from_data(TensorData::from(bc_points_x.as_slice()), &device)
            .unsqueeze_dim(1);
        let y_bc = Tensor::<B, 1>::from_data(TensorData::from(bc_points_y.as_slice()), &device)
            .unsqueeze_dim(1);
        let z_bc = Tensor::<B, 1>::from_data(TensorData::from(bc_points_z.as_slice()), &device)
            .unsqueeze_dim(1);
        let t_bc = Tensor::<B, 1>::from_data(TensorData::from(bc_points_t.as_slice()), &device)
            .unsqueeze_dim(1);

        // Evaluate PINN at boundary points
        let u_bc = self.pinn.forward(x_bc, y_bc, z_bc, t_bc);

        // Dirichlet BC: u = 0 on boundary
        // BC loss = MSE(u_bc - 0)²
        u_bc.powf_scalar(2.0).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::{
        config::BurnPINN3DConfig, geometry::Geometry3D,
    };
    use burn::backend::{Autodiff, NdArray};
    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_training_loss_components() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            num_collocation_points: 10,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.1];
        let u_data = vec![0.0];

        let metrics = solver.train(
            &x_data, &y_data, &z_data, &t_data, &u_data, None, &device, 3,
        )?;

        // Verify all loss components are present
        assert_eq!(metrics.total_loss.len(), 3);
        assert_eq!(metrics.data_loss.len(), 3);
        assert_eq!(metrics.pde_loss.len(), 3);
        assert_eq!(metrics.bc_loss.len(), 3);
        assert_eq!(metrics.ic_loss.len(), 3);

        // All losses should be finite
        assert!(metrics.total_loss.iter().all(|&l| l.is_finite()));
        assert!(metrics.data_loss.iter().all(|&l| l.is_finite()));
        assert!(metrics.pde_loss.iter().all(|&l| l.is_finite()));
        Ok(())
    }
}
