use super::core::BurnPINN3DWave;
use super::losses::LossScales;
use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::config::BurnTrainingMetrics3D;
use crate::solver::inverse::pinn::ml::burn_wave_equation_3d::optimizer::SimpleOptimizer3D;
use burn::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use std::time::Instant;

impl<B: AutodiffBackend> BurnPINN3DWave<B> {
    /// Train the PINN on reference data
    ///
    /// # Arguments
    ///
    /// * `x_data` - X-coordinates of training data
    /// * `y_data` - Y-coordinates of training data
    /// * `z_data` - Z-coordinates of training data
    /// * `t_data` - Time coordinates of training data
    /// * `u_data` - Observed displacement/pressure values
    /// * `v_data` - Optional initial velocity values (∂u/∂t at t=0)
    /// * `device` - Device for tensor operations
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    ///
    /// Training metrics including loss history and training time
    ///
    /// # Type Constraints
    ///
    /// Requires `B: AutodiffBackend` for gradient computation
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let x_data = vec![0.5, 0.6, 0.7];
    /// let y_data = vec![0.5, 0.5, 0.5];
    /// let z_data = vec![0.5, 0.5, 0.5];
    /// let t_data = vec![0.1, 0.2, 0.3];
    /// let u_data = vec![0.0, 0.1, 0.0];
    /// let v_data = None; // Optional velocity IC
    ///
    /// let metrics = solver.train(
    ///     &x_data, &y_data, &z_data, &t_data, &u_data, v_data.as_deref(),
    ///     &device, 1000
    /// )?;
    /// ```
    pub fn train(
        &mut self,
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
        v_data: Option<&[f32]>,
        device: &B::Device,
        epochs: usize,
    ) -> KwaversResult<BurnTrainingMetrics3D>
    where
        B: AutodiffBackend,
    {
        let start_time = Instant::now();
        let mut metrics = BurnTrainingMetrics3D::default();

        let n_data = x_data.len();
        if n_data == 0 {
            return Err(KwaversError::InvalidInput(
                "Training data must be non-empty".into(),
            ));
        }
        if y_data.len() != n_data
            || z_data.len() != n_data
            || t_data.len() != n_data
            || u_data.len() != n_data
        {
            return Err(KwaversError::InvalidInput(
                "x_data, y_data, z_data, t_data, and u_data must have equal length".into(),
            ));
        }

        let x_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(x_data.to_vec(), [n_data, 1]), device);
        let y_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(y_data.to_vec(), [n_data, 1]), device);
        let z_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(z_data.to_vec(), [n_data, 1]), device);
        let t_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(t_data.to_vec(), [n_data, 1]), device);
        let u_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(u_data.to_vec(), [n_data, 1]), device);

        // Generate collocation points for PDE residual
        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            self.generate_collocation_points(&self.config.0, device);
        let (x_ic, y_ic, z_ic, t_ic, u_ic) = Self::extract_initial_condition_tensors(
            x_data, y_data, z_data, t_data, u_data, device,
        )?;

        // Extract velocity initial conditions if provided
        let v_ic_opt = if let Some(v_data) = v_data {
            Some(Self::extract_velocity_initial_condition_tensor(
                x_data, y_data, z_data, t_data, v_data, device,
            )?)
        } else {
            None
        };

        // Adaptive learning rate: start with configured rate, decay on stagnation
        let mut current_lr = self.config.0.learning_rate as f32;
        let min_lr = (self.config.0.learning_rate * 0.001) as f32;
        let lr_decay_factor = 0.95_f32;
        let lr_decay_patience = 10;
        let mut epochs_without_improvement = 0;
        let mut best_total_loss = f64::INFINITY;

        // Loss normalization: track moving averages to normalize loss components
        let mut loss_scales = LossScales {
            data_scale: 1.0,
            pde_scale: 1.0,
            bc_scale: 1.0,
            ic_scale: 1.0,
            ema_alpha: 0.1, // Exponential moving average factor
        };

        // Training loop with physics-informed loss
        for epoch in 0..epochs {
            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) = self.compute_physics_loss(
                x_data_tensor.clone(),
                y_data_tensor.clone(),
                z_data_tensor.clone(),
                t_data_tensor.clone(),
                u_data_tensor.clone(),
                x_colloc.clone(),
                y_colloc.clone(),
                z_colloc.clone(),
                t_colloc.clone(),
                x_ic.clone(),
                y_ic.clone(),
                z_ic.clone(),
                t_ic.clone(),
                u_ic.clone(),
                v_ic_opt.as_ref(),
                &self.config.0.loss_weights,
                &mut loss_scales,
            )?;

            // Convert to f64 for metrics
            let total_val = Self::scalar_f32(&total_loss)? as f64;
            let data_val = Self::scalar_f32(&data_loss)? as f64;
            let pde_val = Self::scalar_f32(&pde_loss)? as f64;
            let bc_val = Self::scalar_f32(&bc_loss)? as f64;
            let ic_val = Self::scalar_f32(&ic_loss)? as f64;

            // Check for NaN/Inf - early stopping for numerical instability
            if !total_val.is_finite()
                || !data_val.is_finite()
                || !pde_val.is_finite()
                || !bc_val.is_finite()
                || !ic_val.is_finite()
            {
                log::error!(
                    "Numerical instability detected at epoch {}: total={:.6e}, data={:.6e}, pde={:.6e}, bc={:.6e}, ic={:.6e}",
                    epoch, total_val, data_val, pde_val, bc_val, ic_val
                );
                return Err(KwaversError::InvalidInput(format!(
                    "Training diverged at epoch {} (NaN/Inf detected)",
                    epoch
                )));
            }

            metrics.total_loss.push(total_val);
            metrics.data_loss.push(data_val);
            metrics.pde_loss.push(pde_val);
            metrics.bc_loss.push(bc_val);
            metrics.ic_loss.push(ic_val);
            metrics.epochs_completed = epoch + 1;

            // Backward pass to compute gradients
            let grads = total_loss.backward();

            // Update learning rate in optimizer (adaptive LR)
            self.optimizer.0 = SimpleOptimizer3D::new(current_lr);

            // Optimizer step with gradients
            self.pinn = self.optimizer.0.step(self.pinn.clone(), &grads);

            // Gradient diagnostics infrastructure ready but disabled due to Burn API limitation
            // The GradientDiagnostics struct is available for future use when Burn exposes
            // parameter introspection. For now, we rely on:
            // 1. Loss monitoring (already implemented)
            // 2. Adaptive LR (prevents explosion via rate reduction)
            // 3. EMA loss normalization (prevents component dominance)
            //
            // KNOWN_LIMITATION: Gradient norm logging blocked on Burn parameter introspection API

            // Adaptive learning rate: decay if no improvement
            if total_val < best_total_loss * 0.999 {
                // 0.1% improvement threshold
                best_total_loss = total_val;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
                if epochs_without_improvement >= lr_decay_patience {
                    let old_lr = current_lr;
                    current_lr = (current_lr * lr_decay_factor).max(min_lr);
                    if current_lr != old_lr {
                        log::info!(
                            "Learning rate decayed: {:.6e} → {:.6e} (no improvement for {} epochs)",
                            old_lr,
                            current_lr,
                            lr_decay_patience
                        );
                    }
                    epochs_without_improvement = 0;
                }
            }

            if epoch % 100 == 0 {
                log::info!(
                    "Epoch {}/{}: total={:.6e}, data={:.6e}, pde={:.6e}, bc={:.6e}, ic={:.6e}, lr={:.6e}",
                    epoch,
                    epochs,
                    total_val,
                    data_val,
                    pde_val,
                    bc_val,
                    ic_val,
                    current_lr
                );
            }
        }

        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        Ok(metrics)
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
    fn test_solver_training_smoke() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            num_collocation_points: 10,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_data = vec![0.5, 0.5, 0.5];
        let y_data = vec![0.5, 0.5, 0.5];
        let z_data = vec![0.5, 0.5, 0.5];
        let t_data = vec![0.0, 0.1, 0.2];
        let u_data = vec![1.0, 0.9, 0.8];

        let metrics = solver.train(
            &x_data, &y_data, &z_data, &t_data, &u_data, None, &device, 5,
        )?;
        assert_eq!(metrics.epochs_completed, 5);
        assert_eq!(metrics.total_loss.len(), 5);
        Ok(())
    }
}
