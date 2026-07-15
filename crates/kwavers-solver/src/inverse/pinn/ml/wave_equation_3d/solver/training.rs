use super::core::PinnWave3D;
use super::losses::LossScales;
use crate::inverse::pinn::ml::wave_equation_3d::config::TrainingMetrics3D;
use crate::inverse::pinn::ml::wave_equation_3d::optimizer::SimpleOptimizer3D;
use coeus_autograd::Var;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::time::Instant;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave3D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Train the PINN on reference data
    ///
    /// Returns training metrics including loss history and training time.
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    // Independent training data slices and hyperparameters with no cohesive
    // sub-grouping; bundling would not clarify the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
        v_data: Option<&[f32]>,
        epochs: usize,
    ) -> KwaversResult<TrainingMetrics3D> {
        let start_time = Instant::now();
        let mut metrics = TrainingMetrics3D::default();

        let n_data = (x_data.len());
        if n_data == 0 {
            return Err(KwaversError::InvalidInput(
                "Training data must be non-empty".into(),
            ));
        }
        if (y_data.len()) != n_data
            || (z_data.len()) != n_data
            || (t_data.len()) != n_data
            || (u_data.len()) != n_data
        {
            return Err(KwaversError::InvalidInput(
                "x_data, y_data, z_data, t_data, and u_data must have equal length".into(),
            ));
        }

        let backend = B::default();
        let mk = |v: &[f32]| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n_data, 1], v, &backend),
                false,
            )
        };
        let x_data_var = mk(x_data);
        let y_data_var = mk(y_data);
        let z_data_var = mk(z_data);
        let t_data_var = mk(t_data);
        let u_data_var = mk(u_data);

        // Generate collocation points for PDE residual
        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            self.generate_collocation_points(&self.config.clone());
        let (x_ic, y_ic, z_ic, t_ic, u_ic) =
            Self::extract_initial_condition_tensors(x_data, y_data, z_data, t_data, u_data)?;

        let v_ic_opt = if let Some(v_data) = v_data {
            Some(Self::extract_velocity_initial_condition_tensor(
                x_data, y_data, z_data, t_data, v_data,
            )?)
        } else {
            None
        };

        // Adaptive learning rate: start with configured rate, decay on stagnation
        let mut current_lr = self.config.learning_rate as f32;
        let min_lr = (self.config.learning_rate * 0.001) as f32;
        let lr_decay_factor = 0.95_f32;
        let lr_decay_patience = 10;
        let mut epochs_without_improvement = 0;
        let mut best_total_loss = f64::INFINITY;

        let mut loss_scales = LossScales {
            data_scale: 1.0,
            pde_scale: 1.0,
            bc_scale: 1.0,
            ic_scale: 1.0,
            ema_alpha: 0.1,
        };

        for epoch in 0..epochs {
            for p in self.pinn.parameters() {
                p.zero_grad();
            }

            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) = self.compute_physics_loss(
                &x_data_var,
                &y_data_var,
                &z_data_var,
                &t_data_var,
                &u_data_var,
                &x_colloc,
                &y_colloc,
                &z_colloc,
                &t_colloc,
                &x_ic,
                &y_ic,
                &z_ic,
                &t_ic,
                &u_ic,
                v_ic_opt.as_ref(),
                &self.config.loss_weights,
                &mut loss_scales,
            )?;

            let total_val = Self::extract_scalar(&total_loss)? as f64;
            let data_val = Self::extract_scalar(&data_loss)? as f64;
            let pde_val = Self::extract_scalar(&pde_loss)? as f64;
            let bc_val = Self::extract_scalar(&bc_loss)? as f64;
            let ic_val = Self::extract_scalar(&ic_loss)? as f64;

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
            total_loss.backward();

            // Update learning rate in optimizer (adaptive LR)
            self.optimizer = SimpleOptimizer3D::new(current_lr);

            // Optimizer step with accumulated gradients
            self.pinn = self.optimizer.step(self.pinn.clone());

            // Adaptive learning rate: decay if no improvement
            if total_val < best_total_loss * 0.999 {
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
    use crate::inverse::pinn::ml::wave_equation_3d::{config::PinnConfig3D, geometry::Geometry3D};
    type TestBackend = coeus_core::MoiraiBackend;

    #[test]
    fn test_solver_training_smoke() -> KwaversResult<()> {
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

        let x_data = vec![0.5, 0.5, 0.5];
        let y_data = vec![0.5, 0.5, 0.5];
        let z_data = vec![0.5, 0.5, 0.5];
        let t_data = vec![0.0, 0.1, 0.2];
        let u_data = vec![1.0, 0.9, 0.8];

        let metrics = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, None, 5)?;
        assert_eq!(metrics.epochs_completed, 5);
        assert_eq!((metrics.total_loss.len()), 5);
        Ok(())
    }
}
