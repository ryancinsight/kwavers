use ndarray::{Array1, Array2};

use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::core::error::{KwaversError, KwaversResult};

use super::super::{
    config::BurnPINNConfig, network::BurnPINN1DWave, optimizer::SimpleOptimizer,
    types::BurnTrainingMetrics,
};

/// PINN trainer for 1D wave equation with physics-informed learning
#[derive(Debug)]
pub struct BurnPINNTrainer<B: AutodiffBackend> {
    pub(super) pinn: BurnPINN1DWave<B>,
    pub(super) optimizer: SimpleOptimizer,
    pub(super) config: BurnPINNConfig,
}

impl<B: AutodiffBackend> BurnPINNTrainer<B> {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: BurnPINNConfig, device: &B::Device) -> KwaversResult<Self> {
        config.validate()?;
        let pinn = BurnPINN1DWave::<B>::new(config.clone(), device)?;
        let optimizer = SimpleOptimizer::new(config.learning_rate as f32);
        Ok(Self {
            pinn,
            optimizer,
            config,
        })
    }

    /// Train.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn train(
        &mut self,
        x_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        device: &B::Device,
        epochs: usize,
    ) -> KwaversResult<BurnTrainingMetrics> {
        self.train_with_callback(
            x_data,
            t_data,
            u_data,
            wave_speed,
            device,
            epochs,
            |_, _| true,
        )
    }
    /// Train with callback.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    pub fn train_with_callback<F>(
        &mut self,
        x_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        device: &B::Device,
        epochs: usize,
        mut callback: F,
    ) -> KwaversResult<BurnTrainingMetrics>
    where
        F: FnMut(usize, &BurnTrainingMetrics) -> bool,
    {
        use std::time::Instant;

        if x_data.len() != t_data.len() || x_data.len() != u_data.nrows() {
            return Err(KwaversError::InvalidInput(
                "Data dimensions must match: x_data.len() == t_data.len() == u_data.nrows()".into(),
            ));
        }

        if u_data.ncols() != 1 {
            return Err(KwaversError::InvalidInput(
                "u_data must have shape [N, 1]".into(),
            ));
        }

        let start_time = Instant::now();
        let mut metrics = BurnTrainingMetrics::new();

        let n_data = x_data.len();
        let x_data_vec: Vec<f32> = x_data.iter().map(|&v| v as f32).collect();
        let t_data_vec: Vec<f32> = t_data.iter().map(|&v| v as f32).collect();
        let u_data_vec: Vec<f32> = u_data.iter().map(|&v| v as f32).collect();

        let x_data_tensor =
            Tensor::<B, 1>::from_floats(x_data_vec.as_slice(), device).reshape([n_data, 1]);
        let t_data_tensor =
            Tensor::<B, 1>::from_floats(t_data_vec.as_slice(), device).reshape([n_data, 1]);
        let u_data_tensor =
            Tensor::<B, 1>::from_floats(u_data_vec.as_slice(), device).reshape([n_data, 1]);

        let n_colloc = self.config.num_collocation_points;
        let x_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32) * 2.0 - 1.0)
            .collect();
        let t_colloc_vec: Vec<f32> = (0..n_colloc).map(|i| i as f32 / n_colloc as f32).collect();

        let x_colloc_tensor =
            Tensor::<B, 1>::from_floats(x_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);
        let t_colloc_tensor =
            Tensor::<B, 1>::from_floats(t_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);

        let n_bc = 10;
        let x_bc_vec: Vec<f32> = vec![-1.0; n_bc / 2]
            .into_iter()
            .chain(vec![1.0; n_bc / 2])
            .collect();
        let t_bc_vec: Vec<f32> = vec![0.0; n_bc];
        let u_bc_vec: Vec<f32> = vec![0.0; n_bc];

        let x_bc_tensor =
            Tensor::<B, 1>::from_floats(x_bc_vec.as_slice(), device).reshape([n_bc, 1]);
        let t_bc_tensor =
            Tensor::<B, 1>::from_floats(t_bc_vec.as_slice(), device).reshape([n_bc, 1]);
        let u_bc_tensor =
            Tensor::<B, 1>::from_floats(u_bc_vec.as_slice(), device).reshape([n_bc, 1]);

        for epoch in 0..epochs {
            let (total_loss, data_loss, pde_loss, bc_loss) = self.pinn.compute_physics_loss(
                x_data_tensor.clone(),
                t_data_tensor.clone(),
                u_data_tensor.clone(),
                x_colloc_tensor.clone(),
                t_colloc_tensor.clone(),
                x_bc_tensor.clone(),
                t_bc_tensor.clone(),
                u_bc_tensor.clone(),
                wave_speed,
                self.config.loss_weights,
            );

            let total_val = total_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let data_val = data_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let pde_val = pde_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let bc_val = bc_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;

            if !total_val.is_finite()
                || !data_val.is_finite()
                || !pde_val.is_finite()
                || !bc_val.is_finite()
            {
                return Err(KwaversError::Numerical(
                    crate::core::error::NumericalError::NaN {
                        operation: "training_loss".to_string(),
                        inputs: format!(
                            "epoch {}: total={}, data={}, pde={}, bc={}",
                            epoch, total_val, data_val, pde_val, bc_val
                        ),
                    },
                ));
            }

            metrics.record_epoch(total_val, data_val, pde_val, bc_val);

            let grads = total_loss.backward();
            self.pinn = self.optimizer.step(self.pinn.clone(), &grads);

            if epoch % 100 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: total_loss={:.6e}, data_loss={:.6e}, pde_loss={:.6e}, bc_loss={:.6e}",
                    epoch + 1,
                    epochs,
                    total_val,
                    data_val,
                    pde_val,
                    bc_val
                );
            }

            if !callback(epoch, &metrics) {
                break;
            }
        }

        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        metrics.epochs_completed = epochs;

        Ok(metrics)
    }

    pub fn pinn(&self) -> &BurnPINN1DWave<B> {
        &self.pinn
    }

    pub fn pinn_mut(&mut self) -> &mut BurnPINN1DWave<B> {
        &mut self.pinn
    }

    pub fn optimizer(&self) -> &SimpleOptimizer {
        &self.optimizer
    }

    pub fn config(&self) -> &BurnPINNConfig {
        &self.config
    }
}
