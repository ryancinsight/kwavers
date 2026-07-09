use leto::{
    Array1,
    Array2,
};

use coeus_autograd::Var;

use kwavers_core::error::{KwaversError, KwaversResult};

use super::super::{
    config::PinnConfig, network::PinnWave1D, optimizer::SimpleOptimizer, types::TrainingMetrics,
};

/// PINN trainer for 1D wave equation with physics-informed learning
#[derive(Debug)]
pub struct PinnTrainer<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    pub(super) pinn: PinnWave1D<B>,
    pub(super) optimizer: SimpleOptimizer,
    pub(super) config: PinnConfig,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnTrainer<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: PinnConfig) -> KwaversResult<Self> {
        config.validate()?;
        let pinn = PinnWave1D::<B>::new(config.clone())?;
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
        epochs: usize,
    ) -> KwaversResult<TrainingMetrics> {
        self.train_with_callback(x_data, t_data, u_data, wave_speed, epochs, |_, _| true)
    }
    /// Train with callback.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    // Independent training data arrays, hyperparameters, and a callback with no
    // cohesive sub-grouping; bundling would not clarify the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn train_with_callback<F>(
        &mut self,
        x_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        epochs: usize,
        mut callback: F,
    ) -> KwaversResult<TrainingMetrics>
    where
        F: FnMut(usize, &TrainingMetrics) -> bool,
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
        let mut metrics = TrainingMetrics::new();
        let backend = B::default();

        let n_data = x_data.len();
        let x_data_vec: Vec<f32> = x_data.iter().map(|&v| v as f32).collect();
        let t_data_vec: Vec<f32> = t_data.iter().map(|&v| v as f32).collect();
        let u_data_vec: Vec<f32> = u_data.iter().map(|&v| v as f32).collect();

        let x_data_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_data, 1], &x_data_vec, &backend),
            false,
        );
        let t_data_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_data, 1], &t_data_vec, &backend),
            false,
        );
        let u_data_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_data, 1], &u_data_vec, &backend),
            false,
        );

        let n_colloc = self.config.num_collocation_points;
        let x_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32) * 2.0 - 1.0)
            .collect();
        let t_colloc_vec: Vec<f32> = (0..n_colloc).map(|i| i as f32 / n_colloc as f32).collect();

        let x_colloc_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_colloc, 1], &x_colloc_vec, &backend),
            false,
        );
        let t_colloc_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_colloc, 1], &t_colloc_vec, &backend),
            false,
        );

        let n_bc = 10;
        let x_bc_vec: Vec<f32> = vec![-1.0; n_bc / 2]
            .into_iter()
            .chain(vec![1.0; n_bc / 2])
            .collect();
        let t_bc_vec: Vec<f32> = vec![0.0; n_bc];
        let u_bc_vec: Vec<f32> = vec![0.0; n_bc];

        let x_bc_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_bc, 1], &x_bc_vec, &backend),
            false,
        );
        let t_bc_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_bc, 1], &t_bc_vec, &backend),
            false,
        );
        let u_bc_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_bc, 1], &u_bc_vec, &backend),
            false,
        );

        for epoch in 0..epochs {
            // Var gradients accumulate across `backward()` calls (unlike the previous autograd path's
            // per-call-graph `Gradients`, which is implicitly fresh each step);
            // zero them before this epoch's forward+backward or they carry over.
            for p in self.pinn.parameters() {
                p.zero_grad();
            }

            let (total_loss, data_loss, pde_loss, bc_loss) = self.pinn.compute_physics_loss(
                &x_data_var,
                &t_data_var,
                &u_data_var,
                &x_colloc_var,
                &t_colloc_var,
                &x_bc_var,
                &t_bc_var,
                &u_bc_var,
                wave_speed,
                self.config.loss_weights,
            );

            let total_val = total_loss.tensor.as_slice()[0] as f64;
            let data_val = data_loss.tensor.as_slice()[0] as f64;
            let pde_val = pde_loss.tensor.as_slice()[0] as f64;
            let bc_val = bc_loss.tensor.as_slice()[0] as f64;

            if !total_val.is_finite()
                || !data_val.is_finite()
                || !pde_val.is_finite()
                || !bc_val.is_finite()
            {
                return Err(KwaversError::Numerical(
                    kwavers_core::error::NumericalError::NaN {
                        operation: "training_loss".to_string(),
                        inputs: format!(
                            "epoch {}: total={}, data={}, pde={}, bc={}",
                            epoch, total_val, data_val, pde_val, bc_val
                        ),
                    },
                ));
            }

            metrics.record_epoch(total_val, data_val, pde_val, bc_val);

            total_loss.backward();
            self.pinn = self.optimizer.step(self.pinn.clone());

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

    pub fn pinn(&self) -> &PinnWave1D<B> {
        &self.pinn
    }

    pub fn pinn_mut(&mut self) -> &mut PinnWave1D<B> {
        &mut self.pinn
    }

    pub fn optimizer(&self) -> &SimpleOptimizer {
        &self.optimizer
    }

    pub fn config(&self) -> &PinnConfig {
        &self.config
    }
}
