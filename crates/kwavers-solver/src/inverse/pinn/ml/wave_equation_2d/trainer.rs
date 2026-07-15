use super::config::{PinnConfig2D, TrainingMetrics2D};
use super::geometry::WaveGeometry2D;
use super::model::PinnWave2D;
use super::optimizer::SimpleOptimizer2D;
use coeus_autograd::Var;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2};
use std::f64::consts::PI;

/// Training state for the 2D PINN.
#[derive(Debug)]
pub struct PinnTrainer2D<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// The neural network
    pub pinn: PinnWave2D<B>,
    /// The geometry definition
    pub geometry: WaveGeometry2D,
    /// Simple optimizer for parameter updates
    pub optimizer: SimpleOptimizer2D,
}

fn var_col<B: coeus_ops::BackendOps<f32> + Default>(vals: &[f32], backend: &B) -> Var<f32, B> {
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![(vals.len()), 1], vals, backend),
        false,
    )
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnTrainer2D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// New trainer.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn new_trainer(config: PinnConfig2D, geometry: WaveGeometry2D) -> KwaversResult<Self> {
        let pinn = PinnWave2D::new(config.clone())?;
        let optimizer = SimpleOptimizer2D::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            geometry,
            optimizer,
        })
    }

    /// Train.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    // Independent training data arrays and hyperparameters with no cohesive
    // sub-grouping; bundling would not clarify the call site.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        config: &PinnConfig2D,
        epochs: usize,
    ) -> KwaversResult<TrainingMetrics2D> {
        self.train_with_callback(
            x_data,
            y_data,
            t_data,
            u_data,
            wave_speed,
            config,
            epochs,
            |_, _| true,
        )
    }
    /// Train with callback.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    /// # Errors
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Returns [`crate::KwaversError::Numerical`] if the precondition for a Numerical-class constraint is violated.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn train_with_callback<F>(
        &mut self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        config: &PinnConfig2D,
        epochs: usize,
        mut callback: F,
    ) -> KwaversResult<TrainingMetrics2D>
    where
        F: FnMut(usize, &TrainingMetrics2D) -> bool,
    {
        use std::time::Instant;

        if (x_data.len()) != (y_data.len())
            || (x_data.len()) != (t_data.len())
            || (x_data.len()) != u_data.shape()[0]
        {
            return Err(KwaversError::InvalidInput(
                "Data dimensions must match".into(),
            ));
        }

        let start_time = Instant::now();
        let mut metrics = TrainingMetrics2D {
            total_loss: Vec::with_capacity(epochs),
            data_loss: Vec::with_capacity(epochs),
            pde_loss: Vec::with_capacity(epochs),
            bc_loss: Vec::with_capacity(epochs),
            ic_loss: Vec::with_capacity(epochs),
            training_time_secs: 0.0,
            epochs_completed: 0,
        };

        let backend = B::default();

        let x_data_vec: Vec<f32> = x_data.iter().map(|&v| v as f32).collect();
        let y_data_vec: Vec<f32> = y_data.iter().map(|&v| v as f32).collect();
        let t_data_vec: Vec<f32> = t_data.iter().map(|&v| v as f32).collect();
        let u_data_vec: Vec<f32> = u_data.iter().map(|&v| v as f32).collect();

        let x_data_var = var_col(&x_data_vec, &backend);
        let y_data_var = var_col(&y_data_vec, &backend);
        let t_data_var = var_col(&t_data_vec, &backend);
        let u_data_var = var_col(&u_data_vec, &backend);

        let n_colloc = config.num_collocation_points;
        let (x_colloc, y_colloc) = self.geometry.sample_points(n_colloc);
        let t_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32) * 2.0 - 1.0)
            .collect();
        let x_colloc_vec: Vec<f32> = x_colloc.iter().map(|&v| v as f32).collect();
        let y_colloc_vec: Vec<f32> = y_colloc.iter().map(|&v| v as f32).collect();

        let x_colloc_var = var_col(&x_colloc_vec, &backend);
        let y_colloc_var = var_col(&y_colloc_vec, &backend);
        let t_colloc_var = var_col(&t_colloc_vec, &backend);

        let (x_bc, y_bc, t_bc, u_bc) = self.generate_boundary_conditions(config, &backend);
        let (x_ic, y_ic, t_ic, u_ic) = self.generate_initial_conditions(config, &backend);

        for epoch in 0..epochs {
            // Var gradients accumulate across `backward()` calls (unlike the previous autograd path's
            // per-call-graph `Gradients`, implicitly fresh each step); zero
            // them before this epoch's forward+backward or they carry over.
            for p in self.pinn.parameters() {
                p.zero_grad();
            }

            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) =
                self.pinn.compute_physics_loss(
                    &x_data_var,
                    &y_data_var,
                    &t_data_var,
                    &u_data_var,
                    &x_colloc_var,
                    &y_colloc_var,
                    &t_colloc_var,
                    &x_bc,
                    &y_bc,
                    &t_bc,
                    &u_bc,
                    &x_ic,
                    &y_ic,
                    &t_ic,
                    &u_ic,
                    wave_speed,
                    config.loss_weights,
                );

            let total_val = total_loss.tensor.as_slice()[0] as f64;
            let data_val = data_loss.tensor.as_slice()[0] as f64;
            let pde_val = pde_loss.tensor.as_slice()[0] as f64;
            let bc_val = bc_loss.tensor.as_slice()[0] as f64;
            let ic_val = ic_loss.tensor.as_slice()[0] as f64;

            if !total_val.is_finite()
                || !data_val.is_finite()
                || !pde_val.is_finite()
                || !bc_val.is_finite()
                || !ic_val.is_finite()
            {
                return Err(KwaversError::Numerical(
                    kwavers_core::error::NumericalError::NaN {
                        operation: "training_loss_2d".to_string(),
                        inputs: format!(
                            "epoch {}: total={}, data={}, pde={}, bc={}, ic={}",
                            epoch, total_val, data_val, pde_val, bc_val, ic_val
                        ),
                    },
                ));
            }

            metrics.total_loss.push(total_val);
            metrics.data_loss.push(data_val);
            metrics.pde_loss.push(pde_val);
            metrics.bc_loss.push(bc_val);
            metrics.ic_loss.push(ic_val);
            metrics.epochs_completed = epoch + 1;

            total_loss.backward();
            self.pinn = self.optimizer.step(self.pinn.clone());

            if epoch % 100 == 0 {
                log::info!("Epoch {}/{}: total_loss={:.6e}", epoch, epochs, total_val);
            }

            if !callback(epoch, &metrics) {
                break;
            }
        }

        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        Ok(metrics)
    }

    #[allow(clippy::type_complexity)] // (x, y, t, u) boundary-condition tensors, no cohesive grouping
    fn generate_boundary_conditions(
        &self,
        _config: &PinnConfig2D,
        backend: &B,
    ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>, Var<f32, B>) {
        let n_bc = 50;
        let mut x_bc = Vec::new();
        let mut y_bc = Vec::new();
        let mut t_bc = Vec::new();
        let mut u_bc = Vec::new();

        let (x_min, x_max, y_min, y_max) = self.geometry.bounding_box();

        // Bottom
        for i in 0..n_bc {
            let x = x_min + (x_max - x_min) * (i as f64) / (n_bc - 1) as f64;
            let y = y_min;
            let t = (i as f64) / (n_bc - 1) as f64;
            let u = 0.0;
            x_bc.push(x as f32);
            y_bc.push(y as f32);
            t_bc.push(t as f32);
            u_bc.push(u as f32);
        }

        // Top
        for i in 0..n_bc {
            let x = x_min + (x_max - x_min) * (i as f64) / (n_bc - 1) as f64;
            let y = y_max;
            let t = (i as f64) / (n_bc - 1) as f64;
            let u = 0.0;
            x_bc.push(x as f32);
            y_bc.push(y as f32);
            t_bc.push(t as f32);
            u_bc.push(u as f32);
        }

        // Left
        for i in 0..n_bc {
            let x = x_min;
            let y = y_min + (y_max - y_min) * (i as f64) / (n_bc - 1) as f64;
            let t = (i as f64) / (n_bc - 1) as f64;
            let u = 0.0;
            x_bc.push(x as f32);
            y_bc.push(y as f32);
            t_bc.push(t as f32);
            u_bc.push(u as f32);
        }

        // Right
        for i in 0..n_bc {
            let x = x_max;
            let y = y_min + (y_max - y_min) * (i as f64) / (n_bc - 1) as f64;
            let t = (i as f64) / (n_bc - 1) as f64;
            let u = 0.0;
            x_bc.push(x as f32);
            y_bc.push(y as f32);
            t_bc.push(t as f32);
            u_bc.push(u as f32);
        }

        (
            var_col(&x_bc, backend),
            var_col(&y_bc, backend),
            var_col(&t_bc, backend),
            var_col(&u_bc, backend),
        )
    }

    #[allow(clippy::type_complexity)] // (x, y, t, u) initial-condition tensors, no cohesive grouping
    fn generate_initial_conditions(
        &self,
        _config: &PinnConfig2D,
        backend: &B,
    ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>, Var<f32, B>) {
        let n_ic = 200;
        let (x_ic, y_ic) = self.geometry.sample_points(n_ic);

        let x_ic_vec: Vec<f32> = x_ic.iter().map(|&v| v as f32).collect();
        let y_ic_vec: Vec<f32> = y_ic.iter().map(|&v| v as f32).collect();
        let t_ic_vec: Vec<f32> = vec![0.0; n_ic];

        let u_ic_vec: Vec<f32> = x_ic_vec
            .iter()
            .zip(y_ic_vec.iter())
            .map(|(&x, &y)| (x as f64 * PI).sin() * (y as f64 * PI).sin())
            .map(|v| v as f32)
            .collect();

        (
            var_col(&x_ic_vec, backend),
            var_col(&y_ic_vec, backend),
            var_col(&t_ic_vec, backend),
            var_col(&u_ic_vec, backend),
        )
    }

    pub fn pinn(&self) -> &PinnWave2D<B> {
        &self.pinn
    }
}
