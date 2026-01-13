use super::config::{BurnPINN2DConfig, BurnTrainingMetrics2D};
use super::geometry::Geometry2D;
use super::model::BurnPINN2DWave;
use super::optimizer::SimpleOptimizer2D;
use crate::core::error::{KwaversError, KwaversResult};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Training state for Burn-based 2D PINN
#[derive(Debug)]
pub struct BurnPINN2DTrainer<B: AutodiffBackend> {
    /// The neural network
    pub pinn: BurnPINN2DWave<B>,
    /// The geometry definition
    pub geometry: Geometry2D,
    /// Simple optimizer for parameter updates
    pub optimizer: SimpleOptimizer2D,
}

impl<B: AutodiffBackend> BurnPINN2DTrainer<B> {
    pub fn new_trainer(
        config: BurnPINN2DConfig,
        geometry: Geometry2D,
        device: &B::Device,
    ) -> KwaversResult<Self> {
        let pinn = BurnPINN2DWave::new(config.clone(), device)?;
        let optimizer = SimpleOptimizer2D::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            geometry,
            optimizer,
        })
    }

    pub fn train(
        &mut self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        config: &BurnPINN2DConfig,
        device: &B::Device,
        epochs: usize,
    ) -> KwaversResult<BurnTrainingMetrics2D> {
        use std::time::Instant;

        if x_data.len() != y_data.len()
            || x_data.len() != t_data.len()
            || x_data.len() != u_data.nrows()
        {
            return Err(KwaversError::InvalidInput(
                "Data dimensions must match".into(),
            ));
        }

        let start_time = Instant::now();
        let mut metrics = BurnTrainingMetrics2D {
            total_loss: Vec::with_capacity(epochs),
            data_loss: Vec::with_capacity(epochs),
            pde_loss: Vec::with_capacity(epochs),
            bc_loss: Vec::with_capacity(epochs),
            ic_loss: Vec::with_capacity(epochs),
            training_time_secs: 0.0,
            epochs_completed: 0,
        };

        let n_data = x_data.len();
        let x_data_vec: Vec<f32> = x_data.iter().map(|&v| v as f32).collect();
        let y_data_vec: Vec<f32> = y_data.iter().map(|&v| v as f32).collect();
        let t_data_vec: Vec<f32> = t_data.iter().map(|&v| v as f32).collect();
        let u_data_vec: Vec<f32> = u_data.iter().map(|&v| v as f32).collect();

        let x_data_tensor =
            Tensor::<B, 1>::from_floats(x_data_vec.as_slice(), device).reshape([n_data, 1]);
        let y_data_tensor =
            Tensor::<B, 1>::from_floats(y_data_vec.as_slice(), device).reshape([n_data, 1]);
        let t_data_tensor =
            Tensor::<B, 1>::from_floats(t_data_vec.as_slice(), device).reshape([n_data, 1]);
        let u_data_tensor =
            Tensor::<B, 1>::from_floats(u_data_vec.as_slice(), device).reshape([n_data, 1]);

        let n_colloc = config.num_collocation_points;
        let (x_colloc, y_colloc) = self.geometry.sample_points(n_colloc);
        let t_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32) * 2.0 - 1.0)
            .collect();
        let x_colloc_vec: Vec<f32> = x_colloc.iter().map(|&v| v as f32).collect();
        let y_colloc_vec: Vec<f32> = y_colloc.iter().map(|&v| v as f32).collect();

        let x_colloc_tensor =
            Tensor::<B, 1>::from_floats(x_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);
        let y_colloc_tensor =
            Tensor::<B, 1>::from_floats(y_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);
        let t_colloc_tensor =
            Tensor::<B, 1>::from_floats(t_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);

        let (x_bc, y_bc, t_bc, u_bc) = self.generate_boundary_conditions(config, device);
        let (x_ic, y_ic, t_ic, u_ic) = self.generate_initial_conditions(config, device);

        for epoch in 0..epochs {
            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) =
                self.pinn.compute_physics_loss(
                    x_data_tensor.clone(),
                    y_data_tensor.clone(),
                    t_data_tensor.clone(),
                    u_data_tensor.clone(),
                    x_colloc_tensor.clone(),
                    y_colloc_tensor.clone(),
                    t_colloc_tensor.clone(),
                    x_bc.clone(),
                    y_bc.clone(),
                    t_bc.clone(),
                    u_bc.clone(),
                    x_ic.clone(),
                    y_ic.clone(),
                    t_ic.clone(),
                    u_ic.clone(),
                    wave_speed,
                    config.loss_weights,
                );

            let total_val = total_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let data_val = data_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let pde_val = pde_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let bc_val = bc_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let ic_val = ic_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;

            metrics.total_loss.push(total_val);
            metrics.data_loss.push(data_val);
            metrics.pde_loss.push(pde_val);
            metrics.bc_loss.push(bc_val);
            metrics.ic_loss.push(ic_val);
            metrics.epochs_completed = epoch + 1;

            let grads = total_loss.backward();
            self.pinn = self.optimizer.step(self.pinn.clone(), &grads);

            if epoch % 100 == 0 {
                log::info!("Epoch {}/{}: total_loss={:.6e}", epoch, epochs, total_val);
            }
        }

        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        Ok(metrics)
    }

    fn generate_boundary_conditions(
        &self,
        _config: &BurnPINN2DConfig,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
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

        let x_bc_tensor =
            Tensor::<B, 1>::from_floats(x_bc.as_slice(), device).reshape([x_bc.len(), 1]);
        let y_bc_tensor =
            Tensor::<B, 1>::from_floats(y_bc.as_slice(), device).reshape([y_bc.len(), 1]);
        let t_bc_tensor =
            Tensor::<B, 1>::from_floats(t_bc.as_slice(), device).reshape([t_bc.len(), 1]);
        let u_bc_tensor =
            Tensor::<B, 1>::from_floats(u_bc.as_slice(), device).reshape([u_bc.len(), 1]);

        (x_bc_tensor, y_bc_tensor, t_bc_tensor, u_bc_tensor)
    }

    fn generate_initial_conditions(
        &self,
        _config: &BurnPINN2DConfig,
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
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

        let x_ic_tensor =
            Tensor::<B, 1>::from_floats(x_ic_vec.as_slice(), device).reshape([n_ic, 1]);
        let y_ic_tensor =
            Tensor::<B, 1>::from_floats(y_ic_vec.as_slice(), device).reshape([n_ic, 1]);
        let t_ic_tensor =
            Tensor::<B, 1>::from_floats(t_ic_vec.as_slice(), device).reshape([n_ic, 1]);
        let u_ic_tensor =
            Tensor::<B, 1>::from_floats(u_ic_vec.as_slice(), device).reshape([n_ic, 1]);

        (x_ic_tensor, y_ic_tensor, t_ic_tensor, u_ic_tensor)
    }

    pub fn pinn(&self) -> &BurnPINN2DWave<B> {
        &self.pinn
    }
}
