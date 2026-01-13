//! Application layer: PINN solver orchestration for 3D wave equation
//!
//! This module implements the high-level solver that orchestrates training and inference
//! for the 3D wave equation PINN. It combines the network, optimizer, and physics-informed
//! loss computation into a unified workflow.
//!
//! ## Solver Responsibilities
//!
//! - Orchestrate training loop with physics-informed loss
//! - Manage collocation point generation for PDE residual
//! - Coordinate network, optimizer, and geometry
//! - Provide prediction interface for trained models
//!
//! ## Training Workflow
//!
//! 1. Convert training data to tensors
//! 2. Generate collocation points for PDE residual
//! 3. For each epoch:
//!    - Compute physics-informed loss (data + PDE + BC + IC)
//!    - Backpropagate gradients
//!    - Update network parameters via optimizer
//! 4. Return training metrics
//!
//! ## Loss Components
//!
//! - **Data loss**: MSE between predictions and observations
//! - **PDE loss**: MSE of wave equation residual at collocation points
//! - **BC loss**: Boundary condition violations
//! - **IC loss**: Initial condition violations

use burn::module::{Ignored, Module};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;
use std::time::Instant;

use super::config::{BurnLossWeights3D, BurnPINN3DConfig, BurnTrainingMetrics3D};
use super::geometry::Geometry3D;
use super::network::PINN3DNetwork;
use super::optimizer::SimpleOptimizer3D;
use super::wavespeed::WaveSpeedFn3D;

/// Main solver for 3D wave equation PINN
///
/// Orchestrates training and prediction by coordinating the network, optimizer,
/// geometry, and wave speed function.
///
/// # Type Parameters
///
/// * `B` - Backend type (e.g., NdArray, Autodiff<NdArray>, WGPU)
///
/// # Fields
///
/// * `pinn` - Neural network module
/// * `geometry` - Domain geometry (rectangular, spherical, cylindrical)
/// * `wave_speed_fn` - Wave speed function c(x, y, z)
/// * `optimizer` - Simple SGD optimizer
/// * `config` - Training configuration
#[derive(Module, Debug)]
pub struct BurnPINN3DWave<B: Backend> {
    /// Neural network for wave equation solution
    pub pinn: PINN3DNetwork<B>,
    /// Geometry definition (wrapped in Ignored for Module trait)
    pub geometry: Ignored<Geometry3D>,
    /// Wave speed function c(x,y,z)
    pub wave_speed_fn: Option<WaveSpeedFn3D<B>>,
    /// Simple optimizer for parameter updates
    pub optimizer: Ignored<SimpleOptimizer3D>,
    /// Configuration (wrapped in Ignored)
    pub config: Ignored<BurnPINN3DConfig>,
    /// Backend type marker
    _backend: PhantomData<B>,
}

impl<B: Backend> BurnPINN3DWave<B> {
    /// Create a new 3D PINN solver
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration (hidden layers, learning rate, etc.)
    /// * `geometry` - Domain geometry
    /// * `wave_speed_fn` - Function c(x, y, z) returning wave speed
    /// * `device` - Target device for network parameters
    ///
    /// # Returns
    ///
    /// A new `BurnPINN3DWave` solver instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::{
    ///     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
    /// };
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let config = BurnPINN3DConfig::default();
    /// let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    /// let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;
    ///
    /// let solver = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device);
    /// ```
    pub fn new<F>(
        config: BurnPINN3DConfig,
        geometry: Geometry3D,
        wave_speed_fn: F,
        device: &B::Device,
    ) -> Self
    where
        F: Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
    {
        let pinn = PINN3DNetwork::new(&config, device);
        let optimizer = SimpleOptimizer3D::new(config.learning_rate as f32);

        Self {
            pinn,
            geometry: Ignored(geometry),
            wave_speed_fn: Some(WaveSpeedFn3D::new(std::sync::Arc::new(wave_speed_fn))),
            optimizer: Ignored(optimizer),
            config: Ignored(config),
            _backend: PhantomData,
        }
    }

    /// Get wave speed at a specific location
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate (meters)
    /// * `y` - Y-coordinate (meters)
    /// * `z` - Z-coordinate (meters)
    ///
    /// # Returns
    ///
    /// Wave speed c(x, y, z) in m/s, or 343.0 (air) as default
    pub fn get_wave_speed(&self, x: f32, y: f32, z: f32) -> f32 {
        self.wave_speed_fn
            .as_ref()
            .map(|f| f.get(x, y, z))
            .unwrap_or(343.0)
    }

    /// Train the PINN on reference data
    ///
    /// # Arguments
    ///
    /// * `x_data` - X-coordinates of training data
    /// * `y_data` - Y-coordinates of training data
    /// * `z_data` - Z-coordinates of training data
    /// * `t_data` - Time coordinates of training data
    /// * `u_data` - Observed displacement/pressure values
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
    ///
    /// let metrics = solver.train(
    ///     &x_data, &y_data, &z_data, &t_data, &u_data,
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
        device: &B::Device,
        epochs: usize,
    ) -> Result<BurnTrainingMetrics3D, String>
    where
        B: AutodiffBackend,
    {
        let start_time = Instant::now();
        let mut metrics = BurnTrainingMetrics3D::default();

        // Convert data to tensors
        let x_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(x_data), device).unsqueeze_dim(1);
        let y_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(y_data), device).unsqueeze_dim(1);
        let z_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(z_data), device).unsqueeze_dim(1);
        let t_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(t_data), device).unsqueeze_dim(1);
        let u_data_tensor =
            Tensor::<B, 2>::from_data(TensorData::from(u_data), device).unsqueeze_dim(1);

        // Generate collocation points for PDE residual
        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            self.generate_collocation_points(&self.config.0, device);

        // Training loop with physics-informed loss
        for epoch in 0..epochs {
            // Compute physics-informed loss
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
                &self.config.0.loss_weights,
            );

            // Convert to f64 for metrics
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

            // Perform optimizer step to update model parameters
            let grads = total_loss.backward();
            self.pinn = self.optimizer.0.step(self.pinn.clone(), &grads);

            if epoch % 100 == 0 {
                log::info!(
                    "Epoch {}/{}: total_loss={:.6e}, data_loss={:.6e}, pde_loss={:.6e}, bc_loss={:.6e}, ic_loss={:.6e}",
                    epoch,
                    epochs,
                    metrics.total_loss.last().unwrap(),
                    metrics.data_loss.last().unwrap(),
                    metrics.pde_loss.last().unwrap(),
                    metrics.bc_loss.last().unwrap(),
                    metrics.ic_loss.last().unwrap()
                );
            }
        }

        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        Ok(metrics)
    }

    /// Make predictions at new points
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinates for prediction
    /// * `y` - Y-coordinates for prediction
    /// * `z` - Z-coordinates for prediction
    /// * `t` - Time coordinates for prediction
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// Predicted displacement/pressure values u(x, y, z, t)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let x_test = vec![0.5, 0.6];
    /// let y_test = vec![0.5, 0.5];
    /// let z_test = vec![0.5, 0.5];
    /// let t_test = vec![0.5, 0.5];
    ///
    /// let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
    /// ```
    pub fn predict(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        t: &[f32],
        device: &B::Device,
    ) -> Result<Vec<f32>, String> {
        let x_tensor = Tensor::<B, 2>::from_data(TensorData::from(x), device).unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 2>::from_data(TensorData::from(y), device).unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 2>::from_data(TensorData::from(z), device).unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 2>::from_data(TensorData::from(t), device).unsqueeze_dim(1);

        let u_pred = self.pinn.forward(x_tensor, y_tensor, z_tensor, t_tensor);
        let u_vec = u_pred.into_data().as_slice::<f32>().unwrap().to_vec();

        Ok(u_vec)
    }

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
    /// - **bc_loss**: Boundary condition violations (placeholder: zero)
    /// - **ic_loss**: Initial condition violations (placeholder: zero)
    /// - **total_loss**: Weighted sum of all components
    fn compute_physics_loss(
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
        weights: &BurnLossWeights3D,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
    ) {
        // Data loss: MSE between predictions and training data
        let u_pred = self.pinn.forward(x_data, y_data, z_data, t_data);
        let data_loss = (u_pred.clone() - u_data).powf_scalar(2.0).mean();

        // PDE loss: MSE of PDE residual at collocation points
        let pde_residual =
            self.pinn
                .compute_pde_residual(x_colloc, y_colloc, z_colloc, t_colloc, |x, y, z| {
                    self.get_wave_speed(x, y, z)
                });
        let pde_loss = pde_residual.powf_scalar(2.0).mean();

        // Boundary condition loss (placeholder: to be implemented with BC enforcement)
        let bc_loss = Tensor::<B, 1>::zeros([1], &u_pred.device());

        // Initial condition loss (placeholder: to be implemented with IC enforcement)
        let ic_loss = Tensor::<B, 1>::zeros([1], &u_pred.device());

        // Total weighted loss
        let total_loss = weights.data_weight * data_loss.clone()
            + weights.pde_weight * pde_loss.clone()
            + weights.bc_weight * bc_loss.clone()
            + weights.ic_weight * ic_loss.clone();

        (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    }

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
    fn generate_collocation_points(
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

        let x_tensor = Tensor::<B, 2>::from_data(TensorData::from(x_points.as_slice()), device)
            .unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 2>::from_data(TensorData::from(y_points.as_slice()), device)
            .unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 2>::from_data(TensorData::from(z_points.as_slice()), device)
            .unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 2>::from_data(TensorData::from(t_points.as_slice()), device)
            .unsqueeze_dim(1);

        (x_tensor, y_tensor, z_tensor, t_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_solver_creation() {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);

        assert!(!solver.pinn.hidden_layers.is_empty());
        assert!(solver.wave_speed_fn.is_some());
    }

    #[test]
    fn test_solver_get_wave_speed() {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, z: f32| if z < 0.5 { 1500.0 } else { 3000.0 };

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);

        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.3), 1500.0);
        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7), 3000.0);
    }

    #[test]
    fn test_solver_training_smoke() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            num_collocation_points: 10,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);

        // Minimal training data
        let x_data = vec![0.5, 0.6];
        let y_data = vec![0.5, 0.5];
        let z_data = vec![0.5, 0.5];
        let t_data = vec![0.1, 0.2];
        let u_data = vec![0.0, 0.0];

        let result = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 5);

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.epochs_completed, 5);
        assert_eq!(metrics.total_loss.len(), 5);
    }

    #[test]
    fn test_solver_prediction() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);

        let x_test = vec![0.5, 0.6];
        let y_test = vec![0.5, 0.5];
        let z_test = vec![0.5, 0.5];
        let t_test = vec![0.1, 0.2];

        let result = solver.predict(&x_test, &y_test, &z_test, &t_test, &device);

        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_collocation_points_generation() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device);

        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Should generate approximately the requested number (some may be filtered)
        let n_generated = x_colloc.shape().dims[0];
        assert!(n_generated > 0 && n_generated <= config.num_collocation_points);
        assert_eq!(y_colloc.shape().dims[0], n_generated);
        assert_eq!(z_colloc.shape().dims[0], n_generated);
        assert_eq!(t_colloc.shape().dims[0], n_generated);
    }

    #[test]
    fn test_collocation_points_spherical_geometry() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 100,
            ..Default::default()
        };
        let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device);

        let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Spherical geometry filters many points, so expect fewer than requested
        let n_generated = x_colloc.shape().dims[0];
        assert!(n_generated > 0);
        assert!(n_generated < config.num_collocation_points);
    }

    #[test]
    fn test_training_loss_components() {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            num_collocation_points: 10,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let mut solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);

        let x_data = vec![0.5];
        let y_data = vec![0.5];
        let z_data = vec![0.5];
        let t_data = vec![0.1];
        let u_data = vec![0.0];

        let result = solver.train(&x_data, &y_data, &z_data, &t_data, &u_data, &device, 3);

        assert!(result.is_ok());
        let metrics = result.unwrap();

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
    }
}
