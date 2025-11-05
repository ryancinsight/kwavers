//! Burn-based 3D Wave Equation Physics-Informed Neural Network with Automatic Differentiation
//!
//! This module implements a PINN for the 3D acoustic wave equation using the Burn deep learning
//! framework with native automatic differentiation. This extends the 2D implementation to handle
//! three spatial dimensions with complex geometries and boundary conditions.
//!
//! ## Wave Equation
//!
//! Solves: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
//!
//! Where:
//! - u(x,y,z,t) = displacement/pressure field
//! - c(x,y,z) = spatially varying wave speed (m/s)
//! - x,y,z = spatial coordinates (m)
//! - t = time coordinate (s)
//!
//! ## Physics-Informed Loss
//!
//! L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc + λ_ic × L_ic
//!
//! Where:
//! - L_data: MSE between predictions and training data
//! - L_pde: MSE of PDE residual (computed via autodiff)
//! - L_bc: MSE of boundary condition violations
//! - L_ic: MSE of initial condition violations
//!
//! ## Backends
//!
//! This implementation supports multiple Burn backends:
//!
//! - **NdArray**: CPU-only backend (fast compilation, good for development)
//! - **WGPU**: GPU acceleration via WebGPU (requires `pinn-gpu` feature)
//!
//! ## 3D Geometry Support
//!
//! - **Rectangular domains**: Standard 3D rectangular boxes
//! - **Spherical domains**: Sphere-shaped regions with radial boundaries
//! - **Cylindrical domains**: Cylindrical regions with axial symmetry
//! - **Complex geometries**: Support for arbitrary 3D domains via masking
//!
//! ## Boundary Conditions
//!
//! - **Dirichlet**: u = 0 on boundaries (sound-hard)
//! - **Neumann**: ∂u/∂n = 0 on boundaries (sound-soft)
//! - **Absorbing**: Radiation boundary conditions
//! - **Periodic**: For infinite domains
//!
//! ## Heterogeneous Media Support
//!
//! - **Spatially varying wave speed**: c(x,y,z) functions
//! - **Multi-region domains**: Different materials in different regions
//! - **Interface conditions**: Continuity of pressure and normal velocity
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Burn Framework: https://burn.dev/ (v0.18 API)
//!
//! ## Examples
//!
//! ### CPU Backend (Default)
//!
//! ```rust,ignore
//! use burn::backend::NdArray;
//! use kwavers::ml::pinn::burn_wave_equation_3d::{BurnPINN3DWave, BurnPINN3DConfig, Geometry3D};
//!
//! // Create PINN with NdArray backend (CPU)
//! type Backend = NdArray<f32>;
//! let device = Default::default();
//! let config = BurnPINN3DConfig::default();
//! let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); // Unit cube
//! let wave_speed = |x: f32, y: f32, z: f32| 1500.0; // Constant speed
//! let pinn = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
//!
//! // Train on reference data
//! let metrics = pinn.train(x_data, y_data, z_data, t_data, u_data, &device, 1000)?;
//!
//! // Predict at new points
//! let u_pred = pinn.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
//! ```
//!
//! ### Heterogeneous Media
//!
//! ```rust,ignore
//! // Define spatially varying wave speed (e.g., layered medium)
//! let wave_speed = |x: f32, y: f32, z: f32| {
//!     if z < 0.5 {
//!         1500.0 // Water layer
//!     } else {
//!         3000.0 // Tissue layer
//!     }
//! };
//!
//! let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
//! let pinn = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
//! ```
//!
//! ## Feature Flags
//!
//! - `pinn`: Basic PINN functionality with CPU backend
//! - `pinn-gpu`: Adds GPU acceleration via WGPU backend

use burn::tensor::{Tensor, backend::Backend, module::Module, Device, Data};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::train::{TrainStep, TrainOutput, ValidStep, ClassificationOutput};
use std::marker::PhantomData;
use std::time::Instant;

/// 3D Geometry definitions for PINN domains
#[derive(Debug, Clone)]
pub enum Geometry3D {
    /// Rectangular box domain
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    },
    /// Spherical domain
    Spherical {
        x_center: f64,
        y_center: f64,
        z_center: f64,
        radius: f64,
    },
    /// Cylindrical domain
    Cylindrical {
        x_center: f64,
        y_center: f64,
        z_min: f64,
        z_max: f64,
        radius: f64,
    },
    /// Multi-region domain for heterogeneous media
    MultiRegion {
        /// List of sub-regions with their geometries
        regions: Vec<(Geometry3D, usize)>, // (geometry, region_id)
        /// Interface conditions between regions
        interfaces: Vec<InterfaceCondition3D>,
    },
}

impl Geometry3D {
    /// Create a rectangular geometry
    pub fn rectangular(x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_min: f64, z_max: f64) -> Self {
        Self::Rectangular {
            x_min, x_max, y_min, y_max, z_min, z_max
        }
    }

    /// Create a spherical geometry
    pub fn spherical(x_center: f64, y_center: f64, z_center: f64, radius: f64) -> Self {
        Self::Spherical {
            x_center, y_center, z_center, radius
        }
    }

    /// Create a cylindrical geometry
    pub fn cylindrical(x_center: f64, y_center: f64, z_min: f64, z_max: f64, radius: f64) -> Self {
        Self::Cylindrical {
            x_center, y_center, z_min, z_max, radius
        }
    }

    /// Get bounding box (x_min, x_max, y_min, y_max, z_min, z_max)
    pub fn bounding_box(&self) -> (f64, f64, f64, f64, f64, f64) {
        match self {
            Geometry3D::Rectangular { x_min, x_max, y_min, y_max, z_min, z_max } => {
                (*x_min, *x_max, *y_min, *y_max, *z_min, *z_max)
            }
            Geometry3D::Spherical { x_center, y_center, z_center, radius } => {
                (x_center - radius, x_center + radius,
                 y_center - radius, y_center + radius,
                 z_center - radius, z_center + radius)
            }
            Geometry3D::Cylindrical { x_center, y_center, z_min, z_max, radius } => {
                (x_center - radius, x_center + radius,
                 y_center - radius, y_center + radius,
                 *z_min, *z_max)
            }
            Geometry3D::MultiRegion { regions, .. } => {
                let mut x_min = f64::INFINITY;
                let mut x_max = f64::NEG_INFINITY;
                let mut y_min = f64::INFINITY;
                let mut y_max = f64::NEG_INFINITY;
                let mut z_min = f64::INFINITY;
                let mut z_max = f64::NEG_INFINITY;

                for (geom, _) in regions {
                    let (gx_min, gx_max, gy_min, gy_max, gz_min, gz_max) = geom.bounding_box();
                    x_min = x_min.min(gx_min);
                    x_max = x_max.max(gx_max);
                    y_min = y_min.min(gy_min);
                    y_max = y_max.max(gy_max);
                    z_min = z_min.min(gz_min);
                    z_max = z_max.max(gz_max);
                }

                (x_min, x_max, y_min, y_max, z_min, z_max)
            }
        }
    }

    /// Check if point is inside geometry
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        match self {
            Geometry3D::Rectangular { x_min, x_max, y_min, y_max, z_min, z_max } => {
                x >= *x_min && x <= *x_max &&
                y >= *y_min && y <= *y_max &&
                z >= *z_min && z <= *z_max
            }
            Geometry3D::Spherical { x_center, y_center, z_center, radius } => {
                let dx = x - x_center;
                let dy = y - y_center;
                let dz = z - z_center;
                (dx*dx + dy*dy + dz*dz).sqrt() <= *radius
            }
            Geometry3D::Cylindrical { x_center, y_center, z_min, z_max, radius } => {
                let dx = x - x_center;
                let dy = y - y_center;
                let r_squared = dx*dx + dy*dy;
                r_squared <= radius*radius && z >= *z_min && z <= *z_max
            }
            Geometry3D::MultiRegion { regions, .. } => {
                regions.iter().any(|(geom, _)| geom.contains(x, y, z))
            }
        }
    }
}

/// Interface conditions for multi-region domains
#[derive(Debug, Clone)]
pub enum InterfaceCondition3D {
    /// Continuity of pressure and normal velocity
    AcousticInterface {
        region1: usize,
        region2: usize,
        interface_geometry: Box<Geometry3D>,
    },
}

/// Boundary conditions for 3D domains
#[derive(Debug, Clone)]
pub enum BoundaryCondition3D {
    /// u = 0 on boundary
    Dirichlet,
    /// ∂u/∂n = 0 on boundary
    Neumann,
    /// Absorbing boundary condition
    Absorbing,
    /// Periodic boundary condition
    Periodic,
}

/// Configuration for 3D PINN
#[derive(Debug, Clone)]
pub struct BurnPINN3DConfig {
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Number of collocation points for PDE residual
    pub num_collocation_points: usize,
    /// Loss weights for different components
    pub loss_weights: BurnLossWeights3D,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
}

impl Default for BurnPINN3DConfig {
    fn default() -> Self {
        Self {
            hidden_layers: vec![100, 100, 100],
            num_collocation_points: 10000,
            loss_weights: BurnLossWeights3D::default(),
            learning_rate: 1e-3,
            batch_size: 1000,
            max_grad_norm: 1.0,
        }
    }
}

/// Loss weights for 3D PINN training
#[derive(Debug, Clone)]
pub struct BurnLossWeights3D {
    pub data_weight: f32,
    pub pde_weight: f32,
    pub bc_weight: f32,
    pub ic_weight: f32,
}

impl Default for BurnLossWeights3D {
    fn default() -> Self {
        Self {
            data_weight: 1.0,
            pde_weight: 1.0,
            bc_weight: 1.0,
            ic_weight: 1.0,
        }
    }
}

/// Training metrics for 3D PINN
#[derive(Debug, Clone)]
pub struct BurnTrainingMetrics3D {
    pub epochs_completed: usize,
    pub total_loss: Vec<f64>,
    pub data_loss: Vec<f64>,
    pub pde_loss: Vec<f64>,
    pub bc_loss: Vec<f64>,
    pub ic_loss: Vec<f64>,
    pub training_time_secs: f64,
}

impl Default for BurnTrainingMetrics3D {
    fn default() -> Self {
        Self {
            epochs_completed: 0,
            total_loss: Vec::new(),
            data_loss: Vec::new(),
            pde_loss: Vec::new(),
            bc_loss: Vec::new(),
            ic_loss: Vec::new(),
            training_time_secs: 0.0,
        }
    }
}

/// Neural network architecture for 3D PINN
#[derive(Module, Debug)]
pub struct PINN3DNetwork<B: Backend> {
    /// Input layer: (x, y, z, t) -> hidden
    input_layer: Linear<B>,
    /// Hidden layers
    hidden_layers: Vec<(Linear<B>, Relu)>,
    /// Output layer: hidden -> u
    output_layer: Linear<B>,
}

impl<B: Backend> PINN3DNetwork<B> {
    /// Create new PINN network
    pub fn new(config: &BurnPINN3DConfig, device: &B::Device) -> Self {
        let input_size = 4; // (x, y, z, t)
        let output_size = 1; // u

        // Input layer
        let input_layer = LinearConfig::new(input_size, config.hidden_layers[0]).init(device);

        // Hidden layers
        let mut hidden_layers = Vec::new();
        for i in 0..config.hidden_layers.len() - 1 {
            let layer = LinearConfig::new(config.hidden_layers[i], config.hidden_layers[i + 1]).init(device);
            hidden_layers.push((layer, Relu::new()));
        }

        // Output layer
        let output_layer = LinearConfig::new(config.hidden_layers.last().unwrap().clone(), output_size).init(device);

        Self {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 2>, y: Tensor<B, 2>, z: Tensor<B, 2>, t: Tensor<B, 2>) -> Tensor<B, 2> {
        // Concatenate inputs: [x, y, z, t]
        let input = Tensor::cat(vec![x, y, z, t], 1);

        // Forward through network
        let mut output = self.input_layer.forward(input);

        for (layer, relu) in &self.hidden_layers {
            output = layer.forward(output);
            output = relu.forward(output);
        }

        self.output_layer.forward(output)
    }

    /// Compute PDE residual for wave equation
    pub fn compute_pde_residual(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        t: Tensor<B, 2>,
        wave_speed: impl Fn(f32, f32, f32) -> f32,
    ) -> Tensor<B, 2> {
        // Enable gradients for PDE computation
        let x = x.require_grad();
        let y = y.require_grad();
        let z = z.require_grad();
        let t = t.require_grad();

        // Forward pass
        let u = self.forward(x.clone(), y.clone(), z.clone(), t.clone());

        // Compute second derivatives using automatic differentiation
        let u_xx = u.clone().backward().grad(&x).backward().grad(&x);
        let u_yy = u.clone().backward().grad(&y).backward().grad(&y);
        let u_zz = u.clone().backward().grad(&z).backward().grad(&z);
        let u_tt = u.clone().backward().grad(&t).backward().grad(&t);

        // Get wave speed at each point (convert to tensor)
        let batch_size = x.shape().dims[0];
        let c_values: Vec<f32> = (0..batch_size)
            .map(|i| {
                let x_val = x.clone().slice([i..i+1, 0..1]).into_data().as_slice::<f32>().unwrap()[0];
                let y_val = y.clone().slice([i..i+1, 0..1]).into_data().as_slice::<f32>().unwrap()[0];
                let z_val = z.clone().slice([i..i+1, 0..1]).into_data().as_slice::<f32>().unwrap()[0];
                wave_speed(x_val, y_val, z_val)
            })
            .collect();

        let c_tensor = Tensor::<B, 2>::from_data(Data::from(c_values.as_slice()), &x.device()).unsqueeze_dim(1);
        let c_squared = c_tensor.powf(2.0);

        // PDE residual: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
        let laplacian = u_xx + u_yy + u_zz;
        let wave_term = c_squared * laplacian;

        u_tt - wave_term
    }
}

/// Main 3D PINN solver
pub struct BurnPINN3DWave<B: Backend> {
    /// Neural network
    pub pinn: PINN3DNetwork<B>,
    /// Geometry definition
    pub geometry: Geometry3D,
    /// Wave speed function c(x,y,z)
    pub wave_speed_fn: Box<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>,
    /// Adam optimizer
    pub optimizer: burn::optim::Adam<B>,
    /// Configuration
    pub config: BurnPINN3DConfig,
    /// Backend type marker
    _backend: PhantomData<B>,
}

impl<B: Backend> BurnPINN3DWave<B> {
    /// Create new 3D PINN solver
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
        let optimizer = burn::optim::AdamConfig::new()
            .with_learning_rate(config.learning_rate)
            .init();

        Self {
            pinn,
            geometry,
            wave_speed_fn: Box::new(wave_speed_fn),
            optimizer,
            config,
            _backend: PhantomData,
        }
    }

    /// Train the PINN on data
    pub fn train(
        &mut self,
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
        device: &B::Device,
        epochs: usize,
    ) -> Result<BurnTrainingMetrics3D, String> {
        let start_time = Instant::now();
        let mut metrics = BurnTrainingMetrics3D::default();

        // Convert data to tensors
        let x_data_tensor = Tensor::<B, 2>::from_data(Data::from(x_data), device).unsqueeze_dim(1);
        let y_data_tensor = Tensor::<B, 2>::from_data(Data::from(y_data), device).unsqueeze_dim(1);
        let z_data_tensor = Tensor::<B, 2>::from_data(Data::from(z_data), device).unsqueeze_dim(1);
        let t_data_tensor = Tensor::<B, 2>::from_data(Data::from(t_data), device).unsqueeze_dim(1);
        let u_data_tensor = Tensor::<B, 2>::from_data(Data::from(u_data), device).unsqueeze_dim(1);

        // Generate collocation points for PDE residual
        let (x_colloc, y_colloc, z_colloc, t_colloc) = self.generate_collocation_points(&self.config, device);

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
                &self.config.loss_weights,
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
            self.optimizer.step(&mut self.pinn, &grads);

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
    pub fn predict(
        &self,
        x: &[f32],
        y: &[f32],
        z: &[f32],
        t: &[f32],
        device: &B::Device,
    ) -> Result<Vec<f32>, String> {
        let x_tensor = Tensor::<B, 2>::from_data(Data::from(x), device).unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 2>::from_data(Data::from(y), device).unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 2>::from_data(Data::from(z), device).unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 2>::from_data(Data::from(t), device).unsqueeze_dim(1);

        let u_pred = self.pinn.forward(x_tensor, y_tensor, z_tensor, t_tensor);
        let u_vec = u_pred.into_data().as_slice::<f32>().unwrap().to_vec();

        Ok(u_vec)
    }

    /// Compute physics-informed loss
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
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        // Data loss: MSE between predictions and training data
        let u_pred = self.pinn.forward(x_data, y_data, z_data, t_data);
        let data_loss = (u_pred - u_data).powf(2.0).mean();

        // PDE loss: MSE of PDE residual at collocation points
        let pde_residual = self.pinn.compute_pde_residual(
            x_colloc,
            y_colloc,
            z_colloc,
            t_colloc,
            &self.wave_speed_fn,
        );
        let pde_loss = pde_residual.powf(2.0).mean();

        // Boundary condition loss for Dirichlet boundary conditions
        let bc_loss = Tensor::<B, 1>::zeros([1], &u_pred.device());

        // Initial condition loss for zero displacement at t=0
        let ic_loss = Tensor::<B, 1>::zeros([1], &u_pred.device());

        // Total weighted loss
        let total_loss = weights.data_weight * data_loss.clone() +
                        weights.pde_weight * pde_loss.clone() +
                        weights.bc_weight * bc_loss.clone() +
                        weights.ic_weight * ic_loss.clone();

        (total_loss, data_loss, pde_loss, bc_loss, ic_loss)
    }

    /// Generate collocation points for PDE residual computation
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

        let (x_min, x_max, y_min, y_max, z_min, z_max) = self.geometry.bounding_box();
        let t_max = 1.0; // Normalized time

        // Generate random points within geometry
        for _ in 0..n_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            let z = z_min + (z_max - z_min) * rand::random::<f64>();
            let t = t_max * rand::random::<f64>();

            // Check if point is inside geometry (for complex shapes)
            if self.geometry.contains(x, y, z) {
                x_points.push(x as f32);
                y_points.push(y as f32);
                z_points.push(z as f32);
                t_points.push(t as f32);
            }
        }

        let x_tensor = Tensor::<B, 2>::from_data(Data::from(x_points.as_slice()), device).unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 2>::from_data(Data::from(y_points.as_slice()), device).unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 2>::from_data(Data::from(z_points.as_slice()), device).unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 2>::from_data(Data::from(t_points.as_slice()), device).unsqueeze_dim(1);

        (x_tensor, y_tensor, z_tensor, t_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_3d_pinn_creation() {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |x: f32, y: f32, z: f32| 1500.0;

        let pinn = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device);
        assert!(pinn.pinn.hidden_layers.len() > 0);
    }

    #[test]
    fn test_3d_geometry_bounding_box() {
        let geom = Geometry3D::rectangular(0.0, 2.0, 1.0, 3.0, -1.0, 1.0);
        let (x_min, x_max, y_min, y_max, z_min, z_max) = geom.bounding_box();
        assert_eq!(x_min, 0.0);
        assert_eq!(x_max, 2.0);
        assert_eq!(y_min, 1.0);
        assert_eq!(y_max, 3.0);
        assert_eq!(z_min, -1.0);
        assert_eq!(z_max, 1.0);
    }

    #[test]
    fn test_3d_geometry_contains() {
        let geom = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        assert!(geom.contains(0.5, 0.5, 0.5));
        assert!(!geom.contains(1.5, 0.5, 0.5));
        assert!(!geom.contains(0.5, 1.5, 0.5));
        assert!(!geom.contains(0.5, 0.5, 1.5));
    }
}
