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

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};

use super::config::{BurnLossWeights3D, BurnPINN3DConfig, BurnTrainingMetrics3D};
use super::geometry::Geometry3D;
use super::network::PINN3DNetwork;
use super::optimizer::SimpleOptimizer3D;
use super::wavespeed::WaveSpeedFn3D;

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
struct LossScales {
    data_scale: f32,
    pde_scale: f32,
    bc_scale: f32,
    ic_scale: f32,
    ema_alpha: f32,
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

/// Gradient diagnostics for monitoring training stability
///
/// Tracks parameter update magnitudes as a proxy for gradient norms.
/// This provides insight into gradient flow without requiring direct
/// access to Burn's opaque Gradients type.
///
/// # Mathematical Specification
///
/// Parameter update norm: ||Δθ||₂ = ||θ_new - θ_old||₂
/// Relative update: ||Δθ||₂ / (||θ_old||₂ + ε)
///
/// These metrics help detect:
/// - Gradient explosion (large updates)
/// - Vanishing gradients (tiny updates)
/// - Training stagnation (near-zero updates)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for future Burn API gradient introspection
struct GradientDiagnostics {
    /// L2 norm of parameter updates ||Δθ||₂
    pub update_norm: f64,
    /// Relative update magnitude ||Δθ||₂ / ||θ||₂
    pub relative_update: f64,
    /// Maximum absolute parameter change
    pub max_update: f64,
}

impl GradientDiagnostics {
    /// Compute diagnostics by comparing old and new parameters
    ///
    /// # Arguments
    ///
    /// * `old_params` - Parameters before optimizer step
    /// * `new_params` - Parameters after optimizer step
    ///
    /// # Returns
    ///
    /// Gradient diagnostics including update norms and relative changes
    #[allow(dead_code)] // Reserved for future use when Burn exposes parameter access
    fn compute<B: Backend>(
        old_params: &[Tensor<B, 2>],
        new_params: &[Tensor<B, 2>],
    ) -> KwaversResult<Self> {
        if old_params.len() != new_params.len() {
            return Err(KwaversError::InvalidInput(
                "Parameter count mismatch between old and new".into(),
            ));
        }

        let mut update_norm_sq = 0.0_f64;
        let mut param_norm_sq = 0.0_f64;
        let mut max_update = 0.0_f64;

        for (old, new) in old_params.iter().zip(new_params.iter()) {
            // Compute parameter difference: Δθ = θ_new - θ_old
            let diff = new.clone().sub(old.clone());

            // Extract values for norm computation
            let diff_data = diff.to_data();
            let old_data = old.to_data();

            let diff_values: Vec<f32> = diff_data
                .to_vec()
                .map_err(|_| KwaversError::InvalidInput("Failed to extract diff values".into()))?;

            let old_values: Vec<f32> = old_data
                .to_vec()
                .map_err(|_| KwaversError::InvalidInput("Failed to extract param values".into()))?;

            // Accumulate squared norms
            for &val in &diff_values {
                let val_f64 = val as f64;
                update_norm_sq += val_f64 * val_f64;
                max_update = max_update.max(val_f64.abs());
            }

            for &val in &old_values {
                let val_f64 = val as f64;
                param_norm_sq += val_f64 * val_f64;
            }
        }

        let update_norm = update_norm_sq.sqrt();
        let param_norm = param_norm_sq.sqrt();
        let relative_update = if param_norm > 1e-12 {
            update_norm / param_norm
        } else {
            update_norm // If params are near-zero, just report absolute
        };

        Ok(Self {
            update_norm,
            relative_update,
            max_update,
        })
    }
}

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
    /// use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::{
    ///     BurnPINN3DWave, BurnPINN3DConfig, Geometry3D
    /// };
    ///
    /// type Backend = NdArray<f32>;
    /// let device = Default::default();
    /// let config = BurnPINN3DConfig::default();
    /// let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    /// let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;
    ///
    /// let solver = BurnPINN3DWave::<Backend>::new(config, geometry, wave_speed, &device)?;
    /// ```
    pub fn new<F>(
        config: BurnPINN3DConfig,
        geometry: Geometry3D,
        wave_speed_fn: F,
        device: &B::Device,
    ) -> KwaversResult<Self>
    where
        F: Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
    {
        config.validate()?;
        let pinn = PINN3DNetwork::new(&config, device)?;
        let optimizer = SimpleOptimizer3D::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            geometry: Ignored(geometry),
            wave_speed_fn: Some(WaveSpeedFn3D::new(std::sync::Arc::new(wave_speed_fn))),
            optimizer: Ignored(optimizer),
            config: Ignored(config),
            _backend: PhantomData,
        })
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
    /// Wave speed c(x, y, z) in m/s
    pub fn get_wave_speed(&self, x: f32, y: f32, z: f32) -> KwaversResult<f32> {
        let wave_speed = self
            .wave_speed_fn
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Wave speed function is missing".into()))?
            .get(x, y, z);
        if !wave_speed.is_finite() || wave_speed <= 0.0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "wave_speed".to_string(),
                value: wave_speed as f64,
                reason: "must be finite and > 0".to_string(),
            }));
        }
        Ok(wave_speed)
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
    ) -> KwaversResult<Vec<f32>> {
        let n = x.len();
        if n == 0 {
            return Err(KwaversError::InvalidInput(
                "Prediction inputs must be non-empty".into(),
            ));
        }
        if y.len() != n || z.len() != n || t.len() != n {
            return Err(KwaversError::InvalidInput(
                "x, y, z, and t must have equal length".into(),
            ));
        }

        let x_tensor = Tensor::<B, 2>::from_data(TensorData::new(x.to_vec(), [n, 1]), device);
        let y_tensor = Tensor::<B, 2>::from_data(TensorData::new(y.to_vec(), [n, 1]), device);
        let z_tensor = Tensor::<B, 2>::from_data(TensorData::new(z.to_vec(), [n, 1]), device);
        let t_tensor = Tensor::<B, 2>::from_data(TensorData::new(t.to_vec(), [n, 1]), device);

        let u_pred = self.pinn.forward(x_tensor, y_tensor, z_tensor, t_tensor);
        let u_vec = Self::tensor_column_vec_f32(&u_pred)?;

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
    /// - **bc_loss**: Boundary condition violations
    /// - **ic_loss**: Initial condition violations
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
        let data_loss_val = Self::scalar_f32(&data_loss_raw).unwrap_or(1.0);
        let pde_loss_val = Self::scalar_f32(&pde_loss_raw).unwrap_or(1.0);
        let bc_loss_val = Self::scalar_f32(&bc_loss_raw).unwrap_or(1.0);
        let ic_loss_val = Self::scalar_f32(&ic_loss_raw).unwrap_or(1.0);

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

    /// Compute temporal derivative ∂u/∂t at t=0 via forward finite difference
    ///
    /// Uses forward difference: ∂u/∂t(0) ≈ (u(ε) - u(0)) / ε
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinates at t=0
    /// * `y` - Y-coordinates at t=0
    /// * `z` - Z-coordinates at t=0
    /// * `t` - Time coordinates (should be t=0)
    ///
    /// # Returns
    ///
    /// Tensor containing ∂u/∂t values at the specified points
    fn compute_temporal_derivative_at_t0(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        t: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let eps = 1e-3_f32;

        // u(t=0)
        let u_t0 = self
            .pinn
            .forward(x.clone(), y.clone(), z.clone(), t.clone());

        // u(t=ε)
        let t_eps = t.add_scalar(eps);
        let u_t_eps = self.pinn.forward(x, y, z, t_eps);

        // Forward difference: ∂u/∂t ≈ (u(ε) - u(0)) / ε
        u_t_eps.sub(u_t0).div_scalar(eps)
    }

    /// Extract velocity initial condition tensor from training data
    ///
    /// Finds all points at t=0 and extracts their velocity values.
    ///
    /// # Arguments
    ///
    /// * `x_data` - X-coordinates of training data
    /// * `y_data` - Y-coordinates of training data
    /// * `z_data` - Z-coordinates of training data
    /// * `t_data` - Time coordinates of training data
    /// * `v_data` - Velocity values (∂u/∂t)
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// Tensor [n_ic, 1] containing velocity IC values
    fn extract_velocity_initial_condition_tensor(
        _x_data: &[f32],
        _y_data: &[f32],
        _z_data: &[f32],
        t_data: &[f32],
        v_data: &[f32],
        device: &B::Device,
    ) -> KwaversResult<Tensor<B, 2>> {
        if v_data.len() != t_data.len() {
            return Err(KwaversError::InvalidInput(
                "v_data and t_data must have equal length".into(),
            ));
        }

        let min_t = t_data.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
        if !min_t.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Training time coordinates must be finite".into(),
            ));
        }

        let eps = 1e-6_f32;
        let mut v_ic = Vec::new();

        for i in 0..t_data.len() {
            if (t_data[i] - min_t).abs() <= eps {
                v_ic.push(v_data[i]);
            }
        }

        if v_ic.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No initial-condition velocity samples found in training data".into(),
            ));
        }

        let n_ic = v_ic.len();
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(v_ic, [n_ic, 1]),
            device,
        ))
    }

    fn extract_initial_condition_tensors(
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
        device: &B::Device,
    ) -> KwaversResult<(
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
    )> {
        let min_t = t_data.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
        if !min_t.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Training time coordinates must be finite".into(),
            ));
        }

        let eps = 1e-6_f32;
        let mut x_ic = Vec::new();
        let mut y_ic = Vec::new();
        let mut z_ic = Vec::new();
        let mut u_ic = Vec::new();

        for i in 0..t_data.len() {
            if (t_data[i] - min_t).abs() <= eps {
                x_ic.push(x_data[i]);
                y_ic.push(y_data[i]);
                z_ic.push(z_data[i]);
                u_ic.push(u_data[i]);
            }
        }

        if x_ic.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No initial-condition samples found in training data".into(),
            ));
        }

        let n_ic = x_ic.len();
        let t_ic = vec![min_t; n_ic];

        Ok((
            Tensor::<B, 2>::from_data(TensorData::new(x_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(y_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(z_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(t_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(u_ic, [n_ic, 1]), device),
        ))
    }

    /// Extract all network parameters as a vector of tensors
    ///
    /// Used for gradient diagnostics by comparing parameters before/after optimizer step.
    /// This provides a workaround for Burn's opaque Gradients type.
    ///
    /// # Returns
    ///
    /// Vector of parameter tensors (weights and biases from all layers)
    #[allow(dead_code)] // Reserved for future gradient diagnostics when Burn API ready
    fn extract_parameters(&self) -> Vec<Tensor<B, 2>> {
        // Note: Burn's Module trait doesn't expose internal parameters directly.
        // As a workaround, we use num_params() to get a rough estimate of total parameters.
        // For now, return an empty vector - gradient diagnostics will be disabled
        // until Burn provides parameter introspection API.
        //
        // KNOWN_LIMITATION: Blocked on Burn parameter introspection API.
        // Alternative: Track loss history and use loss gradient as proxy.
        Vec::new()
    }

    fn scalar_f32(t: &Tensor<B, 1>) -> KwaversResult<f32> {
        let data = t.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            })
        })?;
        if slice.len() != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "len=1".to_string(),
                    actual: format!("len={}", slice.len()),
                },
            ));
        }
        slice.first().copied().ok_or_else(|| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_scalar_extract".to_string(),
                reason: "missing scalar element".to_string(),
            })
        })
    }

    fn tensor_column_vec_f32(t: &Tensor<B, 2>) -> KwaversResult<Vec<f32>> {
        let shape = t.shape();
        let dims = shape.dims.as_slice();
        let [n, m] = dims else {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{dims:?}"),
                },
            ));
        };
        if *m != 1 {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: "[N, 1]".to_string(),
                    actual: format!("{dims:?}"),
                },
            ));
        }
        let data = t.clone().into_data();
        let slice = data.as_slice::<f32>().map_err(|e| {
            KwaversError::System(SystemError::InvalidOperation {
                operation: "tensor_to_f32_slice".to_string(),
                reason: format!("{e:?}"),
            })
        })?;
        if slice.len() != *n {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("len={n}"),
                    actual: format!("len={}", slice.len()),
                },
            ));
        }
        Ok(slice.to_vec())
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
    pub(crate) fn generate_collocation_points(
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

        if x_points.is_empty() {
            let (x, y, z) = self.geometry.0.interior_point();
            x_points.push(x as f32);
            y_points.push(y as f32);
            z_points.push(z as f32);
            t_points.push(0.0);
        }

        let x_tensor = Tensor::<B, 1>::from_data(TensorData::from(x_points.as_slice()), device)
            .unsqueeze_dim(1);
        let y_tensor = Tensor::<B, 1>::from_data(TensorData::from(y_points.as_slice()), device)
            .unsqueeze_dim(1);
        let z_tensor = Tensor::<B, 1>::from_data(TensorData::from(z_points.as_slice()), device)
            .unsqueeze_dim(1);
        let t_tensor = Tensor::<B, 1>::from_data(TensorData::from(t_points.as_slice()), device)
            .unsqueeze_dim(1);

        (x_tensor, y_tensor, z_tensor, t_tensor)
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
    fn compute_bc_loss_internal(
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
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_solver_creation() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        assert!(solver.pinn.hidden_layer_count() > 0);
        assert!(solver.wave_speed_fn.is_some());
        Ok(())
    }

    #[test]
    fn test_solver_get_wave_speed() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig::default();
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, z: f32| if z < 0.5 { 1500.0 } else { 3000.0 };

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.3)?, 1500.0);
        assert_eq!(solver.get_wave_speed(0.5, 0.5, 0.7)?, 3000.0);
        Ok(())
    }

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

    #[test]
    fn test_solver_prediction() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            hidden_layers: vec![8],
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver = BurnPINN3DWave::<TestBackend>::new(config, geometry, wave_speed, &device)?;

        let x_test = vec![0.5, 0.6];
        let y_test = vec![0.5, 0.5];
        let z_test = vec![0.5, 0.5];
        let t_test = vec![0.1, 0.2];

        let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p.is_finite()));
        Ok(())
    }

    #[test]
    fn test_collocation_points_generation() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 50,
            ..Default::default()
        };
        let geometry = Geometry3D::rectangular(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

        let (x_colloc, y_colloc, z_colloc, t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Should generate approximately the requested number (some may be filtered)
        let n_generated = match x_colloc.shape().dims.as_slice() {
            [n, _] => *n,
            dims => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "generate_collocation_points".to_string(),
                    reason: format!("Expected 2D tensor shape, got dims {dims:?}"),
                }));
            }
        };
        assert!(n_generated > 0 && n_generated <= config.num_collocation_points);
        for (name, tensor) in [
            ("y_colloc", &y_colloc),
            ("z_colloc", &z_colloc),
            ("t_colloc", &t_colloc),
        ] {
            let n = match tensor.shape().dims.as_slice() {
                [n, _] => *n,
                dims => {
                    return Err(KwaversError::System(SystemError::InvalidOperation {
                        operation: "generate_collocation_points".to_string(),
                        reason: format!("Expected 2D tensor shape for {name}, got dims {dims:?}"),
                    }));
                }
            };
            assert_eq!(n, n_generated);
        }
        Ok(())
    }

    #[test]
    fn test_collocation_points_spherical_geometry() -> KwaversResult<()> {
        let device = Default::default();
        let config = BurnPINN3DConfig {
            num_collocation_points: 100,
            ..Default::default()
        };
        let geometry = Geometry3D::spherical(0.5, 0.5, 0.5, 0.3);
        let wave_speed = |_x: f32, _y: f32, _z: f32| 1500.0;

        let solver =
            BurnPINN3DWave::<TestBackend>::new(config.clone(), geometry, wave_speed, &device)?;

        let (x_colloc, _y_colloc, _z_colloc, _t_colloc) =
            solver.generate_collocation_points(&config, &device);

        // Spherical geometry filters many points, so expect fewer than requested
        let n_generated = match x_colloc.shape().dims.as_slice() {
            [n, _] => *n,
            dims => {
                return Err(KwaversError::System(SystemError::InvalidOperation {
                    operation: "generate_collocation_points".to_string(),
                    reason: format!("Expected 2D tensor shape, got dims {dims:?}"),
                }));
            }
        };
        assert!(n_generated > 0);
        assert!(n_generated < config.num_collocation_points);
        Ok(())
    }

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
