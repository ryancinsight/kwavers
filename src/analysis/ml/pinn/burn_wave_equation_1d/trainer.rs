//! Training orchestration for 1D Wave Equation PINN
//!
//! This module implements the training loop and orchestration for physics-informed neural
//! networks solving the 1D acoustic wave equation. It coordinates data preparation,
//! collocation point generation, loss computation, optimization, and metrics tracking.
//!
//! ## Training Architecture
//!
//! The trainer implements a multi-objective optimization strategy combining:
//! 1. **Data Fidelity**: Matching observed/reference data points
//! 2. **Physics Constraint**: Enforcing wave equation PDE via collocation points
//! 3. **Boundary Conditions**: Satisfying domain boundary constraints
//!
//! ## Training Algorithm
//!
//! **Theorem (PINN Training Convergence)**: Under suitable conditions (network capacity,
//! learning rate, collocation density), the PINN training converges to a solution that
//! simultaneously fits data and satisfies the PDE (Raissi et al. 2019).
//!
//! ### Algorithm Steps
//!
//! For each epoch t = 1 to T:
//! 1. **Data Preparation**: Convert training data to tensors
//! 2. **Collocation Generation**: Sample N_colloc points in domain
//! 3. **Boundary Setup**: Define boundary condition points
//! 4. **Forward Pass**: Compute network predictions at all points
//! 5. **Loss Computation**: Calculate L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc
//! 6. **Backward Pass**: Compute gradients ∇L via autodiff
//! 7. **Parameter Update**: θ_new = θ_old - α·∇L
//! 8. **Metrics Recording**: Track loss components and convergence
//!
//! ## Collocation Point Strategy
//!
//! **Theorem (Coverage)**: Uniform collocation point distribution provides consistent
//! PDE constraint enforcement across the domain (quasi-Monte Carlo theory).
//!
//! Current implementation uses uniform grid sampling:
//! - x ∈ [-1, 1]: Normalized spatial domain
//! - t ∈ [0, T_max]: Time domain (normalized to [0, 1] for training)
//!
//! Future extensions:
//! - Adaptive refinement (sample more where residual is large)
//! - Latin hypercube sampling (better space-filling)
//! - Importance sampling (focus on high-error regions)
//!
//! ## Convergence Monitoring
//!
//! The trainer tracks multiple metrics to assess convergence:
//! - **Total Loss**: Overall objective function
//! - **Data Loss**: Prediction error on training data
//! - **PDE Loss**: Physics constraint violation
//! - **BC Loss**: Boundary condition violation
//!
//! **Convergence Criterion**: Training is considered converged when:
//! - Relative loss change < ε (e.g., 1e-6) over N epochs (e.g., 100)
//! - All loss components below target thresholds
//! - No numerical instabilities (NaN/Inf)
//!
//! ## References
//!
//! 1. **Raissi et al. (2019)**: "Physics-informed neural networks"
//!    Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//!
//! 2. **Karniadakis et al. (2021)**: "Physics-informed machine learning"
//!    Nature Reviews Physics, 3:422-440. DOI: 10.1038/s42254-021-00314-5
//!
//! 3. **Wang et al. (2021)**: "Understanding and mitigating gradient flow pathologies in
//!    physics-informed neural networks" SIAM Journal on Scientific Computing, 43(5).
//!
//! ## Examples
//!
//! ### Basic Training
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, NdArray};
//! use ndarray::Array1;
//!
//! type Backend = Autodiff<NdArray<f32>>;
//!
//! let device = Default::default();
//! let config = BurnPINNConfig::default();
//! let mut trainer = BurnPINNTrainer::<Backend>::new(config, &device)?;
//!
//! // Generate or load training data
//! let x_data = Array1::linspace(-1.0, 1.0, 100);
//! let t_data = Array1::linspace(0.0, 1.0, 100);
//! let u_data = generate_reference_solution(&x_data, &t_data);
//!
//! // Train with physics-informed loss
//! let metrics = trainer.train(
//!     &x_data,
//!     &t_data,
//!     &u_data,
//!     343.0,  // wave speed (m/s)
//!     &device,
//!     1000    // epochs
//! )?;
//!
//! // Check convergence
//! println!("Final total loss: {:.6e}", metrics.total_loss.last().unwrap());
//! println!("Training time: {:.2}s", metrics.training_time_secs);
//! ```
//!
//! ### GPU-Accelerated Training
//!
//! ```rust,ignore
//! use burn::backend::{Autodiff, Wgpu};
//!
//! type Backend = Autodiff<Wgpu<f32>>;
//!
//! // Initialize GPU device
//! let device = pollster::block_on(Wgpu::<f32>::default())?;
//!
//! // Use larger network for GPU
//! let config = BurnPINNConfig {
//!     hidden_layers: vec![100, 100, 100, 100],
//!     num_collocation_points: 50000,
//!     ..Default::default()
//! };
//!
//! let mut trainer = BurnPINNTrainer::<Backend>::new(config, &device)?;
//!
//! // Training is automatically GPU-accelerated
//! let metrics = trainer.train(&x_data, &t_data, &u_data, 343.0, &device, 5000)?;
//! ```

use ndarray::{Array1, Array2};

use burn::tensor::{backend::AutodiffBackend, Tensor};

use crate::core::error::{KwaversError, KwaversResult};

use super::{
    config::BurnPINNConfig, network::BurnPINN1DWave, optimizer::SimpleOptimizer,
    types::BurnTrainingMetrics,
};

/// PINN trainer for 1D wave equation with physics-informed learning
///
/// Orchestrates the training process by:
/// - Managing network and optimizer state
/// - Preparing training, collocation, and boundary data
/// - Computing physics-informed loss
/// - Updating network parameters
/// - Tracking training metrics and convergence
///
/// ## Design Pattern: Facade + Template Method
///
/// The trainer provides a high-level interface (Facade) that simplifies the complex
/// subsystem of PINN training. The `train()` method implements the Template Method
/// pattern, defining the training algorithm structure while allowing flexibility
/// in loss computation and optimization strategies.
///
/// ## Type Parameters
///
/// - `B`: Burn backend with autodiff support (e.g., Autodiff<NdArray<f32>>)
///
/// ## Examples
///
/// ```rust,ignore
/// use burn::backend::{Autodiff, NdArray};
///
/// type Backend = Autodiff<NdArray<f32>>;
///
/// let device = Default::default();
/// let config = BurnPINNConfig::default();
/// let mut trainer = BurnPINNTrainer::<Backend>::new(config, &device)?;
///
/// let metrics = trainer.train(
///     &x_data, &t_data, &u_data,
///     343.0,  // wave speed
///     &device,
///     1000    // epochs
/// )?;
/// ```
pub struct BurnPINNTrainer<B: AutodiffBackend> {
    /// The physics-informed neural network
    pinn: BurnPINN1DWave<B>,

    /// Gradient descent optimizer for parameter updates
    optimizer: SimpleOptimizer,

    /// Training configuration (cached for data generation)
    config: BurnPINNConfig,
}

impl<B: AutodiffBackend> BurnPINNTrainer<B> {
    /// Create a new PINN trainer for 1D wave equation
    ///
    /// Initializes the neural network and optimizer with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Network architecture and training hyperparameters
    /// * `device` - Computation device (CPU or GPU)
    ///
    /// # Returns
    ///
    /// New trainer instance ready for training
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Configuration is invalid (validated by BurnPINNConfig)
    /// - Network initialization fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::{Autodiff, NdArray};
    ///
    /// type Backend = Autodiff<NdArray<f32>>;
    ///
    /// let device = Default::default();
    /// let config = BurnPINNConfig {
    ///     hidden_layers: vec![50, 50, 50],
    ///     learning_rate: 0.001,
    ///     num_collocation_points: 10000,
    ///     ..Default::default()
    /// };
    ///
    /// let trainer = BurnPINNTrainer::<Backend>::new(config, &device)?;
    /// ```
    pub fn new(config: BurnPINNConfig, device: &B::Device) -> KwaversResult<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize network
        let pinn = BurnPINN1DWave::<B>::new(config.clone(), device)?;

        // Initialize optimizer with configured learning rate
        let optimizer = SimpleOptimizer::new(config.learning_rate as f32);

        Ok(Self {
            pinn,
            optimizer,
            config,
        })
    }

    /// Train the PINN using physics-informed loss with automatic differentiation
    ///
    /// Performs multi-objective optimization to simultaneously:
    /// 1. Fit training data (data fidelity)
    /// 2. Satisfy wave equation PDE (physics constraint)
    /// 3. Enforce boundary conditions
    ///
    /// ## Training Process
    ///
    /// For each epoch:
    /// 1. Convert data to tensors
    /// 2. Generate collocation points for PDE residual
    /// 3. Set up boundary condition points
    /// 4. Compute physics-informed loss (data + PDE + BC)
    /// 5. Backpropagate gradients
    /// 6. Update network parameters
    /// 7. Record metrics
    ///
    /// ## Convergence
    ///
    /// Training typically converges when:
    /// - Data loss < 1e-3 (good fit to observations)
    /// - PDE loss < 1e-4 (good physics adherence)
    /// - BC loss < 1e-5 (strict boundary enforcement)
    ///
    /// # Arguments
    ///
    /// * `x_data` - Spatial coordinates of training data [N] in meters
    /// * `t_data` - Time coordinates of training data [N] in seconds
    /// * `u_data` - Field values at training points [N, 1] in Pa or m
    /// * `wave_speed` - Speed of sound in medium (m/s), e.g., 343 for air at 20°C
    /// * `device` - Computation device
    /// * `epochs` - Number of training iterations
    ///
    /// # Returns
    ///
    /// Training metrics with loss history and convergence information
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Data dimensions don't match (x, t, u must have compatible shapes)
    /// - Numerical instabilities occur (NaN/Inf in loss)
    /// - Device errors during tensor operations
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::{Autodiff, NdArray};
    /// use ndarray::Array1;
    ///
    /// type Backend = Autodiff<NdArray<f32>>;
    ///
    /// let device = Default::default();
    /// let mut trainer = BurnPINNTrainer::<Backend>::new(config, &device)?;
    ///
    /// // Training data from FDTD simulation or analytical solution
    /// let x_data = Array1::linspace(-1.0, 1.0, 100);
    /// let t_data = Array1::linspace(0.0, 0.1, 100);
    /// let u_data = reference_solution(&x_data, &t_data);
    ///
    /// // Train for 1000 epochs
    /// let metrics = trainer.train(
    ///     &x_data,
    ///     &t_data,
    ///     &u_data,
    ///     343.0,  // wave speed in air
    ///     &device,
    ///     1000
    /// )?;
    ///
    /// // Check convergence
    /// if metrics.is_converged(100, 1e-6) {
    ///     println!("Training converged!");
    /// }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - GPU backend provides significant speedup for large networks and datasets
    /// - Use `num_collocation_points` > 10,000 for good PDE constraint enforcement
    /// - Larger networks (more layers/neurons) may require more epochs to converge
    /// - Learning rate tuning is critical: too high causes divergence, too low is slow
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        x_data: &Array1<f64>,
        t_data: &Array1<f64>,
        u_data: &Array2<f64>,
        wave_speed: f64,
        device: &B::Device,
        epochs: usize,
    ) -> KwaversResult<BurnTrainingMetrics> {
        use std::time::Instant;

        // Validate input dimensions
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

        // Initialize metrics tracking
        let mut metrics = BurnTrainingMetrics::new();

        // Convert training data to tensors
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

        // Generate collocation points for PDE residual enforcement
        // Use uniform grid sampling in normalized domain [-1, 1] × [0, 1]
        let n_colloc = self.config.num_collocation_points;
        let x_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32) * 2.0 - 1.0) // Map to [-1, 1]
            .collect();
        let t_colloc_vec: Vec<f32> = (0..n_colloc)
            .map(|i| (i as f32 / n_colloc as f32)) // Map to [0, 1]
            .collect();

        let x_colloc_tensor =
            Tensor::<B, 1>::from_floats(x_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);
        let t_colloc_tensor =
            Tensor::<B, 1>::from_floats(t_colloc_vec.as_slice(), device).reshape([n_colloc, 1]);

        // Boundary conditions: Dirichlet BC at spatial boundaries
        // x = -1 and x = 1, at t = 0 (initial condition)
        let n_bc = 10; // Number of boundary points
        let x_bc_vec: Vec<f32> = vec![-1.0; n_bc / 2]
            .into_iter()
            .chain(vec![1.0; n_bc / 2])
            .collect();
        let t_bc_vec: Vec<f32> = vec![0.0; n_bc]; // Initial time
        let u_bc_vec: Vec<f32> = vec![0.0; n_bc]; // Zero Dirichlet BC

        let x_bc_tensor =
            Tensor::<B, 1>::from_floats(x_bc_vec.as_slice(), device).reshape([n_bc, 1]);
        let t_bc_tensor =
            Tensor::<B, 1>::from_floats(t_bc_vec.as_slice(), device).reshape([n_bc, 1]);
        let u_bc_tensor =
            Tensor::<B, 1>::from_floats(u_bc_vec.as_slice(), device).reshape([n_bc, 1]);

        // Training loop: Template Method pattern
        for epoch in 0..epochs {
            // Compute physics-informed loss (data + PDE + boundary)
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

            // Extract scalar loss values for metrics
            let total_val = total_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let data_val = data_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let pde_val = pde_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let bc_val = bc_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;

            // Check for numerical instabilities
            if !total_val.is_finite()
                || !data_val.is_finite()
                || !pde_val.is_finite()
                || !bc_val.is_finite()
            {
                return Err(KwaversError::NumericalInstability(format!(
                    "Encountered NaN/Inf in loss at epoch {}: total={}, data={}, pde={}, bc={}",
                    epoch, total_val, data_val, pde_val, bc_val
                )));
            }

            // Record epoch metrics
            metrics.record_epoch(total_val, data_val, pde_val, bc_val);

            // Compute gradients via backpropagation
            let grads = total_loss.backward();

            // Update network parameters: θ_new = θ_old - α·∇L
            self.pinn = self.optimizer.step(self.pinn.clone(), &grads);

            // Periodic logging (every 100 epochs)
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
        }

        // Finalize metrics
        metrics.training_time_secs = start_time.elapsed().as_secs_f64();
        metrics.epochs_completed = epochs;

        Ok(metrics)
    }

    /// Get reference to the trained PINN network
    ///
    /// Provides access to the underlying neural network for inference,
    /// serialization, or further analysis.
    ///
    /// # Returns
    ///
    /// Immutable reference to the PINN network
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // After training
    /// let metrics = trainer.train(...)?;
    ///
    /// // Get trained network for inference
    /// let pinn = trainer.pinn();
    /// let predictions = pinn.predict(&x_test, &t_test, &device)?;
    /// ```
    pub fn pinn(&self) -> &BurnPINN1DWave<B> {
        &self.pinn
    }

    /// Get mutable reference to the PINN network
    ///
    /// Allows modification of the network, e.g., for fine-tuning or
    /// transfer learning scenarios.
    ///
    /// # Returns
    ///
    /// Mutable reference to the PINN network
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Fine-tune on new data
    /// let pinn_mut = trainer.pinn_mut();
    /// // ... modify network or continue training
    /// ```
    pub fn pinn_mut(&mut self) -> &mut BurnPINN1DWave<B> {
        &mut self.pinn
    }

    /// Get reference to the optimizer
    ///
    /// Provides access to optimizer state for inspection or modification.
    ///
    /// # Returns
    ///
    /// Immutable reference to the optimizer
    pub fn optimizer(&self) -> &SimpleOptimizer {
        &self.optimizer
    }

    /// Get reference to the training configuration
    ///
    /// # Returns
    ///
    /// Immutable reference to the configuration
    pub fn config(&self) -> &BurnPINNConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use ndarray::Array1;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_trainer_creation() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };

        let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_trainer_with_invalid_config() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![], // Invalid: empty
            ..Default::default()
        };

        let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device);
        assert!(trainer.is_err());
    }

    #[test]
    fn test_train_basic() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            learning_rate: 0.01,
            num_collocation_points: 100,
            ..Default::default()
        };

        let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        // Simple synthetic data
        let n = 20;
        let x_data = Array1::linspace(-1.0, 1.0, n);
        let t_data = Array1::linspace(0.0, 0.1, n);
        let u_data = Array2::zeros((n, 1)); // Zero initial condition

        // Short training run
        let result = trainer.train(&x_data, &t_data, &u_data, 343.0, &device, 10);

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.epochs_completed, 10);
        assert_eq!(metrics.total_loss.len(), 10);
        assert!(metrics.training_time_secs > 0.0);
    }

    #[test]
    fn test_train_mismatched_dimensions() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };

        let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        // Mismatched dimensions
        let x_data = Array1::linspace(-1.0, 1.0, 20);
        let t_data = Array1::linspace(0.0, 0.1, 30); // Different length
        let u_data = Array2::zeros((20, 1));

        let result = trainer.train(&x_data, &t_data, &u_data, 343.0, &device, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_invalid_u_shape() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };

        let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        let n = 20;
        let x_data = Array1::linspace(-1.0, 1.0, n);
        let t_data = Array1::linspace(0.0, 0.1, n);
        let u_data = Array2::zeros((n, 2)); // Wrong: should be [n, 1]

        let result = trainer.train(&x_data, &t_data, &u_data, 343.0, &device, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_metrics_recording() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![5, 5],
            learning_rate: 0.01,
            num_collocation_points: 50,
            ..Default::default()
        };

        let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        let n = 10;
        let x_data = Array1::linspace(-1.0, 1.0, n);
        let t_data = Array1::linspace(0.0, 0.1, n);
        let u_data = Array2::zeros((n, 1));

        let metrics = trainer
            .train(&x_data, &t_data, &u_data, 343.0, &device, 5)
            .unwrap();

        // Check all metrics recorded
        assert_eq!(metrics.total_loss.len(), 5);
        assert_eq!(metrics.data_loss.len(), 5);
        assert_eq!(metrics.pde_loss.len(), 5);
        assert_eq!(metrics.bc_loss.len(), 5);

        // All losses should be finite
        for &loss in &metrics.total_loss {
            assert!(loss.is_finite());
        }
        for &loss in &metrics.data_loss {
            assert!(loss.is_finite());
        }
        for &loss in &metrics.pde_loss {
            assert!(loss.is_finite());
        }
        for &loss in &metrics.bc_loss {
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_pinn_accessor() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };

        let trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        // Should be able to access PINN
        let _pinn = trainer.pinn();
    }

    #[test]
    fn test_optimizer_accessor() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            learning_rate: 0.001,
            ..Default::default()
        };

        let trainer = BurnPINNTrainer::<TestBackend>::new(config.clone(), &device).unwrap();

        // Check optimizer has correct learning rate
        let optimizer = trainer.optimizer();
        assert_eq!(optimizer.learning_rate(), config.learning_rate as f32);
    }

    #[test]
    fn test_config_accessor() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            learning_rate: 0.001,
            num_collocation_points: 5000,
            ..Default::default()
        };

        let trainer = BurnPINNTrainer::<TestBackend>::new(config.clone(), &device).unwrap();

        // Check config matches
        let trainer_config = trainer.config();
        assert_eq!(trainer_config.hidden_layers, config.hidden_layers);
        assert_eq!(trainer_config.learning_rate, config.learning_rate);
        assert_eq!(
            trainer_config.num_collocation_points,
            config.num_collocation_points
        );
    }

    #[test]
    fn test_multiple_training_runs() {
        let device = Default::default();
        let config = BurnPINNConfig {
            hidden_layers: vec![5, 5],
            learning_rate: 0.01,
            ..Default::default()
        };

        let mut trainer = BurnPINNTrainer::<TestBackend>::new(config, &device).unwrap();

        let n = 10;
        let x_data = Array1::linspace(-1.0, 1.0, n);
        let t_data = Array1::linspace(0.0, 0.1, n);
        let u_data = Array2::zeros((n, 1));

        // First training run
        let metrics1 = trainer
            .train(&x_data, &t_data, &u_data, 343.0, &device, 5)
            .unwrap();
        assert_eq!(metrics1.epochs_completed, 5);

        // Second training run (continues training)
        let metrics2 = trainer
            .train(&x_data, &t_data, &u_data, 343.0, &device, 5)
            .unwrap();
        assert_eq!(metrics2.epochs_completed, 5);
    }
}
