//! Configuration and hyperparameters for 2D Elastic Wave PINN
//!
//! This module defines the training configuration, hyperparameters, and loss weights
//! for physics-informed neural networks solving 2D elastic wave equations.
//!
//! # Configuration Philosophy
//!
//! Configuration is separated into:
//! - **Architecture**: Network structure (layers, neurons, activation)
//! - **Training**: Optimization parameters (learning rate, epochs, batch size)
//! - **Physics**: Domain-specific parameters (collocation points, loss weights)
//! - **Regularization**: Techniques to improve generalization
//!
//! # Mathematical Basis
//!
//! Loss weights follow the principle of dimensional homogeneity and physical significance:
//!
//! ```text
//! L_total = λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic + λ_data·L_data
//! ```
//!
//! Default weights are chosen to:
//! - Enforce physical constraints (PDE, BC, IC) strongly
//! - Allow data fitting where measurements are available
//! - Balance competing objectives during training
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::elastic_2d::Config;
//!
//! // Default configuration (suitable for most problems)
//! let config = Config::default();
//!
//! // Custom configuration for high-frequency problems
//! let config = Config {
//!     hidden_layers: vec![128, 128, 128, 128, 128, 128],
//!     learning_rate: 5e-4,
//!     n_collocation_interior: 20000,
//!     loss_weights: LossWeights {
//!         pde: 10.0,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//!
//! // Inverse problem (material parameter estimation)
//! let config = Config {
//!     optimize_lambda: true,
//!     optimize_mu: true,
//!     loss_weights: LossWeights {
//!         data: 100.0,  // Strong data fitting
//!         pde: 1.0,
//!         ..Default::default()
//!     },
//!     ..Default::default()
//! };
//! ```

use serde::{Deserialize, Serialize};

/// Training configuration for 2D Elastic Wave PINN
///
/// This struct contains all hyperparameters needed to configure and train
/// a physics-informed neural network for 2D elastic wave propagation.
///
/// # Architecture Parameters
///
/// - `hidden_layers`: Network depth and width (e.g., `[100, 100, 100, 100]`)
/// - `activation`: Activation function (tanh, sin, or adaptive)
///
/// # Training Parameters
///
/// - `learning_rate`: Initial learning rate for optimizer
/// - `n_epochs`: Number of training iterations
/// - `batch_size`: Mini-batch size (None = full batch)
/// - `scheduler`: Learning rate schedule
///
/// # Physics Parameters
///
/// - `n_collocation_*`: Number of collocation points for PDE residual
/// - `loss_weights`: Relative importance of loss components
/// - `adaptive_sampling`: Enable adaptive refinement based on residuals
///
/// # Material Parameters (Inverse Problems)
///
/// - `optimize_lambda`: Learn Lamé's first parameter λ(x,y)
/// - `optimize_mu`: Learn shear modulus μ(x,y)
/// - `optimize_rho`: Learn density ρ(x,y)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // ============ Architecture ============
    /// Hidden layer sizes (e.g., [100, 100, 100, 100] for 4 layers of 100 neurons)
    ///
    /// **Guidance**:
    /// - Smooth problems: 3-4 layers, 50-100 neurons
    /// - High-frequency/complex: 5-8 layers, 100-200 neurons
    /// - Sharp interfaces: Deeper networks with smaller layers
    pub hidden_layers: Vec<usize>,

    /// Activation function for hidden layers
    ///
    /// **Options**:
    /// - `Tanh`: Standard choice, smooth gradients
    /// - `Sin`: Periodic activation, good for wave problems
    /// - `Swish`: Self-gated, improved training dynamics
    /// - `Adaptive`: Learnable activation (experimental)
    pub activation: ActivationFunction,

    // ============ Training ============
    /// Initial learning rate for optimizer (typical: 1e-3 to 1e-4)
    pub learning_rate: f64,

    /// Number of training epochs
    ///
    /// **Guidance**:
    /// - Forward problems: 5,000-20,000 epochs
    /// - Inverse problems: 10,000-50,000 epochs
    /// - Monitor convergence; early stopping recommended
    pub n_epochs: usize,

    /// Mini-batch size for stochastic training
    ///
    /// - `None`: Full-batch gradient descent (default, more stable)
    /// - `Some(n)`: Mini-batch training (faster, more memory-efficient)
    ///
    /// **Trade-off**: Full-batch is more stable for physics losses,
    /// mini-batch allows larger problems and can escape local minima.
    pub batch_size: Option<usize>,

    /// Learning rate schedule strategy
    pub scheduler: LearningRateScheduler,

    /// Optimizer type
    pub optimizer: OptimizerType,

    // ============ Collocation Points ============
    /// Number of interior collocation points for PDE residual
    ///
    /// **Guidance**:
    /// - Simple domains: 5,000-10,000 points
    /// - Complex/multi-region: 10,000-50,000 points
    /// - 3D problems: 50,000-100,000 points
    pub n_collocation_interior: usize,

    /// Number of boundary collocation points per boundary segment
    pub n_collocation_boundary: usize,

    /// Number of initial condition points (at t=0)
    pub n_collocation_initial: usize,

    /// Sampling strategy for collocation points
    pub sampling_strategy: SamplingStrategy,

    /// Enable adaptive refinement of collocation points
    ///
    /// If enabled, points are added in regions with high PDE residual
    /// during training. Improves accuracy but increases computational cost.
    pub adaptive_sampling: bool,

    /// Residual threshold for adaptive refinement (if enabled)
    ///
    /// Points are added where `|residual| > threshold * mean(|residual|)`
    pub adaptive_threshold: f64,

    // ============ Loss Weights ============
    /// Relative weights for loss function components
    pub loss_weights: LossWeights,

    // ============ Material Parameters (Inverse Problems) ============
    /// Optimize Lamé's first parameter λ(x,y) during training
    ///
    /// If true, λ is treated as a learnable parameter (spatially varying or constant).
    pub optimize_lambda: bool,

    /// Initial guess for λ (Pa) if optimizing
    pub lambda_init: Option<f64>,

    /// Optimize shear modulus μ(x,y) during training
    pub optimize_mu: bool,

    /// Initial guess for μ (Pa) if optimizing
    pub mu_init: Option<f64>,

    /// Optimize density ρ(x,y) during training
    pub optimize_rho: bool,

    /// Initial guess for ρ (kg/m³) if optimizing
    pub rho_init: Option<f64>,

    // ============ Regularization ============
    /// L2 regularization penalty on network weights
    ///
    /// Loss term: `λ_reg * ||θ||²` where θ are network parameters
    pub weight_decay: f64,

    /// Gradient clipping threshold (None = no clipping)
    ///
    /// Prevents exploding gradients by clipping to `[-threshold, threshold]`
    pub gradient_clip: Option<f64>,

    // ============ Checkpointing ============
    /// Checkpoint interval (epochs)
    ///
    /// Save model state every N epochs. 0 = no checkpointing.
    pub checkpoint_interval: usize,

    /// Directory for saving checkpoints
    pub checkpoint_dir: Option<String>,

    // ============ Logging ============
    /// Print training metrics every N epochs
    pub log_interval: usize,

    /// Enable detailed logging (loss components, gradient norms, etc.)
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Architecture
            hidden_layers: vec![100, 100, 100, 100],
            activation: ActivationFunction::Tanh,

            // Training
            learning_rate: 1e-3,
            n_epochs: 10000,
            batch_size: None,
            scheduler: LearningRateScheduler::Exponential { decay_rate: 0.95 },
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },

            // Collocation
            n_collocation_interior: 10000,
            n_collocation_boundary: 1000,
            n_collocation_initial: 1000,
            sampling_strategy: SamplingStrategy::LatinHypercube,
            adaptive_sampling: false,
            adaptive_threshold: 2.0,

            // Loss weights
            loss_weights: LossWeights::default(),

            // Material parameters (forward problem by default)
            optimize_lambda: false,
            lambda_init: None,
            optimize_mu: false,
            mu_init: None,
            optimize_rho: false,
            rho_init: None,

            // Regularization
            weight_decay: 0.0,
            gradient_clip: None,

            // Checkpointing
            checkpoint_interval: 1000,
            checkpoint_dir: None,

            // Logging
            log_interval: 100,
            verbose: false,
        }
    }
}

/// Loss function weights for physics-informed training
///
/// These weights control the relative importance of different loss components.
/// The total loss is:
///
/// ```text
/// L = λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic + λ_data·L_data + λ_interface·L_interface
/// ```
///
/// # Weight Selection Guidelines
///
/// ## Forward Problems (known material properties)
/// - `pde`: 1.0-10.0 (enforce PDE strongly)
/// - `boundary`: 10.0-100.0 (critical for well-posedness)
/// - `initial`: 10.0-100.0 (critical for time-dependent problems)
/// - `data`: 0.0-1.0 (limited or no data)
///
/// ## Inverse Problems (estimate material properties)
/// - `pde`: 0.1-1.0 (relax PDE to allow parameter learning)
/// - `boundary`: 1.0-10.0 (still important for well-posedness)
/// - `initial`: 1.0-10.0 (still important)
/// - `data`: 10.0-1000.0 (strong data fitting drives parameter estimation)
///
/// ## Multi-Region Problems
/// - `interface`: 1.0-10.0 (enforce continuity at material interfaces)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossWeights {
    /// Weight for PDE residual loss (λ_pde)
    ///
    /// Enforces: `ρ ∂²u/∂t² = ∇·σ + f`
    pub pde: f64,

    /// Weight for boundary condition loss (λ_bc)
    ///
    /// Enforces boundary conditions (Dirichlet, Neumann, etc.)
    pub boundary: f64,

    /// Weight for initial condition loss (λ_ic)
    ///
    /// Enforces: `u(x,y,0) = u₀(x,y)` and `∂u/∂t(x,y,0) = v₀(x,y)`
    pub initial: f64,

    /// Weight for data fitting loss (λ_data)
    ///
    /// Fits model to observed displacement measurements
    pub data: f64,

    /// Weight for interface condition loss (λ_interface)
    ///
    /// Enforces continuity of displacement and traction at material interfaces
    pub interface: f64,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            pde: 1.0,
            boundary: 10.0,
            initial: 10.0,
            data: 1.0,
            interface: 10.0,
        }
    }
}

/// Activation function for neural network hidden layers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationFunction {
    /// Hyperbolic tangent: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    ///
    /// **Properties**:
    /// - Smooth, bounded to [-1, 1]
    /// - Zero-centered output
    /// - Standard choice for PINNs
    Tanh,

    /// Sine: sin(x)
    ///
    /// **Properties**:
    /// - Periodic, smooth
    /// - Natural choice for wave equations
    /// - Can capture high-frequency features
    /// - May improve spectral bias
    Sin,

    /// Swish (SiLU): x · σ(x) where σ is sigmoid
    ///
    /// **Properties**:
    /// - Self-gated, smooth, non-monotonic
    /// - Improved training dynamics vs ReLU
    /// - Differentiable everywhere
    Swish,

    /// Adaptive activation (learnable parameters)
    ///
    /// **Experimental**: Parameters learned during training
    Adaptive,
}

/// Learning rate scheduling strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LearningRateScheduler {
    /// Constant learning rate (no decay)
    Constant,

    /// Exponential decay: lr(t) = lr_0 * decay_rate^(t/decay_steps)
    Exponential {
        /// Decay rate per step (typical: 0.9-0.99)
        decay_rate: f64,
    },

    /// Step decay: reduce LR by factor every N epochs
    Step {
        /// Factor to multiply LR by at each step (e.g., 0.1)
        factor: f64,
        /// Number of epochs between steps
        step_size: usize,
    },

    /// Cosine annealing: lr(t) = lr_min + 0.5*(lr_0 - lr_min)*(1 + cos(πt/T))
    CosineAnnealing {
        /// Minimum learning rate
        lr_min: f64,
    },

    /// Reduce on plateau: decrease LR when loss stops improving
    ReduceOnPlateau {
        /// Factor to reduce LR by (e.g., 0.5)
        factor: f64,
        /// Number of epochs with no improvement before reducing
        patience: usize,
        /// Minimum change in loss to qualify as improvement
        threshold: f64,
    },
}

/// Optimizer type for training
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD {
        /// Momentum coefficient (0 = no momentum)
        momentum: f64,
    },

    /// Adam optimizer (Adaptive Moment Estimation)
    ///
    /// Default choice for most problems. Adapts learning rates per parameter.
    Adam {
        /// Exponential decay rate for first moment (typical: 0.9)
        beta1: f64,
        /// Exponential decay rate for second moment (typical: 0.999)
        beta2: f64,
        /// Small constant for numerical stability (typical: 1e-8)
        epsilon: f64,
    },

    /// AdamW (Adam with decoupled weight decay)
    ///
    /// Better regularization than standard Adam.
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },

    /// L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
    ///
    /// Second-order optimizer. More expensive per iteration but can converge faster.
    /// Recommended for final fine-tuning after Adam.
    LBFGS {
        /// Number of correction pairs to store
        history_size: usize,
        /// Line search tolerance
        tolerance: f64,
    },
}

/// Sampling strategy for collocation points
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    ///
    /// Simplest approach, but may leave gaps in coverage.
    Uniform,

    /// Latin Hypercube Sampling
    ///
    /// Better space-filling properties than uniform sampling.
    /// Ensures each dimension is evenly sampled.
    LatinHypercube,

    /// Sobol quasi-random sequence
    ///
    /// Low-discrepancy sequence with excellent space-filling properties.
    /// Best for high-dimensional problems.
    Sobol,

    /// Adaptive refinement based on residuals
    ///
    /// Start with coarse sampling, refine where residuals are high.
    /// Most expensive but most accurate.
    AdaptiveRefinement,
}

impl Config {
    /// Create configuration for a forward problem (known material properties)
    ///
    /// # Arguments
    ///
    /// * `lambda` - Lamé's first parameter (Pa)
    /// * `mu` - Shear modulus (Pa)
    /// * `rho` - Density (kg/m³)
    pub fn forward_problem(lambda: f64, mu: f64, rho: f64) -> Self {
        Self {
            optimize_lambda: false,
            lambda_init: Some(lambda),
            optimize_mu: false,
            mu_init: Some(mu),
            optimize_rho: false,
            rho_init: Some(rho),
            loss_weights: LossWeights {
                pde: 1.0,
                boundary: 10.0,
                initial: 10.0,
                data: 0.0, // No data fitting
                interface: 10.0,
            },
            ..Default::default()
        }
    }

    /// Create configuration for an inverse problem (estimate material properties)
    ///
    /// # Arguments
    ///
    /// * `lambda_guess` - Initial guess for λ (Pa)
    /// * `mu_guess` - Initial guess for μ (Pa)
    /// * `rho_guess` - Initial guess for ρ (kg/m³)
    pub fn inverse_problem(lambda_guess: f64, mu_guess: f64, rho_guess: f64) -> Self {
        Self {
            optimize_lambda: true,
            lambda_init: Some(lambda_guess),
            optimize_mu: true,
            mu_init: Some(mu_guess),
            optimize_rho: true,
            rho_init: Some(rho_guess),
            loss_weights: LossWeights {
                pde: 0.1, // Relax PDE enforcement
                boundary: 1.0,
                initial: 1.0,
                data: 100.0, // Strong data fitting
                interface: 1.0,
            },
            n_epochs: 30000,     // Inverse problems need more iterations
            learning_rate: 5e-4, // Lower LR for stability
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(msg)` if invalid
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_layers.is_empty() {
            return Err("Must have at least one hidden layer".to_string());
        }

        if self.hidden_layers.contains(&0) {
            return Err("Hidden layer sizes must be positive".to_string());
        }

        if self.learning_rate <= 0.0 || self.learning_rate >= 1.0 {
            return Err("Learning rate must be in (0, 1)".to_string());
        }

        if self.n_epochs == 0 {
            return Err("Number of epochs must be positive".to_string());
        }

        if self.n_collocation_interior == 0 {
            return Err("Number of interior collocation points must be positive".to_string());
        }

        if self.loss_weights.pde < 0.0
            || self.loss_weights.boundary < 0.0
            || self.loss_weights.initial < 0.0
            || self.loss_weights.data < 0.0
            || self.loss_weights.interface < 0.0
        {
            return Err("Loss weights must be non-negative".to_string());
        }

        if self.optimize_lambda && self.lambda_init.is_none() {
            return Err("lambda_init required when optimize_lambda is true".to_string());
        }

        if self.optimize_mu && self.mu_init.is_none() {
            return Err("mu_init required when optimize_mu is true".to_string());
        }

        if self.optimize_rho && self.rho_init.is_none() {
            return Err("rho_init required when optimize_rho is true".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_forward_problem_config() {
        let config = Config::forward_problem(1e9, 5e8, 1000.0);
        assert!(config.validate().is_ok());
        assert!(!config.optimize_lambda);
        assert!(!config.optimize_mu);
        assert!(!config.optimize_rho);
        assert_eq!(config.lambda_init, Some(1e9));
        assert_eq!(config.loss_weights.data, 0.0);
    }

    #[test]
    fn test_inverse_problem_config() {
        let config = Config::inverse_problem(1e9, 5e8, 1000.0);
        assert!(config.validate().is_ok());
        assert!(config.optimize_lambda);
        assert!(config.optimize_mu);
        assert!(config.optimize_rho);
        assert!(config.loss_weights.data > config.loss_weights.pde);
    }

    #[test]
    fn test_invalid_empty_layers() {
        let config = Config {
            hidden_layers: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_zero_neurons() {
        let config = Config {
            hidden_layers: vec![100, 0, 100],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let config = Config {
            learning_rate: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = Config {
            learning_rate: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_optimize_without_init() {
        let config = Config {
            optimize_lambda: true,
            lambda_init: None,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_loss_weights_default() {
        let weights = LossWeights::default();
        assert_eq!(weights.pde, 1.0);
        assert_eq!(weights.boundary, 10.0);
        assert_eq!(weights.initial, 10.0);
        assert_eq!(weights.data, 1.0);
        assert_eq!(weights.interface, 10.0);
    }

    #[test]
    fn test_sampling_strategy_equality() {
        assert_eq!(SamplingStrategy::Uniform, SamplingStrategy::Uniform);
        assert_ne!(SamplingStrategy::Uniform, SamplingStrategy::Sobol);
    }

    #[test]
    fn test_activation_function_equality() {
        assert_eq!(ActivationFunction::Tanh, ActivationFunction::Tanh);
        assert_ne!(ActivationFunction::Tanh, ActivationFunction::Sin);
    }
}
