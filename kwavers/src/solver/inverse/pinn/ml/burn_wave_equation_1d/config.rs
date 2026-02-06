//! Configuration types for Burn-based 1D Wave Equation PINN
//!
//! This module provides configuration structures for the Physics-Informed Neural Network
//! implementation using the Burn deep learning framework. All configurations include
//! validation and domain-specific presets.
//!
//! ## Architecture Configuration
//!
//! The PINN architecture is configured via layer sizes and hyperparameters:
//! - Input layer: 2 inputs (x, t) → first hidden layer
//! - Hidden layers: Configurable depth and width with tanh activation
//! - Output layer: last hidden layer → 1 output (u)
//!
//! ## Loss Function Configuration
//!
//! Physics-informed loss combines three components:
//! - Data loss: MSE between predictions and training data
//! - PDE loss: MSE of wave equation residual
//! - Boundary loss: MSE of boundary condition violations
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework
//!   for solving forward and inverse problems involving nonlinear partial differential equations"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//!
//! ## Examples
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::ml::burn_wave_equation_1d::config::*;
//!
//! // Default configuration for standard training
//! let config = BurnPINNConfig::default();
//!
//! // Custom configuration for high-accuracy training
//! let config = BurnPINNConfig {
//!     hidden_layers: vec![100, 100, 100, 100],
//!     learning_rate: 1e-4,
//!     num_collocation_points: 50000,
//!     ..Default::default()
//! };
//!
//! // GPU-optimized configuration
//! let config = BurnPINNConfig::for_gpu();
//! ```

use crate::core::error::{KwaversError, KwaversResult};

/// Configuration for Burn-based 1D Wave Equation PINN
///
/// This structure defines the neural network architecture and training hyperparameters
/// for physics-informed learning of the 1D acoustic wave equation.
///
/// ## Architecture
///
/// The network architecture is defined by:
/// - `hidden_layers`: Vector of hidden layer sizes (e.g., [50, 50, 50, 50])
/// - Input layer: Always 2 inputs (spatial x, temporal t)
/// - Output layer: Always 1 output (field u)
///
/// ## Hyperparameters
///
/// Training is controlled by:
/// - `learning_rate`: Step size for gradient descent (typical: 1e-3 to 1e-4)
/// - `num_collocation_points`: Number of points for PDE residual sampling
/// - `loss_weights`: Relative importance of data, PDE, and boundary losses
///
/// ## Validation
///
/// Configurations are validated via `validate()` to ensure:
/// - Non-empty hidden layers
/// - Positive learning rate
/// - Sufficient collocation points (≥ 100)
/// - Valid loss weights
///
/// # Examples
///
/// ```rust,ignore
/// // Standard configuration
/// let config = BurnPINNConfig::default();
/// assert!(config.validate().is_ok());
///
/// // Custom deep network
/// let config = BurnPINNConfig {
///     hidden_layers: vec![128, 128, 128, 128, 128],
///     learning_rate: 5e-4,
///     num_collocation_points: 20000,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BurnPINNConfig {
    /// Hidden layer sizes (e.g., [50, 50, 50, 50] for 4 layers of 50 neurons each)
    ///
    /// Larger networks can capture more complex solutions but require more computation.
    /// Typical choices:
    /// - Small: [20, 20, 20] for simple problems
    /// - Medium: [50, 50, 50, 50] for standard training (default)
    /// - Large: [100, 100, 100, 100] for high accuracy or complex domains
    pub hidden_layers: Vec<usize>,

    /// Learning rate for gradient descent optimization
    ///
    /// Controls the step size for parameter updates: θ = θ - α∇L
    /// Typical values:
    /// - 1e-3: Standard starting point (default)
    /// - 1e-4: For fine-tuning or unstable training
    /// - 1e-2: For initial exploration (may be unstable)
    ///
    /// **Note**: Too high learning rates cause divergence; too low causes slow convergence
    pub learning_rate: f64,

    /// Loss function weights for physics-informed training
    ///
    /// Controls the relative importance of data fitting, PDE constraints, and boundary conditions
    pub loss_weights: BurnLossWeights,

    /// Number of collocation points for PDE residual sampling
    ///
    /// These points are distributed throughout the spatiotemporal domain to enforce
    /// the wave equation constraint. More points improve PDE enforcement but increase
    /// computational cost.
    ///
    /// Typical values:
    /// - 1,000: Minimum for basic training
    /// - 10,000: Standard training (default)
    /// - 50,000+: High-accuracy training or GPU acceleration
    ///
    /// **Theorem**: Collocation points sample the PDE residual to ensure the neural network
    /// satisfies ∂²u/∂t² = c²∂²u/∂x² throughout the domain (not just at data points)
    pub num_collocation_points: usize,
}

impl Default for BurnPINNConfig {
    /// Default configuration for standard PINN training
    ///
    /// Provides balanced settings suitable for most 1D wave equation problems:
    /// - 4 hidden layers of 50 neurons each
    /// - Learning rate of 1e-3
    /// - 10,000 collocation points
    /// - Balanced loss weights (1:1:10 for data:pde:boundary)
    fn default() -> Self {
        Self {
            hidden_layers: vec![50, 50, 50, 50],
            learning_rate: 1e-3,
            loss_weights: BurnLossWeights::default(),
            num_collocation_points: 10_000,
        }
    }
}

impl BurnPINNConfig {
    /// Create a GPU-optimized configuration
    ///
    /// Uses larger network and more collocation points to leverage GPU parallelism.
    /// Suitable for WGPU backend with sufficient GPU memory.
    ///
    /// # Returns
    ///
    /// Configuration with:
    /// - 5 hidden layers of 100 neurons each
    /// - Learning rate of 5e-4 (lower for stability with larger network)
    /// - 50,000 collocation points
    /// - Balanced loss weights
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::{Autodiff, Wgpu};
    /// use kwavers::solver::inverse::pinn::ml::burn_wave_equation_1d::config::BurnPINNConfig;
    ///
    /// type Backend = Autodiff<Wgpu<f32>>;
    /// let device = pollster::block_on(Wgpu::<f32>::default())?;
    /// let config = BurnPINNConfig::for_gpu();
    /// let pinn = BurnPINN1DWave::<Backend>::new(config, &device)?;
    /// ```
    pub fn for_gpu() -> Self {
        Self {
            hidden_layers: vec![100, 100, 100, 100, 100],
            learning_rate: 5e-4,
            loss_weights: BurnLossWeights::default(),
            num_collocation_points: 50_000,
        }
    }

    /// Create a lightweight configuration for rapid prototyping
    ///
    /// Uses smaller network and fewer collocation points for fast iteration.
    /// Suitable for development, debugging, and proof-of-concept work.
    ///
    /// # Returns
    ///
    /// Configuration with:
    /// - 3 hidden layers of 20 neurons each
    /// - Learning rate of 1e-3
    /// - 1,000 collocation points
    /// - Balanced loss weights
    ///
    /// **Warning**: This configuration prioritizes speed over accuracy and should
    /// not be used for production or publication-quality results.
    pub fn for_prototyping() -> Self {
        Self {
            hidden_layers: vec![20, 20, 20],
            learning_rate: 1e-3,
            loss_weights: BurnLossWeights::default(),
            num_collocation_points: 1_000,
        }
    }

    /// Validate configuration parameters
    ///
    /// Ensures all configuration values are within valid ranges and satisfy
    /// mathematical constraints for stable training.
    ///
    /// # Validation Rules
    ///
    /// - Hidden layers: Must have at least one layer
    /// - Learning rate: Must be positive and finite
    /// - Collocation points: Must be at least 100 for reasonable PDE enforcement
    /// - Loss weights: Must all be non-negative and finite
    ///
    /// # Returns
    ///
    /// - `Ok(())` if configuration is valid
    /// - `Err(KwaversError::InvalidInput)` with descriptive message otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = BurnPINNConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = BurnPINNConfig {
    ///     hidden_layers: vec![],  // Invalid: empty
    ///     ..Default::default()
    /// };
    /// assert!(bad_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> KwaversResult<()> {
        // Validate hidden layers
        if self.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Configuration must have at least one hidden layer".into(),
            ));
        }

        for (i, &size) in self.hidden_layers.iter().enumerate() {
            if size == 0 {
                return Err(KwaversError::InvalidInput(format!(
                    "Hidden layer {} has size 0 (must be positive)",
                    i
                )));
            }
        }

        // Validate learning rate
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Learning rate must be positive and finite (got {})",
                self.learning_rate
            )));
        }

        // Validate collocation points
        if self.num_collocation_points < 100 {
            return Err(KwaversError::InvalidInput(format!(
                "Number of collocation points must be at least 100 for reasonable PDE enforcement (got {})",
                self.num_collocation_points
            )));
        }

        // Validate loss weights
        self.loss_weights.validate()?;

        Ok(())
    }

    /// Get total number of network parameters
    ///
    /// Computes the total number of trainable parameters in the network defined
    /// by this configuration. Useful for memory estimation and complexity analysis.
    ///
    /// # Formula
    ///
    /// For a network with layers [n₀, n₁, ..., nₖ]:
    /// - Input layer: 2 × n₀ + n₀ (weights + biases)
    /// - Hidden layers: Σᵢ (nᵢ × nᵢ₊₁ + nᵢ₊₁)
    /// - Output layer: nₖ × 1 + 1
    ///
    /// # Returns
    ///
    /// Total number of trainable parameters
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = BurnPINNConfig::default(); // [50, 50, 50, 50]
    /// let params = config.num_parameters();
    /// // Input: 2*50 + 50 = 150
    /// // Hidden: 50*50 + 50 = 2550 (x3 layers)
    /// // Output: 50*1 + 1 = 51
    /// // Total: 150 + 7650 + 51 = 7851
    /// assert_eq!(params, 7851);
    /// ```
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Input layer: 2 inputs → first hidden layer
        let first_hidden = self.hidden_layers[0];
        total += 2 * first_hidden + first_hidden; // weights + biases

        // Hidden layers
        for i in 0..self.hidden_layers.len() - 1 {
            let in_size = self.hidden_layers[i];
            let out_size = self.hidden_layers[i + 1];
            total += in_size * out_size + out_size; // weights + biases
        }

        // Output layer: last hidden → 1 output
        let last_hidden = self.hidden_layers[self.hidden_layers.len() - 1];
        total += last_hidden + 1; // weights + biases

        total
    }
}

/// Loss function weight configuration for physics-informed training
///
/// Controls the relative importance of three loss components in the total loss:
///
/// **L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc**
///
/// Where:
/// - L_data: Mean squared error between predictions and training data
/// - L_pde: Mean squared error of PDE residual (wave equation violation)
/// - L_bc: Mean squared error of boundary condition violations
///
/// ## Tuning Guidelines
///
/// - **Increase `data`** if predictions don't match training data well
/// - **Increase `pde`** if the solution violates the wave equation (non-physical behavior)
/// - **Increase `boundary`** if boundary conditions are not satisfied
///
/// Default weights (1.0, 1.0, 10.0) prioritize boundary enforcement, which is critical
/// for well-posed wave equation problems.
///
/// ## Theoretical Foundation
///
/// The multi-objective loss formulation enables the neural network to simultaneously:
/// 1. Fit observed data (data-driven component)
/// 2. Satisfy governing PDEs (physics-informed component)
/// 3. Enforce boundary/initial conditions (constraint component)
///
/// This approach was introduced by Raissi et al. (2019) for physics-informed neural networks.
///
/// # Examples
///
/// ```rust,ignore
/// // Default: Balanced with strong boundary enforcement
/// let weights = BurnLossWeights::default();
/// assert_eq!(weights.data, 1.0);
/// assert_eq!(weights.pde, 1.0);
/// assert_eq!(weights.boundary, 10.0);
///
/// // Data-driven: Prioritize fitting observations
/// let weights = BurnLossWeights {
///     data: 10.0,
///     pde: 1.0,
///     boundary: 5.0,
/// };
///
/// // Physics-driven: Prioritize PDE satisfaction
/// let weights = BurnLossWeights {
///     data: 0.1,
///     pde: 10.0,
///     boundary: 10.0,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BurnLossWeights {
    /// Weight for data fitting loss (λ_data)
    ///
    /// Controls importance of matching training data.
    /// Higher values enforce closer fit to observations.
    ///
    /// **Typical range**: 0.1 to 10.0
    pub data: f64,

    /// Weight for PDE residual loss (λ_pde)
    ///
    /// Controls importance of satisfying the wave equation ∂²u/∂t² = c²∂²u/∂x².
    /// Higher values enforce stronger physics constraints.
    ///
    /// **Typical range**: 0.1 to 10.0
    pub pde: f64,

    /// Weight for boundary condition loss (λ_bc)
    ///
    /// Controls importance of satisfying boundary conditions.
    /// Higher values enforce stronger boundary constraint satisfaction.
    ///
    /// **Typical range**: 1.0 to 100.0 (often higher than data/pde)
    ///
    /// **Note**: Boundary conditions are critical for well-posed wave equation problems,
    /// so this weight is typically set higher than data and pde weights.
    pub boundary: f64,
}

impl Default for BurnLossWeights {
    /// Default loss weights with strong boundary enforcement
    ///
    /// Returns balanced weights suitable for most wave equation problems:
    /// - data: 1.0 (standard data fitting)
    /// - pde: 1.0 (standard physics constraint)
    /// - boundary: 10.0 (strong boundary enforcement)
    ///
    /// The higher boundary weight ensures well-posed problems with proper
    /// boundary condition satisfaction, which is essential for accurate wave
    /// propagation simulation.
    fn default() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 10.0,
        }
    }
}

impl BurnLossWeights {
    /// Validate loss weights
    ///
    /// Ensures all weights are non-negative and finite. Negative or infinite
    /// weights would produce invalid loss functions.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all weights are valid
    /// - `Err(KwaversError::InvalidInput)` with descriptive message otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let weights = BurnLossWeights::default();
    /// assert!(weights.validate().is_ok());
    ///
    /// let bad_weights = BurnLossWeights {
    ///     data: -1.0,  // Invalid: negative
    ///     ..Default::default()
    /// };
    /// assert!(bad_weights.validate().is_err());
    /// ```
    pub fn validate(&self) -> KwaversResult<()> {
        if self.data < 0.0 || !self.data.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Data loss weight must be non-negative and finite (got {})",
                self.data
            )));
        }

        if self.pde < 0.0 || !self.pde.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "PDE loss weight must be non-negative and finite (got {})",
                self.pde
            )));
        }

        if self.boundary < 0.0 || !self.boundary.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "Boundary loss weight must be non-negative and finite (got {})",
                self.boundary
            )));
        }

        Ok(())
    }

    /// Create data-driven weights that prioritize fitting observations
    ///
    /// Suitable for problems where data quality is high and physics constraints
    /// are less critical or already well-satisfied.
    ///
    /// # Returns
    ///
    /// Weights with data=10.0, pde=1.0, boundary=5.0
    pub fn data_driven() -> Self {
        Self {
            data: 10.0,
            pde: 1.0,
            boundary: 5.0,
        }
    }

    /// Create physics-driven weights that prioritize PDE satisfaction
    ///
    /// Suitable for problems with sparse or noisy data where enforcing
    /// physical laws is more important than exact data fitting.
    ///
    /// # Returns
    ///
    /// Weights with data=0.1, pde=10.0, boundary=10.0
    pub fn physics_driven() -> Self {
        Self {
            data: 0.1,
            pde: 10.0,
            boundary: 10.0,
        }
    }

    /// Create balanced weights with equal importance for all components
    ///
    /// Suitable for initial exploration or when the relative importance
    /// of different loss components is unknown.
    ///
    /// # Returns
    ///
    /// Weights with data=1.0, pde=1.0, boundary=1.0
    pub fn balanced() -> Self {
        Self {
            data: 1.0,
            pde: 1.0,
            boundary: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BurnPINNConfig::default();
        assert_eq!(config.hidden_layers, vec![50, 50, 50, 50]);
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.num_collocation_points, 10_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpu_config() {
        let config = BurnPINNConfig::for_gpu();
        assert_eq!(config.hidden_layers, vec![100, 100, 100, 100, 100]);
        assert_eq!(config.learning_rate, 5e-4);
        assert_eq!(config.num_collocation_points, 50_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_prototyping_config() {
        let config = BurnPINNConfig::for_prototyping();
        assert_eq!(config.hidden_layers, vec![20, 20, 20]);
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.num_collocation_points, 1_000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_empty_layers() {
        let config = BurnPINNConfig {
            hidden_layers: vec![],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_zero_layer_size() {
        let config = BurnPINNConfig {
            hidden_layers: vec![50, 0, 50],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_negative_learning_rate() {
        let config = BurnPINNConfig {
            learning_rate: -0.001,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_insufficient_collocation_points() {
        let config = BurnPINNConfig {
            num_collocation_points: 50,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_num_parameters_default() {
        let config = BurnPINNConfig::default(); // [50, 50, 50, 50]
        let params = config.num_parameters();
        // Input: 2*50 + 50 = 150
        // Hidden 1: 50*50 + 50 = 2550
        // Hidden 2: 50*50 + 50 = 2550
        // Hidden 3: 50*50 + 50 = 2550
        // Output: 50*1 + 1 = 51
        // Total: 150 + 2550 + 2550 + 2550 + 51 = 7851
        assert_eq!(params, 7851);
    }

    #[test]
    fn test_num_parameters_simple() {
        let config = BurnPINNConfig {
            hidden_layers: vec![10, 10],
            ..Default::default()
        };
        let params = config.num_parameters();
        // Input: 2*10 + 10 = 30
        // Hidden: 10*10 + 10 = 110
        // Output: 10*1 + 1 = 11
        // Total: 30 + 110 + 11 = 151
        assert_eq!(params, 151);
    }

    #[test]
    fn test_default_loss_weights() {
        let weights = BurnLossWeights::default();
        assert_eq!(weights.data, 1.0);
        assert_eq!(weights.pde, 1.0);
        assert_eq!(weights.boundary, 10.0);
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_data_driven_weights() {
        let weights = BurnLossWeights::data_driven();
        assert_eq!(weights.data, 10.0);
        assert_eq!(weights.pde, 1.0);
        assert_eq!(weights.boundary, 5.0);
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_physics_driven_weights() {
        let weights = BurnLossWeights::physics_driven();
        assert_eq!(weights.data, 0.1);
        assert_eq!(weights.pde, 10.0);
        assert_eq!(weights.boundary, 10.0);
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_balanced_weights() {
        let weights = BurnLossWeights::balanced();
        assert_eq!(weights.data, 1.0);
        assert_eq!(weights.pde, 1.0);
        assert_eq!(weights.boundary, 1.0);
        assert!(weights.validate().is_ok());
    }

    #[test]
    fn test_loss_weights_validation_negative() {
        let weights = BurnLossWeights {
            data: -1.0,
            ..Default::default()
        };
        assert!(weights.validate().is_err());

        let weights = BurnLossWeights {
            pde: -1.0,
            ..Default::default()
        };
        assert!(weights.validate().is_err());

        let weights = BurnLossWeights {
            boundary: -1.0,
            ..Default::default()
        };
        assert!(weights.validate().is_err());
    }

    #[test]
    fn test_loss_weights_validation_infinite() {
        let weights = BurnLossWeights {
            data: f64::INFINITY,
            ..Default::default()
        };
        assert!(weights.validate().is_err());
    }
}
