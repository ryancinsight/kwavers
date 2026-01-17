//! Configuration types for 3D wave equation PINN
//!
//! This module defines configuration structures for the 3D PINN solver, including:
//! - Network architecture configuration
//! - Loss component weights for physics-informed training
//! - Training metrics and progress tracking
//!
//! All types are pure domain logic with sensible defaults for common use cases.

use crate::core::error::{KwaversError, KwaversResult};

/// Configuration for 3D PINN solver
///
/// Defines network architecture, training hyperparameters, and loss weights
/// for physics-informed training of the 3D wave equation.
///
/// # Default Configuration
///
/// The default configuration provides a reasonable starting point:
/// - 3-layer MLP with 100 neurons per layer
/// - 10,000 collocation points for PDE residual
/// - Equal loss weights (1.0 for all components)
/// - Learning rate: 1e-3
/// - Batch size: 1,000
/// - Gradient clipping at norm 1.0
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::BurnPINN3DConfig;
///
/// // Use default configuration
/// let config = BurnPINN3DConfig::default();
///
/// // Custom configuration
/// let config = BurnPINN3DConfig {
///     hidden_layers: vec![200, 200, 200, 200],  // 4-layer network
///     num_collocation_points: 20000,             // More PDE sampling
///     learning_rate: 5e-4,                       // Lower learning rate
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BurnPINN3DConfig {
    /// Hidden layer sizes for the neural network
    ///
    /// Defines the architecture of the multi-layer perceptron.
    /// Each element specifies the number of neurons in that layer.
    ///
    /// **Default**: `vec![100, 100, 100]` (3-layer MLP)
    pub hidden_layers: Vec<usize>,

    /// Number of collocation points for PDE residual computation
    ///
    /// Determines how many points are randomly sampled within the domain
    /// to evaluate the PDE residual during training. Higher values provide
    /// better coverage but increase computational cost.
    ///
    /// **Default**: 10,000
    ///
    /// **Recommended range**: 1,000 - 50,000
    pub num_collocation_points: usize,

    /// Loss weights for different components
    ///
    /// Balances the contribution of data fitting, PDE residual, boundary
    /// conditions, and initial conditions to the total loss.
    ///
    /// **Default**: Equal weights (1.0 for all components)
    pub loss_weights: BurnLossWeights3D,

    /// Learning rate for optimization
    ///
    /// Controls the step size for gradient descent updates.
    ///
    /// **Default**: 1e-3
    ///
    /// **Recommended range**: 1e-5 to 1e-2
    pub learning_rate: f64,

    /// Batch size for training
    ///
    /// Number of data points processed per training iteration.
    ///
    /// **Default**: 1,000
    pub batch_size: usize,

    /// Maximum gradient norm for clipping
    ///
    /// Prevents gradient explosion by clipping gradients with norm
    /// exceeding this threshold.
    ///
    /// **Default**: 1.0
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

impl BurnPINN3DConfig {
    pub fn validate(&self) -> KwaversResult<()> {
        if self.hidden_layers.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Configuration must have at least one hidden layer".into(),
            ));
        }
        for (i, &size) in self.hidden_layers.iter().enumerate() {
            if size == 0 {
                return Err(KwaversError::InvalidInput(format!(
                    "Hidden layer size at index {i} must be non-zero"
                )));
            }
        }

        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Learning rate must be positive and finite (got {})",
                self.learning_rate
            )));
        }

        if self.num_collocation_points == 0 {
            return Err(KwaversError::InvalidInput(
                "Number of collocation points must be non-zero".into(),
            ));
        }

        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "Batch size must be non-zero".into(),
            ));
        }

        if !self.max_grad_norm.is_finite() || self.max_grad_norm <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Max grad norm must be positive and finite (got {})",
                self.max_grad_norm
            )));
        }

        self.loss_weights.validate()
    }
}

/// Loss weights for 3D PINN training
///
/// Defines the relative importance of different loss components in the
/// physics-informed training objective.
///
/// # Total Loss
///
/// ```text
/// L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc + λ_ic × L_ic
/// ```
///
/// Where:
/// - L_data: Mean squared error on training data
/// - L_pde: Mean squared error of PDE residual
/// - L_bc: Mean squared error of boundary condition violations
/// - L_ic: Mean squared error of initial condition violations
///
/// # Tuning Guidelines
///
/// - **Increase `pde_weight`**: If PDE residual is not decreasing
/// - **Increase `data_weight`**: If predictions don't match training data
/// - **Increase `bc_weight`**: If boundary conditions are violated
/// - **Increase `ic_weight`**: If initial conditions are not satisfied
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::BurnLossWeights3D;
///
/// // Default: equal weights
/// let weights = BurnLossWeights3D::default();
///
/// // Emphasize PDE residual
/// let weights = BurnLossWeights3D {
///     pde_weight: 10.0,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BurnLossWeights3D {
    /// Weight for data fitting loss (L_data)
    ///
    /// Controls how strongly the network fits training data points.
    ///
    /// **Default**: 1.0
    pub data_weight: f32,

    /// Weight for PDE residual loss (L_pde)
    ///
    /// Controls how strongly the network satisfies the wave equation.
    ///
    /// **Default**: 1.0
    pub pde_weight: f32,

    /// Weight for boundary condition loss (L_bc)
    ///
    /// Controls how strongly boundary conditions are enforced.
    ///
    /// **Default**: 1.0
    pub bc_weight: f32,

    /// Weight for initial condition loss (L_ic)
    ///
    /// Controls how strongly initial conditions are satisfied.
    ///
    /// **Default**: 1.0
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

impl BurnLossWeights3D {
    pub fn validate(&self) -> KwaversResult<()> {
        for (name, v) in [
            ("data_weight", self.data_weight),
            ("pde_weight", self.pde_weight),
            ("bc_weight", self.bc_weight),
            ("ic_weight", self.ic_weight),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be non-negative and finite (got {v})"
                )));
            }
        }
        Ok(())
    }
}

/// Training metrics for 3D PINN
///
/// Tracks training progress including loss components and timing information.
/// Useful for monitoring convergence and diagnosing training issues.
///
/// # Examples
///
/// ```rust,ignore
/// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::BurnTrainingMetrics3D;
///
/// let mut metrics = BurnTrainingMetrics3D::default();
///
/// // During training, metrics are updated
/// metrics.epochs_completed += 1;
/// metrics.total_loss.push(0.01234);
/// metrics.pde_loss.push(0.00567);
/// // ...
///
/// // After training
/// println!("Trained for {} epochs in {:.2}s",
///     metrics.epochs_completed,
///     metrics.training_time_secs
/// );
/// ```
#[derive(Debug, Clone)]
pub struct BurnTrainingMetrics3D {
    /// Number of training epochs completed
    pub epochs_completed: usize,

    /// Total loss at each epoch
    ///
    /// L_total = λ_data × L_data + λ_pde × L_pde + λ_bc × L_bc + λ_ic × L_ic
    pub total_loss: Vec<f64>,

    /// Data fitting loss at each epoch
    ///
    /// L_data = MSE(u_pred, u_data)
    pub data_loss: Vec<f64>,

    /// PDE residual loss at each epoch
    ///
    /// L_pde = MSE(R(x,y,z,t)) where R is the wave equation residual
    pub pde_loss: Vec<f64>,

    /// Boundary condition loss at each epoch
    ///
    /// L_bc = MSE(boundary violations)
    pub bc_loss: Vec<f64>,

    /// Initial condition loss at each epoch
    ///
    /// L_ic = MSE(initial condition violations)
    pub ic_loss: Vec<f64>,

    /// Total training time in seconds
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BurnPINN3DConfig::default();
        assert_eq!(config.hidden_layers, vec![100, 100, 100]);
        assert_eq!(config.num_collocation_points, 10000);
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.max_grad_norm, 1.0);
    }

    #[test]
    fn test_loss_weights_default() {
        let weights = BurnLossWeights3D::default();
        assert_eq!(weights.data_weight, 1.0);
        assert_eq!(weights.pde_weight, 1.0);
        assert_eq!(weights.bc_weight, 1.0);
        assert_eq!(weights.ic_weight, 1.0);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = BurnTrainingMetrics3D::default();
        assert_eq!(metrics.epochs_completed, 0);
        assert!(metrics.total_loss.is_empty());
        assert!(metrics.data_loss.is_empty());
        assert!(metrics.pde_loss.is_empty());
        assert!(metrics.bc_loss.is_empty());
        assert!(metrics.ic_loss.is_empty());
        assert_eq!(metrics.training_time_secs, 0.0);
    }

    #[test]
    fn test_config_custom() {
        let config = BurnPINN3DConfig {
            hidden_layers: vec![200, 200],
            num_collocation_points: 5000,
            learning_rate: 5e-4,
            batch_size: 500,
            max_grad_norm: 0.5,
            ..Default::default()
        };

        assert_eq!(config.hidden_layers, vec![200, 200]);
        assert_eq!(config.num_collocation_points, 5000);
        assert_eq!(config.learning_rate, 5e-4);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.max_grad_norm, 0.5);
    }

    #[test]
    fn test_loss_weights_custom() {
        let weights = BurnLossWeights3D {
            data_weight: 2.0,
            pde_weight: 10.0,
            bc_weight: 5.0,
            ic_weight: 3.0,
        };

        assert_eq!(weights.data_weight, 2.0);
        assert_eq!(weights.pde_weight, 10.0);
        assert_eq!(weights.bc_weight, 5.0);
        assert_eq!(weights.ic_weight, 3.0);
    }

    #[test]
    fn test_metrics_update() {
        let mut metrics = BurnTrainingMetrics3D {
            epochs_completed: 100,
            ..Default::default()
        };

        metrics.total_loss.push(0.1);
        metrics.total_loss.push(0.05);
        metrics.data_loss.push(0.04);
        metrics.pde_loss.push(0.03);
        metrics.bc_loss.push(0.02);
        metrics.ic_loss.push(0.01);
        metrics.training_time_secs = 123.45;

        assert_eq!(metrics.epochs_completed, 100);
        assert_eq!(metrics.total_loss.len(), 2);
        assert_eq!(metrics.total_loss[1], 0.05);
        assert_eq!(metrics.training_time_secs, 123.45);
    }

    #[test]
    fn test_config_clone() {
        let config1 = BurnPINN3DConfig::default();
        let config2 = config1.clone();

        assert_eq!(config1.hidden_layers, config2.hidden_layers);
        assert_eq!(
            config1.num_collocation_points,
            config2.num_collocation_points
        );
        assert_eq!(config1.learning_rate, config2.learning_rate);
    }

    #[test]
    fn test_metrics_clone() {
        let mut metrics1 = BurnTrainingMetrics3D {
            epochs_completed: 50,
            ..Default::default()
        };
        metrics1.total_loss.push(0.123);

        let metrics2 = metrics1.clone();

        assert_eq!(metrics1.epochs_completed, metrics2.epochs_completed);
        assert_eq!(metrics1.total_loss, metrics2.total_loss);
    }
}
