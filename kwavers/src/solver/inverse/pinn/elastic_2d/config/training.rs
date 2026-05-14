use super::{
    ActivationFunction, LearningRateScheduler, LossWeights, OptimizerType, SamplingStrategy,
};
use serde::{Deserialize, Serialize};

/// Training configuration for 2D Elastic Wave PINN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Architecture
    pub hidden_layers: Vec<usize>,
    pub activation: ActivationFunction,
    // Training
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: Option<usize>,
    pub scheduler: LearningRateScheduler,
    pub optimizer: OptimizerType,
    // Collocation
    pub n_collocation_interior: usize,
    pub n_collocation_boundary: usize,
    pub n_collocation_initial: usize,
    pub sampling_strategy: SamplingStrategy,
    pub adaptive_sampling: bool,
    pub adaptive_threshold: f64,
    // Loss
    pub loss_weights: LossWeights,
    // Material (inverse)
    pub optimize_lambda: bool,
    pub lambda_init: Option<f64>,
    pub optimize_mu: bool,
    pub mu_init: Option<f64>,
    pub optimize_rho: bool,
    pub rho_init: Option<f64>,
    // Regularization
    pub weight_decay: f64,
    pub gradient_clip: Option<f64>,
    // Checkpointing
    pub checkpoint_interval: usize,
    pub checkpoint_dir: Option<String>,
    // Logging
    pub log_interval: usize,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden_layers: vec![100, 100, 100, 100],
            activation: ActivationFunction::Tanh,
            learning_rate: 1e-3,
            n_epochs: 10000,
            batch_size: None,
            scheduler: LearningRateScheduler::Exponential { decay_rate: 0.95 },
            optimizer: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            n_collocation_interior: 10000,
            n_collocation_boundary: 1000,
            n_collocation_initial: 1000,
            sampling_strategy: SamplingStrategy::LatinHypercube,
            adaptive_sampling: false,
            adaptive_threshold: 2.0,
            loss_weights: LossWeights::default(),
            optimize_lambda: false,
            lambda_init: None,
            optimize_mu: false,
            mu_init: None,
            optimize_rho: false,
            rho_init: None,
            weight_decay: 0.0,
            gradient_clip: None,
            checkpoint_interval: 1000,
            checkpoint_dir: None,
            log_interval: 100,
            verbose: false,
        }
    }
}

impl Config {
    #[must_use]
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
                data: 0.0,
                interface: 10.0,
            },
            ..Default::default()
        }
    }

    #[must_use]
    pub fn inverse_problem(lambda_guess: f64, mu_guess: f64, rho_guess: f64) -> Self {
        Self {
            optimize_lambda: true,
            lambda_init: Some(lambda_guess),
            optimize_mu: true,
            mu_init: Some(mu_guess),
            optimize_rho: true,
            rho_init: Some(rho_guess),
            loss_weights: LossWeights {
                pde: 0.1,
                boundary: 1.0,
                initial: 1.0,
                data: 100.0,
                interface: 1.0,
            },
            n_epochs: 30000,
            learning_rate: 5e-4,
            ..Default::default()
        }
    }
    /// Validate.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_layers.is_empty() {
            return Err("Must have at least one hidden layer".to_owned());
        }
        if self.hidden_layers.contains(&0) {
            return Err("Hidden layer sizes must be positive".to_owned());
        }
        if self.learning_rate <= 0.0 || self.learning_rate >= 1.0 {
            return Err("Learning rate must be in (0, 1)".to_owned());
        }
        if self.n_epochs == 0 {
            return Err("Number of epochs must be positive".to_owned());
        }
        if self.n_collocation_interior == 0 {
            return Err("Number of interior collocation points must be positive".to_owned());
        }
        if self.loss_weights.pde < 0.0
            || self.loss_weights.boundary < 0.0
            || self.loss_weights.initial < 0.0
            || self.loss_weights.data < 0.0
            || self.loss_weights.interface < 0.0
        {
            return Err("Loss weights must be non-negative".to_owned());
        }
        if self.optimize_lambda && self.lambda_init.is_none() {
            return Err("lambda_init required when optimize_lambda is true".to_owned());
        }
        if self.optimize_mu && self.mu_init.is_none() {
            return Err("mu_init required when optimize_mu is true".to_owned());
        }
        if self.optimize_rho && self.rho_init.is_none() {
            return Err("rho_init required when optimize_rho is true".to_owned());
        }
        Ok(())
    }
}
