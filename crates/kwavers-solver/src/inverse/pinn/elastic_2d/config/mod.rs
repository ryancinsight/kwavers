//! Configuration and hyperparameters for 2D Elastic Wave PINN.
//!
//! L_total = λ_pde·L_pde + λ_bc·L_bc + λ_ic·L_ic + λ_data·L_data

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;
mod training;

pub use training::Config;

/// Loss function weights for physics-informed training.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LossWeights {
    pub pde: f64,
    pub boundary: f64,
    pub initial: f64,
    pub data: f64,
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

/// Activation function for hidden layers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ElasticPinnActivationFunction {
    Tanh,
    Sin,
    Swish,
    Adaptive,
}

/// Learning rate scheduling strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ElasticPinnLrScheduler {
    Constant,
    Exponential {
        decay_rate: f64,
    },
    Step {
        factor: f64,
        step_size: usize,
    },
    CosineAnnealing {
        lr_min: f64,
    },
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
        threshold: f64,
    },
}

/// Optimizer type for training.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ElasticPinnOptimizerType {
    SGD {
        momentum: f64,
    },
    Adam {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    AdamW {
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    LBFGS {
        history_size: usize,
        tolerance: f64,
    },
}
