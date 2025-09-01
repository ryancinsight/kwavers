//! Parameter optimization using reinforcement learning

use super::convergence_predictor::ConvergencePredictor;
use super::neural_network::NeuralNetwork;
use ndarray::Array1;
use std::collections::VecDeque;

/// Experience tuple for reinforcement learning
#[derive(Clone, Debug)]
pub struct OptimizationExperience {
    pub state: Array1<f64>,
    pub action: Array1<f64>,
    pub reward: f64,
    pub next_state: Option<Array1<f64>>,
    pub done: bool,
}

/// Parameter optimizer using deep reinforcement learning
#[derive(Debug)]
pub struct ParameterOptimizer {
    learning_rate: f64,
    exploration_rate: f64,
    experience_buffer: VecDeque<OptimizationExperience>,
    neural_network: NeuralNetwork,
    convergence_predictor: ConvergencePredictor,
}

impl ParameterOptimizer {
    /// Create a new parameter optimizer
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            learning_rate: 0.001,
            exploration_rate: 0.1,
            experience_buffer: VecDeque::with_capacity(10000),
            neural_network: NeuralNetwork::new(state_dim, 128, action_dim, 0.001),
            convergence_predictor: ConvergencePredictor::new(100),
        }
    }
}
