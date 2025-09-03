//! Convergence prediction for optimization algorithms

use ndarray::Array1;

/// Convergence predictor using time series analysis
#[derive(Debug)]
pub struct ConvergencePredictor {
    history_size: usize,
    loss_history: Vec<f64>,
    gradient_history: Vec<Array1<f64>>,
}

impl ConvergencePredictor {
    /// Create a new convergence predictor
    #[must_use]
    pub fn new(history_size: usize) -> Self {
        Self {
            history_size,
            loss_history: Vec::with_capacity(history_size),
            gradient_history: Vec::with_capacity(history_size),
        }
    }
}
