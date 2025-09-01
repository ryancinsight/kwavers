//! Neural network implementation for parameter optimization

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;

/// Neural network for parameter optimization
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    /// Create a new neural network with specified dimensions
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, learning_rate: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Xavier initialization
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        Self {
            weights1: Array2::from_shape_fn((hidden_dim, input_dim), |_| {
                rng.r#gen::<f64>() * scale1 - scale1 / 2.0
            }),
            bias1: Array1::zeros(hidden_dim),
            weights2: Array2::from_shape_fn((output_dim, hidden_dim), |_| {
                rng.r#gen::<f64>() * scale2 - scale2 / 2.0
            }),
            bias2: Array1::zeros(output_dim),
            learning_rate,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        if input.len() != self.weights1.ncols() {
            return Err(KwaversError::Physics(
                crate::error::PhysicsError::DimensionMismatch,
            ));
        }

        // Hidden layer: ReLU(W1 * input + b1)
        let hidden = self.weights1.dot(input) + &self.bias1;
        let hidden_activated = hidden.mapv(|x| x.max(0.0)); // ReLU activation

        // Output layer: W2 * hidden + b2 (no activation for regression)
        let output = self.weights2.dot(&hidden_activated) + &self.bias2;

        Ok(output)
    }

    /// Update weights using gradient descent with proper backpropagation
    pub fn update_weights(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
        learning_rate: f64,
    ) -> KwaversResult<()> {
        // Forward pass with intermediate values
        let z1 = self.weights1.dot(input) + &self.bias1;
        let a1 = z1.mapv(Self::relu);
        let z2 = self.weights2.dot(&a1) + &self.bias2;
        let prediction = z2.clone(); // Linear output for regression

        // Compute loss gradient
        let error = &prediction - target;

        // Backpropagation
        // Output layer gradients
        let delta2 = error; // For MSE loss with linear output
        let grad_w2 = delta2
            .clone()
            .insert_axis(Axis(1))
            .dot(&a1.clone().insert_axis(Axis(0)));
        let grad_b2 = delta2.clone();

        // Hidden layer gradients
        let delta1 = self.weights2.t().dot(&delta2) * z1.mapv(Self::relu_derivative);
        let grad_w1 = delta1
            .clone()
            .insert_axis(Axis(1))
            .dot(&input.clone().insert_axis(Axis(0)));
        let grad_b1 = delta1;

        // Update weights and biases with gradient descent
        self.weights2 -= &(learning_rate * grad_w2);
        self.bias2 -= &(learning_rate * grad_b2);
        self.weights1 -= &(learning_rate * grad_w1);
        self.bias1 -= &(learning_rate * grad_b1);

        Ok(())
    }

    /// ReLU activation function
    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    /// ReLU derivative
    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
