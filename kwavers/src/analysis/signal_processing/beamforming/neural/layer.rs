//! Neural network layer primitives for beamforming.
//!
//! This module provides the basic building blocks for constructing neural networks
//! for adaptive beamforming: dense (fully-connected) layers with configurable
//! activation functions.
//!
//! ## Layer Architecture
//!
//! A dense layer performs the transformation:
//! ```text
//! y = activation(Wx + b)
//! ```
//!
//! where:
//! - W: weight matrix (input_size × output_size)
//! - x: input vector (batch × input_size)
//! - b: bias vector (output_size)
//! - activation: non-linear function (tanh)
//!
//! ## Weight Initialization
//!
//! Xavier/Glorot uniform initialization:
//! ```text
//! limit = √(6 / (n_in + n_out))
//! W ~ U(-limit, +limit)
//! b = 0
//! ```
//!
//! This initialization ensures:
//! - Proper gradient flow during backpropagation
//! - Prevents vanishing/exploding gradients
//! - Symmetric distribution around zero
//!
//! ## Mathematical Foundation
//!
//! The Xavier initialization maintains variance across layers:
//! ```text
//! Var(W) = 2 / (n_in + n_out)
//! ```
//!
//! For tanh activation (range [-1, 1]):
//! ```text
//! tanh(x) = (e^x - e^-x) / (e^x + e^-x)
//! tanh'(x) = 1 - tanh²(x)
//! ```
//!
//! ## References
//!
//! - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
//! - He et al. (2015): "Delving Deep into Rectifiers"
//! - LeCun et al. (1998): "Efficient BackProp"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};
use rand::distributions::{Distribution, Uniform};

/// Single dense (fully-connected) layer in a neural network.
///
/// Implements a linear transformation followed by tanh activation.
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// Weight matrix (input_size × output_size)
    weights: Array2<f32>,
    /// Bias vector (output_size)
    biases: Array1<f32>,
    /// Input feature dimension
    input_size: usize,
    /// Output feature dimension
    output_size: usize,
}

impl NeuralLayer {
    /// Create a new dense layer with Xavier/Glorot initialization.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    ///
    /// # Initialization Strategy
    ///
    /// **Weights:**
    /// ```text
    /// limit = √(6 / (n_in + n_out))
    /// W_{ij} ~ U(-limit, +limit)
    /// ```
    ///
    /// **Biases:**
    /// ```text
    /// b_i = 0  ∀i
    /// ```
    ///
    /// # Invariants
    ///
    /// - input_size > 0
    /// - output_size > 0
    /// - weights.shape == (input_size, output_size)
    /// - biases.len() == output_size
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = NeuralLayer::new(64, 32)?;
    /// assert_eq!(layer.input_size(), 64);
    /// assert_eq!(layer.output_size(), 32);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> KwaversResult<Self> {
        if input_size == 0 || output_size == 0 {
            return Err(KwaversError::InvalidInput(
                "Layer sizes must be > 0".to_string(),
            ));
        }

        // Xavier/Glorot uniform initialization
        // Ensures variance is maintained: Var(W) = 2/(n_in + n_out)
        let limit = (6.0 / (input_size as f64 + output_size as f64)).sqrt();
        let dist = Uniform::new(-limit, limit);

        let mut rng = rand::thread_rng();
        let mut weights_data = Vec::with_capacity(input_size * output_size);

        for _ in 0..input_size * output_size {
            weights_data.push(dist.sample(&mut rng) as f32);
        }

        let weights =
            Array2::from_shape_vec((input_size, output_size), weights_data).map_err(|e| {
                KwaversError::InternalError(format!("Failed to create weight matrix: {}", e))
            })?;

        // Zero initialization for biases (standard practice)
        let biases = Array1::zeros(output_size);

        Ok(Self {
            weights,
            biases,
            input_size,
            output_size,
        })
    }

    /// Get input feature dimension.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get output feature dimension.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Access weight matrix (for inspection/testing).
    pub fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    /// Access bias vector (for inspection/testing).
    pub fn biases(&self) -> &Array1<f32> {
        &self.biases
    }

    /// Forward pass through the layer.
    ///
    /// Applies linear transformation followed by tanh activation:
    /// ```text
    /// y = tanh(Wx + b)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (frame × spatial × features)
    ///
    /// # Returns
    ///
    /// Output tensor (frame × spatial × output_features) after transformation.
    ///
    /// # Process
    ///
    /// 1. Reshape input: (d0, d1, d2) → (d0·d1, d2)
    /// 2. Matrix multiplication: (batch, in) × (in, out) → (batch, out)
    /// 3. Add biases: broadcast across batch dimension
    /// 4. Apply activation: tanh element-wise
    /// 5. Reshape output: (d0·d1, out) → (d0, d1, out)
    ///
    /// # Mathematical Definition
    ///
    /// For input x ∈ ℝ^(batch × n_in):
    /// ```text
    /// z = xW + b          (linear)
    /// y = tanh(z)         (activation)
    /// ```
    ///
    /// where:
    /// - W ∈ ℝ^(n_in × n_out): weight matrix
    /// - b ∈ ℝ^(n_out): bias vector
    /// - tanh: hyperbolic tangent, range [-1, 1]
    ///
    /// # Invariants
    ///
    /// - input.dim().2 == self.input_size (feature dimension must match)
    /// - output.dim() == (input.dim().0, input.dim().1, self.output_size)
    /// - All output values in [-1, 1] (tanh bounds)
    pub fn forward(&self, input: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let (d0, d1, d2) = input.dim();

        // Validate input dimension
        if d2 != self.input_size {
            return Err(KwaversError::DimensionMismatch(format!(
                "Layer expects input size {}, got {}",
                self.input_size, d2
            )));
        }

        // Reshape to (Batch, InputSize) where Batch = d0 × d1
        // This flattens the spatial dimensions for matrix multiplication
        let flattened_input = input
            .to_shape((d0 * d1, d2))
            .map_err(|e| KwaversError::InternalError(format!("Input reshape failed: {}", e)))?;

        // Linear transformation: (Batch, In) × (In, Out) → (Batch, Out)
        let linear_output = flattened_input.dot(&self.weights);

        // Add biases: broadcast across batch dimension
        // b ∈ ℝ^(out) is broadcast to match (batch, out)
        let output_with_bias = linear_output + &self.biases;

        // Apply tanh activation element-wise
        // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        let activated_output = output_with_bias.mapv(|x| x.tanh());

        // Reshape back to (d0, d1, output_size)
        // Restores spatial structure
        let output = activated_output
            .to_shape((d0, d1, self.output_size))
            .map_err(|e| KwaversError::InternalError(format!("Output reshape failed: {}", e)))?
            .to_owned();

        Ok(output)
    }

    /// Adapt layer weights using simple gradient descent.
    ///
    /// **WARNING:** This is a simplified placeholder implementation.
    /// Production code should use proper backpropagation with layer-specific
    /// gradients computed via the chain rule.
    ///
    /// # Arguments
    ///
    /// * `gradient` - Effective gradient (learning_rate × error_gradient)
    ///
    /// # Update Rule
    ///
    /// Simplified SGD with fixed step size:
    /// ```text
    /// W_new = W_old - 0.01 · gradient
    /// b_new = b_old - 0.01 · gradient
    /// ```
    ///
    /// # Limitations
    ///
    /// - Does not use actual gradients from backpropagation
    /// - Applies same update to all weights (ignores layer position)
    /// - No momentum, adaptive learning rate, or regularization
    /// - For demonstration purposes only
    ///
    /// # Proper Backpropagation
    ///
    /// True gradient descent requires:
    /// ```text
    /// δ^l = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^l)     (backprop error)
    /// ∂L/∂W^l = a^(l-1) (δ^l)^T              (weight gradient)
    /// ∂L/∂b^l = δ^l                          (bias gradient)
    /// W_new = W_old - η · ∂L/∂W^l           (update)
    /// ```
    pub fn adapt(&mut self, gradient: f32) -> KwaversResult<()> {
        // Simplified SGD update with fixed step size
        let step_size = 0.01 * gradient;
        self.weights.mapv_inplace(|w| w - step_size);
        self.biases.mapv_inplace(|b| b - step_size);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_layer_creation() {
        let layer = NeuralLayer::new(64, 32).unwrap();
        assert_eq!(layer.input_size(), 64);
        assert_eq!(layer.output_size(), 32);
        assert_eq!(layer.weights().dim(), (64, 32));
        assert_eq!(layer.biases().len(), 32);
    }

    #[test]
    fn test_neural_layer_zero_sizes() {
        assert!(NeuralLayer::new(0, 32).is_err());
        assert!(NeuralLayer::new(64, 0).is_err());
        assert!(NeuralLayer::new(0, 0).is_err());
    }

    #[test]
    fn test_xavier_initialization_bounds() {
        let layer = NeuralLayer::new(64, 32).unwrap();
        let limit = (6.0_f64 / (64.0_f64 + 32.0_f64)).sqrt();

        // All weights should be within [-limit, limit]
        for &weight in layer.weights().iter() {
            assert!(weight >= -limit as f32);
            assert!(weight <= limit as f32);
        }

        // All biases should be zero
        for &bias in layer.biases().iter() {
            assert_eq!(bias, 0.0);
        }
    }

    #[test]
    fn test_neural_layer_forward() {
        let layer = NeuralLayer::new(8, 4).unwrap();
        let input = Array3::ones((2, 3, 8)); // Batch=(2×3)=6, Features=8

        let output = layer.forward(&input).unwrap();

        assert_eq!(output.dim(), (2, 3, 4));
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_neural_layer_activation_range() {
        let layer = NeuralLayer::new(4, 4).unwrap();
        let input = Array3::from_elem((2, 2, 4), 100.0); // Very large input

        let output = layer.forward(&input).unwrap();

        // Tanh saturates: output must be in [-1, 1]
        for &val in output.iter() {
            assert!(val >= -1.0, "Output {} < -1.0", val);
            assert!(val <= 1.0, "Output {} > 1.0", val);
        }
    }

    #[test]
    fn test_neural_layer_dimension_mismatch() {
        let layer = NeuralLayer::new(8, 4).unwrap();
        let wrong_input = Array3::ones((2, 3, 16)); // Wrong feature size (16 instead of 8)

        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        if let Err(KwaversError::DimensionMismatch(msg)) = result {
            assert!(msg.contains("expects input size 8"));
            assert!(msg.contains("got 16"));
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[test]
    fn test_neural_layer_forward_shape_preservation() {
        let layer = NeuralLayer::new(16, 8).unwrap();
        let input = Array3::from_elem((5, 7, 16), 0.5);

        let output = layer.forward(&input).unwrap();

        // Spatial dimensions preserved, feature dimension transformed
        assert_eq!(output.dim().0, 5);
        assert_eq!(output.dim().1, 7);
        assert_eq!(output.dim().2, 8);
    }

    #[test]
    fn test_neural_layer_adaptation() {
        let mut layer = NeuralLayer::new(4, 2).unwrap();
        let initial_weights = layer.weights().clone();

        layer.adapt(0.5).unwrap();

        // Weights should have changed
        let updated_weights = layer.weights();
        assert_ne!(initial_weights, *updated_weights);
    }

    #[test]
    fn test_tanh_activation_zero_input() {
        let layer = NeuralLayer::new(4, 4).unwrap();
        let input = Array3::zeros((2, 2, 4));

        let output = layer.forward(&input).unwrap();

        // For zero input: tanh(b) where b ≈ 0 → tanh(0) = 0
        // (biases initialized to zero)
        assert!(output.iter().all(|&x| x.abs() < 0.1));
    }

    #[test]
    fn test_layer_linearity_before_activation() {
        // Test that doubling input approximately doubles pre-activation output
        // (ignoring saturation effects)
        let layer = NeuralLayer::new(4, 4).unwrap();
        let input1 = Array3::from_elem((2, 2, 4), 0.01);
        let input2 = Array3::from_elem((2, 2, 4), 0.02);

        let output1 = layer.forward(&input1).unwrap();
        let output2 = layer.forward(&input2).unwrap();

        // For small inputs, tanh(x) ≈ x (linear regime)
        // So output should be approximately linear in input
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..4 {
                    let ratio = output2[[i, j, k]] / (output1[[i, j, k]] + 1e-8);
                    // Should be close to 2.0 in linear regime
                    assert!(ratio > 1.5 && ratio < 2.5, "Ratio: {}", ratio);
                }
            }
        }
    }
}
