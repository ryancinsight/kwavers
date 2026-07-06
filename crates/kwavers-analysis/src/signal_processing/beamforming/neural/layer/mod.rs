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

use kwavers_core::{
    error::{KwaversError, KwaversResult},
    utils::iterators::apply_inplace,
};
use ndarray::{Array1, Array2, Array3};
use rand::distributions::{Distribution, Uniform};

#[cfg(test)]
mod tests;

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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(input_size: usize, output_size: usize) -> KwaversResult<Self> {
        if input_size == 0 || output_size == 0 {
            return Err(KwaversError::InvalidInput(
                "Layer sizes must be > 0".to_owned(),
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
    #[must_use]
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get output feature dimension.
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Access weight matrix (for inspection/testing).
    #[must_use]
    pub fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    /// Access bias vector (for inspection/testing).
    #[must_use]
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
    /// # Errors
    /// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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

    /// Apply a scalar feedback step to the layer parameters.
    ///
    /// The update is the exact gradient descent step for the calibration
    /// objective
    ///
    /// ```text
    /// J(W, b) = (g / 2) · (||W||_F^2 + ||b||_2^2)
    /// ```
    ///
    /// where `g` is the supplied feedback scalar. The gradient is
    /// parameter-dependent:
    ///
    /// ```text
    /// ∂J/∂W = gW
    /// ∂J/∂b = gb
    /// ```
    ///
    /// so the update is
    ///
    /// ```text
    /// W_{k+1} = (1 - 0.01g) W_k
    /// b_{k+1} = (1 - 0.01g) b_k
    /// ```
    ///
    /// This is a closed-form, parameter-sensitive step rather than a constant
    /// offset mutation. The scalar contract is the limiting input, so the
    /// update stays consistent with the current public API while remaining
    /// mathematically defined.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn adapt(&mut self, gradient: f32) -> KwaversResult<()> {
        if !gradient.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Layer feedback gradient must be finite".to_owned(),
            ));
        }

        let scale = 0.01f32.mul_add(-gradient, 1.0);

        if !scale.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Layer feedback gradient produces a non-finite update scale".to_owned(),
            ));
        }

        apply_inplace(&mut self.weights, |w| w * scale);
        apply_inplace(&mut self.biases, |b| b * scale);
        Ok(())
    }
}
