//! Neural network architecture for beamforming optimization.
//!
//! This module implements a simple feedforward neural network for adaptive
//! beamforming weight computation. The network learns to map input features
//! (RF data, steering angles, signal quality metrics) to optimal beamforming
//! parameters.
//!
//! ## Architecture
//!
//! ```text
//! Input Features → Dense Layer 1 → Tanh → Dense Layer 2 → Tanh → Output
//! ```
//!
//! ## Initialization
//!
//! Weights are initialized using Xavier/Glorot initialization:
//! ```text
//! W ~ U(-√(6/(n_in + n_out)), +√(6/(n_in + n_out)))
//! ```
//! This ensures proper gradient flow during training.
//!
//! ## Activation
//!
//! Hyperbolic tangent (tanh) activation:
//! - Range: [-1, 1]
//! - Smooth, differentiable
//! - Zero-centered (better convergence than sigmoid)
//!
//! ## References
//!
//! - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
//! - LeCun et al. (1998): "Efficient BackProp"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

use super::layer::NeuralLayer;
use super::types::BeamformingFeedback;

#[cfg(feature = "pinn")]
use super::physics::PhysicsConstraints;

/// Feedforward neural network for beamforming optimization.
#[derive(Debug)]
pub struct NeuralBeamformingNetwork {
    /// Network layers
    layers: Vec<NeuralLayer>,
    /// Architecture specification (layer sizes)
    architecture: Vec<usize>,
}

impl NeuralBeamformingNetwork {
    /// Create a new neural network with specified architecture.
    ///
    /// # Arguments
    ///
    /// * `architecture` - Layer sizes [input_size, hidden1, hidden2, ..., output_size]
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create network: 128 → 64 → 32 → 16
    /// let net = NeuralBeamformingNetwork::new(&[128, 64, 32, 16])?;
    /// ```
    ///
    /// # Invariants
    ///
    /// - architecture.len() >= 2 (at least input and output layers)
    /// - All layer sizes > 0
    pub fn new(architecture: &[usize]) -> KwaversResult<Self> {
        if architecture.len() < 2 {
            return Err(KwaversError::InvalidInput(
                "Architecture must have at least 2 layers (input and output)".to_string(),
            ));
        }

        if architecture.contains(&0) {
            return Err(KwaversError::InvalidInput(
                "All layer sizes must be > 0".to_string(),
            ));
        }

        let mut layers = Vec::new();

        // Create layers for each adjacent pair
        for i in 0..architecture.len() - 1 {
            layers.push(NeuralLayer::new(architecture[i], architecture[i + 1])?);
        }

        Ok(Self {
            layers,
            architecture: architecture.to_vec(),
        })
    }

    /// Get network architecture specification.
    pub fn architecture(&self) -> &[usize] {
        &self.architecture
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector (6 summary statistics)
    /// * `steering_angles` - Beam steering angles (degrees)
    ///
    /// # Returns
    ///
    /// Network output as 3D image volume.
    ///
    /// # Process
    ///
    /// 1. Concatenate input features with steering angle
    /// 2. Forward through each layer with tanh activation
    /// 3. Return output (no final activation)
    pub fn forward(
        &self,
        features: &ndarray::Array1<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        // Concatenate features with steering angle into 3D input
        let input = self.concatenate_features(features, steering_angles)?;
        let mut output = input;

        // Forward through layers
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    /// Forward pass with physics-informed constraints.
    ///
    /// Applies physical consistency constraints after the neural network output
    /// to ensure acoustic reciprocity, spatial coherence, and sparsity.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector (6 summary statistics)
    /// * `steering_angles` - Beam steering angles
    /// * `constraints` - Physics constraints to enforce
    ///
    /// # Returns
    ///
    /// Physics-constrained network output.
    #[cfg(feature = "pinn")]
    pub fn forward_physics_informed(
        &self,
        features: &ndarray::Array1<f32>,
        steering_angles: &[f64],
        constraints: &PhysicsConstraints,
    ) -> KwaversResult<Array3<f32>> {
        let unconstrained = self.forward(features, steering_angles)?;
        constraints.apply(&unconstrained)
    }

    /// Adapt network weights based on performance feedback.
    ///
    /// Simple gradient descent update (placeholder for full backpropagation).
    /// TODO_AUDIT: P2 - Advanced Neural Beamforming - Implement full deep learning beamforming with attention mechanisms and transformer architectures
    /// DEPENDS ON: analysis/beamforming/neural/transformer.rs, analysis/beamforming/neural/attention.rs, analysis/beamforming/neural/contrastive.rs
    /// MISSING: Transformer-based beamforming with self-attention on channel correlations
    /// MISSING: Contrastive learning for improved generalization across transducer geometries
    /// MISSING: Meta-learning for fast adaptation to new imaging scenarios
    /// MISSING: Physics-informed neural networks with wave equation constraints
    /// MISSING: Uncertainty quantification using Monte Carlo dropout
    /// MISSING: Multi-task learning for simultaneous beamforming and aberration correction
    ///
    /// # Arguments
    ///
    /// * `feedback` - Performance metrics and error gradients
    /// * `learning_rate` - Step size for weight updates
    ///
    /// # Update Rule
    ///
    /// ```text
    /// W_new = W_old - η · ∇L
    /// ```
    /// where η is the learning_rate and ∇L is approximated by error_gradient.
    pub fn adapt(
        &mut self,
        feedback: &BeamformingFeedback,
        learning_rate: f64,
    ) -> KwaversResult<()> {
        // Simplified adaptation: scale gradient by feedback
        let effective_gradient = (learning_rate * feedback.error_gradient) as f32;

        for layer in &mut self.layers {
            layer.adapt(effective_gradient)?;
        }

        Ok(())
    }

    /// Concatenate feature vector with steering angle into network input.
    ///
    /// # Process
    ///
    /// 1. Take 6 feature statistics
    /// 2. Append steering angle as 7th feature
    /// 3. Reshape to (1, 1, 7) for layer processing
    ///
    /// # Arguments
    ///
    /// * `features` - Feature vector (6 elements: mean, std, gradient, laplacian, entropy, peak)
    /// * `steering_angles` - Beam steering angles to encode
    ///
    /// # Returns
    ///
    /// Input tensor shaped (1, 1, 7) for network processing.
    fn concatenate_features(
        &self,
        features: &ndarray::Array1<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        if features.len() != 6 {
            return Err(KwaversError::InvalidInput(format!(
                "Expected 6 features, got {}",
                features.len()
            )));
        }

        if steering_angles.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No steering angles provided".to_string(),
            ));
        }

        // Create input vector: [6 features + 1 angle] = 7 elements
        let mut input_vec = features.to_vec();
        input_vec.push(steering_angles[0] as f32); // Use first steering angle

        // Reshape to (1, 1, 7) for layer processing
        // This represents 1 batch item with 1 spatial location and 7 features
        Array3::from_shape_vec((1, 1, 7), input_vec)
            .map_err(|e| KwaversError::InternalError(format!("Failed to reshape input: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = NeuralBeamformingNetwork::new(&[64, 32, 16, 8]).unwrap();
        assert_eq!(net.architecture(), &[64, 32, 16, 8]);
    }

    #[test]
    fn test_network_invalid_architecture() {
        // Single layer (no hidden layer)
        let result = NeuralBeamformingNetwork::new(&[64]);
        assert!(result.is_err());

        // Zero-sized layer
        let result = NeuralBeamformingNetwork::new(&[64, 0, 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_feature_concatenation() {
        let net = NeuralBeamformingNetwork::new(&[7, 5]).unwrap();

        // 6 feature statistics + 1 angle = 7 input features
        use ndarray::Array1;
        let features = Array1::from_vec(vec![0.5, 0.1, 0.2, 0.05, 0.3, 0.8]);
        let angles = vec![15.0];

        let concatenated = net.concatenate_features(&features, &angles).unwrap();

        // Should be (1, 1, 7): 6 features + 1 angle
        assert_eq!(concatenated.dim(), (1, 1, 7));
    }

    #[test]
    fn test_network_forward() {
        let net = NeuralBeamformingNetwork::new(&[7, 4, 2]).unwrap();

        // 6 feature statistics
        use ndarray::Array1;
        let features = Array1::from_vec(vec![0.5, 0.1, 0.2, 0.05, 0.3, 0.8]);
        let angles = vec![15.0];

        let output = net.forward(&features, &angles).unwrap();

        assert_eq!(output.dim().2, 2); // Output size = 2
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_network_adaptation() {
        let mut net = NeuralBeamformingNetwork::new(&[4, 2]).unwrap();

        let feedback = BeamformingFeedback {
            improvement: 0.1,
            error_gradient: 0.5,
            signal_quality: 0.8,
        };

        let result = net.adapt(&feedback, 0.01);
        assert!(result.is_ok());
    }
}
