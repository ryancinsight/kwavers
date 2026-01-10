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
use ndarray::{Array3, Axis};

use super::layer::NeuralLayer;
use super::physics::PhysicsConstraints;
use super::types::BeamformingFeedback;

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

        if architecture.iter().any(|&size| size == 0) {
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
    /// * `features` - Input feature maps
    /// * `steering_angles` - Beam steering angles (degrees)
    ///
    /// # Returns
    ///
    /// Network output as 3D image volume.
    ///
    /// # Process
    ///
    /// 1. Concatenate input features
    /// 2. Forward through each layer with tanh activation
    /// 3. Return output (no final activation)
    pub fn forward(
        &self,
        features: &[Array3<f32>],
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        // Concatenate features and flatten for neural network input
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
    /// * `features` - Input feature maps
    /// * `steering_angles` - Beam steering angles
    /// * `constraints` - Physics constraints to enforce
    ///
    /// # Returns
    ///
    /// Physics-constrained network output.
    #[cfg(feature = "pinn")]
    pub fn forward_physics_informed(
        &self,
        features: &[Array3<f32>],
        steering_angles: &[f64],
        constraints: &PhysicsConstraints,
    ) -> KwaversResult<Array3<f32>> {
        let unconstrained = self.forward(features, steering_angles)?;
        constraints.apply(&unconstrained)
    }

    /// Adapt network weights based on performance feedback.
    ///
    /// Simple gradient descent update (placeholder for full backpropagation).
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

    /// Concatenate multiple feature maps into a single tensor.
    ///
    /// # Process
    ///
    /// 1. Concatenate all feature arrays along the feature dimension (axis 2)
    /// 2. Append steering angle information as an additional feature channel
    ///
    /// # Arguments
    ///
    /// * `features` - Array of feature maps (each frame × lateral × features)
    /// * `steering_angles` - Beam steering angles to encode
    ///
    /// # Returns
    ///
    /// Concatenated feature tensor.
    fn concatenate_features(
        &self,
        features: &[Array3<f32>],
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        if features.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if steering_angles.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No steering angles provided".to_string(),
            ));
        }

        // Start with first feature map
        let mut concatenated = features[0].clone();

        // Concatenate remaining features
        for feature in features.iter().skip(1) {
            concatenated.append(Axis(2), feature.view()).map_err(|e| {
                KwaversError::InternalError(format!("Feature concatenation failed: {}", e))
            })?;
        }

        // Encode steering angles as additional feature channels
        let angle_feature = Array3::from_elem(
            (
                concatenated.shape()[0],
                concatenated.shape()[1],
                steering_angles.len(),
            ),
            steering_angles[0] as f32, // Simplified: broadcast first angle
        );

        concatenated
            .append(Axis(2), angle_feature.view())
            .map_err(|e| {
                KwaversError::InternalError(format!("Angle concatenation failed: {}", e))
            })?;

        Ok(concatenated)
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
        let net = NeuralBeamformingNetwork::new(&[10, 5]).unwrap();

        let feature1 = Array3::ones((2, 3, 3));
        let feature2 = Array3::from_elem((2, 3, 2), 2.0);
        let features = vec![feature1, feature2];
        let angles = vec![15.0, 30.0];

        let concatenated = net.concatenate_features(&features, &angles).unwrap();

        // 3 + 2 + 2 (angles) = 7 features
        assert_eq!(concatenated.dim(), (2, 3, 7));
    }

    #[test]
    fn test_network_forward() {
        let net = NeuralBeamformingNetwork::new(&[8, 4, 2]).unwrap();

        let features = vec![Array3::ones((2, 3, 6))];
        let angles = vec![15.0, 30.0];

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
