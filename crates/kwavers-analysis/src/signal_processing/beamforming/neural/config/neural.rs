use kwavers_core::error::{KwaversError, KwaversResult};

use super::{
    AdaptationParameters, NeuralBeamformingMode, NeuralBeamformingPhysicsParams, SensorGeometry,
};

/// Main configuration for neural beamforming.
///
/// ## Configuration Validation
///
/// The `validate()` method checks:
/// - Network architecture has at least 2 layers
/// - Physics weights are non-negative
/// - Learning rate is positive
/// - Sensor geometry is valid (>= 2 elements)
#[derive(Debug, Clone)]
pub struct NeuralBeamformingConfig {
    /// Processing mode (neural-only, hybrid, PINN, adaptive).
    pub mode: NeuralBeamformingMode,

    /// Neural network layer sizes [input, hidden..., output].
    pub network_architecture: Vec<usize>,

    /// Physics constraint parameters.
    pub physics_parameters: NeuralBeamformingPhysicsParams,

    /// Adaptation and learning parameters.
    pub adaptation_parameters: AdaptationParameters,

    /// Sensor array geometry.
    pub sensor_geometry: SensorGeometry,

    /// Batch size for parallel processing. Range: [1, 1024]
    pub batch_size: usize,
}

impl NeuralBeamformingConfig {
    /// Validate configuration parameters.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.network_architecture.len() < 2 {
            return Err(KwaversError::InvalidInput(
                "Network architecture must have at least 2 layers (input and output)".to_owned(),
            ));
        }

        if self.physics_parameters.reciprocity_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Reciprocity weight must be non-negative".to_owned(),
            ));
        }

        if self.physics_parameters.coherence_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Coherence weight must be non-negative".to_owned(),
            ));
        }

        if self.physics_parameters.sparsity_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sparsity weight must be non-negative".to_owned(),
            ));
        }

        if self.adaptation_parameters.learning_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Learning rate must be positive".to_owned(),
            ));
        }

        if self.adaptation_parameters.uncertainty_threshold < 0.0
            || self.adaptation_parameters.uncertainty_threshold > 1.0
        {
            return Err(KwaversError::InvalidInput(
                "Uncertainty threshold must be in range [0, 1]".to_owned(),
            ));
        }

        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "Batch size must be positive".to_owned(),
            ));
        }

        if self.sensor_geometry.num_elements() < 2 {
            return Err(KwaversError::InvalidInput(
                "Sensor array must have at least 2 elements".to_owned(),
            ));
        }

        Ok(())
    }
}

impl Default for NeuralBeamformingConfig {
    fn default() -> Self {
        Self {
            mode: NeuralBeamformingMode::default(),
            network_architecture: vec![7, 32, 16, 1],
            physics_parameters: NeuralBeamformingPhysicsParams::default(),
            adaptation_parameters: AdaptationParameters::default(),
            sensor_geometry: SensorGeometry::default(),
            batch_size: 32,
        }
    }
}
