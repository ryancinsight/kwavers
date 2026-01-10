//! Configuration types for neural beamforming.
//!
//! This module defines configuration structures and enums for controlling
//! neural beamforming behavior, including processing modes, network architecture,
//! physics constraints, and adaptation parameters.
//!
//! ## Configuration Hierarchy
//!
//! ```text
//! NeuralBeamformingConfig
//! ├── mode: NeuralBeamformingMode
//! ├── network_architecture: Vec<usize>
//! ├── physics_parameters: PhysicsParameters
//! ├── adaptation_parameters: AdaptationParameters
//! └── sensor_geometry: SensorGeometry
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::neural::{
//!     NeuralBeamformingConfig, NeuralBeamformingMode,
//! };
//!
//! // Create default configuration
//! let mut config = NeuralBeamformingConfig::default();
//!
//! // Customize for specific application
//! config.mode = NeuralBeamformingMode::Hybrid;
//! config.network_architecture = vec![6, 64, 32, 1];
//! config.physics_parameters.reciprocity_weight = 1.0;
//!
//! // Use for beamformer creation
//! let beamformer = NeuralBeamformer::new(config)?;
//! ```

/// Neural beamforming processing modes.
///
/// Different modes trade off between computational cost, image quality,
/// and physical consistency.
///
/// ## Mode Selection Guidelines
///
/// - **NeuralOnly**: Fastest, purely data-driven, may violate physics
/// - **Hybrid**: Good balance, combines traditional + neural refinement
/// - **PhysicsInformed**: Best quality, enforces wave equation constraints
/// - **Adaptive**: Robust, switches based on signal quality metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NeuralBeamformingMode {
    /// Pure neural network beamforming without traditional preprocessing.
    ///
    /// **Use case**: Fast inference when training data is abundant
    /// **Speed**: ★★★★★
    /// **Quality**: ★★★☆☆
    /// **Physics consistency**: ★★☆☆☆
    NeuralOnly,

    /// Hybrid approach: traditional delay-and-sum followed by neural refinement.
    ///
    /// **Use case**: General-purpose imaging with quality enhancement
    /// **Speed**: ★★★★☆
    /// **Quality**: ★★★★☆
    /// **Physics consistency**: ★★★★☆
    #[default]
    Hybrid,

    /// Physics-informed neural networks (PINN) enforcing wave equation.
    ///
    /// **Use case**: High-quality imaging requiring physical plausibility
    /// **Speed**: ★★☆☆☆
    /// **Quality**: ★★★★★
    /// **Physics consistency**: ★★★★★
    ///
    /// **Requires**: `pinn` feature flag
    #[cfg(feature = "pinn")]
    PhysicsInformed,

    /// Adaptive mode switching based on real-time signal quality assessment.
    ///
    /// **Use case**: Robust imaging under varying conditions
    /// **Speed**: ★★★★☆ (varies)
    /// **Quality**: ★★★★☆
    /// **Physics consistency**: ★★★★☆
    Adaptive,
}

/// Physics constraint parameters for neural beamforming.
///
/// Controls the strength of various physics-based regularization terms
/// applied during neural network processing.
///
/// ## Constraint Types
///
/// - **Reciprocity**: Time-reversal symmetry H(A→B) = H(B→A)
/// - **Coherence**: Spatial smoothness ∇²I
/// - **Sparsity**: Focused point-spread function via L1 penalty
///
/// ## Tuning Guidelines
///
/// - **High reciprocity** (>1.0): Strong physical consistency, may over-smooth
/// - **High coherence** (>0.5): Smooth images, reduced noise, may blur details
/// - **High sparsity** (>0.1): Sharp features, may introduce artifacts
#[derive(Debug, Clone)]
pub struct PhysicsParameters {
    /// Weight for reciprocity constraint (time-reversal symmetry).
    ///
    /// Range: [0.0, 10.0]
    /// Default: 1.0
    pub reciprocity_weight: f64,

    /// Weight for coherence constraint (spatial smoothness).
    ///
    /// Range: [0.0, 5.0]
    /// Default: 0.5
    pub coherence_weight: f64,

    /// Weight for sparsity constraint (L1 regularization).
    ///
    /// Range: [0.0, 1.0]
    /// Default: 0.1
    pub sparsity_weight: f64,

    /// Coherence diffusion coefficient for smoothing.
    ///
    /// Controls the rate of spatial averaging.
    /// Range: [0.0, 1.0]
    /// Default: 0.01
    pub diffusion_coefficient: f64,

    /// Sparsity soft threshold for L1 penalty.
    ///
    /// Values below threshold are set to zero.
    /// Range: [0.0, 1.0]
    /// Default: 0.05
    pub soft_threshold: f64,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            reciprocity_weight: 1.0,
            coherence_weight: 0.5,
            sparsity_weight: 0.1,
            diffusion_coefficient: 0.01,
            soft_threshold: 0.05,
        }
    }
}

/// Adaptation parameters for learning and feedback.
///
/// Controls how the neural network adapts based on performance feedback
/// and signal quality metrics.
#[derive(Debug, Clone)]
pub struct AdaptationParameters {
    /// Learning rate for online adaptation.
    ///
    /// Range: [1e-6, 1e-2]
    /// Default: 1e-4
    pub learning_rate: f64,

    /// Uncertainty threshold for mode switching (adaptive mode).
    ///
    /// When average uncertainty exceeds this value, switch to more robust mode.
    /// Range: [0.0, 1.0]
    /// Default: 0.3
    pub uncertainty_threshold: f64,

    /// Signal quality threshold for mode selection.
    ///
    /// Based on coherence factor (CF). High CF → neural-only, Low CF → hybrid.
    /// Range: [0.0, 1.0]
    /// Default: 0.7
    pub quality_threshold: f64,

    /// Enable online learning during processing.
    ///
    /// If true, network weights are updated based on feedback.
    /// Default: false
    pub enable_online_learning: bool,

    /// Number of adaptation iterations per feedback cycle.
    ///
    /// Range: [1, 100]
    /// Default: 10
    pub adaptation_iterations: usize,

    /// Momentum coefficient for gradient descent.
    ///
    /// Range: [0.0, 0.99]
    /// Default: 0.9
    pub momentum: f64,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            uncertainty_threshold: 0.3,
            quality_threshold: 0.7,
            enable_online_learning: false,
            adaptation_iterations: 10,
            momentum: 0.9,
        }
    }
}

/// Sensor array geometry specification.
///
/// Defines the spatial arrangement of sensor elements for beamforming
/// delay and apodization calculations.
#[derive(Debug, Clone)]
pub struct SensorGeometry {
    /// 3D positions of sensor elements [x, y, z] in meters.
    ///
    /// Coordinate system:
    /// - X: Lateral (array axis)
    /// - Y: Elevation (perpendicular to imaging plane)
    /// - Z: Axial (depth/propagation direction)
    pub positions: Vec<[f64; 3]>,

    /// Sampling frequency in Hz.
    ///
    /// Typical range: [1 MHz, 100 MHz]
    pub sampling_frequency: f64,

    /// Speed of sound in medium (m/s).
    ///
    /// Default: 1540 m/s (soft tissue)
    pub sound_speed: f64,
}

impl SensorGeometry {
    /// Create linear array geometry.
    ///
    /// # Arguments
    ///
    /// - `num_elements`: Number of array elements
    /// - `pitch`: Element spacing in meters
    /// - `sampling_frequency`: Sampling rate in Hz
    /// - `sound_speed`: Speed of sound in m/s
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let geometry = SensorGeometry::linear_array(
    ///     64,           // 64 elements
    ///     0.0003,       // 300 μm pitch
    ///     40e6,         // 40 MHz sampling
    ///     1540.0,       // tissue sound speed
    /// );
    /// ```
    pub fn linear_array(
        num_elements: usize,
        pitch: f64,
        sampling_frequency: f64,
        sound_speed: f64,
    ) -> Self {
        let positions: Vec<[f64; 3]> = (0..num_elements)
            .map(|i| {
                let x = (i as f64 - (num_elements - 1) as f64 / 2.0) * pitch;
                [x, 0.0, 0.0]
            })
            .collect();

        Self {
            positions,
            sampling_frequency,
            sound_speed,
        }
    }

    /// Create phased array geometry (2D).
    ///
    /// # Arguments
    ///
    /// - `nx`: Number of elements in X (lateral)
    /// - `ny`: Number of elements in Y (elevation)
    /// - `pitch_x`: Element spacing in X (m)
    /// - `pitch_y`: Element spacing in Y (m)
    /// - `sampling_frequency`: Sampling rate in Hz
    /// - `sound_speed`: Speed of sound in m/s
    pub fn phased_array(
        nx: usize,
        ny: usize,
        pitch_x: f64,
        pitch_y: f64,
        sampling_frequency: f64,
        sound_speed: f64,
    ) -> Self {
        let mut positions = Vec::with_capacity(nx * ny);

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 - (nx - 1) as f64 / 2.0) * pitch_x;
                let y = (j as f64 - (ny - 1) as f64 / 2.0) * pitch_y;
                positions.push([x, y, 0.0]);
            }
        }

        Self {
            positions,
            sampling_frequency,
            sound_speed,
        }
    }

    /// Get number of sensor elements.
    pub fn num_elements(&self) -> usize {
        self.positions.len()
    }
}

impl Default for SensorGeometry {
    fn default() -> Self {
        Self::linear_array(
            64,     // 64 elements
            0.0003, // 300 μm pitch
            40e6,   // 40 MHz
            1540.0, // soft tissue
        )
    }
}

/// Main configuration for neural beamforming.
///
/// Aggregates all configuration parameters for neural beamforming processing.
///
/// ## Configuration Validation
///
/// The `validate()` method checks:
/// - Network architecture has at least 2 layers
/// - Physics weights are non-negative
/// - Learning rate is positive
/// - Sensor geometry is valid (>= 2 elements)
///
/// ## Example
///
/// ```rust,ignore
/// let config = NeuralBeamformingConfig {
///     mode: NeuralBeamformingMode::Hybrid,
///     network_architecture: vec![6, 32, 16, 1],
///     physics_parameters: PhysicsParameters::default(),
///     adaptation_parameters: AdaptationParameters::default(),
///     sensor_geometry: SensorGeometry::linear_array(128, 0.0003, 40e6, 1540.0),
///     batch_size: 32,
/// };
///
/// config.validate()?;
/// ```
#[derive(Debug, Clone)]
pub struct NeuralBeamformingConfig {
    /// Processing mode (neural-only, hybrid, PINN, adaptive).
    pub mode: NeuralBeamformingMode,

    /// Neural network layer sizes [input, hidden..., output].
    ///
    /// Example: `[6, 32, 16, 1]` creates:
    /// - Input layer: 6 features
    /// - Hidden layer 1: 32 neurons
    /// - Hidden layer 2: 16 neurons
    /// - Output layer: 1 value (beamformed intensity)
    pub network_architecture: Vec<usize>,

    /// Physics constraint parameters.
    pub physics_parameters: PhysicsParameters,

    /// Adaptation and learning parameters.
    pub adaptation_parameters: AdaptationParameters,

    /// Sensor array geometry.
    pub sensor_geometry: SensorGeometry,

    /// Batch size for parallel processing.
    ///
    /// Larger batches improve GPU utilization but increase memory usage.
    /// Range: [1, 1024]
    /// Default: 32
    pub batch_size: usize,
}

impl NeuralBeamformingConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Network architecture has < 2 layers
    /// - Any physics weight is negative
    /// - Learning rate is non-positive
    /// - Batch size is zero
    /// - Sensor geometry has < 2 elements
    pub fn validate(&self) -> crate::domain::core::error::KwaversResult<()> {
        use crate::core::error::KwaversError;

        // Network architecture validation
        if self.network_architecture.len() < 2 {
            return Err(KwaversError::InvalidInput(
                "Network architecture must have at least 2 layers (input and output)".to_string(),
            ));
        }

        // Physics parameters validation
        if self.physics_parameters.reciprocity_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Reciprocity weight must be non-negative".to_string(),
            ));
        }

        if self.physics_parameters.coherence_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Coherence weight must be non-negative".to_string(),
            ));
        }

        if self.physics_parameters.sparsity_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sparsity weight must be non-negative".to_string(),
            ));
        }

        // Adaptation parameters validation
        if self.adaptation_parameters.learning_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Learning rate must be positive".to_string(),
            ));
        }

        if self.adaptation_parameters.uncertainty_threshold < 0.0
            || self.adaptation_parameters.uncertainty_threshold > 1.0
        {
            return Err(KwaversError::InvalidInput(
                "Uncertainty threshold must be in range [0, 1]".to_string(),
            ));
        }

        // Batch size validation
        if self.batch_size == 0 {
            return Err(KwaversError::InvalidInput(
                "Batch size must be positive".to_string(),
            ));
        }

        // Sensor geometry validation
        if self.sensor_geometry.num_elements() < 2 {
            return Err(KwaversError::InvalidInput(
                "Sensor array must have at least 2 elements".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for NeuralBeamformingConfig {
    fn default() -> Self {
        Self {
            mode: NeuralBeamformingMode::default(),
            // Architecture: [6 features + 1 angle, 32 hidden, 16 hidden, 1 output]
            network_architecture: vec![7, 32, 16, 1],
            physics_parameters: PhysicsParameters::default(),
            adaptation_parameters: AdaptationParameters::default(),
            sensor_geometry: SensorGeometry::default(),
            batch_size: 32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_default() {
        assert_eq!(
            NeuralBeamformingMode::default(),
            NeuralBeamformingMode::Hybrid
        );
    }

    #[test]
    fn test_physics_parameters_default() {
        let params = PhysicsParameters::default();
        assert_eq!(params.reciprocity_weight, 1.0);
        assert_eq!(params.coherence_weight, 0.5);
        assert_eq!(params.sparsity_weight, 0.1);
    }

    #[test]
    fn test_sensor_geometry_linear() {
        let geometry = SensorGeometry::linear_array(64, 0.0003, 40e6, 1540.0);
        assert_eq!(geometry.num_elements(), 64);
        assert_eq!(geometry.sampling_frequency, 40e6);

        // Check symmetry around origin
        assert!((geometry.positions[31][0] + geometry.positions[32][0]).abs() < 1e-10);
    }

    #[test]
    fn test_sensor_geometry_phased() {
        let geometry = SensorGeometry::phased_array(8, 8, 0.0003, 0.0003, 40e6, 1540.0);
        assert_eq!(geometry.num_elements(), 64);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = NeuralBeamformingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_invalid_architecture() {
        let config = NeuralBeamformingConfig {
            network_architecture: vec![5],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_physics_weight() {
        let mut config = NeuralBeamformingConfig::default();
        config.physics_parameters.reciprocity_weight = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_learning_rate() {
        let mut config = NeuralBeamformingConfig::default();
        config.adaptation_parameters.learning_rate = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_batch_size() {
        let config = NeuralBeamformingConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_sensor_count() {
        let mut config = NeuralBeamformingConfig::default();
        config.sensor_geometry.positions = vec![[0.0, 0.0, 0.0]]; // Only 1 element
        assert!(config.validate().is_err());
    }
}
