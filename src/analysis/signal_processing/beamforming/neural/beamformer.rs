//! High-level neural beamformer implementation.
//!
//! This module provides the main `NeuralBeamformer` struct that orchestrates
//! all neural beamforming operations, including traditional delay-and-sum,
//! neural network refinement, physics constraints, and uncertainty quantification.
//!
//! ## Architecture
//!
//! ```text
//! NeuralBeamformer
//! ├── Configuration (mode, network, physics, adaptation)
//! ├── Neural Network (optional, mode-dependent)
//! ├── Physics Constraints (reciprocity, coherence, sparsity)
//! ├── Uncertainty Estimator (dropout MC, local variance)
//! └── Performance Metrics (processing time, confidence, quality)
//! ```
//!
//! ## Processing Pipeline
//!
//! ```text
//! RF Data → Traditional DAS → Feature Extraction → Neural Network
//!                                                         ↓
//!           ← Uncertainty Estimation ← Physics Constraints ←
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::neural::{
//!     NeuralBeamformer, NeuralBeamformingConfig, NeuralBeamformingMode,
//! };
//! use ndarray::Array4;
//!
//! // Configure beamformer
//! let mut config = NeuralBeamformingConfig::default();
//! config.mode = NeuralBeamformingMode::Hybrid;
//!
//! // Create beamformer
//! let mut beamformer = NeuralBeamformer::new(config)?;
//!
//! // Process RF data
//! let rf_data = Array4::<f32>::zeros((1, 64, 1024, 1)); // (frames, channels, samples, 1)
//! let steering_angles = vec![0.0]; // Plane wave
//! let result = beamformer.process(&rf_data, &steering_angles)?;
//!
//! println!("Confidence: {:.2}", result.confidence);
//! println!("Processing mode: {}", result.processing_mode);
//! ```

use ndarray::{Array3, Array4};

use crate::core::error::{KwaversError, KwaversResult};

use super::config::{NeuralBeamformingConfig, NeuralBeamformingMode};
use super::features;
use super::network::NeuralBeamformingNetwork;
use super::physics::PhysicsConstraints;
use super::types::{BeamformingFeedback, HybridBeamformingMetrics, HybridBeamformingResult};
use super::uncertainty::UncertaintyEstimator;

/// Main neural beamformer processor.
///
/// Orchestrates all neural beamforming operations including traditional
/// delay-and-sum, neural network processing, physics constraints, and
/// uncertainty quantification.
#[derive(Debug)]
pub struct NeuralBeamformer {
    /// Configuration parameters
    config: NeuralBeamformingConfig,

    /// Neural network (optional, mode-dependent)
    neural_network: Option<NeuralBeamformingNetwork>,

    /// Physics constraints enforcer
    physics_constraints: PhysicsConstraints,

    /// Uncertainty estimator
    uncertainty_estimator: UncertaintyEstimator,

    /// Performance metrics
    metrics: HybridBeamformingMetrics,
}

impl NeuralBeamformer {
    /// Create a new neural beamformer with given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Beamforming configuration (validated on construction)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Configuration validation fails
    /// - Network initialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = NeuralBeamformingConfig::default();
    /// let beamformer = NeuralBeamformer::new(config)?;
    /// ```
    pub fn new(config: NeuralBeamformingConfig) -> KwaversResult<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize neural network for modes that require it
        let neural_network = match config.mode {
            NeuralBeamformingMode::NeuralOnly
            | NeuralBeamformingMode::Hybrid
            | NeuralBeamformingMode::Adaptive => {
                Some(NeuralBeamformingNetwork::new(&config.network_architecture)?)
            }
            #[cfg(feature = "pinn")]
            NeuralBeamformingMode::PhysicsInformed => {
                Some(NeuralBeamformingNetwork::new(&config.network_architecture)?)
            }
        };

        // Initialize physics constraints
        let physics_constraints = PhysicsConstraints::new(
            config.physics_parameters.reciprocity_weight,
            config.physics_parameters.coherence_weight,
            config.physics_parameters.sparsity_weight,
        );

        // Initialize uncertainty estimator
        let uncertainty_estimator = UncertaintyEstimator::new();

        Ok(Self {
            config,
            neural_network,
            physics_constraints,
            uncertainty_estimator,
            metrics: HybridBeamformingMetrics::default(),
        })
    }

    /// Process RF data through neural beamforming pipeline.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Raw RF data (frames, channels, samples, 1)
    /// * `steering_angles` - Beam steering angles in radians
    ///
    /// # Returns
    ///
    /// Beamformed result with image, uncertainty, confidence, and mode info.
    ///
    /// # Processing Steps
    ///
    /// 1. Traditional beamforming (DAS)
    /// 2. Feature extraction
    /// 3. Neural network processing (mode-dependent)
    /// 4. Physics constraint application
    /// 5. Uncertainty estimation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let rf_data = Array4::<f32>::zeros((1, 64, 1024, 1));
    /// let angles = vec![0.0]; // Plane wave
    /// let result = beamformer.process(&rf_data, &angles)?;
    /// ```
    pub fn process(
        &mut self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let start_time = std::time::Instant::now();

        // Dispatch to appropriate processing mode
        let result = match self.config.mode {
            NeuralBeamformingMode::NeuralOnly => {
                self.process_neural_only(rf_data, steering_angles)?
            }
            NeuralBeamformingMode::Hybrid => self.process_hybrid(rf_data, steering_angles)?,
            #[cfg(feature = "pinn")]
            NeuralBeamformingMode::PhysicsInformed => {
                self.process_physics_informed(rf_data, steering_angles)?
            }
            NeuralBeamformingMode::Adaptive => self.process_adaptive(rf_data, steering_angles)?,
        };

        // Update metrics
        let processing_time = start_time.elapsed().as_secs_f64();
        self.metrics.update(processing_time, result.confidence);

        Ok(result)
    }

    /// Pure neural network beamforming.
    ///
    /// Applies neural network directly to extracted features without
    /// traditional preprocessing.
    fn process_neural_only(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let network = self
            .neural_network
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Neural network not initialized".into()))?;

        // Generate base image using traditional DAS for feature extraction
        let base_image = self.traditional_beamforming(rf_data, steering_angles)?;

        // Extract features
        let features = features::extract_all_features(&base_image);

        // Apply neural network
        let beamformed = network.forward(&features, steering_angles)?;

        // Estimate uncertainty
        let uncertainty = self.uncertainty_estimator.estimate(&beamformed)?;
        let mean_uncertainty = uncertainty.mean().unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: beamformed,
            uncertainty: Some(uncertainty),
            confidence: 1.0 - mean_uncertainty,
            processing_mode: "NeuralOnly".to_string(),
        })
    }

    /// Hybrid beamforming: traditional DAS + neural refinement.
    ///
    /// Combines traditional delay-and-sum with neural network enhancement
    /// and physics constraints for optimal quality/robustness trade-off.
    fn process_hybrid(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let network = self
            .neural_network
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Neural network not initialized".into()))?;

        // Traditional beamforming
        let base_image = self.traditional_beamforming(rf_data, steering_angles)?;

        // Extract features
        let features = features::extract_all_features(&base_image);

        // Apply neural refinement
        let refined = network.forward(&features, steering_angles)?;

        // Apply physics constraints
        let constrained = self.physics_constraints.apply(&refined)?;

        // Estimate uncertainty
        let uncertainty = self.uncertainty_estimator.estimate(&constrained)?;
        let mean_uncertainty = uncertainty.mean().unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: constrained,
            uncertainty: Some(uncertainty),
            confidence: 0.9 - mean_uncertainty * 0.1,
            processing_mode: "Hybrid".to_string(),
        })
    }

    /// Physics-informed neural network beamforming.
    ///
    /// Enforces acoustic wave equation constraints during neural network
    /// processing for maximum physical consistency.
    #[cfg(feature = "pinn")]
    fn process_physics_informed(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        let network = self
            .neural_network
            .as_ref()
            .ok_or_else(|| KwaversError::InvalidInput("Neural network not initialized".into()))?;

        // Traditional beamforming
        let base_image = self.traditional_beamforming(rf_data, steering_angles)?;

        // Extract features
        let features = features::extract_all_features(&base_image);

        // Apply PINN with physics constraints
        let beamformed = network.forward_physics_informed(
            &features,
            steering_angles,
            &self.physics_constraints,
        )?;

        // Estimate uncertainty
        let uncertainty = self.uncertainty_estimator.estimate(&beamformed)?;
        let mean_uncertainty = uncertainty.mean().unwrap_or(0.0) as f64;

        Ok(HybridBeamformingResult {
            image: beamformed,
            uncertainty: Some(uncertainty),
            confidence: 0.95 - mean_uncertainty * 0.05,
            processing_mode: "PhysicsInformed".to_string(),
        })
    }

    /// Adaptive beamforming with automatic mode selection.
    ///
    /// Assesses signal quality and switches between neural-only and hybrid
    /// modes based on coherence metrics.
    fn process_adaptive(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<HybridBeamformingResult> {
        // Assess signal quality
        let signal_quality = self.assess_signal_quality(rf_data)?;
        let quality_threshold = self.config.adaptation_parameters.quality_threshold;

        // Select mode based on quality
        if signal_quality > quality_threshold {
            // High quality: use fast neural-only
            let mut result = self.process_neural_only(rf_data, steering_angles)?;
            result.processing_mode = "Adaptive(Neural)".to_string();
            Ok(result)
        } else {
            // Low quality: use robust hybrid
            let mut result = self.process_hybrid(rf_data, steering_angles)?;
            result.processing_mode = "Adaptive(Hybrid)".to_string();
            Ok(result)
        }
    }

    /// Traditional delay-and-sum beamforming.
    ///
    /// Computes baseline image using geometric focusing delays.
    /// This serves as input for neural refinement.
    fn traditional_beamforming(
        &self,
        rf_data: &Array4<f32>,
        steering_angles: &[f64],
    ) -> KwaversResult<Array3<f32>> {
        let (frames, channels, samples, _) = rf_data.dim();
        let num_angles = steering_angles.len();

        // Output: (frames, angles, samples)
        let mut image = Array3::<f32>::zeros((frames, num_angles, samples));

        let positions = &self.config.sensor_geometry.positions;
        let c = self.config.sensor_geometry.sound_speed;

        // Simple plane-wave DAS beamforming
        for f in 0..frames {
            for (a_idx, &angle) in steering_angles.iter().enumerate() {
                // Compute steering delays for plane wave
                let delays: Vec<f64> = positions
                    .iter()
                    .map(|pos| pos[0] * angle.sin() / c)
                    .collect();

                // Apply delays and sum (simplified version)
                for s in 0..samples {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for ch in 0..channels.min(positions.len()) {
                        // Simplified delay application (no interpolation)
                        let delay_samples =
                            (delays[ch] * self.config.sensor_geometry.sampling_frequency) as isize;
                        let sample_idx = (s as isize + delay_samples) as usize;

                        if sample_idx < samples {
                            sum += rf_data[[f, ch, sample_idx, 0]];
                            count += 1;
                        }
                    }

                    if count > 0 {
                        image[[f, a_idx, s]] = sum / count as f32;
                    }
                }
            }
        }

        Ok(image)
    }

    /// Assess signal quality using coherence factor.
    ///
    /// Computes the coherence of received signals across channels to
    /// determine whether neural-only or hybrid mode is more appropriate.
    ///
    /// # Returns
    ///
    /// Quality metric in [0, 1] where higher values indicate better signal quality.
    fn assess_signal_quality(&self, rf_data: &Array4<f32>) -> KwaversResult<f64> {
        let (frames, channels, samples, _) = rf_data.dim();

        if channels == 0 || samples == 0 {
            return Ok(0.0);
        }

        let mut total_cf = 0.0;
        let mut count = 0;

        // Compute coherence factor: CF = |Sum(s_i)|^2 / (N * Sum(|s_i|^2))
        let stride = 1.max(samples / 100); // Sample every ~1% of data

        for f in 0..frames {
            for s in (0..samples).step_by(stride) {
                let mut sum_sig = 0.0;
                let mut sum_sq_energy = 0.0;

                for c in 0..channels {
                    let val = rf_data[[f, c, s, 0]];
                    sum_sig += val;
                    sum_sq_energy += val * val;
                }

                if sum_sq_energy > 1e-10 {
                    let coherent_energy = sum_sig * sum_sig;
                    let incoherent_energy = channels as f32 * sum_sq_energy;
                    total_cf += (coherent_energy / incoherent_energy) as f64;
                }
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_cf / count as f64
        } else {
            0.0
        })
    }

    /// Adapt beamformer based on performance feedback.
    ///
    /// Updates neural network weights and physics constraint parameters
    /// based on observed performance metrics.
    ///
    /// # Arguments
    ///
    /// * `feedback` - Performance feedback with improvement and error metrics
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let feedback = BeamformingFeedback {
    ///     improvement: 0.05,
    ///     error_gradient: 0.02,
    ///     signal_quality: 0.85,
    /// };
    /// beamformer.adapt(&feedback)?;
    /// ```
    pub fn adapt(&mut self, feedback: &BeamformingFeedback) -> KwaversResult<()> {
        // Update neural network if available and online learning enabled
        if self.config.adaptation_parameters.enable_online_learning {
            if let Some(network) = &mut self.neural_network {
                network.adapt(feedback, self.config.adaptation_parameters.learning_rate)?;
            }
        }

        // Update physics constraints
        self.physics_constraints.update(feedback)?;

        Ok(())
    }

    /// Get current performance metrics.
    ///
    /// Returns reference to accumulated metrics including processing times,
    /// confidence scores, and frame counts.
    pub fn metrics(&self) -> &HybridBeamformingMetrics {
        &self.metrics
    }

    /// Get configuration reference.
    pub fn config(&self) -> &NeuralBeamformingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rf_data() -> Array4<f32> {
        Array4::from_elem((1, 64, 1024, 1), 0.1)
    }

    #[test]
    fn test_beamformer_creation() {
        let config = NeuralBeamformingConfig::default();
        let beamformer = NeuralBeamformer::new(config);
        assert!(beamformer.is_ok());
    }

    #[test]
    fn test_beamformer_creation_invalid_config() {
        let mut config = NeuralBeamformingConfig::default();
        config.network_architecture = vec![]; // Invalid
        let beamformer = NeuralBeamformer::new(config);
        assert!(beamformer.is_err());
    }

    #[test]
    fn test_process_hybrid() {
        let config = NeuralBeamformingConfig {
            mode: NeuralBeamformingMode::Hybrid,
            ..Default::default()
        };
        let mut beamformer = NeuralBeamformer::new(config).unwrap();

        let rf_data = create_test_rf_data();
        let angles = vec![0.0];
        let result = beamformer.process(&rf_data, &angles);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.processing_mode, "Hybrid");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_process_neural_only() {
        let config = NeuralBeamformingConfig {
            mode: NeuralBeamformingMode::NeuralOnly,
            ..Default::default()
        };
        let mut beamformer = NeuralBeamformer::new(config).unwrap();

        let rf_data = create_test_rf_data();
        let angles = vec![0.0];
        let result = beamformer.process(&rf_data, &angles);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.processing_mode, "NeuralOnly");
    }

    #[test]
    fn test_process_adaptive() {
        let config = NeuralBeamformingConfig {
            mode: NeuralBeamformingMode::Adaptive,
            ..Default::default()
        };
        let mut beamformer = NeuralBeamformer::new(config).unwrap();

        let rf_data = create_test_rf_data();
        let angles = vec![0.0];
        let result = beamformer.process(&rf_data, &angles);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.processing_mode.starts_with("Adaptive"));
    }

    #[test]
    fn test_signal_quality_assessment() {
        let config = NeuralBeamformingConfig::default();
        let beamformer = NeuralBeamformer::new(config).unwrap();

        let rf_data = create_test_rf_data();
        let quality = beamformer.assess_signal_quality(&rf_data);

        assert!(quality.is_ok());
        let quality = quality.unwrap();
        assert!(quality >= 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_adaptation() {
        let config = NeuralBeamformingConfig::default();
        let mut beamformer = NeuralBeamformer::new(config).unwrap();

        let feedback = BeamformingFeedback {
            improvement: 0.05,
            error_gradient: 0.02,
            signal_quality: 0.85,
        };

        let result = beamformer.adapt(&feedback);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics_tracking() {
        let config = NeuralBeamformingConfig::default();
        let mut beamformer = NeuralBeamformer::new(config).unwrap();

        let rf_data = create_test_rf_data();
        let angles = vec![0.0];

        // Process multiple frames
        for _ in 0..3 {
            let _ = beamformer.process(&rf_data, &angles);
        }

        let metrics = beamformer.metrics();
        assert_eq!(metrics.total_frames_processed, 3);
        assert!(metrics.average_processing_time > 0.0);
    }
}
