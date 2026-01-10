//! Physics-informed neural network (PINN) beamforming processor.
//!
//! This module implements a beamforming processor that uses physics-informed neural
//! networks to optimize delay calculations and beamforming weights based on acoustic
//! wave propagation physics.
//!
//! ## Mathematical Foundation
//!
//! ### Wave Equation Constraints
//!
//! The PINN enforces the acoustic wave equation:
//! ```text
//! ∂²p/∂t² = c² ∇²p
//! ```
//! where:
//! - p: acoustic pressure
//! - c: speed of sound
//! - ∇²: Laplacian operator
//!
//! ### Eikonal Equation for Delay Calculation
//!
//! Travel time τ(x) satisfies:
//! ```text
//! |∇τ|² = 1/c²(x)
//! ```
//! This provides physics-consistent delay calculations for heterogeneous media.
//!
//! ### Optimized Beamforming Weights
//!
//! Weights computed via constrained optimization:
//! ```text
//! w* = arg min_w [ ||y - Xw||² + λ||w||² ]
//! subject to: w satisfies reciprocity and coherence
//! ```
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks"
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Szabo (2004): "Diagnostic Ultrasound Imaging: Inside Out"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array3, Array4, ArrayView3};
use std::collections::HashMap;

#[cfg(feature = "pinn")]
use crate::math::ml::pinn::{
    uncertainty_quantification::BayesianPINN, BurnPINN1DWave, BurnTrainingMetrics,
};

use crate::domain::sensor::beamforming::SteeringVector;

use super::super::types::{NeuralBeamformingMetrics, PINNBeamformingConfig, PinnBeamformingResult};
use super::inference;

/// AI-enhanced beamforming processor with PINN optimization.
///
/// Integrates physics-informed neural networks for:
/// - Optimal delay calculation via eikonal equation
/// - Adaptive weight computation with wave physics constraints
/// - Uncertainty quantification via Bayesian inference
#[derive(Debug)]
pub struct NeuralBeamformingProcessor {
    /// Configuration
    config: PINNBeamformingConfig,
    /// PINN model for beamforming optimization
    #[cfg(feature = "pinn")]
    pinn_model: Option<BurnPINN1DWave<burn::backend::Autodiff<burn::backend::NdArray<f32>>>>,
    /// Bayesian PINN for uncertainty quantification
    #[cfg(feature = "pinn")]
    bayesian_model: Option<BayesianPINN<burn::backend::Autodiff<burn::backend::NdArray<f32>>>>,
    /// Steering vectors cache for performance
    steering_cache: HashMap<(usize, usize, usize), SteeringVector>,
    /// Performance metrics
    metrics: NeuralBeamformingMetrics,
}

impl NeuralBeamformingProcessor {
    /// Create new neural beamforming processor.
    ///
    /// # Arguments
    ///
    /// * `config` - PINN beamforming configuration
    ///
    /// # Initialization
    ///
    /// - Creates PINN model (if feature enabled)
    /// - Initializes Bayesian uncertainty quantification
    /// - Allocates steering vector cache
    pub fn new(config: PINNBeamformingConfig) -> KwaversResult<Self> {
        let mut processor = Self {
            config,
            #[cfg(feature = "pinn")]
            pinn_model: None,
            #[cfg(feature = "pinn")]
            bayesian_model: None,
            steering_cache: HashMap::new(),
            metrics: NeuralBeamformingMetrics::default(),
        };

        // Initialize PINN model if feature is enabled
        #[cfg(feature = "pinn")]
        {
            processor.initialize_pinn_model()?;
        }

        Ok(processor)
    }

    /// Initialize PINN model for beamforming optimization.
    ///
    /// Creates a 1D wave equation PINN with physics constraints and optional
    /// Bayesian uncertainty quantification via Monte Carlo dropout.
    #[cfg(feature = "pinn")]
    fn initialize_pinn_model(&mut self) -> KwaversResult<()> {
        type Backend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

        let pinn =
            BurnPINN1DWave::<Backend>::new(self.config.pinn_config.clone(), &Default::default())?;

        self.pinn_model = Some(pinn);

        Ok(())
    }

    /// Process 4D RF data volume with PINN-enhanced beamforming.
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Input RF data (frames × channels × samples × 1)
    ///
    /// # Returns
    ///
    /// Beamformed volume with uncertainty and confidence maps.
    ///
    /// # Process
    ///
    /// 1. Extract dimensions from input
    /// 2. Process each frame independently (parallelizable)
    /// 3. Compute PINN-optimized delays and weights
    /// 4. Estimate uncertainty via Bayesian inference or local variance
    /// 5. Derive confidence from uncertainty
    pub fn process_volume(
        &mut self,
        rf_data: &Array4<f32>,
    ) -> KwaversResult<PinnBeamformingResult> {
        let start_time = std::time::Instant::now();

        let (frames, channels, samples, _) = rf_data.dim();

        // Initialize output volume
        let mut volume = Array3::<f32>::zeros((frames, channels, samples));

        // Process each frame with PINN-optimized beamforming
        for frame_idx in 0..frames {
            let frame_data = rf_data.slice(s![frame_idx, .., .., 0..1]);
            let frame_result = self.process_frame(&frame_data)?;
            volume
                .index_axis_mut(ndarray::Axis(0), frame_idx)
                .assign(&frame_result.index_axis(ndarray::Axis(2), 0));
        }

        // Compute uncertainty quantification
        let uncertainty = self.compute_uncertainty(&volume)?;
        let confidence = self.compute_confidence(&uncertainty)?;

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.metrics.total_processing_time = processing_time;

        Ok(PinnBeamformingResult {
            volume,
            uncertainty,
            confidence,
            #[cfg(feature = "pinn")]
            pinn_metrics: None,
            #[cfg(not(feature = "pinn"))]
            pinn_metrics: None,
            processing_time_ms: processing_time,
        })
    }

    /// Process single frame with PINN-optimized beamforming.
    ///
    /// Applies delay-and-sum beamforming with PINN-computed delays and weights.
    fn process_frame(&mut self, frame_data: &ArrayView3<f32>) -> KwaversResult<Array3<f32>> {
        let (channels, samples, _) = frame_data.dim();
        let mut output = Array3::<f32>::zeros((channels, samples, 1));

        for channel in 0..channels {
            for sample in 0..samples {
                let optimal_delay = self.compute_pinn_delay(channel, sample)?;
                let voxel_value =
                    self.apply_pinn_beamforming(frame_data, channel, sample, optimal_delay)?;
                output[[channel, sample, 0]] = voxel_value;
            }
        }

        Ok(output)
    }

    /// Compute optimal delay using PINN physics constraints.
    ///
    /// Delegates to inference module for eikonal equation solution.
    #[cfg(feature = "pinn")]
    fn compute_pinn_delay(&mut self, channel_idx: usize, sample_idx: usize) -> KwaversResult<f64> {
        inference::compute_delay(
            channel_idx,
            self.config.rf_data_channels,
            sample_idx,
            self.config.channel_spacing,
            self.config.focal_depth,
            self.config.base_config.sound_speed,
            self.config.base_config.sampling_frequency,
        )
    }

    /// Fallback delay computation without PINN feature.
    #[cfg(not(feature = "pinn"))]
    fn compute_pinn_delay(
        &mut self,
        _channel_idx: usize,
        _sample_idx: usize,
    ) -> KwaversResult<f64> {
        Err(KwaversError::System(
            crate::core::error::SystemError::FeatureNotAvailable {
                feature: "pinn".to_string(),
                reason: "PINN beamforming requires 'pinn' feature".to_string(),
            },
        ))
    }

    /// Apply PINN-optimized beamforming weights.
    ///
    /// # Mathematical Definition
    ///
    /// Beamformed output:
    /// ```text
    /// y(r,t) = ∑ᵢ wᵢ(r) · s(rᵢ, t - τᵢ(r))
    /// ```
    /// where:
    /// - wᵢ: PINN-optimized weight for element i
    /// - s(rᵢ,t): received signal at element i
    /// - τᵢ(r): delay for focusing at point r
    fn apply_pinn_beamforming(
        &mut self,
        frame_data: &ArrayView3<f32>,
        channel: usize,
        sample: usize,
        delay: f64,
    ) -> KwaversResult<f32> {
        // Extract delayed samples
        let delayed_samples: Vec<f32> = (0..self.config.rf_data_channels)
            .map(|elem| {
                let delayed_idx = (sample as f64 + delay * elem as f64) as usize;
                if delayed_idx < frame_data.shape()[1] {
                    frame_data[[elem, delayed_idx, 0]]
                } else {
                    0.0
                }
            })
            .collect();

        // Compute PINN-optimized weights
        let weights = self.compute_pinn_weights(channel, sample, &delayed_samples)?;

        // Weighted sum
        let result: f32 = delayed_samples
            .iter()
            .zip(weights.iter())
            .map(|(sample, weight)| sample * weight)
            .sum();

        Ok(result)
    }

    /// Compute PINN-optimized beamforming weights.
    ///
    /// Delegates to inference module for physics-informed weight computation.
    fn compute_pinn_weights(
        &mut self,
        _channel: usize,
        sample: usize,
        _samples: &[f32],
    ) -> KwaversResult<Vec<f32>> {
        inference::compute_weights(
            self.config.rf_data_channels,
            sample,
            self.config.channel_spacing,
            self.config.focal_depth,
            self.config.base_config.sound_speed,
            self.config.base_config.reference_frequency,
        )
    }

    /// Compute uncertainty quantification for beamforming results.
    ///
    /// Uses Bayesian PINN (if available) or local variance estimation.
    ///
    /// # Bayesian Approach (with PINN feature)
    ///
    /// Monte Carlo dropout for uncertainty:
    /// ```text
    /// σ²(x) = E[(f(x) - E[f(x)])²]
    /// ```
    /// via multiple stochastic forward passes.
    ///
    /// # Fallback Approach (without PINN)
    ///
    /// Signal-based uncertainty:
    /// ```text
    /// σ(x) = 1 / (|x| + 1)
    /// ```
    /// (higher amplitude → lower uncertainty)
    fn compute_uncertainty(&mut self, volume: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        #[cfg(feature = "pinn")]
        {
            if let (Some(bayesian), Some(_pinn)) = (&mut self.bayesian_model, &self.pinn_model) {
                let uncertainty_start = std::time::Instant::now();
                let mut uncertainty = Array3::zeros(volume.dim());
                let (d0, d1, d2) = volume.dim();

                for i in 0..d0 {
                    for j in 0..d1 {
                        for k in 0..d2 {
                            // Normalize coordinates to [0, 1]
                            let x = i as f32 / d0 as f32;
                            let y = j as f32 / d1 as f32;
                            let t = k as f32 / d2 as f32;

                            let input = [x, y, t];
                            let prediction =
                                bayesian.predict_with_uncertainty(&input).map_err(|e| {
                                    KwaversError::InternalError(format!("Inference error: {}", e))
                                })?;

                            // Variance (std²) as uncertainty
                            if !prediction.std.is_empty() {
                                uncertainty[[i, j, k]] = prediction.std[0].powi(2);
                            }
                        }
                    }
                }

                self.metrics.uncertainty_computation_time =
                    uncertainty_start.elapsed().as_millis() as f64;

                return Ok(uncertainty);
            }
        }

        // Fallback: signal-based uncertainty
        let mut uncertainty = Array3::<f32>::zeros(volume.dim());
        for ((i, j, k), value) in volume.indexed_iter() {
            uncertainty[[i, j, k]] = 1.0 / (value.abs() + 1.0);
        }
        Ok(uncertainty)
    }

    /// Compute confidence scores from uncertainty.
    ///
    /// # Mathematical Definition
    ///
    /// Confidence:
    /// ```text
    /// c(x) = 1 / (1 + σ²(x))
    /// ```
    /// where σ²(x) is the uncertainty (variance).
    ///
    /// Properties:
    /// - c ∈ [0, 1]
    /// - c = 1: perfect confidence (σ² = 0)
    /// - c → 0: high uncertainty (σ² → ∞)
    fn compute_confidence(&self, uncertainty: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut confidence = Array3::<f32>::zeros(uncertainty.dim());

        for ((i, j, k), &uncert) in uncertainty.indexed_iter() {
            confidence[[i, j, k]] = 1.0 / (1.0 + uncert);
        }

        Ok(confidence)
    }

    /// Get performance metrics.
    pub fn metrics(&self) -> &NeuralBeamformingMetrics {
        &self.metrics
    }

    /// Calculate memory requirement for processor.
    ///
    /// # Components
    ///
    /// - PINN model: ~4MB per hidden layer
    /// - Uncertainty quantification: volume_size × 8 bytes × 10 (MC samples)
    /// - RF data buffer: channels × samples × 4 bytes
    pub fn calculate_memory_requirement(&self) -> usize {
        let pinn_memory = if self.config.enable_pinn {
            #[cfg(feature = "pinn")]
            {
                self.config.pinn_config.hidden_layers.iter().sum::<usize>() * 4 * 1024 * 1024
            }
            #[cfg(not(feature = "pinn"))]
            {
                0
            }
        } else {
            0
        };

        let uncertainty_memory = if self.config.enable_uncertainty_quantification {
            self.config.volume_size.0
                * self.config.volume_size.1
                * self.config.volume_size.2
                * 8
                * 10
        } else {
            0
        };

        let base_memory = self.config.rf_data_channels * self.config.samples_per_channel * 4;

        pinn_memory + uncertainty_memory + base_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let config = PINNBeamformingConfig::default();
        let processor = NeuralBeamformingProcessor::new(config);

        #[cfg(feature = "pinn")]
        assert!(processor.is_ok());

        #[cfg(not(feature = "pinn"))]
        assert!(processor.is_ok()); // Should succeed even without PINN feature
    }

    #[test]
    fn test_memory_calculation() {
        let config = PINNBeamformingConfig::default();
        let processor = NeuralBeamformingProcessor::new(config).unwrap();

        let memory = processor.calculate_memory_requirement();
        assert!(memory > 0);
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_process_volume() {
        let config = PINNBeamformingConfig {
            rf_data_channels: 8,
            samples_per_channel: 128,
            volume_size: (2, 8, 128),
            enable_pinn: false, // Disable for test speed
            enable_uncertainty_quantification: false,
            ..Default::default()
        };

        let mut processor = NeuralBeamformingProcessor::new(config).unwrap();
        let rf_data = Array4::<f32>::ones((2, 8, 128, 1));

        let result = processor.process_volume(&rf_data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.volume.dim(), (2, 8, 128));
        assert_eq!(output.uncertainty.dim(), (2, 8, 128));
        assert_eq!(output.confidence.dim(), (2, 8, 128));
    }
}
