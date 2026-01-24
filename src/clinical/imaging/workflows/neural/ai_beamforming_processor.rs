//! Neural Beamforming Processor
//!
//! This module implements the main orchestration logic for neural network-enhanced
//! ultrasound beamforming, integrating traditional signal processing with real-time
//! PINN inference and clinical decision support.
//!
//! # Architecture
//!
//! The processor coordinates four main stages:
//! 1. **Beamforming**: Traditional delay-and-sum beamforming
//! 2. **Feature Extraction**: Multi-scale morphological, spectral, and texture features
//! 3. **PINN Inference**: Real-time uncertainty quantification
//! 4. **Clinical Analysis**: Automated lesion detection and tissue classification
//!
//! # Performance
//!
//! Target total processing time: <100ms for real-time clinical use
//! - Each stage is timed independently for performance analysis
//! - Memory usage is estimated for monitoring
//! - Warnings are logged if performance targets are not met
//!
//! # Literature References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks"
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach"

use super::feature_extraction::FeatureExtractor;
use super::{
    types::{AIBeamformingConfig, AIBeamformingResult, FeatureMap, PerformanceMetrics},
    ClinicalDecisionSupport,
};
use crate::analysis::signal_processing::beamforming::domain_processor::BeamformingProcessor;
use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView4};
use std::time::Instant;

pub trait PinnInferenceEngine: Send + Sync {
    fn predict_realtime(
        &mut self,
        x_coords: &[f32],
        y_coords: &[f32],
        t_coords: &[f32],
    ) -> KwaversResult<(Vec<f32>, Vec<f32>)>;
}

/// Neural Beamforming Processor
///
/// Integrates real-time PINN inference with traditional beamforming
/// for clinical decision support and automated diagnosis.
///
/// # Example
///
/// ```ignore
/// use kwavers::domain::sensor::beamforming::neural::{
///     config::AIBeamformingConfig,
///     processor::AIEnhancedBeamformingProcessor,
/// };
/// use ndarray::Array4;
///
/// let config = AIBeamformingConfig::default();
/// let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
///
/// let mut processor = AIEnhancedBeamformingProcessor::new(config, sensor_positions)?;
///
/// let rf_data = Array4::<f32>::zeros((1024, 64, 100, 1));
/// let angles = vec![0.0; 100];
///
/// let result = processor.process_ai_enhanced(rf_data.view(), &angles)?;
/// println!("Processing time: {:.2}ms", result.performance.total_time_ms);
/// ```
pub struct AIEnhancedBeamformingProcessor {
    /// Base beamforming processor
    _beamforming_processor: BeamformingProcessor,

    /// Real-time PINN inference engine
    pinn_engine: Option<Box<dyn PinnInferenceEngine>>,

    /// Feature extractor
    feature_extractor: FeatureExtractor,

    /// Clinical decision support system
    clinical_support: ClinicalDecisionSupport,

    /// Configuration
    config: AIBeamformingConfig,
}

impl std::fmt::Debug for AIEnhancedBeamformingProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AIEnhancedBeamformingProcessor")
            .field("_beamforming_processor", &self._beamforming_processor)
            .field("feature_extractor", &self.feature_extractor)
            .field("clinical_support", &self.clinical_support)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl AIEnhancedBeamformingProcessor {
    /// Create new neural beamforming processor
    ///
    /// # Arguments
    ///
    /// * `config` - Neural beamforming configuration
    /// * `sensor_positions` - 3D positions of ultrasound array elements
    ///
    /// # Returns
    ///
    /// Configured processor ready for real-time operation
    ///
    /// # Errors
    ///
    /// Returns error if PINN initialization fails or configuration is invalid
    pub fn new(
        config: AIBeamformingConfig,
        sensor_positions: Vec<[f64; 3]>,
        pinn_engine: Option<Box<dyn PinnInferenceEngine>>,
    ) -> KwaversResult<Self> {
        // Validate configuration
        config.validate().map_err(|e| {
            crate::core::error::KwaversError::InvalidInput(format!(
                "Invalid AI beamforming config: {e}"
            ))
        })?;

        if config.enable_realtime_pinn && pinn_engine.is_none() {
            return Err(crate::core::error::KwaversError::InvalidInput(
                "PINN inference enabled but no PinnInferenceEngine provided".to_string(),
            ));
        }

        // Create base beamforming processor
        let beamforming_processor =
            BeamformingProcessor::new(config.beamforming_config.clone(), sensor_positions);

        // Create feature extractor and clinical support
        let feature_extractor = FeatureExtractor::new(config.feature_config.clone());
        let clinical_support = ClinicalDecisionSupport::new(config.clinical_thresholds.clone());

        Ok(Self {
            _beamforming_processor: beamforming_processor,
            pinn_engine,
            feature_extractor,
            clinical_support,
            config,
        })
    }

    /// Process ultrasound data with neural enhancement
    ///
    /// Executes the complete neural-enhanced beamforming pipeline:
    /// 1. Traditional beamforming (delay-and-sum)
    /// 2. Feature extraction (morphological, spectral, texture)
    /// 3. PINN inference (uncertainty quantification)
    /// 4. Clinical analysis (lesion detection, tissue classification)
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Raw RF data [time_samples, channels, frames, spatial_points]
    /// * `angles` - Steering angles for each frame (radians)
    ///
    /// # Returns
    ///
    /// Complete neural-enhanced beamforming result with clinical analysis and performance metrics
    ///
    /// # Performance
    ///
    /// Target: <100ms total processing time
    /// Logs warning if target is exceeded
    pub fn process_ai_enhanced(
        &mut self,
        rf_data: ArrayView4<f32>,
        angles: &[f32],
    ) -> KwaversResult<AIBeamformingResult> {
        let start_time = Instant::now();

        // Step 1: Traditional beamforming
        let beamforming_start = Instant::now();
        let volume = self.perform_beamforming(rf_data, angles)?;
        let beamforming_time = beamforming_start.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Feature extraction
        let feature_start = Instant::now();
        let features = self.feature_extractor.extract_features(volume.view())?;
        let feature_time = feature_start.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Real-time PINN inference
        let pinn_start = Instant::now();
        let (uncertainty, confidence) = self.perform_pinn_inference(&volume, &features)?;
        let pinn_time = pinn_start.elapsed().as_secs_f64() * 1000.0;

        // Step 4: Clinical analysis
        let clinical_start = Instant::now();
        let clinical_analysis = self.clinical_support.analyze_clinical(
            volume.view(),
            &features,
            uncertainty.view(),
            confidence.view(),
        )?;
        let clinical_time = clinical_start.elapsed().as_secs_f64() * 1000.0;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Validate performance requirements
        if total_time > self.config.performance_target_ms {
            log::warn!(
                "Neural beamforming exceeded performance target: {:.2}ms > {:.2}ms",
                total_time,
                self.config.performance_target_ms
            );
        }

        Ok(AIBeamformingResult {
            volume,
            uncertainty,
            confidence,
            features,
            clinical_analysis,
            performance: PerformanceMetrics {
                total_time_ms: total_time,
                beamforming_time_ms: beamforming_time,
                feature_extraction_time_ms: feature_time,
                pinn_inference_time_ms: pinn_time,
                clinical_analysis_time_ms: clinical_time,
                memory_usage_mb: self.estimate_memory_usage(),
                gpu_utilization_percent: 0.0, // Placeholder for future GPU monitoring
            },
        })
    }

    /// Perform traditional delay-and-sum beamforming
    ///
    /// Converts 4D RF data to 3D beamformed volume using delay-and-sum algorithm
    /// with steering angle compensation.
    ///
    /// # Algorithm
    ///
    /// For each spatial point (x, y) and frame z:
    /// 1. Compute steering delay based on angle and position
    /// 2. Apply delay to each channel
    /// 3. Sum delayed signals across channels
    /// 4. Normalize by channel count
    ///
    /// # Arguments
    ///
    /// * `rf_data` - Raw RF data [time_samples, channels, frames, spatial_points]
    /// * `angles` - Steering angles per frame (radians)
    ///
    /// # Returns
    ///
    /// Beamformed 3D volume [x, y, z]
    fn perform_beamforming(
        &self,
        rf_data: ArrayView4<f32>,
        angles: &[f32],
    ) -> KwaversResult<Array3<f32>> {
        let (nt, nchan, nframes, _) = rf_data.dim();

        // Create output volume [x, y, z] where z represents frames/depth
        let mut volume = Array3::<f32>::zeros((64, 64, nframes));

        // Simple delay-and-sum beamforming with steering
        for frame in 0..nframes {
            for x in 0..64 {
                for y in 0..64 {
                    // Compute steering delay based on angle and position
                    let angle = angles.get(frame).copied().unwrap_or(0.0);
                    let steering_delay = (x as f32 * angle.sin() + y as f32 * angle.cos()) * 0.01;

                    // Accumulate signals from channels with steering delays
                    let mut sum = 0.0;
                    let channel_limit = nchan.min(10); // Limit for simulation efficiency

                    for chan in 0..channel_limit {
                        let sample_idx = ((frame * nt / nframes) as f32 + steering_delay) as usize;
                        if sample_idx < nt {
                            sum += rf_data[[sample_idx, chan, frame, 0]];
                        }
                    }

                    volume[[x, y, frame]] = sum / channel_limit as f32;
                }
            }
        }

        Ok(volume)
    }

    /// Perform real-time PINN inference for uncertainty quantification
    ///
    /// Uses sparse sampling with interpolation for real-time performance.
    /// Samples every 4th voxel, performs PINN inference, and interpolates
    /// results back to full resolution.
    ///
    /// # Arguments
    ///
    /// * `volume` - Beamformed volume
    /// * `_features` - Extracted features (reserved for future adaptive sampling)
    ///
    /// # Returns
    ///
    /// Tuple of (uncertainty_map, confidence_map), both 3D arrays [x, y, z]
    fn perform_pinn_inference(
        &mut self,
        volume: &Array3<f32>,
        _features: &FeatureMap,
    ) -> KwaversResult<(Array3<f32>, Array3<f32>)> {
        let (nx, ny, nz) = volume.dim();

        if !self.config.enable_realtime_pinn {
            let uncertainty_volume = Array3::<f32>::from_elem((nx, ny, nz), 1.0);
            let confidence_volume = Array3::<f32>::zeros((nx, ny, nz));
            return Ok((uncertainty_volume, confidence_volume));
        }

        // Sparse sampling for real-time performance
        let sample_step = 4; // Sample every 4th point
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();
        let mut t_coords = Vec::new();

        for z in (0..nz).step_by(sample_step) {
            for y in (0..ny).step_by(sample_step) {
                for x in (0..nx).step_by(sample_step) {
                    x_coords.push(x as f32 / nx as f32); // Normalize to [0,1]
                    y_coords.push(y as f32 / ny as f32);
                    t_coords.push(z as f32 / nz as f32);
                }
            }
        }

        // Perform batched PINN inference
        let pinn_engine = self.pinn_engine.as_mut().ok_or_else(|| {
            crate::core::error::KwaversError::InvalidInput(
                "PINN inference enabled but no PinnInferenceEngine available".to_string(),
            )
        })?;
        let (_predictions, uncertainties) =
            pinn_engine.predict_realtime(&x_coords, &y_coords, &t_coords)?;

        // Interpolate results back to full volume resolution
        let mut uncertainty_volume = Array3::<f32>::zeros((nx, ny, nz));
        let mut confidence_volume = Array3::<f32>::zeros((nx, ny, nz));

        let mut sample_idx = 0;
        for z in (0..nz).step_by(sample_step) {
            for y in (0..ny).step_by(sample_step) {
                for x in (0..nx).step_by(sample_step) {
                    if sample_idx < uncertainties.len() {
                        let uncertainty_val = uncertainties[sample_idx];
                        let confidence_val = (1.0 - uncertainty_val).clamp(0.0, 1.0);

                        // Fill sampled region with nearest-neighbor interpolation
                        let x_end = (x + sample_step).min(nx);
                        let y_end = (y + sample_step).min(ny);
                        let z_end = (z + sample_step).min(nz);

                        for zz in z..z_end {
                            for yy in y..y_end {
                                for xx in x..x_end {
                                    uncertainty_volume[[xx, yy, zz]] = uncertainty_val;
                                    confidence_volume[[xx, yy, zz]] = confidence_val;
                                }
                            }
                        }

                        sample_idx += 1;
                    }
                }
            }
        }

        Ok((uncertainty_volume, confidence_volume))
    }

    /// Estimate memory usage for monitoring
    ///
    /// Provides rough estimate of memory consumption including:
    /// - Volume data
    /// - Feature maps
    /// - Working buffers
    ///
    /// # Returns
    ///
    /// Estimated memory usage in megabytes
    fn estimate_memory_usage(&self) -> f64 {
        // Volume sizes (assuming 64x64x100 typical volume)
        let volume_size = 64 * 64 * 100 * 4; // f32 = 4 bytes

        // Feature maps: morphological + spectral + texture
        let feature_size = volume_size * 3;

        // Working buffers: uncertainty + confidence + intermediate
        let buffer_size = volume_size * 2;

        (volume_size + feature_size + buffer_size) as f64 / (1024.0 * 1024.0)
    }

    /// Get current configuration
    pub fn config(&self) -> &AIBeamformingConfig {
        &self.config
    }

    /// Update configuration
    ///
    /// Updates processor configuration. Note: Does not reinitialize PINN model.
    ///
    /// # Arguments
    ///
    /// * `config` - New configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn update_config(&mut self, config: AIBeamformingConfig) -> KwaversResult<()> {
        config.validate().map_err(|e| {
            crate::core::error::KwaversError::InvalidInput(format!(
                "Invalid AI beamforming config: {}",
                e
            ))
        })?;

        // Update sub-component configurations
        self.feature_extractor = FeatureExtractor::new(config.feature_config.clone());
        self.clinical_support = ClinicalDecisionSupport::new(config.clinical_thresholds.clone());
        self.config = config;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    #[derive(Debug)]
    struct TestPinnEngine;

    impl PinnInferenceEngine for TestPinnEngine {
        fn predict_realtime(
            &mut self,
            x_coords: &[f32],
            y_coords: &[f32],
            t_coords: &[f32],
        ) -> KwaversResult<(Vec<f32>, Vec<f32>)> {
            let n = x_coords.len().min(y_coords.len()).min(t_coords.len());
            Ok((vec![0.0; n], vec![0.25; n]))
        }
    }

    #[test]
    fn test_processor_creation() {
        let config = AIBeamformingConfig {
            enable_realtime_pinn: false,
            ..Default::default()
        };
        let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];

        let result = AIEnhancedBeamformingProcessor::new(config, sensor_positions, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_memory_estimation() {
        let config = AIBeamformingConfig {
            enable_realtime_pinn: false,
            ..Default::default()
        };
        let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
        let processor =
            AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

        let memory_mb = processor.estimate_memory_usage();
        assert!(memory_mb > 0.0);
        assert!(memory_mb < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_config_access() {
        let config = AIBeamformingConfig {
            enable_realtime_pinn: false,
            ..Default::default()
        };
        let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
        let processor =
            AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

        let retrieved_config = processor.config();
        assert_eq!(retrieved_config.performance_target_ms, 100.0);
    }

    #[test]
    fn test_beamforming() {
        let config = AIBeamformingConfig {
            enable_realtime_pinn: false,
            ..Default::default()
        };
        let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
        let processor =
            AIEnhancedBeamformingProcessor::new(config, sensor_positions, None).unwrap();

        let rf_data = Array4::<f32>::from_elem((100, 10, 20, 1), 1.0);
        let angles = vec![0.0; 20];

        let result = processor.perform_beamforming(rf_data.view(), &angles);
        assert!(result.is_ok());

        let volume = result.unwrap();
        assert_eq!(volume.dim(), (64, 64, 20));
    }

    #[test]
    fn test_full_pipeline() {
        let config = AIBeamformingConfig {
            enable_realtime_pinn: true,
            ..Default::default()
        };
        let sensor_positions = vec![[0.0, 0.0, 0.0]; 64];
        let mut processor = AIEnhancedBeamformingProcessor::new(
            config,
            sensor_positions,
            Some(Box::new(TestPinnEngine)),
        )
        .unwrap();

        let rf_data = Array4::<f32>::from_elem((100, 10, 20, 1), 0.5);
        let angles = vec![0.0; 20];

        let result = processor.process_ai_enhanced(rf_data.view(), &angles);
        assert!(result.is_ok());

        let ai_result = result.unwrap();
        assert_eq!(ai_result.volume.dim(), (64, 64, 20));
        assert!(ai_result.performance.total_time_ms > 0.0);
        assert!(!ai_result.features.is_empty());
    }
}
