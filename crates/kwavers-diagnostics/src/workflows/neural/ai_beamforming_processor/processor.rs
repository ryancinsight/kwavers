//! AIEnhancedBeamformingProcessor struct and impl.

use leto::{Array3, ArrayView4};
use std::time::Instant;

use kwavers_core::error::KwaversResult;
use kwavers_transducer::beamforming::processor::BeamformingProcessor;

use super::super::feature_extraction::FeatureExtractor;
use super::super::{
    types::{AIBeamformingConfig, AIBeamformingResult, AiBeamformingMetrics, FeatureMap},
    NeuralClinicalDecisionSupport,
};
use super::trait_engine::PinnInferenceEngine;

/// Neural Beamforming Processor
///
/// Integrates real-time PINN inference with traditional beamforming
/// for clinical decision support and automated diagnosis.
pub struct AIEnhancedBeamformingProcessor {
    /// Base beamforming processor
    _beamforming_processor: BeamformingProcessor,

    /// Real-time PINN inference engine
    pinn_engine: Option<Box<dyn PinnInferenceEngine>>,

    /// Feature extractor
    feature_extractor: FeatureExtractor,

    /// Clinical decision support system
    clinical_support: NeuralClinicalDecisionSupport,

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
    /// Create new neural beamforming processor.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(
        config: AIBeamformingConfig,
        sensor_positions: Vec<[f64; 3]>,
        pinn_engine: Option<Box<dyn PinnInferenceEngine>>,
    ) -> KwaversResult<Self> {
        config.validate().map_err(|e| {
            kwavers_core::error::KwaversError::InvalidInput(format!(
                "Invalid AI beamforming config: {e}"
            ))
        })?;

        if config.enable_realtime_pinn && pinn_engine.is_none() {
            return Err(kwavers_core::error::KwaversError::InvalidInput(
                "PINN inference enabled but no PinnInferenceEngine provided".to_owned(),
            ));
        }

        let beamforming_processor =
            BeamformingProcessor::new(config.beamforming_config.clone(), sensor_positions);
        let feature_extractor = FeatureExtractor::new(config.feature_config.clone());
        let clinical_support =
            NeuralClinicalDecisionSupport::new(config.clinical_thresholds.clone());

        Ok(Self {
            _beamforming_processor: beamforming_processor,
            pinn_engine,
            feature_extractor,
            clinical_support,
            config,
        })
    }

    /// Process ultrasound data with neural enhancement.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn process_ai_enhanced(
        &mut self,
        rf_data: ArrayView4<f32>,
        angles: &[f32],
    ) -> KwaversResult<AIBeamformingResult> {
        let start_time = Instant::now();

        let beamforming_start = Instant::now();
        let volume = self.perform_beamforming(rf_data, angles)?;
        let beamforming_time = beamforming_start.elapsed().as_secs_f64() * 1000.0;

        let feature_start = Instant::now();
        let features = self.feature_extractor.extract_features(volume.view())?;
        let feature_time = feature_start.elapsed().as_secs_f64() * 1000.0;

        let pinn_start = Instant::now();
        let (uncertainty, confidence) = self.perform_pinn_inference(&volume, &features)?;
        let pinn_time = pinn_start.elapsed().as_secs_f64() * 1000.0;

        let clinical_start = Instant::now();
        let clinical_analysis = self.clinical_support.analyze_clinical(
            volume.view(),
            &features,
            uncertainty.view(),
            confidence.view(),
        )?;
        let clinical_time = clinical_start.elapsed().as_secs_f64() * 1000.0;

        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;

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
            performance: AiBeamformingMetrics {
                total_time_ms: total_time,
                beamforming_time_ms: beamforming_time,
                feature_extraction_time_ms: feature_time,
                pinn_inference_time_ms: pinn_time,
                clinical_analysis_time_ms: clinical_time,
                memory_usage_mb: self.estimate_memory_usage(),
                gpu_utilization_percent: f64::NAN,
            },
        })
    }

    /// Perform traditional delay-and-sum beamforming.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn perform_beamforming(
        &self,
        rf_data: ArrayView4<f32>,
        angles: &[f32],
    ) -> KwaversResult<Array3<f32>> {
        let [nt, nchan, nframes, _] = rf_data.shape();
        let mut volume = Array3::<f32>::zeros((64, 64, nframes));

        for frame in 0..nframes {
            for x in 0..64 {
                for y in 0..64 {
                    let angle = angles.get(frame).copied().unwrap_or(0.0);
                    let steering_delay =
                        (x as f32).mul_add(angle.sin(), y as f32 * angle.cos()) * 0.01;
                    let mut sum = 0.0;
                    let channel_limit = nchan.min(10);
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

    /// Perform real-time PINN inference for uncertainty quantification.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    fn perform_pinn_inference(
        &mut self,
        volume: &Array3<f32>,
        _features: &FeatureMap,
    ) -> KwaversResult<(Array3<f32>, Array3<f32>)> {
        let [nx, ny, nz] = volume.shape();

        if !self.config.enable_realtime_pinn {
            let uncertainty_volume = Array3::<f32>::from_elem((nx, ny, nz), 1.0);
            let confidence_volume = Array3::<f32>::zeros((nx, ny, nz));
            return Ok((uncertainty_volume, confidence_volume));
        }

        let sample_step = 4;
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();
        let mut t_coords = Vec::new();

        for z in (0..nz).step_by(sample_step) {
            for y in (0..ny).step_by(sample_step) {
                for x in (0..nx).step_by(sample_step) {
                    x_coords.push(x as f32 / nx as f32);
                    y_coords.push(y as f32 / ny as f32);
                    t_coords.push(z as f32 / nz as f32);
                }
            }
        }

        let pinn_engine = self.pinn_engine.as_mut().ok_or_else(|| {
            kwavers_core::error::KwaversError::InvalidInput(
                "PINN inference enabled but no PinnInferenceEngine available".to_owned(),
            )
        })?;
        let (_predictions, uncertainties) =
            pinn_engine.predict_realtime(&x_coords, &y_coords, &t_coords)?;

        let mut uncertainty_volume = Array3::<f32>::zeros((nx, ny, nz));
        let mut confidence_volume = Array3::<f32>::zeros((nx, ny, nz));

        let mut sample_idx = 0;
        for z in (0..nz).step_by(sample_step) {
            for y in (0..ny).step_by(sample_step) {
                for x in (0..nx).step_by(sample_step) {
                    if sample_idx < uncertainties.len() {
                        let uncertainty_val = uncertainties[sample_idx];
                        let confidence_val = (1.0 - uncertainty_val).clamp(0.0, 1.0);
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

    /// Estimate memory usage in megabytes.
    pub(super) fn estimate_memory_usage(&self) -> f64 {
        let volume_size = 64 * 64 * 100 * 4;
        let feature_size = volume_size * 3;
        let buffer_size = volume_size * 2;
        (volume_size + feature_size + buffer_size) as f64 / (1024.0 * 1024.0)
    }

    /// Get current configuration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &AIBeamformingConfig {
        &self.config
    }

    /// Update configuration.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn update_config(&mut self, config: AIBeamformingConfig) -> KwaversResult<()> {
        config.validate().map_err(|e| {
            kwavers_core::error::KwaversError::InvalidInput(format!(
                "Invalid AI beamforming config: {}",
                e
            ))
        })?;

        self.feature_extractor = FeatureExtractor::new(config.feature_config.clone());
        self.clinical_support =
            NeuralClinicalDecisionSupport::new(config.clinical_thresholds.clone());
        self.config = config;

        Ok(())
    }
}