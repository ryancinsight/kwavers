//! AI-enhanced beamforming configuration, result, and supporting types.

use aequitas::systems::si::quantities::{Dimensionless, Length, Time};
use aequitas::systems::si::units::{Millimeter, Millisecond};
use kwavers_analysis::signal_processing::beamforming::neural::config::FeatureConfig;
use kwavers_transducer::beamforming::BeamformingConfig;
use leto::Array3;
use std::collections::HashMap;

use super::ClinicalAnalysis;

/// Multi-scale feature maps organized by category for clinical decision support.
#[derive(Debug)]
pub struct FeatureMap {
    /// Morphological features (size, shape, boundaries).
    pub morphological: HashMap<String, Array3<f32>>,

    /// Spectral features (frequency content, bandwidth).
    pub spectral: HashMap<String, Array3<f32>>,

    /// Texture features (speckle statistics, homogeneity).
    pub texture: HashMap<String, Array3<f32>>,
}

impl FeatureMap {
    /// Create new empty feature map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            morphological: HashMap::new(),
            spectral: HashMap::new(),
            texture: HashMap::new(),
        }
    }

    /// Return true if the feature map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.morphological.is_empty() && self.spectral.is_empty() && self.texture.is_empty()
    }

    /// Total number of feature channels.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.morphological.len() + self.spectral.len() + self.texture.len()
    }
}

impl Default for FeatureMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Threshold values for clinical decision support algorithms.
#[derive(Debug, Clone)]
pub struct ClinicalThresholds {
    /// Lesion detection confidence threshold.
    pub lesion_confidence_threshold: Dimensionless<f32>,

    /// Contrast abnormality threshold.
    pub contrast_abnormality_threshold: Dimensionless<f32>,

    /// Tissue uncertainty threshold.
    pub tissue_uncertainty_threshold: Dimensionless<f32>,

    /// Speckle anomaly threshold.
    pub speckle_anomaly_threshold: Dimensionless<f32>,

    /// Segmentation sensitivity.
    pub segmentation_sensitivity: Dimensionless<f32>,

    /// Isotropic voxel edge length as an Aequitas length quantity.
    pub voxel_size: Length<f32>,
}

impl Default for ClinicalThresholds {
    fn default() -> Self {
        Self {
            lesion_confidence_threshold: Dimensionless::from_base(0.85),
            contrast_abnormality_threshold: Dimensionless::from_base(2.0),
            tissue_uncertainty_threshold: Dimensionless::from_base(0.3),
            speckle_anomaly_threshold: Dimensionless::from_base(0.7),
            segmentation_sensitivity: Dimensionless::from_base(0.5),
            voxel_size: Length::from_unit::<Millimeter>(0.5),
        }
    }
}

impl ClinicalThresholds {
    /// High sensitivity preset (more detections, fewer false negatives).
    #[must_use]
    pub fn high_sensitivity() -> Self {
        Self {
            lesion_confidence_threshold: Dimensionless::from_base(0.7),
            contrast_abnormality_threshold: Dimensionless::from_base(1.5),
            tissue_uncertainty_threshold: Dimensionless::from_base(0.4),
            speckle_anomaly_threshold: Dimensionless::from_base(0.6),
            segmentation_sensitivity: Dimensionless::from_base(0.4),
            voxel_size: Length::from_unit::<Millimeter>(0.5),
        }
    }

    /// High specificity preset (fewer detections, fewer false positives).
    #[must_use]
    pub fn high_specificity() -> Self {
        Self {
            lesion_confidence_threshold: Dimensionless::from_base(0.95),
            contrast_abnormality_threshold: Dimensionless::from_base(2.5),
            tissue_uncertainty_threshold: Dimensionless::from_base(0.2),
            speckle_anomaly_threshold: Dimensionless::from_base(0.8),
            segmentation_sensitivity: Dimensionless::from_base(0.6),
            voxel_size: Length::from_unit::<Millimeter>(0.5),
        }
    }

    /// Validate thresholds are in valid ranges.
    /// # Errors
    /// - Returns [`Err`] with a description if any threshold is out of range.
    ///
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(self.lesion_confidence_threshold.as_base()) {
            return Err("Lesion confidence threshold must be in [0, 1]".to_owned());
        }
        if *self.contrast_abnormality_threshold.as_base() < 0.0 {
            return Err("Contrast abnormality threshold must be positive".to_owned());
        }
        if !(0.0..=1.0).contains(self.tissue_uncertainty_threshold.as_base()) {
            return Err("Tissue uncertainty threshold must be in [0, 1]".to_owned());
        }
        if *self.voxel_size.as_base() <= 0.0 {
            return Err("Voxel size must be positive".to_owned());
        }
        Ok(())
    }
}

/// Configuration for neural network-enhanced ultrasound beamforming with
/// clinical decision support integration.
#[derive(Debug, Clone)]
pub struct AIBeamformingConfig {
    /// Base beamforming configuration.
    pub beamforming_config: BeamformingConfig,

    /// Feature extraction configuration.
    pub feature_config: FeatureConfig,

    /// Clinical analysis thresholds.
    pub clinical_thresholds: ClinicalThresholds,

    /// Enable real-time PINN inference.
    pub enable_realtime_pinn: bool,

    /// Maximum processing time as an Aequitas time quantity.
    pub performance_target: Time<f64>,
}

impl Default for AIBeamformingConfig {
    fn default() -> Self {
        Self {
            beamforming_config: BeamformingConfig::default(),
            feature_config: FeatureConfig::default(),
            clinical_thresholds: ClinicalThresholds::default(),
            enable_realtime_pinn: true,
            performance_target: Time::from_unit::<Millisecond>(100.0),
        }
    }
}

impl AIBeamformingConfig {
    /// Validate configuration.
    /// # Errors
    /// - Propagates validation errors from `feature_config` and `clinical_thresholds`.
    ///
    pub fn validate(&self) -> Result<(), String> {
        self.feature_config.validate()?;
        self.clinical_thresholds.validate()?;

        if *self.performance_target.as_base() <= 0.0 {
            return Err("Performance target must be positive".to_owned());
        }

        Ok(())
    }
}

/// Complete result from neural-enhanced beamforming.
#[derive(Debug)]
pub struct AIBeamformingResult {
    /// Reconstructed ultrasound volume [x, y, z].
    pub volume: Array3<f32>,

    /// Uncertainty map from PINN [x, y, z].
    pub uncertainty: Array3<f32>,

    /// Confidence map [x, y, z].
    pub confidence: Array3<f32>,

    /// Extracted feature maps.
    pub features: FeatureMap,

    /// Clinical analysis results.
    pub clinical_analysis: ClinicalAnalysis,

    /// Performance metrics.
    pub performance: AiBeamformingMetrics,
}

/// Execution time and resource usage for each AI beamforming stage.
#[derive(Debug, Clone)]
pub struct AiBeamformingMetrics {
    /// Total processing time as an Aequitas time quantity.
    pub total_time: Time<f64>,

    /// Beamforming stage time as an Aequitas time quantity.
    pub beamforming_time: Time<f64>,

    /// Feature extraction time as an Aequitas time quantity.
    pub feature_extraction_time: Time<f64>,

    /// PINN inference time as an Aequitas time quantity.
    pub pinn_inference_time: Time<f64>,

    /// Clinical analysis time as an Aequitas time quantity.
    pub clinical_analysis_time: Time<f64>,

    /// Estimated memory usage in bytes at the storage-instrumentation boundary.
    pub memory_usage_bytes: u64,

    /// GPU utilization as a dimensionless percentage value [0, 100].
    pub gpu_utilization: Option<Dimensionless<f64>>,
}
