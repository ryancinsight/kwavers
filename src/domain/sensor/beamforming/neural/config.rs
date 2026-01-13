//! Configuration Types for Neural Beamforming
//!
//! This module provides configuration structures for neural network-enhanced
//! ultrasound beamforming with real-time PINN inference and clinical decision support.
//!
//! # Architecture
//!
//! Configuration is organized into three layers:
//! - **Beamforming Config**: Base signal processing parameters
//! - **Feature Config**: Feature extraction and analysis parameters
//! - **Clinical Config**: Clinical thresholds and diagnostic criteria
//!
//! # Literature References
//!
//! - Kendall & Gal (2017): "What uncertainties do we need in Bayesian deep learning?"
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"

use crate::domain::sensor::beamforming::BeamformingConfig;

/// Configuration for Neural Beamforming
///
/// Integrates traditional beamforming parameters with AI inference settings
/// and clinical analysis thresholds for real-time diagnostic support.
///
/// # Performance Requirements
///
/// Target total processing time: <100ms for real-time clinical use
/// - Beamforming: <30ms
/// - Feature extraction: <20ms
/// - PINN inference: <30ms
/// - Clinical analysis: <20ms
#[derive(Debug, Clone)]
pub struct AIBeamformingConfig {
    /// Base beamforming configuration
    pub beamforming_config: BeamformingConfig,

    /// Enable real-time PINN inference for uncertainty quantification
    pub enable_realtime_pinn: bool,

    /// Enable clinical decision support (lesion detection, tissue classification)
    pub enable_clinical_support: bool,

    /// Feature extraction parameters
    pub feature_config: FeatureConfig,

    /// Clinical analysis thresholds
    pub clinical_thresholds: ClinicalThresholds,

    /// Performance target in milliseconds
    ///
    /// Warning will be logged if total processing time exceeds this threshold.
    /// Default: 100.0ms for real-time clinical applications
    pub performance_target_ms: f64,
}

impl Default for AIBeamformingConfig {
    fn default() -> Self {
        Self {
            beamforming_config: BeamformingConfig::default(),
            enable_realtime_pinn: true,
            enable_clinical_support: true,
            feature_config: FeatureConfig::default(),
            clinical_thresholds: ClinicalThresholds::default(),
            performance_target_ms: 100.0,
        }
    }
}

impl AIBeamformingConfig {
    /// Validate configuration consistency
    ///
    /// # Invariants
    ///
    /// - Performance target must be positive
    /// - Feature config must be valid
    /// - Clinical thresholds must be in valid ranges [0, 1] or [0, ∞)
    pub fn validate(&self) -> Result<(), String> {
        if self.performance_target_ms <= 0.0 {
            return Err("Performance target must be positive".to_string());
        }

        self.feature_config.validate()?;
        self.clinical_thresholds.validate()?;

        Ok(())
    }
}

/// Feature Extraction Configuration
///
/// Controls which features are extracted from ultrasound volumes for
/// AI analysis and clinical decision support.
///
/// # Feature Types
///
/// - **Morphological**: Shape, boundaries, gradients (edge detection)
/// - **Spectral**: Frequency content, bandwidth, local frequency
/// - **Texture**: Speckle statistics, homogeneity, entropy
///
/// # Literature References
///
/// - Haralick et al. (1973): "Textural Features for Image Classification"
/// - Mandelbrot (1982): "The Fractal Geometry of Nature"
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Extract morphological features (gradient magnitude, Laplacian)
    pub morphological_features: bool,

    /// Extract spectral features (local frequency, bandwidth)
    pub spectral_features: bool,

    /// Extract texture features (speckle variance, homogeneity)
    pub texture_features: bool,

    /// Feature computation window size (voxels)
    ///
    /// Must be odd and >= 3. Larger windows provide smoother features
    /// but reduce spatial resolution.
    pub window_size: usize,

    /// Overlap between feature windows (0.0 to 1.0)
    ///
    /// Higher overlap increases computation but improves spatial continuity.
    /// Typical values: 0.5 (50% overlap) for clinical applications.
    pub overlap: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            morphological_features: true,
            spectral_features: true,
            texture_features: true,
            window_size: 32,
            overlap: 0.5,
        }
    }
}

impl FeatureConfig {
    /// Validate feature configuration
    ///
    /// # Invariants
    ///
    /// - At least one feature type must be enabled
    /// - Window size must be odd and >= 3
    /// - Overlap must be in range [0.0, 1.0)
    pub fn validate(&self) -> Result<(), String> {
        if !self.morphological_features && !self.spectral_features && !self.texture_features {
            return Err("At least one feature type must be enabled".to_string());
        }

        if self.window_size < 3 {
            return Err("Window size must be >= 3".to_string());
        }

        if self.window_size % 2 == 0 {
            return Err("Window size must be odd".to_string());
        }

        if self.overlap < 0.0 || self.overlap >= 1.0 {
            return Err("Overlap must be in range [0.0, 1.0)".to_string());
        }

        Ok(())
    }
}

/// Clinical Analysis Thresholds
///
/// Defines thresholds for automated clinical decision support including
/// lesion detection, tissue classification, and diagnostic confidence.
///
/// # Clinical Safety
///
/// These thresholds directly affect clinical decision-making. Default values
/// are conservative to minimize false negatives. Site-specific tuning should
/// be performed with clinical validation.
///
/// # Literature References
///
/// - Stavros et al. (1995): "Solid breast nodules: use of sonography"
/// - Burnside et al. (2007): "Differentiating benign from malignant findings"
#[derive(Debug, Clone)]
pub struct ClinicalThresholds {
    /// Lesion detection confidence threshold (0.0 to 1.0)
    ///
    /// Minimum confidence score to report a lesion detection.
    /// Higher values reduce false positives but may miss subtle lesions.
    /// Default: 0.8 (high specificity)
    pub lesion_confidence_threshold: f32,

    /// Tissue classification uncertainty threshold (0.0 to 1.0)
    ///
    /// Maximum acceptable uncertainty for tissue classification.
    /// Lower values enforce higher model confidence.
    /// Default: 0.3 (conservative)
    pub tissue_uncertainty_threshold: f32,

    /// Contrast abnormality threshold (std dev multiplier)
    ///
    /// Intensity contrast required for lesion detection, measured in
    /// standard deviations above local mean.
    /// Default: 2.0 (moderate sensitivity)
    pub contrast_abnormality_threshold: f32,

    /// Speckle anomaly threshold (std dev multiplier)
    ///
    /// Speckle pattern deviation required for texture-based detection.
    /// Default: 1.5 (moderate sensitivity)
    pub speckle_anomaly_threshold: f32,

    /// Segmentation sensitivity (std dev multiplier)
    ///
    /// Controls adaptive thresholding for lesion boundary detection.
    /// Higher values create larger segmentation regions.
    /// Default: 1.0 (balanced)
    pub segmentation_sensitivity: f32,

    /// Voxel size in millimeters
    ///
    /// Physical size of each voxel for accurate volume measurements.
    /// Must match imaging system calibration.
    /// Default: 0.5mm (high-resolution clinical imaging)
    pub voxel_size_mm: f32,
}

impl Default for ClinicalThresholds {
    fn default() -> Self {
        Self {
            lesion_confidence_threshold: 0.8,
            tissue_uncertainty_threshold: 0.3,
            contrast_abnormality_threshold: 2.0,
            speckle_anomaly_threshold: 1.5,
            segmentation_sensitivity: 1.0,
            voxel_size_mm: 0.5,
        }
    }
}

impl ClinicalThresholds {
    /// Validate clinical thresholds
    ///
    /// # Invariants
    ///
    /// - Probability thresholds must be in [0, 1]
    /// - Statistical thresholds must be positive
    /// - Voxel size must be positive
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.lesion_confidence_threshold) {
            return Err("Lesion confidence threshold must be in [0, 1]".to_string());
        }

        if !(0.0..=1.0).contains(&self.tissue_uncertainty_threshold) {
            return Err("Tissue uncertainty threshold must be in [0, 1]".to_string());
        }

        if self.contrast_abnormality_threshold <= 0.0 {
            return Err("Contrast threshold must be positive".to_string());
        }

        if self.speckle_anomaly_threshold <= 0.0 {
            return Err("Speckle threshold must be positive".to_string());
        }

        if self.segmentation_sensitivity <= 0.0 {
            return Err("Segmentation sensitivity must be positive".to_string());
        }

        if self.voxel_size_mm <= 0.0 {
            return Err("Voxel size must be positive".to_string());
        }

        Ok(())
    }

    /// Create high-sensitivity configuration for screening
    ///
    /// Lowers thresholds to maximize detection of subtle abnormalities.
    /// Increases false positive rate but minimizes missed findings.
    pub fn high_sensitivity() -> Self {
        Self {
            lesion_confidence_threshold: 0.6,
            tissue_uncertainty_threshold: 0.4,
            contrast_abnormality_threshold: 1.5,
            speckle_anomaly_threshold: 1.2,
            segmentation_sensitivity: 1.5,
            voxel_size_mm: 0.5,
        }
    }

    /// Create high-specificity configuration for diagnostic confirmation
    ///
    /// Raises thresholds to minimize false positives.
    /// Appropriate for follow-up of previously identified abnormalities.
    pub fn high_specificity() -> Self {
        Self {
            lesion_confidence_threshold: 0.9,
            tissue_uncertainty_threshold: 0.2,
            contrast_abnormality_threshold: 2.5,
            speckle_anomaly_threshold: 2.0,
            segmentation_sensitivity: 0.8,
            voxel_size_mm: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configs_are_valid() {
        let ai_config = AIBeamformingConfig::default();
        assert!(ai_config.validate().is_ok());

        let feature_config = FeatureConfig::default();
        assert!(feature_config.validate().is_ok());

        let clinical_thresholds = ClinicalThresholds::default();
        assert!(clinical_thresholds.validate().is_ok());
    }

    #[test]
    fn test_feature_config_validation() {
        let mut config = FeatureConfig::default();

        // Test: at least one feature type required
        config.morphological_features = false;
        config.spectral_features = false;
        config.texture_features = false;
        assert!(config.validate().is_err());

        // Test: window size must be odd and >= 3
        config.texture_features = true;
        config.window_size = 2;
        assert!(config.validate().is_err());

        config.window_size = 4;
        assert!(config.validate().is_err());

        config.window_size = 3;
        assert!(config.validate().is_ok());

        // Test: overlap must be in [0, 1)
        config.overlap = -0.1;
        assert!(config.validate().is_err());

        config.overlap = 1.0;
        assert!(config.validate().is_err());

        config.overlap = 0.5;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_clinical_thresholds_validation() {
        let mut thresholds = ClinicalThresholds::default();

        // Test: probability thresholds in [0, 1]
        thresholds.lesion_confidence_threshold = 1.5;
        assert!(thresholds.validate().is_err());

        thresholds.lesion_confidence_threshold = -0.1;
        assert!(thresholds.validate().is_err());

        thresholds.lesion_confidence_threshold = 0.8;
        assert!(thresholds.validate().is_ok());

        // Test: statistical thresholds must be positive
        thresholds.contrast_abnormality_threshold = 0.0;
        assert!(thresholds.validate().is_err());

        thresholds.contrast_abnormality_threshold = 2.0;
        thresholds.voxel_size_mm = -1.0;
        assert!(thresholds.validate().is_err());

        thresholds.voxel_size_mm = 0.5;
        assert!(thresholds.validate().is_ok());
    }

    #[test]
    fn test_clinical_presets() {
        let high_sens = ClinicalThresholds::high_sensitivity();
        assert!(high_sens.validate().is_ok());
        assert!(
            high_sens.lesion_confidence_threshold
                < ClinicalThresholds::default().lesion_confidence_threshold
        );

        let high_spec = ClinicalThresholds::high_specificity();
        assert!(high_spec.validate().is_ok());
        assert!(
            high_spec.lesion_confidence_threshold
                > ClinicalThresholds::default().lesion_confidence_threshold
        );
    }

    #[test]
    fn test_ai_config_validation() {
        let mut config = AIBeamformingConfig::default();

        // Test: performance target must be positive
        config.performance_target_ms = 0.0;
        assert!(config.validate().is_err());

        config.performance_target_ms = -10.0;
        assert!(config.validate().is_err());

        config.performance_target_ms = 100.0;
        assert!(config.validate().is_ok());

        // Test: validates nested configs
        config.feature_config.window_size = 2; // Invalid (even)
        assert!(config.validate().is_err());
    }
}
