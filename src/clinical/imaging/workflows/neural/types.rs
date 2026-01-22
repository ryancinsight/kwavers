//! Clinical Analysis Types for Neural Ultrasound
//!
//! This module defines the clinical-specific types used for lesion detection,
//! tissue classification, and diagnostic decision support.

use ndarray::Array3;
use std::collections::HashMap;

/// Clinical Analysis Result
///
/// Complete clinical findings including lesion detection, tissue classification,
/// and diagnostic recommendations.
#[derive(Debug)]
pub struct ClinicalAnalysis {
    /// Detected lesions with characteristics
    pub lesions: Vec<LesionDetection>,

    /// Tissue classification results
    pub tissue_classification: TissueClassification,

    /// Clinical recommendations
    pub recommendations: Vec<String>,

    /// Overall diagnostic confidence [0, 1]
    pub diagnostic_confidence: f32,
}

impl ClinicalAnalysis {
    /// Create empty clinical analysis
    pub fn empty() -> Self {
        Self {
            lesions: Vec::new(),
            tissue_classification: TissueClassification::empty(),
            recommendations: Vec::new(),
            diagnostic_confidence: 0.0,
        }
    }

    /// Check if any lesions were detected
    pub fn has_lesions(&self) -> bool {
        !self.lesions.is_empty()
    }

    /// Count high-confidence lesions (confidence > 0.9)
    pub fn high_confidence_lesion_count(&self) -> usize {
        self.lesions.iter().filter(|l| l.confidence > 0.9).count()
    }
}

/// Lesion Detection Result
///
/// Represents a detected lesion with its characteristics, location,
/// and clinical significance assessment.
#[derive(Debug, Clone)]
pub struct LesionDetection {
    /// Lesion center coordinates (x, y, z) in voxel space
    pub center: (usize, usize, usize),

    /// Lesion size (diameter in mm)
    pub size_mm: f32,

    /// Detection confidence (0-1)
    pub confidence: f32,

    /// Lesion type classification
    pub lesion_type: String,

    /// Clinical significance score (0-1)
    pub clinical_significance: f32,
}

impl LesionDetection {
    /// Check if lesion requires immediate clinical attention
    pub fn requires_urgent_attention(&self) -> bool {
        self.clinical_significance > 0.8 || self.size_mm > 10.0
    }

    /// Get risk category based on size and significance
    pub fn risk_category(&self) -> &str {
        if self.clinical_significance > 0.7 || self.size_mm > 10.0 {
            "HIGH"
        } else if self.clinical_significance > 0.4 || self.size_mm > 5.0 {
            "MODERATE"
        } else {
            "LOW"
        }
    }
}

/// Tissue Classification Result
///
/// Contains tissue type probabilities, dominant tissue per region,
/// and boundary confidence.
#[derive(Debug)]
pub struct TissueClassification {
    /// Tissue type probabilities per voxel
    pub probabilities: HashMap<String, Array3<f32>>,

    /// Dominant tissue type per region [x, y, z]
    pub dominant_tissue: Array3<String>,

    /// Tissue boundary confidence [x, y, z]
    pub boundary_confidence: Array3<f32>,
}

impl TissueClassification {
    /// Create empty tissue classification
    pub fn empty() -> Self {
        Self {
            probabilities: HashMap::new(),
            dominant_tissue: Array3::from_elem((1, 1, 1), String::new()),
            boundary_confidence: Array3::zeros((1, 1, 1)),
        }
    }

    /// Get list of detected tissue types
    pub fn tissue_types(&self) -> Vec<String> {
        self.probabilities.keys().cloned().collect()
    }

    /// Check if suspicious tissue was detected
    pub fn has_suspicious_tissue(&self) -> bool {
        self.probabilities.contains_key("suspicious")
    }
}

/// Feature Map for Clinical Analysis
///
/// Multi-scale feature maps organized by category for clinical decision support.
#[derive(Debug)]
pub struct FeatureMap {
    /// Morphological features (size, shape, boundaries)
    pub morphological: HashMap<String, Array3<f32>>,

    /// Spectral features (frequency content, bandwidth)
    pub spectral: HashMap<String, Array3<f32>>,

    /// Texture features (speckle statistics, homogeneity)
    pub texture: HashMap<String, Array3<f32>>,
}

impl FeatureMap {
    /// Create new empty feature map
    pub fn new() -> Self {
        Self {
            morphological: HashMap::new(),
            spectral: HashMap::new(),
            texture: HashMap::new(),
        }
    }

    /// Check if feature map is empty
    pub fn is_empty(&self) -> bool {
        self.morphological.is_empty() && self.spectral.is_empty() && self.texture.is_empty()
    }

    /// Count total number of features
    pub fn feature_count(&self) -> usize {
        self.morphological.len() + self.spectral.len() + self.texture.len()
    }
}

impl Default for FeatureMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Clinical Thresholds Configuration
///
/// Threshold values for clinical decision support algorithms.
#[derive(Debug, Clone)]
pub struct ClinicalThresholds {
    /// Lesion detection confidence threshold
    pub lesion_confidence_threshold: f32,

    /// Contrast abnormality threshold
    pub contrast_abnormality_threshold: f32,

    /// Tissue uncertainty threshold
    pub tissue_uncertainty_threshold: f32,

    /// Speckle anomaly threshold
    pub speckle_anomaly_threshold: f32,

    /// Segmentation sensitivity
    pub segmentation_sensitivity: f32,

    /// Voxel size in mm
    pub voxel_size_mm: f32,
}

impl Default for ClinicalThresholds {
    fn default() -> Self {
        Self {
            lesion_confidence_threshold: 0.85,
            contrast_abnormality_threshold: 2.0,
            tissue_uncertainty_threshold: 0.3,
            speckle_anomaly_threshold: 0.7,
            segmentation_sensitivity: 0.5,
            voxel_size_mm: 0.5,
        }
    }
}

impl ClinicalThresholds {
    /// High sensitivity preset (more detections, fewer false negatives)
    pub fn high_sensitivity() -> Self {
        Self {
            lesion_confidence_threshold: 0.7,
            contrast_abnormality_threshold: 1.5,
            tissue_uncertainty_threshold: 0.4,
            speckle_anomaly_threshold: 0.6,
            segmentation_sensitivity: 0.4,
            voxel_size_mm: 0.5,
        }
    }

    /// High specificity preset (fewer detections, fewer false positives)
    pub fn high_specificity() -> Self {
        Self {
            lesion_confidence_threshold: 0.95,
            contrast_abnormality_threshold: 2.5,
            tissue_uncertainty_threshold: 0.2,
            speckle_anomaly_threshold: 0.8,
            segmentation_sensitivity: 0.6,
            voxel_size_mm: 0.5,
        }
    }

    /// Validate thresholds are in valid ranges
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.lesion_confidence_threshold) {
            return Err("Lesion confidence threshold must be in [0, 1]".to_string());
        }
        if self.contrast_abnormality_threshold < 0.0 {
            return Err("Contrast abnormality threshold must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.tissue_uncertainty_threshold) {
            return Err("Tissue uncertainty threshold must be in [0, 1]".to_string());
        }
        if self.voxel_size_mm <= 0.0 {
            return Err("Voxel size must be positive".to_string());
        }
        Ok(())
    }
}
