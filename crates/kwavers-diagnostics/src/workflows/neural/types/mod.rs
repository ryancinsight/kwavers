//! Clinical analysis types for neural ultrasound.
//!
//! Core clinical types: lesion detection, tissue classification, diagnostics.
//! AI beamforming and configuration types live in [`ai_beamforming`].

pub mod ai_beamforming;

use leto::Array3;
use std::collections::HashMap;

pub use ai_beamforming::{
    AIBeamformingConfig, AIBeamformingResult, AiBeamformingMetrics, ClinicalThresholds, FeatureMap,
};
// Re-export ClinicalThresholds at this level so callers using `types::ClinicalThresholds`
// continue to resolve correctly after the split.

/// Complete clinical findings including lesion detection, tissue classification,
/// and diagnostic recommendations.
#[derive(Debug)]
pub struct ClinicalAnalysis {
    /// Detected lesions with characteristics.
    pub lesions: Vec<LesionDetection>,

    /// Tissue classification results.
    pub tissue_classification: TissueClassification,

    /// Clinical recommendations.
    pub recommendations: Vec<String>,

    /// Overall diagnostic confidence [0, 1].
    pub diagnostic_confidence: f32,
}

impl ClinicalAnalysis {
    /// Create empty clinical analysis.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            lesions: Vec::new(),
            tissue_classification: TissueClassification::empty(),
            recommendations: Vec::new(),
            diagnostic_confidence: 0.0,
        }
    }

    /// Check if any lesions were detected.
    #[must_use]
    pub fn has_lesions(&self) -> bool {
        !self.lesions.is_empty()
    }

    /// Count high-confidence lesions (confidence > 0.9).
    #[must_use]
    pub fn high_confidence_lesion_count(&self) -> usize {
        self.lesions.iter().filter(|l| l.confidence > 0.9).count()
    }
}

/// Detected lesion with its characteristics, location, and clinical significance.
#[derive(Debug, Clone)]
pub struct LesionDetection {
    /// Lesion center coordinates (x, y, z) in voxel space.
    pub center: (usize, usize, usize),

    /// Lesion size (diameter in mm).
    pub size_mm: f32,

    /// Detection confidence [0, 1].
    pub confidence: f32,

    /// Lesion type classification.
    pub lesion_type: String,

    /// Clinical significance score [0, 1].
    pub clinical_significance: f32,
}

impl LesionDetection {
    /// Return true if lesion requires immediate clinical attention.
    #[must_use]
    pub fn requires_urgent_attention(&self) -> bool {
        self.clinical_significance > 0.8 || self.size_mm > 10.0
    }

    /// Risk category: "HIGH", "MODERATE", or "LOW".
    #[must_use]
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

/// Tissue type probabilities, dominant tissue per region, and boundary confidence.
#[derive(Debug)]
pub struct TissueClassification {
    /// Tissue type probabilities per voxel.
    pub probabilities: HashMap<String, Array3<f32>>,

    /// Dominant tissue type per region [x, y, z].
    pub dominant_tissue: Array3<String>,

    /// Tissue boundary confidence [x, y, z].
    pub boundary_confidence: Array3<f32>,
}

impl TissueClassification {
    /// Create empty tissue classification.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            probabilities: HashMap::new(),
            dominant_tissue: Array3::from_elem((1, 1, 1), String::new()),
            boundary_confidence: Array3::zeros((1, 1, 1)),
        }
    }

    /// List of detected tissue types.
    #[must_use]
    pub fn tissue_types(&self) -> Vec<String> {
        self.probabilities.keys().cloned().collect()
    }

    /// Return true if suspicious tissue was detected.
    #[must_use]
    pub fn has_suspicious_tissue(&self) -> bool {
        self.probabilities.contains_key("suspicious")
    }
}
