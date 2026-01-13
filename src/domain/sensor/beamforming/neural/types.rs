//! Result Types and Data Structures for Neural Beamforming
//!
//! This module defines the output types from neural network-enhanced beamforming
//! including reconstructed volumes, uncertainty quantification, clinical analysis
//! results, and performance metrics.
//!
//! # Architecture
//!
//! Types are organized by concern:
//! - **Result Types**: Complete analysis results with all components
//! - **Feature Types**: Extracted features for clinical analysis
//! - **Clinical Types**: Lesion detection and tissue classification
//! - **Performance Types**: Timing and resource utilization metrics
//!
//! # Literature References
//!
//! - Stavros et al. (1995): "Solid breast nodules: use of sonography to distinguish"
//! - Burnside et al. (2007): "Differentiating benign from malignant solid breast masses"

use ndarray::Array3;
use std::collections::HashMap;

/// Result from Neural Beamforming
///
/// Complete analysis result including reconstructed volume, uncertainty
/// quantification, extracted features, clinical analysis, and performance metrics.
///
/// # Components
///
/// - **volume**: Beamformed ultrasound volume (post-processing)
/// - **uncertainty**: Epistemic uncertainty from PINN inference
/// - **confidence**: Model confidence scores per voxel
/// - **features**: Multi-scale morphological, spectral, and texture features
/// - **clinical_analysis**: Automated lesion detection and tissue classification
/// - **performance**: Processing time and resource utilization
#[derive(Debug)]
pub struct AIBeamformingResult {
    /// Reconstructed volume from beamforming [x, y, z]
    pub volume: Array3<f32>,

    /// Uncertainty map from PINN inference [x, y, z]
    ///
    /// Values in range [0, 1] where higher values indicate greater epistemic
    /// uncertainty in the reconstruction.
    pub uncertainty: Array3<f32>,

    /// Confidence scores from clinical analysis [x, y, z]
    ///
    /// Values in range [0, 1] where higher values indicate greater model
    /// confidence in tissue classification and lesion detection.
    pub confidence: Array3<f32>,

    /// Extracted features for clinical analysis
    pub features: FeatureMap,

    /// Clinical findings and recommendations
    pub clinical_analysis: ClinicalAnalysis,

    /// Processing performance metrics
    pub performance: PerformanceMetrics,
}

/// Extracted Features for Clinical Analysis
///
/// Multi-scale feature maps organized by category. Each feature is a 3D array
/// aligned with the reconstructed volume.
///
/// # Feature Categories
///
/// - **Morphological**: Shape-based features (gradients, edges, boundaries)
/// - **Spectral**: Frequency-domain features (local frequency, bandwidth)
/// - **Texture**: Statistical texture features (speckle, homogeneity, entropy)
///
/// # Literature References
///
/// - Haralick et al. (1973): "Textural Features for Image Classification"
/// - Mallat (1989): "A theory for multiresolution signal decomposition"
#[derive(Debug)]
pub struct FeatureMap {
    /// Morphological features (size, shape, boundaries)
    ///
    /// Common keys:
    /// - "gradient_magnitude": Edge strength
    /// - "laplacian": Second-order derivatives (blobs, ridges)
    pub morphological: HashMap<String, Array3<f32>>,

    /// Spectral features (frequency content, bandwidth)
    ///
    /// Common keys:
    /// - "local_frequency": Dominant frequency per voxel
    /// - "bandwidth": Spectral width
    pub spectral: HashMap<String, Array3<f32>>,

    /// Texture features (speckle statistics, homogeneity)
    ///
    /// Common keys:
    /// - "speckle_variance": Speckle pattern variability
    /// - "homogeneity": Local intensity uniformity
    pub texture: HashMap<String, Array3<f32>>,
}

impl FeatureMap {
    /// Create empty feature map
    pub fn new() -> Self {
        Self {
            morphological: HashMap::new(),
            spectral: HashMap::new(),
            texture: HashMap::new(),
        }
    }

    /// Get total number of features across all categories
    pub fn feature_count(&self) -> usize {
        self.morphological.len() + self.spectral.len() + self.texture.len()
    }

    /// Check if feature map is empty
    pub fn is_empty(&self) -> bool {
        self.feature_count() == 0
    }
}

impl Default for FeatureMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Clinical Analysis Results
///
/// Complete clinical decision support output including automated lesion
/// detection, tissue classification, and diagnostic recommendations.
///
/// # Clinical Safety
///
/// All neural network results are intended for decision support only and require clinical
/// interpretation by qualified medical professionals.
///
/// # Literature References
///
/// - Stavros et al. (1995): "Solid breast nodules: use of sonography"
/// - D'Orsi et al. (2013): "ACR BI-RADS Atlas, Breast Imaging Reporting"
#[derive(Debug)]
pub struct ClinicalAnalysis {
    /// Detected lesions with confidence scores
    pub lesions: Vec<LesionDetection>,

    /// Tissue classification results
    pub tissue_classification: TissueClassification,

    /// Clinical recommendations for follow-up
    pub recommendations: Vec<String>,

    /// Overall diagnostic confidence score [0, 1]
    ///
    /// Aggregate confidence across all detected findings.
    /// Low confidence suggests need for repeat imaging or alternative modality.
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

    /// Get count of high-confidence lesions (confidence > 0.8)
    pub fn high_confidence_lesion_count(&self) -> usize {
        self.lesions.iter().filter(|l| l.confidence > 0.8).count()
    }
}

/// Detected Lesion Information
///
/// Complete characterization of a detected lesion including location, size,
/// type classification, and clinical significance assessment.
///
/// # Literature References
///
/// - Stavros et al. (1995): "Solid breast nodules: use of sonography"
/// - Hong et al. (2005): "Correlation of US findings with histology"
#[derive(Debug, Clone)]
pub struct LesionDetection {
    /// Lesion center coordinates (x, y, z) in voxel space
    pub center: (usize, usize, usize),

    /// Lesion size (diameter in mm)
    ///
    /// Computed from 3D connected component analysis with physical
    /// voxel size calibration.
    pub size_mm: f32,

    /// Detection confidence (0-1)
    ///
    /// Confidence score from multi-feature analysis. Higher values
    /// indicate more reliable detection.
    pub confidence: f32,

    /// Lesion type classification
    ///
    /// Common types: "cyst", "solid", "complex", "calcification"
    pub lesion_type: String,

    /// Clinical significance score (0-1)
    ///
    /// Estimates clinical importance based on size, characteristics,
    /// and confidence. Higher scores suggest priority for follow-up.
    pub clinical_significance: f32,
}

impl LesionDetection {
    /// Check if lesion requires immediate clinical attention
    ///
    /// Criteria: size > 10mm OR clinical significance > 0.8
    pub fn requires_urgent_attention(&self) -> bool {
        self.size_mm > 10.0 || self.clinical_significance > 0.8
    }

    /// Get risk category based on size and characteristics
    pub fn risk_category(&self) -> &str {
        if self.clinical_significance > 0.8 {
            "HIGH"
        } else if self.clinical_significance > 0.5 {
            "MODERATE"
        } else {
            "LOW"
        }
    }
}

/// Tissue Classification Results
///
/// Probabilistic tissue type classification for each voxel with
/// uncertainty quantification and boundary confidence.
///
/// # Literature References
///
/// - Noble & Boukerroui (2006): "Ultrasound image segmentation: a survey"
/// - Huang & Chen (2004): "Breast ultrasound image segmentation"
#[derive(Debug)]
pub struct TissueClassification {
    /// Tissue type probabilities per voxel
    ///
    /// Common tissue types: "normal", "benign", "suspicious", "cyst"
    /// Each value is a 3D probability map [x, y, z] in range [0, 1]
    pub probabilities: HashMap<String, Array3<f32>>,

    /// Dominant tissue type per region [x, y, z]
    ///
    /// String label of most probable tissue type at each voxel
    pub dominant_tissue: Array3<String>,

    /// Tissue boundary confidence [x, y, z]
    ///
    /// Confidence in tissue boundary delineation. Low values indicate
    /// uncertain boundaries requiring clinical interpretation.
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

/// Performance Metrics for Real-Time Processing
///
/// Comprehensive timing and resource utilization metrics for monitoring
/// and optimization of the AI-enhanced beamforming pipeline.
///
/// # Performance Requirements
///
/// Total processing time target: <100ms for real-time clinical use
/// - Beamforming: <30ms
/// - Feature extraction: <20ms
/// - PINN inference: <30ms
/// - Clinical analysis: <20ms
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total processing time (ms)
    pub total_time_ms: f64,

    /// Beamforming time (ms)
    pub beamforming_time_ms: f64,

    /// Feature extraction time (ms)
    pub feature_extraction_time_ms: f64,

    /// PINN inference time (ms)
    pub pinn_inference_time_ms: f64,

    /// Clinical analysis time (ms)
    pub clinical_analysis_time_ms: f64,

    /// Memory usage (MB)
    pub memory_usage_mb: f64,

    /// GPU utilization (percent)
    pub gpu_utilization_percent: f64,
}

impl PerformanceMetrics {
    /// Check if performance meets real-time target (<100ms)
    pub fn meets_realtime_target(&self) -> bool {
        self.total_time_ms < 100.0
    }

    /// Get breakdown of time by component (as percentages)
    pub fn time_breakdown(&self) -> HashMap<String, f64> {
        let mut breakdown = HashMap::new();
        if self.total_time_ms > 0.0 {
            breakdown.insert(
                "beamforming".to_string(),
                (self.beamforming_time_ms / self.total_time_ms) * 100.0,
            );
            breakdown.insert(
                "features".to_string(),
                (self.feature_extraction_time_ms / self.total_time_ms) * 100.0,
            );
            breakdown.insert(
                "pinn".to_string(),
                (self.pinn_inference_time_ms / self.total_time_ms) * 100.0,
            );
            breakdown.insert(
                "clinical".to_string(),
                (self.clinical_analysis_time_ms / self.total_time_ms) * 100.0,
            );
        }
        breakdown
    }

    /// Identify performance bottleneck (slowest component)
    pub fn bottleneck(&self) -> &str {
        let mut max_time = self.beamforming_time_ms;
        let mut bottleneck = "beamforming";

        if self.feature_extraction_time_ms > max_time {
            max_time = self.feature_extraction_time_ms;
            bottleneck = "feature_extraction";
        }
        if self.pinn_inference_time_ms > max_time {
            max_time = self.pinn_inference_time_ms;
            bottleneck = "pinn_inference";
        }
        if self.clinical_analysis_time_ms > max_time {
            bottleneck = "clinical_analysis";
        }

        bottleneck
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_map_creation() {
        let mut features = FeatureMap::new();
        assert!(features.is_empty());
        assert_eq!(features.feature_count(), 0);

        features
            .morphological
            .insert("gradient".to_string(), Array3::zeros((10, 10, 10)));
        assert!(!features.is_empty());
        assert_eq!(features.feature_count(), 1);
    }

    #[test]
    fn test_lesion_detection_risk_assessment() {
        let lesion_low = LesionDetection {
            center: (5, 5, 5),
            size_mm: 3.0,
            confidence: 0.7,
            lesion_type: "cyst".to_string(),
            clinical_significance: 0.3,
        };
        assert!(!lesion_low.requires_urgent_attention());
        assert_eq!(lesion_low.risk_category(), "LOW");

        let lesion_high = LesionDetection {
            center: (5, 5, 5),
            size_mm: 15.0,
            confidence: 0.9,
            lesion_type: "solid".to_string(),
            clinical_significance: 0.85,
        };
        assert!(lesion_high.requires_urgent_attention());
        assert_eq!(lesion_high.risk_category(), "HIGH");
    }

    #[test]
    fn test_clinical_analysis_lesion_counting() {
        let mut analysis = ClinicalAnalysis::empty();
        assert!(!analysis.has_lesions());
        assert_eq!(analysis.high_confidence_lesion_count(), 0);

        analysis.lesions.push(LesionDetection {
            center: (5, 5, 5),
            size_mm: 5.0,
            confidence: 0.9,
            lesion_type: "solid".to_string(),
            clinical_significance: 0.7,
        });

        analysis.lesions.push(LesionDetection {
            center: (10, 10, 10),
            size_mm: 3.0,
            confidence: 0.6,
            lesion_type: "cyst".to_string(),
            clinical_significance: 0.4,
        });

        assert!(analysis.has_lesions());
        assert_eq!(analysis.high_confidence_lesion_count(), 1);
    }

    #[test]
    fn test_tissue_classification_queries() {
        let mut classification = TissueClassification::empty();
        assert!(!classification.has_suspicious_tissue());
        assert!(classification.tissue_types().is_empty());

        classification
            .probabilities
            .insert("normal".to_string(), Array3::zeros((5, 5, 5)));
        classification
            .probabilities
            .insert("suspicious".to_string(), Array3::zeros((5, 5, 5)));

        assert!(classification.has_suspicious_tissue());
        assert_eq!(classification.tissue_types().len(), 2);
    }

    #[test]
    fn test_performance_metrics_analysis() {
        let metrics = PerformanceMetrics {
            total_time_ms: 150.0,
            beamforming_time_ms: 40.0,
            feature_extraction_time_ms: 30.0,
            pinn_inference_time_ms: 50.0,
            clinical_analysis_time_ms: 30.0,
            memory_usage_mb: 128.0,
            gpu_utilization_percent: 75.0,
        };

        assert!(!metrics.meets_realtime_target());
        assert_eq!(metrics.bottleneck(), "pinn_inference");

        let breakdown = metrics.time_breakdown();
        assert!((breakdown["beamforming"] - 26.67).abs() < 0.1);
        assert!((breakdown["pinn"] - 33.33).abs() < 0.1);
    }

    #[test]
    fn test_performance_metrics_realtime() {
        let metrics = PerformanceMetrics {
            total_time_ms: 85.0,
            beamforming_time_ms: 25.0,
            feature_extraction_time_ms: 20.0,
            pinn_inference_time_ms: 25.0,
            clinical_analysis_time_ms: 15.0,
            memory_usage_mb: 64.0,
            gpu_utilization_percent: 60.0,
        };

        assert!(metrics.meets_realtime_target());
    }
}
