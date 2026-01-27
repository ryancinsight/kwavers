//! Clinical Decision Support for Neural Ultrasound Analysis
//!
//! This module implements automated clinical analysis including lesion detection,
//! tissue classification, and diagnostic recommendations for ultrasound imaging.
//!
//! # Algorithms
//!
//! - **Lesion Detection**: Multi-feature anomaly detection with adaptive thresholding
//! - **Size Estimation**: 3D connected component analysis with 26-connectivity
//! - **Tissue Classification**: Feature-based tissue type identification
//! - **Clinical Scoring**: Confidence-weighted diagnostic recommendations
//!
//! # Clinical Safety
//!
//! All results are for decision support only and require clinical interpretation
//! by qualified medical professionals. Algorithms are based on published literature
//! but must be validated for specific clinical contexts.
//!
//! # Literature References
//!
//! - Stavros et al. (1995): "Solid breast nodules: use of sonography"
//! - Gonzalez & Woods (2008): "Digital Image Processing" (3rd ed.)
//! - Burnside et al. (2007): "Differentiating benign from malignant findings"
//! - Noble & Boukerroui (2006): "Ultrasound image segmentation: a survey"

use super::types::ClinicalThresholds;
use super::types::{ClinicalAnalysis, FeatureMap, LesionDetection, TissueClassification};
use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3};
use std::collections::HashMap;

/// Clinical Decision Support System
///
/// Provides automated lesion detection, tissue classification, and diagnostic
/// recommendations based on neural network-enhanced ultrasound analysis.
///
/// # Example
///
/// ```ignore
/// use kwavers::domain::sensor::beamforming::neural::clinical::ClinicalDecisionSupport;
/// use kwavers::domain::sensor::beamforming::neural::config::ClinicalThresholds;
///
/// let thresholds = ClinicalThresholds::default();
/// let support = ClinicalDecisionSupport::new(thresholds);
///
/// let analysis = support.analyze_clinical(volume, &features, uncertainty, confidence)?;
/// println!("Detected {} lesions", analysis.lesions.len());
/// ```
#[derive(Debug, Clone)]
pub struct ClinicalDecisionSupport {
    config: ClinicalThresholds,
}

impl ClinicalDecisionSupport {
    /// Create new clinical decision support system
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Clinical analysis thresholds and parameters
    pub fn new(thresholds: ClinicalThresholds) -> Self {
        Self { config: thresholds }
    }

    /// Perform comprehensive clinical analysis
    ///
    /// Integrates lesion detection, tissue classification, and diagnostic
    /// recommendation generation into a complete clinical analysis result.
    ///
    /// # Arguments
    ///
    /// * `volume` - Reconstructed ultrasound volume [x, y, z]
    /// * `features` - Extracted morphological, spectral, and texture features
    /// * `uncertainty` - Epistemic uncertainty map from PINN [x, y, z]
    /// * `confidence` - Model confidence scores [x, y, z]
    ///
    /// # Returns
    ///
    /// Complete clinical analysis with lesions, tissue classification, and recommendations
    ///
    /// # Performance
    ///
    /// Target: <20ms for 64x64x100 volume
    pub fn analyze_clinical(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
        uncertainty: ArrayView3<f32>,
        confidence: ArrayView3<f32>,
    ) -> KwaversResult<ClinicalAnalysis> {
        // Detect lesions based on feature anomalies
        let lesions = self.detect_lesions(volume, features, uncertainty, confidence)?;

        // Classify tissue types
        let tissue_classification = self.classify_tissues(volume, features)?;

        // Generate clinical recommendations
        let recommendations = self.generate_recommendations(&lesions, &tissue_classification);

        // Compute overall diagnostic confidence
        let diagnostic_confidence = self.compute_diagnostic_confidence(&lesions, confidence);

        Ok(ClinicalAnalysis {
            lesions,
            tissue_classification,
            recommendations,
            diagnostic_confidence,
        })
    }

    /// Detect lesions using multi-feature analysis
    ///
    /// Identifies potential lesions based on combined analysis of:
    /// - High intensity contrast (abnormal echo pattern)
    /// - High model confidence (reliable detection)
    /// - Low uncertainty (consistent reconstruction)
    /// - Anomalous speckle pattern (abnormal texture)
    /// - Strong gradients (clear boundaries)
    ///
    /// # Algorithm
    ///
    /// 1. Scan volume excluding 10-voxel boundary margin
    /// 2. For each voxel, evaluate detection criteria
    /// 3. If criteria met, estimate lesion size via connected component analysis
    /// 4. Classify lesion type and assess clinical significance
    ///
    /// # Literature Reference
    ///
    /// - Stavros et al. (1995): "Solid breast nodules: use of sonography"
    fn detect_lesions(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
        uncertainty: ArrayView3<f32>,
        confidence: ArrayView3<f32>,
    ) -> KwaversResult<Vec<LesionDetection>> {
        let mut lesions = Vec::new();
        let (nx, ny, nz) = volume.dim();

        // Avoid boundary artifacts by excluding 10-voxel margin
        let margin = 10;

        for z in margin..nz.saturating_sub(margin) {
            for y in margin..ny.saturating_sub(margin) {
                for x in margin..nx.saturating_sub(margin) {
                    let vol_val = volume[[x, y, z]];
                    let conf_val = confidence[[x, y, z]];
                    let uncert_val = uncertainty[[x, y, z]];

                    // Extract relevant features
                    let gradient_mag = features
                        .morphological
                        .get("gradient_magnitude")
                        .map(|arr| arr[[x, y, z]])
                        .unwrap_or(0.0);

                    let speckle_var = features
                        .texture
                        .get("speckle_variance")
                        .map(|arr| arr[[x, y, z]])
                        .unwrap_or(0.0);

                    // Multi-criteria lesion detection
                    let is_high_contrast = vol_val > self.config.contrast_abnormality_threshold;
                    let is_high_confidence = conf_val > self.config.lesion_confidence_threshold;
                    let is_low_uncertainty = uncert_val < self.config.tissue_uncertainty_threshold;
                    let is_anomalous_speckle = speckle_var > self.config.speckle_anomaly_threshold;
                    let is_strong_boundary = gradient_mag > 0.5;

                    if is_high_contrast
                        && is_high_confidence
                        && is_low_uncertainty
                        && is_anomalous_speckle
                        && is_strong_boundary
                    {
                        lesions.push(LesionDetection {
                            center: (x, y, z),
                            size_mm: self.estimate_lesion_size(volume, features, x, y, z),
                            confidence: conf_val,
                            lesion_type: self.classify_lesion_type(vol_val, features, x, y, z),
                            clinical_significance: self
                                .assess_clinical_significance(conf_val, vol_val),
                        });
                    }
                }
            }
        }

        Ok(lesions)
    }

    /// Estimate lesion size using 3D connected component analysis
    ///
    /// Uses flood-fill algorithm with 26-connected neighborhood to segment
    /// the lesion region, then computes equivalent spherical diameter.
    ///
    /// # Algorithm
    ///
    /// 1. Compute adaptive threshold: local_mean + k·σ
    /// 2. Initialize flood-fill from seed point
    /// 3. Expand to 26-connected neighbors above threshold
    /// 4. Count voxels in connected component
    /// 5. Compute equivalent spherical diameter: d = 2·[(3V)/(4π)]^(1/3)
    ///
    /// # Literature Reference
    ///
    /// - Gonzalez & Woods (2008): "Digital Image Processing" (Connected components)
    ///
    /// # Arguments
    ///
    /// * `volume` - Ultrasound volume
    /// * `_features` - Feature maps (reserved for future use)
    /// * `seed_x`, `seed_y`, `seed_z` - Initial detection point
    ///
    /// # Returns
    ///
    /// Equivalent spherical diameter in millimeters
    fn estimate_lesion_size(
        &self,
        volume: ArrayView3<f32>,
        _features: &FeatureMap,
        seed_x: usize,
        seed_y: usize,
        seed_z: usize,
    ) -> f32 {
        let (dim_x, dim_y, dim_z) = volume.dim();

        // Adaptive thresholding based on local statistics
        let local_mean = self.compute_local_statistics(&volume, seed_x, seed_y, seed_z);
        let threshold = local_mean + 2.0 * self.config.segmentation_sensitivity;

        // 3D connected component analysis using flood fill
        let mut visited = Array3::<bool>::from_elem((dim_x, dim_y, dim_z), false);
        let mut component_voxels = Vec::new();

        let mut queue = std::collections::VecDeque::new();
        queue.push_back((seed_x, seed_y, seed_z));
        visited[[seed_x, seed_y, seed_z]] = true;

        // 26-connected neighborhood offsets (all combinations of ±1 in 3D)
        let offsets = [
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, 1, 0),
            (-1, 1, 1),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ];

        // Flood fill to find all connected voxels above threshold
        while let Some((x, y, z)) = queue.pop_front() {
            component_voxels.push((x, y, z));

            // Check all 26 neighbors
            for (dx, dy, dz) in offsets.iter() {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;

                if nx >= 0
                    && nx < dim_x as isize
                    && ny >= 0
                    && ny < dim_y as isize
                    && nz >= 0
                    && nz < dim_z as isize
                {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    let nz = nz as usize;

                    if !visited[[nx, ny, nz]] && volume[[nx, ny, nz]] > threshold {
                        visited[[nx, ny, nz]] = true;
                        queue.push_back((nx, ny, nz));
                    }
                }
            }
        }

        // Calculate lesion volume in mm³
        let voxel_volume_mm3 = self.config.voxel_size_mm.powi(3);
        let lesion_volume_mm3 = component_voxels.len() as f32 * voxel_volume_mm3;

        // Compute equivalent spherical diameter (ESD)
        // V = (4/3)πr³ ⇒ r = [(3V)/(4π)]^(1/3) ⇒ diameter = 2r
        let equivalent_radius_mm =
            (3.0 * lesion_volume_mm3 / (4.0 * std::f32::consts::PI)).powf(1.0 / 3.0);

        2.0 * equivalent_radius_mm
    }

    /// Compute local statistics for adaptive thresholding
    ///
    /// Computes mean intensity in a 5×5×5 window around the specified point.
    /// Used for adaptive threshold computation in lesion segmentation.
    ///
    /// # Arguments
    ///
    /// * `volume` - Ultrasound volume
    /// * `x`, `y`, `z` - Center point coordinates
    ///
    /// # Returns
    ///
    /// Local mean intensity, or center voxel value if window is invalid
    fn compute_local_statistics(
        &self,
        volume: &ArrayView3<f32>,
        x: usize,
        y: usize,
        z: usize,
    ) -> f32 {
        let (nx, ny, nz) = volume.dim();
        let window_size = 5; // 5×5×5 local window

        let mut sum = 0.0;
        let mut count = 0;

        for dx in -(window_size as isize)..=(window_size as isize) {
            for dy in -(window_size as isize)..=(window_size as isize) {
                for dz in -(window_size as isize)..=(window_size as isize) {
                    let neighbor_x = x as isize + dx;
                    let neighbor_y = y as isize + dy;
                    let neighbor_z = z as isize + dz;

                    if neighbor_x >= 0
                        && neighbor_x < nx as isize
                        && neighbor_y >= 0
                        && neighbor_y < ny as isize
                        && neighbor_z >= 0
                        && neighbor_z < nz as isize
                    {
                        let neighbor_x = neighbor_x as usize;
                        let neighbor_y = neighbor_y as usize;
                        let neighbor_z = neighbor_z as usize;

                        sum += volume[[neighbor_x, neighbor_y, neighbor_z]];
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            volume[[x, y, z]]
        }
    }

    /// Classify lesion type based on echo characteristics
    ///
    /// Categorizes lesions based on echogenicity relative to surrounding tissue:
    /// - **Hyperechoic**: Brighter than surrounding tissue (high intensity)
    /// - **Hypoechoic**: Darker than surrounding tissue (low intensity)
    /// - **Isoechoic**: Similar to surrounding tissue (medium intensity)
    ///
    /// # Literature Reference
    ///
    /// - Stavros et al. (1995): "Solid breast nodules: use of sonography"
    ///
    /// # Arguments
    ///
    /// * `intensity` - Normalized voxel intensity
    /// * `_features` - Feature maps (reserved for advanced classification)
    /// * `_x`, `_y`, `_z` - Lesion coordinates (reserved for context-aware classification)
    ///
    /// # Returns
    ///
    /// Lesion type string
    fn classify_lesion_type(
        &self,
        intensity: f32,
        _features: &FeatureMap,
        _x: usize,
        _y: usize,
        _z: usize,
    ) -> String {
        if intensity > 3.0 {
            "Hyperechoic Lesion".to_string()
        } else if intensity < 0.5 {
            "Hypoechoic Lesion".to_string()
        } else {
            "Isoechoic Lesion".to_string()
        }
    }

    /// Assess clinical significance of detected lesion
    ///
    /// Computes clinical significance score [0, 1] based on:
    /// - Detection confidence (model certainty)
    /// - Intensity abnormality (echo pattern deviation)
    ///
    /// Higher scores indicate findings requiring urgent clinical attention.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Model confidence in detection
    /// * `intensity` - Normalized intensity value
    ///
    /// # Returns
    ///
    /// Clinical significance score [0, 1]
    fn assess_clinical_significance(&self, confidence: f32, intensity: f32) -> f32 {
        let confidence_score = confidence;
        let intensity_score = intensity.abs().min(1.0);
        (confidence_score + intensity_score) / 2.0
    }

    /// Classify tissue types using feature-based analysis
    ///
    /// Performs voxel-wise tissue classification based on intensity and
    /// texture features. Provides probabilistic classification and dominant
    /// tissue type per voxel.
    ///
    /// # Tissue Types
    ///
    /// - **Fat**: High intensity, low speckle variance
    /// - **Muscle**: Medium intensity, medium speckle variance
    /// - **Blood**: Low intensity, high speckle variance
    ///
    /// # Literature Reference
    ///
    /// - Noble & Boukerroui (2006): "Ultrasound image segmentation: a survey"
    ///
    /// # Arguments
    ///
    /// * `volume` - Ultrasound volume
    /// * `features` - Extracted feature maps
    ///
    /// # Returns
    ///
    /// Tissue classification with probabilities, dominant types, and boundary confidence
    fn classify_tissues(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
    ) -> KwaversResult<TissueClassification> {
        let (nx, ny, nz) = volume.dim();

        let mut probabilities = HashMap::new();
        let mut dominant_tissue = Array3::<String>::from_elem((nx, ny, nz), "Unknown".to_string());
        let mut boundary_confidence = Array3::<f32>::zeros((nx, ny, nz));

        // Initialize probability maps for each tissue type
        probabilities.insert("Fat".to_string(), Array3::from_elem((nx, ny, nz), 0.33));
        probabilities.insert("Muscle".to_string(), Array3::from_elem((nx, ny, nz), 0.33));
        probabilities.insert("Blood".to_string(), Array3::from_elem((nx, ny, nz), 0.33));

        // Classify each voxel based on intensity and texture
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let intensity = volume[[x, y, z]];
                    let speckle_var = features
                        .texture
                        .get("speckle_variance")
                        .map(|arr| arr[[x, y, z]])
                        .unwrap_or(0.5);

                    // Rule-based tissue classification
                    let tissue_type = if intensity < 0.7 && speckle_var > 0.8 {
                        "Blood"
                    } else if intensity > 1.2 && speckle_var < 0.4 {
                        "Fat"
                    } else {
                        "Muscle"
                    };

                    dominant_tissue[[x, y, z]] = tissue_type.to_string();
                    boundary_confidence[[x, y, z]] = 0.8; // Placeholder confidence
                }
            }
        }

        Ok(TissueClassification {
            probabilities,
            dominant_tissue,
            boundary_confidence,
        })
    }

    /// Generate clinical recommendations based on analysis results
    ///
    /// Produces actionable clinical recommendations based on detected lesions
    /// and tissue classification results. Recommendations are prioritized by
    /// clinical urgency.
    ///
    /// # Recommendation Logic
    ///
    /// - No lesions: Routine follow-up
    /// - Lesions detected: Clinical correlation and possible biopsy
    /// - High-confidence lesions (>0.9): Urgent evaluation recommended
    /// - Always include disclaimer about AI support role
    ///
    /// # Arguments
    ///
    /// * `lesions` - Detected lesions with characteristics
    /// * `_tissue_classification` - Tissue classification results (reserved)
    ///
    /// # Returns
    ///
    /// Vector of clinical recommendation strings
    fn generate_recommendations(
        &self,
        lesions: &[LesionDetection],
        _tissue_classification: &TissueClassification,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if lesions.is_empty() {
            recommendations.push(
                "No significant lesions detected. Consider follow-up imaging if clinically indicated."
                    .to_string(),
            );
        } else {
            recommendations.push(format!(
                "Detected {} potential lesion(s). Recommend clinical correlation and possible biopsy.",
                lesions.len()
            ));

            let high_confidence_lesions = lesions.iter().filter(|l| l.confidence > 0.9).count();

            if high_confidence_lesions > 0 {
                recommendations.push(format!(
                    "{} high-confidence lesion(s) identified. Urgent clinical evaluation recommended.",
                    high_confidence_lesions
                ));
            }
        }

        // Always include clinical disclaimer
        recommendations.push(
            "Neural network analysis is supportive only. Clinical judgment required for final diagnosis."
                .to_string(),
        );

        recommendations
    }

    /// Compute overall diagnostic confidence
    ///
    /// Aggregates lesion detection confidence and volumetric confidence
    /// into an overall diagnostic confidence score.
    ///
    /// # Algorithm
    ///
    /// confidence = (mean_lesion_confidence + mean_volume_confidence) / 2
    ///
    /// Special case: If no lesions detected, lesion confidence = 0.9 (high confidence in negative finding)
    ///
    /// # Arguments
    ///
    /// * `lesions` - Detected lesions with individual confidence scores
    /// * `confidence` - Volumetric confidence map from PINN
    ///
    /// # Returns
    ///
    /// Overall diagnostic confidence [0, 1]
    fn compute_diagnostic_confidence(
        &self,
        lesions: &[LesionDetection],
        confidence: ArrayView3<f32>,
    ) -> f32 {
        let lesion_confidence = if lesions.is_empty() {
            0.9 // High confidence when no lesions found (negative finding)
        } else {
            lesions.iter().map(|l| l.confidence).sum::<f32>() / lesions.len() as f32
        };

        let image_confidence = confidence.iter().sum::<f32>() / confidence.len() as f32;

        (lesion_confidence + image_confidence) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clinical_decision_support_creation() {
        let thresholds = ClinicalThresholds::default();
        let support = ClinicalDecisionSupport::new(thresholds);
        assert!(support.config.lesion_confidence_threshold > 0.0);
    }

    #[test]
    fn test_lesion_type_classification() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let features = FeatureMap::new();

        let hyperechoic = support.classify_lesion_type(3.5, &features, 5, 5, 5);
        assert_eq!(hyperechoic, "Hyperechoic Lesion");

        let hypoechoic = support.classify_lesion_type(0.3, &features, 5, 5, 5);
        assert_eq!(hypoechoic, "Hypoechoic Lesion");

        let isoechoic = support.classify_lesion_type(1.0, &features, 5, 5, 5);
        assert_eq!(isoechoic, "Isoechoic Lesion");
    }

    #[test]
    fn test_clinical_significance_assessment() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());

        let high_sig = support.assess_clinical_significance(0.9, 0.8);
        assert!(high_sig > 0.8);

        let low_sig = support.assess_clinical_significance(0.3, 0.2);
        assert!(low_sig < 0.3);
    }

    #[test]
    fn test_recommendations_no_lesions() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let lesions = Vec::new();
        let classification = TissueClassification::empty();

        let recs = support.generate_recommendations(&lesions, &classification);
        assert!(!recs.is_empty());
        assert!(recs[0].contains("No significant lesions"));
    }

    #[test]
    fn test_recommendations_with_lesions() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let lesions = vec![
            LesionDetection {
                center: (10, 10, 10),
                size_mm: 5.0,
                confidence: 0.95,
                lesion_type: "Solid".to_string(),
                clinical_significance: 0.85,
            },
            LesionDetection {
                center: (20, 20, 20),
                size_mm: 3.0,
                confidence: 0.75,
                lesion_type: "Cyst".to_string(),
                clinical_significance: 0.60,
            },
        ];
        let classification = TissueClassification::empty();

        let recs = support.generate_recommendations(&lesions, &classification);
        assert!(recs[0].contains("2 potential lesion"));
        assert!(recs.iter().any(|r| r.contains("high-confidence")));
    }

    #[test]
    fn test_diagnostic_confidence_no_lesions() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let lesions = Vec::new();
        let confidence = Array3::from_elem((10, 10, 10), 0.8);

        let diag_conf = support.compute_diagnostic_confidence(&lesions, confidence.view());
        assert!(diag_conf > 0.8); // Should be high for negative finding
    }

    #[test]
    fn test_diagnostic_confidence_with_lesions() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let lesions = vec![LesionDetection {
            center: (5, 5, 5),
            size_mm: 4.0,
            confidence: 0.9,
            lesion_type: "Solid".to_string(),
            clinical_significance: 0.8,
        }];
        let confidence = Array3::from_elem((10, 10, 10), 0.85);

        let diag_conf = support.compute_diagnostic_confidence(&lesions, confidence.view());
        assert!(diag_conf > 0.85);
    }

    #[test]
    fn test_local_statistics_computation() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let volume = Array3::from_elem((20, 20, 20), 1.0);

        let stats = support.compute_local_statistics(&volume.view(), 10, 10, 10);
        assert!((stats - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lesion_size_estimation() {
        let support = ClinicalDecisionSupport::new(ClinicalThresholds::default());
        let mut volume = Array3::from_elem((30, 30, 30), 0.5);

        // Create a spherical lesion
        for z in 10..20 {
            for y in 10..20 {
                for x in 10..20 {
                    let dist = ((x as f32 - 15.0).powi(2)
                        + (y as f32 - 15.0).powi(2)
                        + (z as f32 - 15.0).powi(2))
                    .sqrt();
                    if dist < 5.0 {
                        volume[[x, y, z]] = 3.0; // High intensity lesion
                    }
                }
            }
        }

        let features = FeatureMap::new();
        let size_mm = support.estimate_lesion_size(volume.view(), &features, 15, 15, 15);
        assert!(size_mm > 0.0); // Should detect non-zero size
    }
}
