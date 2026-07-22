use super::super::types::{ClinicalAnalysis, FeatureMap, LesionDetection, TissueClassification};
use super::NeuralClinicalDecisionSupport;
use kwavers_core::error::KwaversResult;
use leto::{Array3, ArrayView3};
use std::collections::HashMap;

impl NeuralClinicalDecisionSupport {
    /// Perform comprehensive clinical analysis: lesion detection, tissue classification,
    /// recommendations, and overall diagnostic confidence.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn analyze_clinical(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
        uncertainty: ArrayView3<f32>,
        confidence: ArrayView3<f32>,
    ) -> KwaversResult<ClinicalAnalysis> {
        let lesions = self.detect_lesions(volume, features, uncertainty, confidence)?;
        let tissue_classification = self.classify_tissues(volume, features)?;
        let recommendations = self.generate_recommendations(&lesions, &tissue_classification);
        let diagnostic_confidence = self.compute_diagnostic_confidence(&lesions, confidence);

        Ok(ClinicalAnalysis {
            lesions,
            tissue_classification,
            recommendations,
            diagnostic_confidence,
        })
    }

    /// Voxel-wise tissue classification: Fat, Muscle, Blood.
    ///
    /// Reference: Noble & Boukerroui (2006), "Ultrasound image segmentation: a survey".
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn classify_tissues(
        &self,
        volume: ArrayView3<f32>,
        features: &FeatureMap,
    ) -> KwaversResult<TissueClassification> {
        let [nx, ny, nz] = volume.shape();

        let mut probabilities = HashMap::new();
        let mut dominant_tissue = Array3::<String>::from_elem((nx, ny, nz), "Unknown".to_owned());
        let mut boundary_confidence = Array3::<f32>::zeros((nx, ny, nz));

        probabilities.insert("Fat".to_owned(), Array3::from_elem((nx, ny, nz), 0.33));
        probabilities.insert("Muscle".to_owned(), Array3::from_elem((nx, ny, nz), 0.33));
        probabilities.insert("Blood".to_owned(), Array3::from_elem((nx, ny, nz), 0.33));

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let intensity = volume[[x, y, z]];
                    let speckle_var = features
                        .texture
                        .get("speckle_variance")
                        .map_or(0.5, |arr| arr[[x, y, z]]);

                    let tissue_type = if intensity < 0.7 && speckle_var > 0.8 {
                        "Blood"
                    } else if intensity > 1.2 && speckle_var < 0.4 {
                        "Fat"
                    } else {
                        "Muscle"
                    };

                    dominant_tissue[[x, y, z]] = tissue_type.to_owned();

                    let grad_x = if x > 0 && x < nx - 1 {
                        (volume[[x + 1, y, z]] - volume[[x - 1, y, z]]) / 2.0
                    } else {
                        0.0
                    };
                    let grad_y = if y > 0 && y < ny - 1 {
                        (volume[[x, y + 1, z]] - volume[[x, y - 1, z]]) / 2.0
                    } else {
                        0.0
                    };
                    let grad_z = if z > 0 && z < nz - 1 {
                        (volume[[x, y, z + 1]] - volume[[x, y, z - 1]]) / 2.0
                    } else {
                        0.0
                    };
                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                    boundary_confidence[[x, y, z]] = 1.0 / 5.0f32.mul_add(grad_mag, 1.0);
                }
            }
        }

        Ok(TissueClassification {
            probabilities,
            dominant_tissue,
            boundary_confidence,
        })
    }

    /// Generate clinical recommendations: routine follow-up or lesion correlation + urgency.
    pub(super) fn generate_recommendations(
        &self,
        lesions: &[LesionDetection],
        _tissue_classification: &TissueClassification,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if lesions.is_empty() {
            recommendations.push(
                "No significant lesions detected. Consider follow-up imaging if clinically indicated.".to_owned(),
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

        recommendations.push(
            "Neural network analysis is supportive only. Clinical judgment required for final diagnosis.".to_owned(),
        );

        recommendations
    }

    /// Aggregate lesion and volumetric confidence into overall diagnostic confidence [0, 1].
    ///
    /// No-lesion case uses 0.9 (high confidence in negative finding).
    pub(super) fn compute_diagnostic_confidence(
        &self,
        lesions: &[LesionDetection],
        confidence: ArrayView3<f32>,
    ) -> f32 {
        let lesion_confidence = if lesions.is_empty() {
            0.9
        } else {
            lesions.iter().map(|l| l.confidence).sum::<f32>() / lesions.len() as f32
        };

        let image_confidence = confidence.iter().sum::<f32>() / confidence.size() as f32;
        (lesion_confidence + image_confidence) / 2.0
    }
}