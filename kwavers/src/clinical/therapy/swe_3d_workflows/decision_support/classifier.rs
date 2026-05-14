//! `ClinicalDecisionSupport`: liver fibrosis and breast lesion classification.

use std::collections::HashMap;

use super::super::statistics::VolumetricStatistics;
use super::types::{
    BreastLesionClassification, ClassificationConfidence, FibrosisStage, LiverFibrosisStage,
    TissueReference,
};

/// Clinical decision support for 3D SWE classification.
#[derive(Debug)]
pub struct ClinicalDecisionSupport {
    _reference_ranges: HashMap<String, TissueReference>,
    classification_thresholds: HashMap<String, Vec<f64>>,
}

impl Default for ClinicalDecisionSupport {
    fn default() -> Self {
        let mut reference_ranges = HashMap::new();
        let mut classification_thresholds = HashMap::new();

        reference_ranges.insert(
            "liver_normal".to_owned(),
            TissueReference {
                mean_modulus: 5.5,
                std_modulus: 1.2,
                min_modulus: 3.0,
                max_modulus: 8.0,
            },
        );

        reference_ranges.insert(
            "liver_fibrosis_f4".to_owned(),
            TissueReference {
                mean_modulus: 15.0,
                std_modulus: 3.0,
                min_modulus: 10.0,
                max_modulus: 25.0,
            },
        );

        // Liver fibrosis METAVIR thresholds (kPa): F0F1|F2, F2|F3, F3|F4.
        classification_thresholds.insert("liver_metavir".to_owned(), vec![6.5, 8.5, 11.0]);

        reference_ranges.insert(
            "breast_normal".to_owned(),
            TissueReference {
                mean_modulus: 18.0,
                std_modulus: 5.0,
                min_modulus: 10.0,
                max_modulus: 30.0,
            },
        );

        reference_ranges.insert(
            "breast_malignant".to_owned(),
            TissueReference {
                mean_modulus: 120.0,
                std_modulus: 40.0,
                min_modulus: 50.0,
                max_modulus: 200.0,
            },
        );

        Self {
            _reference_ranges: reference_ranges,
            classification_thresholds,
        }
    }
}

impl ClinicalDecisionSupport {
    /// Classify liver fibrosis stage using METAVIR scoring.
    ///
    /// # Panics
    /// - Panics if `liver_metavir` thresholds are missing from the internal map.
    #[must_use]
    pub fn classify_liver_fibrosis(&self, stats: &VolumetricStatistics) -> LiverFibrosisStage {
        let thresholds = self.classification_thresholds.get("liver_metavir").unwrap();
        let mean_kpa = stats.mean_modulus / 1000.0;

        let stage = if mean_kpa < thresholds[0] {
            FibrosisStage::F0F1
        } else if mean_kpa < thresholds[1] {
            FibrosisStage::F2
        } else if mean_kpa < thresholds[2] {
            FibrosisStage::F3
        } else {
            FibrosisStage::F4
        };

        let confidence = if stats.std_modulus / stats.mean_modulus < 0.3 && stats.mean_quality > 0.7
        {
            ClassificationConfidence::High
        } else if stats.std_modulus / stats.mean_modulus < 0.5 && stats.mean_quality > 0.5 {
            ClassificationConfidence::Medium
        } else {
            ClassificationConfidence::Low
        };

        LiverFibrosisStage {
            stage,
            mean_stiffness_kpa: mean_kpa,
            confidence,
            quality_score: stats.mean_quality,
        }
    }

    /// Classify breast lesion using BI-RADS criteria.
    #[must_use]
    pub fn classify_breast_lesion(
        &self,
        stats: &VolumetricStatistics,
    ) -> BreastLesionClassification {
        let mean_kpa = stats.mean_modulus / 1000.0;

        let (birads_category, malignancy_probability) = if mean_kpa < 20.0 {
            (2, 0.0)
        } else if mean_kpa < 50.0 {
            (3, 0.02)
        } else if mean_kpa < 80.0 {
            (4, 0.3)
        } else {
            (5, 0.95)
        };

        let confidence = if stats.std_modulus / stats.mean_modulus < 0.4 && stats.mean_quality > 0.8
        {
            ClassificationConfidence::High
        } else if stats.std_modulus / stats.mean_modulus < 0.6 && stats.mean_quality > 0.6 {
            ClassificationConfidence::Medium
        } else {
            ClassificationConfidence::Low
        };

        BreastLesionClassification {
            birads_category,
            estimated_malignancy_probability: malignancy_probability,
            mean_stiffness_kpa: mean_kpa,
            confidence,
            quality_score: stats.mean_quality,
        }
    }

    /// Generate clinical report string for an organ.
    #[must_use]
    pub fn generate_report(&self, organ: &str, stats: &VolumetricStatistics) -> String {
        let mut report = format!("3D SWE Clinical Report - {}\n", organ.to_uppercase());
        report.push_str(&format!("={}\n\n", "=".repeat(40)));

        report.push_str("Volumetric Analysis:\n");
        report.push_str(&format!("- Valid voxels: {}\n", stats.valid_voxels));
        report.push_str(&format!(
            "- Volume coverage: {:.1}%\n",
            stats.volume_coverage * 100.0
        ));
        report.push_str(&format!("- Mean quality: {:.2}\n", stats.mean_quality));
        report.push_str(&format!(
            "- Mean confidence: {:.2}\n\n",
            stats.mean_confidence
        ));

        report.push_str("Elasticity Results:\n");
        report.push_str(&format!(
            "- Mean Young's modulus: {:.1} kPa\n",
            stats.mean_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Standard deviation: {:.1} kPa\n",
            stats.std_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Range: {:.1} - {:.1} kPa\n",
            stats.min_modulus / 1000.0,
            stats.max_modulus / 1000.0
        ));
        report.push_str(&format!(
            "- Mean shear speed: {:.1} m/s\n\n",
            stats.mean_speed
        ));

        match organ.to_lowercase().as_str() {
            "liver" => {
                let classification = self.classify_liver_fibrosis(stats);
                report.push_str("Liver Fibrosis Assessment (METAVIR):\n");
                report.push_str(&format!("- Stage: {:?}\n", classification.stage));
                report.push_str(&format!("- Confidence: {:?}\n", classification.confidence));
                report.push_str(&format!(
                    "- Quality score: {:.2}\n",
                    classification.quality_score
                ));
            }
            "breast" => {
                let classification = self.classify_breast_lesion(stats);
                report.push_str("Breast Lesion Assessment (BI-RADS):\n");
                report.push_str(&format!("- Category: {}\n", classification.birads_category));
                report.push_str(&format!(
                    "- Estimated malignancy: {:.1}%\n",
                    classification.estimated_malignancy_probability * 100.0
                ));
                report.push_str(&format!("- Confidence: {:?}\n", classification.confidence));
            }
            _ => {
                report.push_str("General tissue assessment completed.\n");
            }
        }

        report.push_str("\nNote: This is an automated analysis. Clinical correlation required.\n");
        report
    }
}
