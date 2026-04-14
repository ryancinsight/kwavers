use super::statistics::VolumetricStatistics;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ClinicalDecisionSupport {
    /// Reference ranges for different tissues
    _reference_ranges: HashMap<String, TissueReference>,
    /// Classification thresholds
    classification_thresholds: HashMap<String, Vec<f64>>,
}

impl Default for ClinicalDecisionSupport {
    fn default() -> Self {
        let mut reference_ranges = HashMap::new();
        let mut classification_thresholds = HashMap::new();

        // Liver reference ranges (kPa)
        reference_ranges.insert(
            "liver_normal".to_string(),
            TissueReference {
                mean_modulus: 5.5, // kPa
                std_modulus: 1.2,
                min_modulus: 3.0,
                max_modulus: 8.0,
            },
        );

        reference_ranges.insert(
            "liver_fibrosis_f4".to_string(),
            TissueReference {
                mean_modulus: 15.0, // kPa
                std_modulus: 3.0,
                min_modulus: 10.0,
                max_modulus: 25.0,
            },
        );

        // Liver fibrosis classification thresholds (kPa)
        classification_thresholds.insert(
            "liver_metavir".to_string(),
            vec![
                6.5,  // F0/F1 vs F2/F3/F4
                8.5,  // F0/F1/F2 vs F3/F4
                11.0, // F0/F1/F2/F3 vs F4
            ],
        );

        // Breast reference ranges (kPa)
        reference_ranges.insert(
            "breast_normal".to_string(),
            TissueReference {
                mean_modulus: 18.0,
                std_modulus: 5.0,
                min_modulus: 10.0,
                max_modulus: 30.0,
            },
        );

        reference_ranges.insert(
            "breast_malignant".to_string(),
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
    /// Classify liver fibrosis stage using METAVIR scoring
    pub fn classify_liver_fibrosis(&self, stats: &VolumetricStatistics) -> LiverFibrosisStage {
        let thresholds = self.classification_thresholds.get("liver_metavir").unwrap();
        let mean_kpa = stats.mean_modulus / 1000.0; // Convert to kPa

        let stage = if mean_kpa < thresholds[0] {
            FibrosisStage::F0F1
        } else if mean_kpa < thresholds[1] {
            FibrosisStage::F2
        } else if mean_kpa < thresholds[2] {
            FibrosisStage::F3
        } else {
            FibrosisStage::F4
        };

        // Confidence based on standard deviation and quality
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

    /// Classify breast lesion using BI-RADS criteria
    pub fn classify_breast_lesion(
        &self,
        stats: &VolumetricStatistics,
    ) -> BreastLesionClassification {
        let mean_kpa = stats.mean_modulus / 1000.0; // Convert to kPa

        // Simplified BI-RADS classification based on stiffness
        let (birads_category, malignancy_probability) = if mean_kpa < 20.0 {
            (2, 0.0) // Benign
        } else if mean_kpa < 50.0 {
            (3, 0.02) // Probably benign
        } else if mean_kpa < 80.0 {
            (4, 0.3) // Suspicious
        } else {
            (5, 0.95) // Highly suggestive of malignancy
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

    /// Generate clinical report
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

        // Organ-specific classification
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

/// Tissue reference ranges for clinical comparison
#[derive(Debug, Clone)]
pub struct TissueReference {
    /// Mean Young's modulus (kPa)
    pub mean_modulus: f64,
    /// Standard deviation (kPa)
    pub std_modulus: f64,
    /// Minimum expected modulus (kPa)
    pub min_modulus: f64,
    /// Maximum expected modulus (kPa)
    pub max_modulus: f64,
}

/// Liver fibrosis classification result
#[derive(Debug, Clone)]
pub struct LiverFibrosisStage {
    /// METAVIR fibrosis stage
    pub stage: FibrosisStage,
    /// Mean stiffness in kPa
    pub mean_stiffness_kpa: f64,
    /// Classification confidence
    pub confidence: ClassificationConfidence,
    /// Quality score (0-1)
    pub quality_score: f64,
}

/// METAVIR fibrosis stages
#[derive(Debug, Clone, Copy)]
pub enum FibrosisStage {
    /// No fibrosis (F0) or mild fibrosis (F1)
    F0F1,
    /// Moderate fibrosis (F2)
    F2,
    /// Severe fibrosis (F3)
    F3,
    /// Cirrhosis (F4)
    F4,
}

/// Breast lesion classification result
#[derive(Debug, Clone)]
pub struct BreastLesionClassification {
    /// BI-RADS category (2-5)
    pub birads_category: u8,
    /// Estimated probability of malignancy (0-1)
    pub estimated_malignancy_probability: f64,
    /// Mean stiffness in kPa
    pub mean_stiffness_kpa: f64,
    /// Classification confidence
    pub confidence: ClassificationConfidence,
    /// Quality score (0-1)
    pub quality_score: f64,
}

/// Classification confidence levels
#[derive(Debug, Clone, Copy)]
pub enum ClassificationConfidence {
    /// High confidence (>80% accuracy expected)
    High,
    /// Medium confidence (60-80% accuracy expected)
    Medium,
    /// Low confidence (<60% accuracy expected)
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liver_fibrosis_classification() {
        let cds = ClinicalDecisionSupport::default();

        // Test normal liver
        let normal_stats = VolumetricStatistics {
            valid_voxels: 1000,
            mean_modulus: 5500.0, // 5.5 kPa
            std_modulus: 1200.0,
            median_modulus: 5500.0,
            min_modulus: 3000.0,
            max_modulus: 8000.0,
            mean_speed: 1.5,
            mean_confidence: 0.9,
            mean_quality: 0.85,
            volume_coverage: 0.95,
        };

        let classification = cds.classify_liver_fibrosis(&normal_stats);
        match classification.stage {
            FibrosisStage::F0F1 => {} // Expected for normal
            _ => panic!("Expected F0F1 for normal liver"),
        }

        // Test cirrhosis
        let cirrhosis_stats = VolumetricStatistics {
            valid_voxels: 1000,
            mean_modulus: 15000.0, // 15 kPa
            std_modulus: 3000.0,
            median_modulus: 15000.0,
            min_modulus: 10000.0,
            max_modulus: 25000.0,
            mean_speed: 2.5,
            mean_confidence: 0.9,
            mean_quality: 0.85,
            volume_coverage: 0.95,
        };

        let classification = cds.classify_liver_fibrosis(&cirrhosis_stats);
        match classification.stage {
            FibrosisStage::F4 => {} // Expected for cirrhosis
            _ => panic!("Expected F4 for cirrhosis"),
        }
    }

    #[test]
    fn test_breast_lesion_classification() {
        let cds = ClinicalDecisionSupport::default();

        // Test benign lesion
        let benign_stats = VolumetricStatistics {
            valid_voxels: 500,
            mean_modulus: 15000.0, // 15 kPa
            std_modulus: 3000.0,
            median_modulus: 15000.0,
            min_modulus: 10000.0,
            max_modulus: 20000.0,
            mean_speed: 2.2,
            mean_confidence: 0.85,
            mean_quality: 0.8,
            volume_coverage: 0.9,
        };

        let classification = cds.classify_breast_lesion(&benign_stats);
        assert_eq!(classification.birads_category, 2); // Benign

        // Test malignant lesion
        let malignant_stats = VolumetricStatistics {
            valid_voxels: 500,
            mean_modulus: 120000.0, // 120 kPa
            std_modulus: 20000.0,
            median_modulus: 120000.0,
            min_modulus: 80000.0,
            max_modulus: 160000.0,
            mean_speed: 8.0,
            mean_confidence: 0.85,
            mean_quality: 0.8,
            volume_coverage: 0.9,
        };

        let classification = cds.classify_breast_lesion(&malignant_stats);
        assert_eq!(classification.birads_category, 5); // Highly suggestive of malignancy
    }

    #[test]
    fn test_clinical_report_generation() {
        let cds = ClinicalDecisionSupport::default();

        let stats = VolumetricStatistics {
            valid_voxels: 1500,
            mean_modulus: 8000.0, // 8 kPa
            std_modulus: 1500.0,
            median_modulus: 8000.0,
            min_modulus: 5000.0,
            max_modulus: 12000.0,
            mean_speed: 1.8,
            mean_confidence: 0.85,
            mean_quality: 0.82,
            volume_coverage: 0.92,
        };

        let report = cds.generate_report("liver", &stats);

        // Check that report contains expected content
        assert!(report.contains("3D SWE Clinical Report"));
        assert!(report.contains("LIVER"));
        assert!(report.contains("Valid voxels: 1500"));
        assert!(report.contains("Mean Young's modulus: 8.0 kPa"));
        assert!(report.contains("Liver Fibrosis Assessment"));
    }
}
