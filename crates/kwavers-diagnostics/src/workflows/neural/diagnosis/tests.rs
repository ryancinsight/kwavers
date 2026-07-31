use super::super::types::{LesionDetection, TissueClassification};
use super::*;
use aequitas::systems::si::quantities::{Dimensionless, Length};
use aequitas::systems::si::units::Millimeter;

fn dimensionless(value: f32) -> Dimensionless<f32> {
    Dimensionless::from_base(value)
}

fn length_mm(value: f32) -> Length<f32> {
    Length::from_unit::<Millimeter>(value)
}

#[test]
fn test_diagnosis_algorithm_creation() {
    let algorithm = DiagnosisAlgorithm::new();
    assert!(algorithm._models.is_empty());

    let default_algorithm = DiagnosisAlgorithm::default();
    assert!(default_algorithm._models.is_empty());
}

#[test]
fn test_diagnosis_no_lesions() {
    let algorithm = DiagnosisAlgorithm::new();
    let features = FeatureMap::new();
    let clinical_data = ClinicalAnalysis {
        lesions: Vec::new(),
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.9),
    };

    let diagnosis = algorithm.diagnose(&features, &clinical_data).unwrap();
    assert!(diagnosis.contains("No significant findings"));
}

#[test]
fn test_diagnosis_single_lesion() {
    let algorithm = DiagnosisAlgorithm::new();
    let features = FeatureMap::new();
    let clinical_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size: length_mm(5.0),
            confidence: dimensionless(0.95),
            lesion_type: "Solid".to_string(),
            clinical_significance: dimensionless(0.85),
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.9),
    };

    let diagnosis = algorithm.diagnose(&features, &clinical_data).unwrap();
    assert!(diagnosis.contains("Single"));
    assert!(diagnosis.contains("high-confidence"));
}

#[test]
fn test_diagnosis_multiple_lesions() {
    let algorithm = DiagnosisAlgorithm::new();
    let features = FeatureMap::new();
    let clinical_data = ClinicalAnalysis {
        lesions: vec![
            LesionDetection {
                center: (10, 10, 10),
                size: length_mm(5.0),
                confidence: dimensionless(0.95),
                lesion_type: "Solid".to_string(),
                clinical_significance: dimensionless(0.85),
            },
            LesionDetection {
                center: (20, 20, 20),
                size: length_mm(3.0),
                confidence: dimensionless(0.90),
                lesion_type: "Cyst".to_string(),
                clinical_significance: dimensionless(0.70),
            },
            LesionDetection {
                center: (30, 30, 30),
                size: length_mm(4.0),
                confidence: dimensionless(0.85),
                lesion_type: "Complex".to_string(),
                clinical_significance: dimensionless(0.75),
            },
        ],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.9),
    };

    let diagnosis = algorithm.diagnose(&features, &clinical_data).unwrap();
    assert!(diagnosis.contains("Multiple") || diagnosis.contains("high-confidence"));
}

#[test]
fn test_priority_assessment() {
    let algorithm = DiagnosisAlgorithm::new();

    // Test URGENT priority
    let urgent_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size: length_mm(5.0),
            confidence: dimensionless(0.95),
            lesion_type: "Solid".to_string(),
            clinical_significance: dimensionless(0.90),
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.9),
    };
    assert_eq!(algorithm.assess_priority(&urgent_data), "URGENT");

    // Test NEGATIVE priority
    let negative_data = ClinicalAnalysis {
        lesions: Vec::new(),
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.95),
    };
    assert_eq!(algorithm.assess_priority(&negative_data), "NEGATIVE");

    // Test HIGH priority (lesion but not high confidence)
    let high_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size: length_mm(3.0),
            confidence: dimensionless(0.75),
            lesion_type: "Cyst".to_string(),
            clinical_significance: dimensionless(0.60),
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.8),
    };
    assert_eq!(algorithm.assess_priority(&high_data), "HIGH");
}

#[test]
fn test_report_generation() {
    let algorithm = DiagnosisAlgorithm::new();
    let clinical_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size: length_mm(5.0),
            confidence: dimensionless(0.95),
            lesion_type: "Solid".to_string(),
            clinical_significance: dimensionless(0.85),
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.9),
    };

    let report = algorithm.generate_report(&clinical_data);
    assert_eq!(report.get("lesion_count").unwrap(), "1");
    assert_eq!(report.get("high_confidence_count").unwrap(), "1");
    assert!(report.contains_key("diagnostic_confidence"));
    assert_eq!(report.get("priority").unwrap(), "URGENT");
    assert!(report.contains_key("lesion_details"));
}

#[test]
fn test_report_no_lesions() {
    let algorithm = DiagnosisAlgorithm::new();
    let clinical_data = ClinicalAnalysis {
        lesions: Vec::new(),
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: dimensionless(0.95),
    };

    let report = algorithm.generate_report(&clinical_data);
    assert_eq!(report.get("lesion_count").unwrap(), "0");
    assert_eq!(report.get("priority").unwrap(), "NEGATIVE");
    assert!(!report.contains_key("lesion_details"));
}
