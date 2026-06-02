use super::super::types::{LesionDetection, TissueClassification};
use super::*;

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
        diagnostic_confidence: 0.9_f32,
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
            size_mm: 5.0_f32,
            confidence: 0.95_f32,
            lesion_type: "Solid".to_string(),
            clinical_significance: 0.85_f32,
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.9_f32,
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
                size_mm: 5.0_f32,
                confidence: 0.95_f32,
                lesion_type: "Solid".to_string(),
                clinical_significance: 0.85_f32,
            },
            LesionDetection {
                center: (20, 20, 20),
                size_mm: 3.0_f32,
                confidence: 0.90_f32,
                lesion_type: "Cyst".to_string(),
                clinical_significance: 0.70_f32,
            },
            LesionDetection {
                center: (30, 30, 30),
                size_mm: 4.0_f32,
                confidence: 0.85_f32,
                lesion_type: "Complex".to_string(),
                clinical_significance: 0.75_f32,
            },
        ],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.9_f32,
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
            size_mm: 5.0_f32,
            confidence: 0.95_f32,
            lesion_type: "Solid".to_string(),
            clinical_significance: 0.90_f32,
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.9_f32,
    };
    assert_eq!(algorithm.assess_priority(&urgent_data), "URGENT");

    // Test NEGATIVE priority
    let negative_data = ClinicalAnalysis {
        lesions: Vec::new(),
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.95_f32,
    };
    assert_eq!(algorithm.assess_priority(&negative_data), "NEGATIVE");

    // Test HIGH priority (lesion but not high confidence)
    let high_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size_mm: 3.0_f32,
            confidence: 0.75_f32,
            lesion_type: "Cyst".to_string(),
            clinical_significance: 0.60_f32,
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.8_f32,
    };
    assert_eq!(algorithm.assess_priority(&high_data), "HIGH");
}

#[test]
fn test_report_generation() {
    let algorithm = DiagnosisAlgorithm::new();
    let clinical_data = ClinicalAnalysis {
        lesions: vec![LesionDetection {
            center: (10, 10, 10),
            size_mm: 5.0_f32,
            confidence: 0.95_f32,
            lesion_type: "Solid".to_string(),
            clinical_significance: 0.85_f32,
        }],
        tissue_classification: TissueClassification::empty(),
        recommendations: Vec::new(),
        diagnostic_confidence: 0.9_f32,
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
        diagnostic_confidence: 0.95_f32,
    };

    let report = algorithm.generate_report(&clinical_data);
    assert_eq!(report.get("lesion_count").unwrap(), "0");
    assert_eq!(report.get("priority").unwrap(), "NEGATIVE");
    assert!(!report.contains_key("lesion_details"));
}
