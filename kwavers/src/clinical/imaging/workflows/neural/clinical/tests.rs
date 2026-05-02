use super::super::types::{ClinicalThresholds, FeatureMap, LesionDetection, TissueClassification};
use super::ClinicalDecisionSupport;
use ndarray::Array3;

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
    assert!(diag_conf > 0.8);
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

    for z in 10..20 {
        for y in 10..20 {
            for x in 10..20 {
                let dist = ((x as f32 - 15.0).powi(2)
                    + (y as f32 - 15.0).powi(2)
                    + (z as f32 - 15.0).powi(2))
                .sqrt();
                if dist < 5.0 {
                    volume[[x, y, z]] = 3.0;
                }
            }
        }
    }

    let features = FeatureMap::new();
    let size_mm = support.estimate_lesion_size(volume.view(), &features, 15, 15, 15);
    assert!(size_mm > 0.0);
}
