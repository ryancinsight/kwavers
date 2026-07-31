use super::super::types::{ClinicalThresholds, FeatureMap, LesionDetection, TissueClassification};
use super::NeuralClinicalDecisionSupport;
use aequitas::systems::si::quantities::{Dimensionless, Length};
use aequitas::systems::si::units::Millimeter;
use leto::Array3;

fn dimensionless(value: f32) -> Dimensionless<f32> {
    Dimensionless::from_base(value)
}

fn length_mm(value: f32) -> Length<f32> {
    Length::from_unit::<Millimeter>(value)
}

#[test]
fn test_clinical_decision_support_creation() {
    let thresholds = ClinicalThresholds::default();
    let support = NeuralClinicalDecisionSupport::new(thresholds);
    assert!(*support.config.lesion_confidence_threshold.as_base() > 0.0);
    assert_eq!(support.config.voxel_size.in_unit::<Millimeter>(), 0.5);
}

#[test]
fn test_lesion_size_unit_boundary() {
    let lesion = LesionDetection {
        center: (0, 0, 0),
        size: length_mm(10.1),
        confidence: dimensionless(0.5),
        lesion_type: "Solid".to_owned(),
        clinical_significance: dimensionless(0.2),
    };

    assert!(lesion.requires_urgent_attention());
    assert_eq!(lesion.risk_category(), "HIGH");
}

#[test]
fn test_lesion_type_classification() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
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
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());

    let high_sig = support.assess_clinical_significance(0.9, 0.8);
    assert!(*high_sig.as_base() > 0.8);

    let low_sig = support.assess_clinical_significance(0.3, 0.2);
    assert!(*low_sig.as_base() < 0.3);
}

#[test]
fn test_recommendations_no_lesions() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
    let lesions = Vec::new();
    let classification = TissueClassification::empty();

    let recs = support.generate_recommendations(&lesions, &classification);
    assert!(!recs.is_empty());
    assert!(recs[0].contains("No significant lesions"));
}

#[test]
fn test_recommendations_with_lesions() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
    let lesions = vec![
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
            confidence: dimensionless(0.75),
            lesion_type: "Cyst".to_string(),
            clinical_significance: dimensionless(0.60),
        },
    ];
    let classification = TissueClassification::empty();

    let recs = support.generate_recommendations(&lesions, &classification);
    assert!(recs[0].contains("2 potential lesion"));
    assert!(recs.iter().any(|r| r.contains("high-confidence")));
}

#[test]
fn test_diagnostic_confidence_no_lesions() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
    let lesions = Vec::new();
    let confidence = Array3::from_elem((10, 10, 10), 0.8);

    let diag_conf = support.compute_diagnostic_confidence(&lesions, confidence.view());
    assert!(diag_conf > 0.8);
}

#[test]
fn test_diagnostic_confidence_with_lesions() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
    let lesions = vec![LesionDetection {
        center: (5, 5, 5),
        size: length_mm(4.0),
        confidence: dimensionless(0.9),
        lesion_type: "Solid".to_string(),
        clinical_significance: dimensionless(0.8),
    }];
    let confidence = Array3::from_elem((10, 10, 10), 0.85);

    let diag_conf = support.compute_diagnostic_confidence(&lesions, confidence.view());
    assert!(diag_conf > 0.85);
}

#[test]
fn test_local_statistics_computation() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
    let volume = Array3::from_elem((20, 20, 20), 1.0);

    let stats = support.compute_local_statistics(&volume.view(), 10, 10, 10);
    assert!((stats - 1.0).abs() < 1e-6);
}

#[test]
fn test_lesion_size_estimation() {
    let support = NeuralClinicalDecisionSupport::new(ClinicalThresholds::default());
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
    assert!(*size_mm.as_base() > 0.0);
}
