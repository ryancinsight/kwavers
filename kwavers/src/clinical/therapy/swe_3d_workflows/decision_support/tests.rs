//! Tests for `Swe3dClinicalDecisionSupport`.

use super::super::statistics::VolumetricStatistics;
use super::classifier::Swe3dClinicalDecisionSupport;
use super::types::FibrosisStage;

#[test]
fn test_liver_fibrosis_classification() {
    let cds = Swe3dClinicalDecisionSupport::default();

    let normal_stats = VolumetricStatistics {
        valid_voxels: 1000,
        mean_modulus: 5500.0,
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
        FibrosisStage::F0F1 => {}
        _ => panic!("Expected F0F1 for normal liver"),
    }

    let cirrhosis_stats = VolumetricStatistics {
        valid_voxels: 1000,
        mean_modulus: 15000.0,
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
        FibrosisStage::F4 => {}
        _ => panic!("Expected F4 for cirrhosis"),
    }
}

#[test]
fn test_breast_lesion_classification() {
    let cds = Swe3dClinicalDecisionSupport::default();

    let benign_stats = VolumetricStatistics {
        valid_voxels: 500,
        mean_modulus: 15000.0,
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
    assert_eq!(classification.birads_category, 2);

    let malignant_stats = VolumetricStatistics {
        valid_voxels: 500,
        mean_modulus: 120000.0,
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
    assert_eq!(classification.birads_category, 5);
}

#[test]
fn test_clinical_report_generation() {
    let cds = Swe3dClinicalDecisionSupport::default();

    let stats = VolumetricStatistics {
        valid_voxels: 1500,
        mean_modulus: 8000.0,
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

    assert!(report.contains("3D SWE Clinical Report"));
    assert!(report.contains("LIVER"));
    assert!(report.contains("Valid voxels: 1500"));
    assert!(report.contains("Mean Young's modulus: 8.0 kPa"));
    assert!(report.contains("Liver Fibrosis Assessment"));
}
