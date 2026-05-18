use crate::clinical::safety::mechanical_index::MechanicalIndexTissueType;
use crate::clinical::therapy::parameters::ClinicalTherapyParameters;

use super::*;

#[test]
fn test_compliance_config_default() {
    let config = ComplianceConfig::default();
    config.validate().unwrap();
}

#[test]
fn test_compliance_validator_creation() {
    let config = ComplianceConfig::default();
    let _validator = EnhancedComplianceValidator::new(config).unwrap();
}

#[test]
fn test_compliance_check_creation() {
    let check = ComplianceCheck::new("Test".to_string(), 50.0, 100.0, "W".to_string(), 80.0);

    assert_eq!(check.status, ComplianceStatus::Compliant);
    assert!((check.percent_of_limit() - 50.0).abs() < 0.1);
}

#[test]
fn test_compliance_check_warning() {
    let check = ComplianceCheck::new("Test".to_string(), 85.0, 100.0, "W".to_string(), 80.0);

    assert_eq!(check.status, ComplianceStatus::Warning);
}

#[test]
fn test_compliance_check_non_compliant() {
    let check = ComplianceCheck::new("Test".to_string(), 105.0, 100.0, "W".to_string(), 80.0);

    assert_eq!(check.status, ComplianceStatus::NonCompliant);
}

#[test]
fn test_audit_parameters_hifu() {
    let config =
        ComplianceConfig::default().with_tissue_type(MechanicalIndexTissueType::SoftTissue);
    let mut validator = EnhancedComplianceValidator::new(config).unwrap();
    let params = ClinicalTherapyParameters::hifu();

    let audit = validator.audit_parameters(&params).unwrap();
    assert!(!audit.checks.is_empty());
}

#[test]
fn test_session_metrics() {
    let config = ComplianceConfig::default();
    let mut validator = EnhancedComplianceValidator::new(config).unwrap();

    validator.start_session();
    let metrics = validator.end_session().unwrap();
    assert!(metrics.session_duration >= 0.0);
}

#[test]
fn test_compliance_report() {
    let config = ComplianceConfig::default();
    let validator = EnhancedComplianceValidator::new(config).unwrap();

    let report = validator.generate_report();
    assert_eq!(report.total_audits, 0);
    assert_eq!(report.compliance_percentage, 100.0);
}

#[test]
fn test_config_builder() {
    let config = ComplianceConfig::default()
        .with_power_limit(100.0)
        .with_intensity_limit(5.0)
        .with_tissue_type(MechanicalIndexTissueType::Brain);

    config.validate().unwrap();
    assert!((config.max_power - 100.0).abs() < 0.1);
}

#[test]
fn test_invalid_config() {
    let mut config = ComplianceConfig::default();
    config.max_power = -1.0;

    assert!(config.validate().is_err());
}
