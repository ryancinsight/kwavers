use crate::clinical::safety::mechanical_index::MechanicalIndexTissueType;
use crate::clinical::therapy::parameters::ClinicalTherapyParameters;
use crate::core::constants::acoustic_parameters::DB_TO_NP;
use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::constants::medical::{IEC_TISSUE_ABSORPTION_DB_CM_MHZ, IEC_TISSUE_SPECIFIC_HEAT};
use crate::core::constants::tissue_acoustics::DENSITY_BLOOD;

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

/// **Test: IEC 62127-1 tissue acoustic impedance in temperature rise estimate**
///
/// Verifies that `estimate_temperature_rise` (called via `audit_parameters`) uses
/// tissue acoustic impedance `Z_t = ρ_t · c_t` (not water impedance) per Nyborg (1988)
/// Eq. 4 and IEC 62127-1:2013 Table A.1.
///
/// Analytical derivation:
/// ```text
/// f      = 1 MHz → f_MHz = 1.0
/// α      = IEC_TISSUE_ABSORPTION_DB_CM_MHZ × 1.0 × 100 × DB_TO_NP   [Np/m]
/// p      = 100 kPa peak; p_rms = p / √2
/// Z_t    = ρ_t · c_t = 1060 × 1540 = 1 632 400 Rayl
/// I_SPTA = p_rms² / Z_t
/// Q      = 2α · I_SPTA
/// t_eff  = treatment_duration × duty_cycle = 1.0 × 1.0 = 1.0 s
/// ΔT     = Q · t_eff / (ρ_t · c_p)
/// ```
#[test]
fn test_temperature_rise_uses_tissue_impedance_not_water() {
    // Build a validator with a large max_temp_rise so the cap is not hit.
    let mut config = ComplianceConfig::default();
    config.max_temp_rise = 100.0;
    let mut validator = EnhancedComplianceValidator::new(config).unwrap();

    let params = ClinicalTherapyParameters {
        frequency: 1.0e6, // 1 MHz
        pressure: 1.0e5,  // 100 kPa peak (sinusoidal)
        duration: 1.0,
        peak_negative_pressure: 1.0e5,
        mechanical_index: 0.1, // well below any limit
        treatment_duration: 1.0,
        duty_cycle: 1.0, // 100% duty — t_eff = 1.0 s
        prf: 1.0,
    };

    let audit = validator.audit_parameters(&params).unwrap();
    let temp_check = audit
        .checks
        .iter()
        .find(|c| c.name == "Maximum Temperature Rise")
        .expect("temperature rise check must be present");

    // Analytical expected value using IEC 62127-1 tissue parameters
    let f_mhz = 1.0_f64;
    let alpha_np_m = IEC_TISSUE_ABSORPTION_DB_CM_MHZ * f_mhz * 100.0 * DB_TO_NP;
    let p_rms = 1.0e5_f64 / std::f64::consts::SQRT_2;
    let z_tissue = DENSITY_BLOOD * SOUND_SPEED_TISSUE; // ≈ 1.63 MRayl
    let i_spta = p_rms * p_rms / z_tissue;
    let q_vol = 2.0 * alpha_np_m * i_spta;
    let t_eff = 1.0_f64; // 1 s × 1.0 duty cycle
    let expected_delta_t = q_vol * t_eff / (DENSITY_BLOOD * IEC_TISSUE_SPECIFIC_HEAT);

    let rel_err = (temp_check.measured - expected_delta_t).abs() / expected_delta_t;
    assert!(
        rel_err < 1e-12,
        "temperature rise {:.6e} °C differs from IEC 62127-1 tissue impedance model {:.6e} °C (rel_err={rel_err:.2e})",
        temp_check.measured,
        expected_delta_t
    );
}
