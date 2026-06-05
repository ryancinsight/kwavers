use super::{
    ClinicalSafetyLevel, ClinicalSafetyLimits, ClinicalSafetyMonitor, ComplianceValidator,
    DoseController, Interlock, InterlockSystem, SystemConfiguration,
};
use crate::therapy::domain_types::ClinicalTherapyParameters;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn test_safety_limits_creation() {
    let limits = ClinicalSafetyLimits::default();
    assert!(limits.max_intensity <= 3.0);
    assert!(limits.max_power <= 100.0);
    assert!(limits.max_temperature_rise <= 5.0);
}

#[test]
fn test_safety_monitor_normal_operation() {
    let limits = ClinicalSafetyLimits::default();
    let mut monitor = ClinicalSafetyMonitor::new(limits);

    let params = ClinicalTherapyParameters {
        frequency: 1.5 * MHZ_TO_HZ,
        pressure: MPA_TO_PA,
        duration: 600.0,
        peak_negative_pressure: MPA_TO_PA,
        treatment_duration: 600.0,
        mechanical_index: 1.2,
        duty_cycle: 0.5,
        prf: 100.0,
    };

    let state = monitor.check_safety(&params);
    assert_eq!(state, ClinicalSafetyLevel::Normal);
    assert!(monitor.violations().is_empty());
}

#[test]
fn test_safety_monitor_critical_violation() {
    let limits = ClinicalSafetyLimits::default();
    let mut monitor = ClinicalSafetyMonitor::new(limits);

    let params = ClinicalTherapyParameters {
        frequency: 1.5 * MHZ_TO_HZ,
        pressure: 3.0 * MPA_TO_PA,
        duration: 600.0,
        peak_negative_pressure: 3.0 * MPA_TO_PA,
        treatment_duration: 600.0,
        mechanical_index: 2.5,
        duty_cycle: 0.5,
        prf: 100.0,
    };

    let state = monitor.check_safety(&params);
    assert_eq!(state, ClinicalSafetyLevel::Critical);
    assert!(!monitor.violations().is_empty());
    assert!(monitor.requires_emergency_shutdown());
}

#[test]
fn test_interlock_system() {
    let mut interlocks = InterlockSystem::new();

    interlocks.add_interlock(
        "power_supply".to_string(),
        Interlock::new("Power supply OK".to_string(), || Ok(true)),
    );

    assert!(interlocks.check_interlocks().unwrap());
    interlocks.enable_system().unwrap();
    assert!(interlocks.is_system_enabled());
}

#[test]
fn test_dose_controller() {
    let limits = ClinicalSafetyLimits::default();
    let mut controller = DoseController::new(limits);

    assert!(controller
        .start_session("patient_001".to_string(), "hifu_ablation".to_string())
        .is_ok());

    let params = ClinicalTherapyParameters::hifu();
    controller.update_dose(100.0, &params).unwrap();

    assert_eq!(controller.accumulated_dose, 100.0);
    assert_eq!(controller.remaining_dose_capacity(), 9900.0);
}

#[test]
fn test_compliance_validator() {
    let mut validator = ComplianceValidator::new();

    let config = SystemConfiguration {
        safety_limits: ClinicalSafetyLimits::default(),
        monitoring_enabled: true,
        interlocks_enabled: true,
        emergency_stop_tested: true,
    };

    let report = validator.validate_compliance(&config).unwrap();
    assert!(report.overall_compliant);
    assert_eq!(report.standard_version, "IEC 60601-2-37:2007+A1:2010");
}
