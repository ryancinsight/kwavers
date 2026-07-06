use super::controller::SafetyController;
use super::types::TherapyAction;
use crate::therapy::therapy_integration::config::TherapyIntegrationSafetyLimits;
use crate::therapy::therapy_integration::state::SafetyMetrics;
use kwavers_core::constants::medical::TI_LIMIT_SOFT_TISSUE;
use std::collections::HashMap;

fn create_test_controller() -> SafetyController {
    let limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1.0,
        max_treatment_time: 600.0,
    };
    SafetyController::new(limits, None)
}

#[test]
fn test_controller_creation() {
    let controller = create_test_controller();
    assert_eq!(controller.last_action, TherapyAction::Continue);
    assert!(!controller.should_stop());
}

#[test]
fn test_thermal_index_violation() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics {
        thermal_index: 6.5,
        ..SafetyMetrics::default()
    };

    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::Stop);
}

#[test]
fn test_thermal_index_warning() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics {
        thermal_index: 5.0, // 83% of TI_LIMIT_SOFT_TISSUE
        ..SafetyMetrics::default()
    };

    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::ReducePower);
}

#[test]
fn test_mechanical_index_safe() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics {
        mechanical_index: 1.5,
        ..SafetyMetrics::default()
    };

    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::Continue);
}

#[test]
fn test_cavitation_dose_exceeds() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics {
        cavitation_dose: 1.1,
        ..SafetyMetrics::default()
    };

    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::Stop);
}

#[test]
fn test_treatment_time_limit() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics::default();
    let action = controller.evaluate_safety(metrics, 600.0).unwrap();
    assert_eq!(action, TherapyAction::Stop);
}

#[test]
fn test_power_reduction_factor() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    assert_eq!(controller.power_reduction_factor(), 1.0); // Continue

    // Trigger ReducePower via thermal index at 83% of limit
    let metrics = SafetyMetrics {
        thermal_index: 5.0, // > 80% of TI_LIMIT_SOFT_TISSUE
        ..SafetyMetrics::default()
    };
    controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(controller.power_reduction_factor(), 0.5);

    // Trigger Stop via thermal index above limit
    let metrics = SafetyMetrics {
        thermal_index: 7.0,
        ..SafetyMetrics::default()
    };
    controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(controller.power_reduction_factor(), 0.0);
}

#[test]
fn test_organ_dose_tracking() {
    let limits = TherapyIntegrationSafetyLimits {
        thermal_index_max: TI_LIMIT_SOFT_TISSUE,
        mechanical_index_max: 1.9,
        cavitation_dose_max: 1.0,
        max_treatment_time: 600.0,
    };

    let mut organ_limits = HashMap::new();
    organ_limits.insert("brain".to_string(), 100.0);

    let mut controller = SafetyController::new(limits, Some(organ_limits));
    controller.start_monitoring(0.0);

    controller.accumulate_organ_dose("brain", 50.0).unwrap();
    controller.accumulate_organ_dose("brain", 40.0).unwrap();

    // 90% of limit → Warning
    let metrics = SafetyMetrics::default();
    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::Warning);

    // Further accumulation past limit → Stop
    controller.accumulate_organ_dose("brain", 15.0).unwrap();
    let metrics = SafetyMetrics::default();
    let action = controller.evaluate_safety(metrics, 1.0).unwrap();
    assert_eq!(action, TherapyAction::Stop);
}

#[test]
fn test_event_summary() {
    let mut controller = create_test_controller();
    controller.start_monitoring(0.0);

    let metrics = SafetyMetrics::default();
    controller.evaluate_safety(metrics, 1.0).unwrap();

    let summary = controller.event_summary();
    assert!(summary.contains("Warnings:"));
    assert!(summary.contains("Power reductions:"));
}
