use super::monitor::ClinicalMonitor;
use super::types::{MonitoringConfig, SafetyEvent, SafetyEventType, SafetySeverity};
use std::time::SystemTime;

#[test]
fn test_monitoring_config_default() {
    let config = MonitoringConfig::default();
    assert!(config.enable_quality_monitoring);
    assert!(config.enable_safety_logging);
}

#[test]
fn test_clinical_monitor_creation() {
    let config = MonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);
    assert_eq!(monitor.performance_metrics.total_frames, 0);
}

#[test]
fn test_frame_quality_recording() {
    let config = MonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor.record_frame_quality(1, 50.0, 25.0, 0.8, 1.5, 0.1).unwrap();
    assert_eq!(monitor.performance_metrics.total_frames, 1);
}

#[test]
fn test_safety_event_logging() {
    let config = MonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    let event = SafetyEvent {
        timestamp: SystemTime::now(),
        event_type: SafetyEventType::TemperatureExceeded,
        parameter_value: 6.0,
        safety_limit: 5.0,
        severity: SafetySeverity::Critical,
        message: "Temperature exceeds limit".to_string(),
    };

    monitor.log_safety_event(event).unwrap();
    assert_eq!(monitor.safety_log().len(), 1);
}

#[test]
fn test_temperature_check() {
    let config = MonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor.check_temperature(42.5, 37.0);
    assert_eq!(monitor.safety_log().len(), 1);

    let event = &monitor.safety_log()[0];
    assert_eq!(event.severity, SafetySeverity::Critical);
}

#[test]
fn test_mechanical_index_check() {
    let config = MonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor.check_mechanical_index(2.0);
    assert_eq!(monitor.safety_log().len(), 1);
}

#[test]
fn test_monitoring_report() {
    let config = MonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);

    let report = monitor.generate_report();
    assert_eq!(report.total_frames_processed, 0);
    assert!(report.uptime_seconds >= 0.0);
}

#[test]
fn test_quality_score_computation() {
    let config = MonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);

    let score = monitor.compute_quality_score(30.0, 1.0, 0.0);
    assert!(score > 90.0);

    let score = monitor.compute_quality_score(5.0, 0.2, 0.8);
    assert!(score < 30.0);
}

#[test]
fn test_safety_event_severity() {
    assert!(SafetySeverity::Critical > SafetySeverity::Warning);
    assert!(SafetySeverity::Urgent > SafetySeverity::Info);
}
