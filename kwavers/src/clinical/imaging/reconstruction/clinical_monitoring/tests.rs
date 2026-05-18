use super::monitor::ClinicalMonitor;
use super::types::{MonitoringConfig, MonitoringSafetyEventType, SafetyEvent, SafetySeverity};
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

    monitor
        .record_frame_quality(1, 50.0, 25.0, 0.8, 1.5, 0.1)
        .unwrap();
    assert_eq!(monitor.performance_metrics.total_frames, 1);
}

#[test]
fn test_safety_event_logging() {
    let config = MonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    let event = SafetyEvent {
        timestamp: SystemTime::now(),
        event_type: MonitoringSafetyEventType::TemperatureExceeded,
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

// ─── Exact quality score formula verification ─────────────────────────────────

/// Perfect inputs (SNR=30 dB, contrast=1.0, artifact=0) → score = 100.
///
/// snr_score   = (30/30 × 100).clamp(0,100) = 100
/// contrast_score = (1.0 × 100).clamp(0,100)  = 100
/// artifact_score = ((1−0) × 100).clamp(0,100) = 100
/// total = (100×0.4 + 100×0.4 + 100×0.2).round() = 100.0
#[test]
fn quality_score_perfect_inputs_is_one_hundred() {
    let monitor = ClinicalMonitor::new(MonitoringConfig::default());
    let score = monitor.compute_quality_score(30.0, 1.0, 0.0);
    assert!((score - 100.0).abs() < 1e-10, "expected 100.0, got {score}");
}

/// Zero inputs (SNR=0 dB, contrast=0, artifact=1) → score = 0.
///
/// snr_score   = 0
/// contrast_score = 0
/// artifact_score = 0
/// total = 0.round() = 0.0
#[test]
fn quality_score_zero_inputs_is_zero() {
    let monitor = ClinicalMonitor::new(MonitoringConfig::default());
    let score = monitor.compute_quality_score(0.0, 0.0, 1.0);
    assert!(score.abs() < 1e-10, "expected 0.0, got {score}");
}

/// Exact mid-range: SNR=15 dB, contrast=0.5, artifact=0 → score = 60.
///
/// snr_score      = (15/30 × 100).clamp(0,100) = 50
/// contrast_score = (0.5 × 100).clamp(0,100)   = 50
/// artifact_score = ((1−0) × 100).clamp(0,100) = 100
/// total = (50×0.4 + 50×0.4 + 100×0.2).round() = (20 + 20 + 20).round() = 60.0
#[test]
fn quality_score_mid_range_exact() {
    let monitor = ClinicalMonitor::new(MonitoringConfig::default());
    let score = monitor.compute_quality_score(15.0, 0.5, 0.0);
    assert!((score - 60.0).abs() < 1e-10, "expected 60.0, got {score}");
}

/// Frame quality recording updates avg_processing_time_ms via running mean.
///
/// Frame 1 at 40 ms, frame 2 at 60 ms:
///   avg after frame 1 = 40.0
///   avg after frame 2 = (40 × 1 + 60) / 2 = 50.0
#[test]
fn quality_recording_running_average_is_exact() {
    let mut monitor = ClinicalMonitor::new(MonitoringConfig::default());
    monitor
        .record_frame_quality(1, 40.0, 30.0, 1.0, 1.0, 0.0)
        .unwrap();
    assert!(
        (monitor.performance_metrics.avg_processing_time_ms - 40.0).abs() < 1e-10,
        "after 1 frame: expected 40.0 ms avg, got {}",
        monitor.performance_metrics.avg_processing_time_ms
    );
    monitor
        .record_frame_quality(2, 60.0, 30.0, 1.0, 1.0, 0.0)
        .unwrap();
    assert!(
        (monitor.performance_metrics.avg_processing_time_ms - 50.0).abs() < 1e-10,
        "after 2 frames: expected 50.0 ms avg, got {}",
        monitor.performance_metrics.avg_processing_time_ms
    );
}
