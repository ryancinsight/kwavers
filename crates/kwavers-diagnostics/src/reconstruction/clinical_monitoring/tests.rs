use super::monitor::ClinicalMonitor;
use super::types::{
    ClinicalMonitoringConfig, MonitoringMetric, MonitoringSafetyEventType, SafetyEvent,
    SafetySeverity,
};
use aequitas::systems::si::quantities::{
    Dimensionless, Length, TemperatureDifference, ThermodynamicTemperature, Time,
};
use aequitas::systems::si::units::{Kelvin, Millimeter, Millisecond};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use std::time::SystemTime;

#[test]
fn test_monitoring_config_default() {
    let config = ClinicalMonitoringConfig::default();
    assert!(config.enable_quality_monitoring);
    assert!(config.enable_safety_logging);
}

#[test]
fn test_clinical_monitor_creation() {
    let config = ClinicalMonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);
    assert_eq!(monitor.performance_metrics.total_frames, 0);
}

#[test]
fn test_frame_quality_recording() {
    let config = ClinicalMonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor
        .record_frame_quality(
            1,
            Time::from_unit::<Millisecond>(50.0),
            Dimensionless::from_base(25.0),
            Dimensionless::from_base(0.8),
            Length::from_unit::<Millimeter>(1.5),
            Dimensionless::from_base(0.1),
        )
        .unwrap();
    assert_eq!(monitor.performance_metrics.total_frames, 1);
}

#[test]
fn test_safety_event_logging() {
    let config = ClinicalMonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    let event = SafetyEvent {
        timestamp: SystemTime::now(),
        event_type: MonitoringSafetyEventType::TemperatureExceeded,
        parameter_value: MonitoringMetric::TemperatureRise(TemperatureDifference::from_unit::<
            Kelvin,
        >(6.0)),
        safety_limit: MonitoringMetric::TemperatureRise(
            TemperatureDifference::from_unit::<Kelvin>(5.0),
        ),
        severity: SafetySeverity::Critical,
        message: "Temperature exceeds limit".to_string(),
    };

    monitor.log_safety_event(event).unwrap();
    let logged = &monitor.safety_log()[0];
    assert_eq!(
        logged.parameter_value,
        MonitoringMetric::TemperatureRise(TemperatureDifference::from_unit::<Kelvin>(6.0),)
    );
    assert_eq!(
        logged.safety_limit,
        MonitoringMetric::TemperatureRise(TemperatureDifference::from_unit::<Kelvin>(5.0),)
    );
}

#[test]
fn test_temperature_check() {
    let config = ClinicalMonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor
        .check_temperature(
            ThermodynamicTemperature::from_base(315.65),
            ThermodynamicTemperature::from_base(BODY_TEMPERATURE_K),
        )
        .unwrap();
    assert_eq!(monitor.safety_log().len(), 1);

    let event = &monitor.safety_log()[0];
    assert_eq!(event.severity, SafetySeverity::Critical);
    assert_eq!(
        event.parameter_value,
        MonitoringMetric::TemperatureRise(TemperatureDifference::from_unit::<Kelvin>(5.5))
    );
    assert_eq!(
        event.safety_limit,
        MonitoringMetric::TemperatureRise(TemperatureDifference::from_unit::<Kelvin>(5.0))
    );
}

#[test]
fn test_mechanical_index_check() {
    let config = ClinicalMonitoringConfig::default();
    let mut monitor = ClinicalMonitor::new(config);

    monitor
        .check_mechanical_index(Dimensionless::from_base(2.0))
        .unwrap();
    let event = &monitor.safety_log()[0];
    assert_eq!(
        event.parameter_value,
        MonitoringMetric::MechanicalIndex(Dimensionless::from_base(2.0))
    );
    assert_eq!(
        event.safety_limit,
        MonitoringMetric::MechanicalIndex(Dimensionless::from_base(1.9))
    );
}

#[test]
fn test_monitoring_report() {
    let config = ClinicalMonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);

    let report = monitor.generate_report();
    assert_eq!(report.total_frames_processed, 0);
    assert!(report.uptime.into_base() >= 0.0);
}

#[test]
fn test_quality_score_computation() {
    let config = ClinicalMonitoringConfig::default();
    let monitor = ClinicalMonitor::new(config);

    let score = monitor.compute_quality_score(
        Dimensionless::from_base(30.0),
        Dimensionless::from_base(1.0),
        Dimensionless::from_base(0.0),
    );
    assert!(score.into_base() > 90.0);

    let score = monitor.compute_quality_score(
        Dimensionless::from_base(5.0),
        Dimensionless::from_base(0.2),
        Dimensionless::from_base(0.8),
    );
    assert!(score.into_base() < 30.0);
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
    let monitor = ClinicalMonitor::new(ClinicalMonitoringConfig::default());
    let score = monitor.compute_quality_score(
        Dimensionless::from_base(30.0),
        Dimensionless::from_base(1.0),
        Dimensionless::from_base(0.0),
    );
    assert!(
        (score.into_base() - 100.0).abs() < 1e-10,
        "expected 100.0, got {}",
        score.into_base()
    );
}

/// Zero inputs (SNR=0 dB, contrast=0, artifact=1) → score = 0.
///
/// snr_score   = 0
/// contrast_score = 0
/// artifact_score = 0
/// total = 0.round() = 0.0
#[test]
fn quality_score_zero_inputs_is_zero() {
    let monitor = ClinicalMonitor::new(ClinicalMonitoringConfig::default());
    let score = monitor.compute_quality_score(
        Dimensionless::from_base(0.0),
        Dimensionless::from_base(0.0),
        Dimensionless::from_base(1.0),
    );
    assert!(
        score.into_base().abs() < 1e-10,
        "expected 0.0, got {}",
        score.into_base()
    );
}

/// Exact mid-range: SNR=15 dB, contrast=0.5, artifact=0 → score = 60.
///
/// snr_score      = (15/30 × 100).clamp(0,100) = 50
/// contrast_score = (0.5 × 100).clamp(0,100)   = 50
/// artifact_score = ((1−0) × 100).clamp(0,100) = 100
/// total = (50×0.4 + 50×0.4 + 100×0.2).round() = (20 + 20 + 20).round() = 60.0
#[test]
fn quality_score_mid_range_exact() {
    let monitor = ClinicalMonitor::new(ClinicalMonitoringConfig::default());
    let score = monitor.compute_quality_score(
        Dimensionless::from_base(15.0),
        Dimensionless::from_base(0.5),
        Dimensionless::from_base(0.0),
    );
    assert!(
        (score.into_base() - 60.0).abs() < 1e-10,
        "expected 60.0, got {}",
        score.into_base()
    );
}

/// Frame quality recording updates the typed processing-time average via running mean.
///
/// Frame 1 at 40 ms, frame 2 at 60 ms:
///   avg after frame 1 = 40.0
///   avg after frame 2 = (40 × 1 + 60) / 2 = 50.0
#[test]
fn quality_recording_running_average_is_exact() {
    let mut monitor = ClinicalMonitor::new(ClinicalMonitoringConfig::default());
    monitor
        .record_frame_quality(
            1,
            Time::from_unit::<Millisecond>(40.0),
            Dimensionless::from_base(30.0),
            Dimensionless::from_base(1.0),
            Length::from_unit::<Millimeter>(1.0),
            Dimensionless::from_base(0.0),
        )
        .unwrap();
    assert!(
        (monitor.performance_metrics.avg_processing_time.into_base() - 0.04).abs() < 1e-10,
        "after 1 frame: expected 40.0 ms avg, got {}",
        monitor.performance_metrics.avg_processing_time.into_base()
    );
    monitor
        .record_frame_quality(
            2,
            Time::from_unit::<Millisecond>(60.0),
            Dimensionless::from_base(30.0),
            Dimensionless::from_base(1.0),
            Length::from_unit::<Millimeter>(1.0),
            Dimensionless::from_base(0.0),
        )
        .unwrap();
    assert!(
        (monitor.performance_metrics.avg_processing_time.into_base() - 0.05).abs() < 1e-10,
        "after 2 frames: expected 50.0 ms avg, got {}",
        monitor.performance_metrics.avg_processing_time.into_base()
    );
}
