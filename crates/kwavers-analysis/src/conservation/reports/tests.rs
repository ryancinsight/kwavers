use super::types::{ConservationReport, ConservationStatus, ReportMetadata};
use crate::conservation::detectors::ViolationStatistics;

#[test]
fn test_conservation_status_display() {
    assert_eq!(ConservationStatus::Excellent.to_string(), "Excellent");
    assert_eq!(ConservationStatus::Good.to_string(), "Good");
    assert_eq!(ConservationStatus::Warning.to_string(), "Warning");
    assert_eq!(ConservationStatus::Critical.to_string(), "Critical");
}

#[test]
fn test_report_generation() {
    let metadata = ReportMetadata {
        title: "Test Simulation".to_string(),
        time_start: 0.0,
        time_end: 1.0,
        timesteps: 100,
        grid_nx: 256,
        grid_ny: 256,
        grid_nz: 256,
        tolerance: 1e-6,
        generated_at: "2026-01-29T00:00:00Z".to_string(),
    };

    let violations = vec![];
    let statistics = ViolationStatistics {
        total_violations: 0,
        critical_violations: 0,
        max_relative_error: 0.0,
        average_relative_error: 0.0,
        violations_per_law: std::collections::HashMap::new(),
    };

    let report =
        ConservationReport::new("Test".to_string(), metadata, &violations, &statistics, 100);

    assert_eq!(report.status, ConservationStatus::Excellent);
    let text = report.to_text();
    assert!(text.contains("Excellent"));
}
