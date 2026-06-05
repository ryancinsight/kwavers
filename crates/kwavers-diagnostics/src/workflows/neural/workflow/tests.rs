use super::*;

#[test]
fn test_workflow_creation() {
    let workflow = RealTimeWorkflow::new();
    assert!(workflow.performance_history.is_empty());
    assert!(workflow.quality_metrics.is_empty());

    let default_workflow = RealTimeWorkflow::default();
    assert!(default_workflow.performance_history.is_empty());
}

#[test]
fn test_median_computation() {
    let workflow = RealTimeWorkflow::new();

    // Odd number of elements
    let values_odd = vec![5.0, 3.0, 7.0, 1.0, 9.0];
    assert_eq!(workflow.compute_median(&values_odd), 5.0);

    // Even number of elements
    let values_even = vec![2.0, 4.0, 6.0, 8.0];
    assert_eq!(workflow.compute_median(&values_even), 5.0); // (4.0 + 6.0) / 2.0
}

#[test]
fn test_performance_stats() {
    let mut workflow = RealTimeWorkflow::new();

    // Add performance data
    workflow
        .performance_history
        .extend_from_slice(&[50.0, 60.0, 55.0, 65.0, 70.0]);
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 60.0);
    workflow
        .quality_metrics
        .insert("diagnostic_confidence".to_string(), 0.9);

    let stats = workflow.get_performance_stats();
    assert_eq!(stats.get("min_time"), Some(&50.0));
    assert_eq!(stats.get("max_time"), Some(&70.0));
    assert_eq!(stats.get("median_time"), Some(&60.0));
    assert_eq!(stats.get("diagnostic_confidence"), Some(&0.9));
}

#[test]
fn test_rolling_window() {
    let mut workflow = RealTimeWorkflow::new();

    // Add 150 measurements
    for i in 0..150 {
        workflow.record_processing_time_ms(i as f64);
    }

    // Should maintain only last 100
    assert_eq!(workflow.performance_history.len(), 100);
    assert_eq!(workflow.performance_history[0], 50.0); // First element is 50th measurement
    assert_eq!(workflow.performance_history[99], 149.0); // Last element is 149th measurement
}

#[test]
fn test_performance_target() {
    let mut workflow = RealTimeWorkflow::new();

    // Meets target
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 85.0);
    assert!(workflow.meets_performance_target());

    // Exceeds target
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 110.0);
    assert!(!workflow.meets_performance_target());
}

#[test]
fn test_health_status() {
    let mut workflow = RealTimeWorkflow::new();

    // EXCELLENT
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 75.0);
    workflow
        .quality_metrics
        .insert("diagnostic_confidence".to_string(), 0.95);
    assert_eq!(workflow.get_health_status(), "EXCELLENT");

    // GOOD
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 90.0);
    workflow
        .quality_metrics
        .insert("diagnostic_confidence".to_string(), 0.85);
    assert_eq!(workflow.get_health_status(), "GOOD");

    // ACCEPTABLE
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 110.0);
    workflow
        .quality_metrics
        .insert("diagnostic_confidence".to_string(), 0.75);
    assert_eq!(workflow.get_health_status(), "ACCEPTABLE");

    // DEGRADED
    workflow
        .quality_metrics
        .insert("avg_processing_time".to_string(), 150.0);
    workflow
        .quality_metrics
        .insert("diagnostic_confidence".to_string(), 0.6);
    assert_eq!(workflow.get_health_status(), "DEGRADED");
}

#[test]
fn test_reset() {
    let mut workflow = RealTimeWorkflow::new();

    workflow.performance_history.push(50.0);
    workflow
        .quality_metrics
        .insert("avg_time".to_string(), 50.0);

    workflow.reset();

    assert!(workflow.performance_history.is_empty());
    assert!(workflow.quality_metrics.is_empty());
}

#[test]
fn test_execution_count() {
    let mut workflow = RealTimeWorkflow::new();
    assert_eq!(workflow.execution_count(), 0);

    workflow
        .performance_history
        .extend_from_slice(&[50.0, 60.0, 70.0]);
    assert_eq!(workflow.execution_count(), 3);
}
