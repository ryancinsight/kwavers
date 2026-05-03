use super::metrics::BurnTrainingMetrics;

#[test]
fn test_new_metrics() {
    let metrics = BurnTrainingMetrics::with_capacity(100);
    assert_eq!(metrics.epochs_completed, 0);
    assert!(metrics.total_loss.is_empty());
    assert_eq!(metrics.training_time_secs, 0.0);
}

#[test]
fn test_record_epoch() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);
    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);

    assert_eq!(metrics.epochs_completed, 1);
    assert_eq!(metrics.total_loss.len(), 1);
    assert_eq!(metrics.total_loss[0], 0.1);
    assert_eq!(metrics.data_loss[0], 0.05);
    assert_eq!(metrics.pde_loss[0], 0.03);
    assert_eq!(metrics.bc_loss[0], 0.02);
}

#[test]
fn test_final_losses() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);
    assert!(metrics.final_total_loss().is_none());

    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    assert_eq!(metrics.final_total_loss(), Some(0.1));
    assert_eq!(metrics.final_data_loss(), Some(0.05));
    assert_eq!(metrics.final_pde_loss(), Some(0.03));
    assert_eq!(metrics.final_bc_loss(), Some(0.02));
}

#[test]
fn test_convergence_detection() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);

    assert!(!metrics.is_converged(1e-6));

    metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    metrics.record_epoch(0.5, 0.25, 0.15, 0.1);
    assert!(!metrics.is_converged(1e-6));

    metrics.record_epoch(0.5000001, 0.25, 0.15, 0.1);
    assert!(metrics.is_converged(1e-5));
}

#[test]
fn test_average_loss_last_n() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);

    assert!(metrics.average_loss_last_n(3).is_none());

    metrics.record_epoch(0.3, 0.1, 0.1, 0.1);
    metrics.record_epoch(0.2, 0.1, 0.05, 0.05);
    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);

    let avg = metrics.average_loss_last_n(3).unwrap();
    assert!((avg - 0.2).abs() < 1e-10);
}

#[test]
fn test_throughput() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);

    assert!(metrics.throughput().is_none());

    for _ in 0..100 {
        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    }
    metrics.training_time_secs = 10.0;

    let throughput = metrics.throughput().unwrap();
    assert_eq!(throughput, 10.0);
}

#[test]
fn test_numerical_issues_detection() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);
    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    assert!(!metrics.has_numerical_issues());

    metrics.record_epoch(f64::NAN, 0.05, 0.03, 0.02);
    assert!(metrics.has_numerical_issues());
}

#[test]
fn test_numerical_issues_infinity() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);
    metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    assert!(!metrics.has_numerical_issues());

    metrics.record_epoch(f64::INFINITY, 0.05, 0.03, 0.02);
    assert!(metrics.has_numerical_issues());
}

#[test]
fn test_loss_reduction_percent() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);

    assert!(metrics.loss_reduction_percent().is_none());

    metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    metrics.record_epoch(0.5, 0.25, 0.15, 0.1);

    let reduction = metrics.loss_reduction_percent().unwrap();
    assert_eq!(reduction, 50.0);
}

#[test]
fn test_training_duration() {
    let mut metrics = BurnTrainingMetrics::with_capacity(10);
    metrics.training_time_secs = 12.5;

    let duration = metrics.training_duration();
    assert_eq!(duration.as_secs_f64(), 12.5);
}
