use super::loss::MetaLoss;
use super::stats::MetaLearningStats;

#[test]
fn test_meta_loss_default() {
    let loss = MetaLoss::default();
    assert_eq!(loss.total_loss, 0.0);
    assert_eq!(loss.physics_loss, 0.0);
    assert_eq!(loss.generalization_score, 0.0);
    assert!(loss.task_losses.is_empty());
}

#[test]
fn test_meta_loss_new() {
    let task_losses = vec![0.1, 0.15, 0.12, 0.13];
    let physics_loss = 0.05;
    let loss = MetaLoss::new(task_losses.clone(), physics_loss);

    assert_eq!(loss.task_losses, task_losses);
    assert_eq!(loss.physics_loss, physics_loss);
    assert!((loss.total_loss - 0.125).abs() < 1e-10);
    assert!(loss.generalization_score > 0.0);
}

#[test]
fn test_meta_loss_generalization_score() {
    // Perfect consistency (all same values)
    let perfect = vec![0.1, 0.1, 0.1, 0.1];
    let loss_perfect = MetaLoss::new(perfect, 0.0);
    assert!(loss_perfect.generalization_score > 0.99);

    // Poor consistency (high variance)
    let poor = vec![0.01, 0.1, 0.5, 1.0];
    let loss_poor = MetaLoss::new(poor, 0.0);
    assert!(loss_poor.generalization_score < 0.7);
}

#[test]
fn test_meta_loss_worst_best_task() {
    let task_losses = vec![0.1, 0.5, 0.2, 0.3];
    let loss = MetaLoss::new(task_losses, 0.0);

    assert_eq!(loss.worst_task_loss(), Some(0.5));
    assert_eq!(loss.best_task_loss(), Some(0.1));
}

#[test]
fn test_meta_loss_std_dev() {
    let task_losses = vec![0.1, 0.2, 0.3, 0.4];
    let loss = MetaLoss::new(task_losses, 0.0);

    let std = loss.task_loss_std_dev();
    assert!(std > 0.0);
    assert!(std < 0.2); // Should be reasonable
}

#[test]
fn test_meta_loss_is_converged() {
    let task_losses = vec![0.01, 0.015, 0.012];
    let loss = MetaLoss::new(task_losses, 0.005);

    assert!(loss.is_converged(0.02, 0.01, 0.8));
    assert!(!loss.is_converged(0.01, 0.01, 0.8)); // Total loss too high
}

#[test]
fn test_stats_default() {
    let stats = MetaLearningStats::default();
    assert_eq!(stats.meta_epochs_completed, 0);
    assert_eq!(stats.total_tasks_processed, 0);
    assert_eq!(stats.average_meta_loss, 0.0);
    assert_eq!(stats.best_generalization_score, 0.0);
}

#[test]
fn test_stats_update() {
    let mut stats = MetaLearningStats::new();

    stats.update(1, 4, 0.5, 0.7);
    assert_eq!(stats.meta_epochs_completed, 1);
    assert_eq!(stats.total_tasks_processed, 4);
    assert_eq!(stats.average_meta_loss, 0.5);
    assert_eq!(stats.best_generalization_score, 0.7);

    stats.update(2, 4, 0.3, 0.8);
    assert_eq!(stats.meta_epochs_completed, 2);
    assert_eq!(stats.total_tasks_processed, 8);
    assert!(stats.best_generalization_score >= 0.8);
}

#[test]
fn test_stats_adaptation_time() {
    let mut stats = MetaLearningStats::new();

    stats.update_adaptation_time(1.0);
    assert_eq!(stats.average_adaptation_time, 1.0);

    stats.update_adaptation_time(2.0);
    // Should be weighted average (closer to 1.0 due to α=0.1)
    assert!(stats.average_adaptation_time > 1.0);
    assert!(stats.average_adaptation_time < 1.5);
}

#[test]
fn test_stats_tasks_per_epoch() {
    let mut stats = MetaLearningStats::new();
    stats.update(2, 8, 0.5, 0.7);
    stats.update(2, 8, 0.4, 0.8);

    assert_eq!(stats.tasks_per_epoch(), 8.0);
}

#[test]
fn test_stats_is_converging() {
    let mut stats = MetaLearningStats::new();
    stats.convergence_rate = 0.001;
    stats.best_generalization_score = 0.85;

    assert!(stats.is_converging(0.01, 0.8));
    assert!(!stats.is_converging(0.0001, 0.8)); // Rate threshold too tight
    assert!(!stats.is_converging(0.01, 0.9)); // Gen threshold too high
}

#[test]
fn test_stats_reset() {
    let mut stats = MetaLearningStats::new();
    stats.update(10, 40, 0.2, 0.9);

    stats.reset();
    assert_eq!(stats.meta_epochs_completed, 0);
    assert_eq!(stats.total_tasks_processed, 0);
    assert_eq!(stats.average_meta_loss, 0.0);
}

#[test]
fn test_generalization_score_edge_cases() {
    // Empty task losses
    let empty = Vec::new();
    let loss_empty = MetaLoss::new(empty, 0.0);
    assert_eq!(loss_empty.generalization_score, 0.0);

    // Single task
    let single = vec![0.1];
    let loss_single = MetaLoss::new(single, 0.0);
    assert_eq!(loss_single.generalization_score, 1.0); // Perfect consistency

    // All zeros
    let zeros = vec![0.0, 0.0, 0.0];
    let loss_zeros = MetaLoss::new(zeros, 0.0);
    assert_eq!(loss_zeros.generalization_score, 1.0);
}
