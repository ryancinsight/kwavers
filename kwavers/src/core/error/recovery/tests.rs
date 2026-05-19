use crate::core::error::context::{ErrorContext, ErrorLocation};

use super::{
    CflViolationRecovery, ConvergenceFailureRecovery, ErrorRecoveryGpuOom, RecoveryAction,
    RecoveryAttempt, RecoveryBuilder, RecoveryStatistics, RecoveryStrategy,
};
use crate::core::error::{KwaversError, NumericalError};

fn create_test_context() -> ErrorContext {
    ErrorContext::new(ErrorLocation::new("test.rs", 1, "test"))
}

#[test]
fn gpu_oom_strategy_detects_oom_errors() {
    let strategy = ErrorRecoveryGpuOom::new();

    let oom_error = KwaversError::ResourceLimitExceeded {
        message: "GPU out of memory".to_string(),
    };
    assert!(strategy.can_handle(&oom_error));

    let gpu_error = KwaversError::GpuError("OOM on device 0".to_string());
    assert!(strategy.can_handle(&gpu_error));

    let other_error = KwaversError::InternalError("some error".to_string());
    assert!(!strategy.can_handle(&other_error));
}

#[test]
fn manager_returns_typed_recovery_action() {
    let mut manager = RecoveryBuilder::new().build();
    let context = create_test_context();
    let error = KwaversError::ResourceLimitExceeded {
        message: "GPU out of memory".to_string(),
    };

    let action = manager.recover_action(&error, &context).unwrap();
    assert_eq!(action, RecoveryAction::CpuFallback);
}

#[test]
fn compatibility_recover_returns_boxed_recovery_action() {
    let mut manager = RecoveryBuilder::new().build();
    let context = create_test_context();
    let error = KwaversError::ResourceLimitExceeded {
        message: "GPU out of memory".to_string(),
    };

    let payload = manager.recover(&error, &context).unwrap();
    let action = payload.downcast::<RecoveryAction>().unwrap();
    assert_eq!(*action, RecoveryAction::CpuFallback);
}

#[test]
fn cfl_strategy_detects_cfl_errors() {
    let strategy = CflViolationRecovery::new();
    let cfl_error = KwaversError::Numerical(NumericalError::InvalidOperation(
        "CFL condition violated".to_string(),
    ));
    assert!(strategy.can_handle(&cfl_error));
}

#[test]
fn convergence_strategy_limits_switches() {
    let strategy = ConvergenceFailureRecovery::new().with_max_switches(2);
    assert_eq!(strategy.switch_count(), 0);
}

#[test]
fn recovery_statistics_calculates_rates() {
    let stats = RecoveryStatistics {
        total_attempts: 10,
        successful_attempts: 9,
        overall_success_rate: 0.9,
        by_strategy: [("ErrorRecoveryGpuOom".to_string(), (5, 5))]
            .into_iter()
            .collect(),
    };

    assert!(stats.meets_threshold());
    assert_eq!(stats.strategy_success_rate("ErrorRecoveryGpuOom"), Some(1.0));
    assert_eq!(stats.strategy_success_rate("NonExistent"), None);
}

#[test]
fn builder_creates_manager_with_strategies() {
    let manager = RecoveryBuilder::new().build();
    let stats = manager.statistics();

    assert_eq!(stats.total_attempts, 0);
    assert_eq!(stats.overall_success_rate, 1.0);
}

#[test]
fn recovery_attempt_tracks_metadata() {
    let error = KwaversError::InternalError("test".to_string());
    let mut attempt = RecoveryAttempt::new("TestStrategy", &error);

    assert!(!attempt.succeeded);
    assert_eq!(attempt.strategy, "TestStrategy");

    attempt.mark_succeeded();
    assert!(attempt.succeeded);

    attempt.mark_failed(&KwaversError::InternalError("fail".to_string()));
    assert!(!attempt.succeeded);
    assert!(attempt.recovery_error.as_ref().unwrap().contains("fail"));
}

#[test]
fn cfl_reduction_factor_bounds() {
    let strategy = CflViolationRecovery::new();

    strategy.set_reduction_factor(0.01);
    assert_eq!(strategy.reduction_factor(), 0.125);

    strategy.set_reduction_factor(1.0);
    assert_eq!(strategy.reduction_factor(), 0.5);

    strategy.set_reduction_factor(0.3);
    assert!(strategy.reduction_factor() >= 0.125);
    assert!(strategy.reduction_factor() <= 0.5);
}
