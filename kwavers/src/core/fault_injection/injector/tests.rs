use super::*;
use crate::core::error::{ErrorContext, KwaversError, RecoveryStrategy};
use crate::core::fault_injection::scenario::{FaultInjectionScenario, InjectionTiming};
use std::any::Any;

#[derive(Debug)]
struct MockRecoveryStrategy {
    should_succeed: bool,
}

impl RecoveryStrategy for MockRecoveryStrategy {
    fn recover(
        &self,
        _error: &KwaversError,
        _context: &ErrorContext,
    ) -> crate::core::error::RecoveryResult {
        if self.should_succeed {
            Ok(Box::new(()) as Box<dyn Any + Send>)
        } else {
            Err(KwaversError::InternalError("Recovery failed".to_string()))
        }
    }

    fn can_handle(&self, error: &KwaversError) -> bool {
        matches!(error, KwaversError::ResourceLimitExceeded { .. })
    }

    fn strategy_name(&self) -> &'static str {
        "MockStrategy"
    }

    fn success_rate(&self) -> f64 {
        0.95
    }
}

#[test]
fn injector_creates_oom_error() {
    let injector = FaultInjector::new(InjectionConfig::default());
    let scenario = FaultInjectionScenario::GpuOomSudden {
        allocation_size_bytes: 1024usize.pow(3), // 1GB
        timing: InjectionTiming::Immediate,
    };

    let result = injector.inject_fault(&scenario);

    assert!(result.injected);
    assert!(result.causal_chain_preserved);
    assert!(matches!(
        result.generated_error,
        Some(KwaversError::ResourceLimitExceeded { .. })
    ));
}

#[test]
fn injector_disabled_returns_no_fault() {
    let config = InjectionConfig {
        enabled: false,
        ..Default::default()
    };
    let injector = FaultInjector::new(config);
    let scenario = FaultInjectionScenario::GpuOomSudden {
        allocation_size_bytes: 1024,
        timing: InjectionTiming::Immediate,
    };

    let result = injector.inject_fault(&scenario);

    assert!(!result.injected);
    assert!(result.generated_error.is_none());
}

#[test]
fn injector_tracks_faults() {
    let injector = FaultInjector::new(InjectionConfig::default());
    let scenario = FaultInjectionScenario::CflViolation {
        overshoot_factor: 1.5,
        timing: InjectionTiming::Immediate,
    };

    let initial_count = injector.injection_count();
    injector.inject_fault(&scenario);
    injector.inject_fault(&scenario);

    assert_eq!(injector.injection_count(), initial_count + 2);
}
