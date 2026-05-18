// Fault Injection Framework for Recovery Validation
//
// This module provides comprehensive fault injection capabilities for validating
// recovery strategies under controlled failure conditions.

pub mod injector;
pub mod recovery_stats;
pub mod scenario;

pub use injector::InjectionConfig;
pub use recovery_stats::RecoveryStats;
pub use scenario::{FaultInjectionScenario, InjectionTiming, RecoveryExpectation};
