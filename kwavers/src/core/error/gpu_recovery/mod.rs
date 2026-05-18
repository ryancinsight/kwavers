//! GPU Error Recovery Strategies for Kwavers
//!
//! This module provides comprehensive GPU fault tolerance mechanisms including:
//! - Device loss recovery with wgpu reinitialization
//! - Out-of-memory recovery with CPU fallback
//! - Timeout recovery with exponential backoff
//! - Validation error recovery with error scope management

pub mod context;
pub mod device_lost;
pub mod fault_injection;
pub mod manager;
pub mod oom;
pub mod telemetry;
pub mod timeout;

pub use context::{GpuRecoveryStrategy, RecoveryContext, RecoveryResult};
pub use device_lost::{DeviceLostRecovery, DeviceRecoveryResult};
pub use fault_injection::{FaultInjectionConfig, FaultInjector, GpuFaultScenario};
pub use manager::GpuRecoveryManager;
pub use oom::{OomRecovery, OomRecoveryConfig, OomRecoveryResult};
pub use telemetry::{
    GlobalTelemetry, GpuRecoveryTelemetry, GpuStrategyLatencies, GpuStrategyRates,
};
pub use timeout::{TimeoutRecovery, TimeoutRecoveryResult};
