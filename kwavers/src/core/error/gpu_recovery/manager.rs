use crate::core::error::{KwaversError, SystemError};
use crate::core::error::gpu::GpuError;

use super::context::{GpuRecoveryStrategy, RecoveryContext, RecoveryResult};
use super::device_lost::DeviceLostRecovery;
use super::oom::{OomRecovery, OomRecoveryConfig};
use super::telemetry::{
    global_telemetry_snapshot, GpuRecoveryTelemetry, GpuStrategyLatencies, GpuStrategyRates,
};
use super::timeout::TimeoutRecovery;

/// Composite GPU Recovery Manager
///
/// Provides unified interface for GPU-specific recovery strategies.
#[derive(Debug)]
pub struct GpuRecoveryManager {
    device_lost: DeviceLostRecovery,
    oom: OomRecovery,
    timeout: TimeoutRecovery,
    custom_strategies: Vec<Box<dyn GpuRecoveryStrategy>>,
}

impl GpuRecoveryManager {
    pub fn new() -> Self {
        Self {
            device_lost: DeviceLostRecovery::new(),
            oom: OomRecovery::new(),
            timeout: TimeoutRecovery::new(),
            custom_strategies: Vec::new(),
        }
    }

    pub fn with_config(oom_config: OomRecoveryConfig, max_retries: u32) -> Self {
        Self {
            device_lost: DeviceLostRecovery::new(),
            oom: OomRecovery::with_config(oom_config),
            timeout: TimeoutRecovery::with_retries(max_retries),
            custom_strategies: Vec::new(),
        }
    }

    pub fn register_strategy(&mut self, strategy: Box<dyn GpuRecoveryStrategy>) {
        self.custom_strategies.push(strategy);
    }

    pub fn recover_gpu_error(&self, error: &GpuError, ctx: &RecoveryContext) -> RecoveryResult {
        if self.device_lost.can_handle(error) {
            return self.device_lost.recover(ctx);
        }

        if self.oom.can_handle(error) {
            return self.oom.recover(ctx);
        }

        if self.timeout.can_handle(error) {
            return self.timeout.recover(ctx);
        }

        for strategy in &self.custom_strategies {
            if strategy.can_handle(error) {
                return strategy.recover(ctx);
            }
        }

        Err(KwaversError::System(SystemError::InvalidOperation {
            operation: "GPU error recovery".to_string(),
            reason: "No recovery strategy can handle this error type".to_string(),
        }))
    }

    pub fn can_recover(&self, error: &GpuError) -> bool {
        self.device_lost.can_handle(error)
            || self.oom.can_handle(error)
            || self.timeout.can_handle(error)
            || self.custom_strategies.iter().any(|s| s.can_handle(error))
    }

    pub fn strategy_rates(&self) -> GpuStrategyRates {
        GpuStrategyRates {
            device_lost: self.device_lost.success_rate(),
            oom: self.oom.success_rate(),
            timeout: self.timeout.success_rate(),
        }
    }

    pub fn meets_threshold(&self) -> bool {
        self.strategy_rates().meets_threshold()
    }

    pub fn strategy_latencies(&self) -> GpuStrategyLatencies {
        GpuStrategyLatencies {
            device_lost: self.device_lost.avg_latency(),
            oom: self.oom.avg_latency(),
            timeout: self.timeout.avg_latency(),
        }
    }

    pub fn telemetry_snapshot(&self) -> GpuRecoveryTelemetry {
        GpuRecoveryTelemetry {
            rates: self.strategy_rates(),
            latencies: self.strategy_latencies(),
            global_stats: global_telemetry_snapshot(),
        }
    }
}

impl Default for GpuRecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}
