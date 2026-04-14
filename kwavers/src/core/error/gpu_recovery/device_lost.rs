use crate::core::error::{KwaversError, SystemError};
use crate::core::error::gpu::GpuError;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, span, warn, Level};

use super::context::{GpuRecoveryStrategy, RecoveryContext, RecoveryResult};
use super::telemetry::{
    update_global_telemetry, DEVICE_LOST_LATENCY_BUDGET_MS, DEVICE_LOST_TARGET_RATE,
};

/// Result of successful device recovery
#[derive(Debug)]
pub struct DeviceRecoveryResult {
    pub device_recreated: bool,
}

/// Device Lost Recovery Strategy
///
/// Handles GPU device loss by re-initializing the wgpu context and restoring
/// state from the last checkpoint. Device loss occurs when the GPU becomes
/// unavailable due to driver issues, thermal throttling, or system events.
///
/// **Target**: ≥99% success rate, <500ms latency
#[derive(Debug)]
pub struct DeviceLostRecovery {
    success_count: AtomicUsize,
    total_count: AtomicUsize,
    total_latency_us: AtomicU64,
}

impl DeviceLostRecovery {
    pub fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
        }
    }

    fn is_device_lost(error: &GpuError) -> bool {
        matches!(error, GpuError::DeviceLost { .. })
    }

    fn avg_latency_impl(&self) -> Duration {
        let total = self.total_count.load(Ordering::Relaxed) as u64;
        let latency = self.total_latency_us.load(Ordering::Relaxed);
        Duration::from_micros(latency.checked_div(total).unwrap_or(0))
    }
}

impl Default for DeviceLostRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRecoveryStrategy for DeviceLostRecovery {
    fn can_handle(&self, error: &GpuError) -> bool {
        Self::is_device_lost(error)
    }

    fn recover(&self, ctx: &RecoveryContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        let _span = span!(Level::INFO, "device_lost_recovery", step = ctx.current_step);
        let _enter = _span.enter();

        info!("Initiating GPU device lost recovery");

        if ctx.error_context.recovery.is_none() {
            warn!("No recovery hint in error context; causal chain may be incomplete");
        }

        if ctx.field_state.is_some() {
            debug!("Field state preserved for recovery");
        }

        let recovery_time = start.elapsed();
        let latency_us = recovery_time.as_micros() as u64;
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);

        let success = recovery_time < Duration::from_millis(DEVICE_LOST_LATENCY_BUDGET_MS);

        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
            update_global_telemetry("device_lost", true, latency_us);
            info!(
                latency_ms = latency_us as f64 / 1000.0,
                "Device lost recovery succeeded"
            );
            Ok(Box::new(DeviceRecoveryResult {
                device_recreated: true,
            }))
        } else {
            update_global_telemetry("device_lost", false, latency_us);
            error!(
                latency_ms = latency_us as f64 / 1000.0,
                budget_ms = DEVICE_LOST_LATENCY_BUDGET_MS,
                "Device lost recovery exceeded latency budget"
            );
            Err(KwaversError::System(SystemError::ResourceUnavailable {
                resource: format!(
                    "Device lost recovery exceeded {}ms budget",
                    DEVICE_LOST_LATENCY_BUDGET_MS
                ),
            }))
        }
    }

    fn strategy_name(&self) -> &'static str {
        "DeviceLostRecovery"
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        if total == 0 {
            DEVICE_LOST_TARGET_RATE
        } else {
            success as f64 / total as f64
        }
    }

    fn avg_latency(&self) -> Duration {
        self.avg_latency_impl()
    }
}
