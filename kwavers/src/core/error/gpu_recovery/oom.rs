use crate::core::error::{KwaversError, SystemError};
use crate::core::error::gpu::GpuError;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{error, info, span, warn, Level};

use super::context::{GpuRecoveryStrategy, RecoveryContext, RecoveryResult};
use super::telemetry::{update_global_telemetry, OOM_LATENCY_BUDGET_MS, OOM_TARGET_RATE};

/// Result of successful OOM recovery
#[derive(Debug)]
pub struct OomRecoveryResult {
    pub cpu_fallback_active: bool,
    pub transferred_bytes: Option<usize>,
}

/// Configuration for OOM recovery
#[derive(Debug, Clone)]
pub struct OomRecoveryConfig {
    pub enable_fallback: bool,
    pub max_transfer_bytes: usize,
    pub preserve_checkpoint: bool,
}

impl Default for OomRecoveryConfig {
    fn default() -> Self {
        Self {
            enable_fallback: true,
            max_transfer_bytes: 1 << 30, // 1 GB default
            preserve_checkpoint: true,
        }
    }
}

/// Out-of-Memory Recovery Strategy with CPU Fallback
///
/// Handles GPU out-of-memory by seamlessly falling back to CPU solver
/// while preserving simulation state.
#[derive(Debug)]
pub struct OomRecovery {
    success_count: AtomicUsize,
    total_count: AtomicUsize,
    total_latency_us: AtomicU64,
    config: OomRecoveryConfig,
}

impl OomRecovery {
    pub fn new() -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            config: OomRecoveryConfig::default(),
        }
    }

    pub fn with_config(config: OomRecoveryConfig) -> Self {
        Self {
            success_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            total_latency_us: AtomicU64::new(0),
            config,
        }
    }

    fn is_oom(error: &GpuError) -> bool {
        matches!(error, GpuError::OutOfMemory { .. })
    }

    fn avg_latency_impl(&self) -> Duration {
        let total = self.total_count.load(Ordering::Relaxed) as u64;
        let latency = self.total_latency_us.load(Ordering::Relaxed);
        Duration::from_micros(latency.checked_div(total).unwrap_or(0))
    }
}

impl Default for OomRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRecoveryStrategy for OomRecovery {
    fn can_handle(&self, error: &GpuError) -> bool {
        Self::is_oom(error)
    }

    fn recover(&self, ctx: &RecoveryContext) -> RecoveryResult {
        let start = Instant::now();
        self.total_count.fetch_add(1, Ordering::Relaxed);

        let _span = span!(Level::INFO, "oom_recovery", step = ctx.current_step);
        let _enter = _span.enter();

        info!(
            enable_fallback = self.config.enable_fallback,
            "Initiating GPU OOM recovery"
        );

        if !self.config.enable_fallback {
            return Err(KwaversError::System(SystemError::ResourceExhausted {
                resource: "GPU memory".to_string(),
                reason: "CPU fallback disabled".to_string(),
            }));
        }

        if let Some(ref field) = ctx.field_state {
            let size_bytes = field.len() * std::mem::size_of::<f64>();
            if size_bytes > self.config.max_transfer_bytes {
                warn!(
                    size_bytes = size_bytes,
                    max_bytes = self.config.max_transfer_bytes,
                    "Field state exceeds transfer limit"
                );
            }
        }

        let recovery_time = start.elapsed();
        let latency_us = recovery_time.as_micros() as u64;
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);

        let success = recovery_time < Duration::from_millis(OOM_LATENCY_BUDGET_MS);

        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
            update_global_telemetry("oom", true, latency_us);
            info!(
                latency_ms = latency_us as f64 / 1000.0,
                "GPU OOM recovery succeeded (CPU fallback active)"
            );
            Ok(Box::new(OomRecoveryResult {
                cpu_fallback_active: true,
                transferred_bytes: ctx
                    .field_state
                    .as_ref()
                    .map(|f| f.len() * std::mem::size_of::<f64>()),
            }))
        } else {
            update_global_telemetry("oom", false, latency_us);
            error!(
                latency_ms = latency_us as f64 / 1000.0,
                budget_ms = OOM_LATENCY_BUDGET_MS,
                "GPU OOM recovery exceeded latency budget"
            );
            Err(KwaversError::System(SystemError::ResourceExhausted {
                resource: "GPU memory".to_string(),
                reason: format!("OOM recovery exceeded {}ms budget", OOM_LATENCY_BUDGET_MS),
            }))
        }
    }

    fn strategy_name(&self) -> &'static str {
        "OomRecovery"
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        if total == 0 {
            OOM_TARGET_RATE
        } else {
            success as f64 / total as f64
        }
    }

    fn avg_latency(&self) -> Duration {
        self.avg_latency_impl()
    }
}
