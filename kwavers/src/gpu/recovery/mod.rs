// GPU Recovery Strategies for Sprint 220
//
// Implements comprehensive GPU fault tolerance mechanisms including device loss
// recovery, OOM fallback to CPU, and timeout handling.
//
// ## Mathematical Specification
//
// **THEOREM: GPU Recovery Success Rate**
// For recovery strategy R and GPU failure type F:
// P(R succeeds | F) ≥ RECOVERY_SUCCESS_THRESHOLD
//
// **Proof**: Each strategy implements specific remediation:
// - DeviceLost: Re-initialize wgpu context, restore from checkpoint
// - OOM: Migrate to CPU solver, preserve simulation state
// - Timeout: Retry with exponential backoff, reduce batch size
//
// **THEOREM: Recovery Latency Bound**
// E[recovery_time] ≤ LATENCY_BUDGET for all strategies
//
// ## Recovery Strategy Targets
//
// | Strategy | Target Rate | Latency Budget | Implementation |
// |----------|-------------|----------------|----------------|
// | GpuDeviceLostRecovery | ≥99% | <500ms | Context re-init |
// | GpuRecoveryOom | ≥95% | <100ms | CPU fallback |
// | GpuTimeoutRecovery | ≥90% | <200ms | Retry w/ backoff |
// | ValidationRecovery | ≥95% | <50ms | Scope recovery |
//
// ## References
//
// - Nygard (2007) "Release It!" ISBN: 978-0978739218
// - wgpu error scopes: https://docs.rs/wgpu/latest/wgpu/struct.Device.html

pub mod checkpoint;
pub mod device_lost;
pub mod error_scope;
pub mod injector;
pub mod manager;
pub mod oom;
pub mod stats;
pub mod timeout;

pub use checkpoint::*;
pub use device_lost::*;
pub use error_scope::*;
pub use injector::*;
pub use manager::*;
pub use oom::*;
pub use stats::*;
pub use timeout::*;

// ── GPU Error Type & Recovery Action ────────────────────────────────────────

/// Discriminant for fault-injection targets.
///
/// Used by [`GpuInjectorFaultInjector`] to produce the correct [`KwaversError`] variant
/// for each simulated GPU failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuErrorType {
    /// GPU ran out of device memory.
    OutOfMemory,
    /// GPU device was lost (driver crash, hot-unplug, TDR).
    DeviceLost,
    /// A GPU operation exceeded the timeout budget.
    Timeout,
}

/// Signal returned from a recovery strategy to the GPU time loop.
///
/// The `recover()` method cannot perform async work (wgpu re-init is async)
/// or mutate the simulation state directly (the strategy holds no solver ref).
/// Instead it returns a `GpuRecoveryAction` that the time loop dispatches.
///
/// Reference: Gamma, E. et al. (1994). *Design Patterns*. Addison-Wesley. §Command pattern.
#[derive(Debug)]
pub enum GpuRecoveryAction {
    /// OOM recovery: caller should switch to CPU solver using the checkpoint state.
    ///
    /// The time loop should:
    /// 1. Construct a CPU PSTD/FDTD solver from the checkpoint fields.
    /// 2. Resume from `checkpoint.step + 1`.
    FallbackToCpu { checkpoint: GpuCheckpoint },

    /// Device-lost recovery: caller must re-initialize wgpu asynchronously, then
    /// re-upload checkpoint fields and resume.
    ///
    /// The time loop should:
    /// 1. Re-request adapter and device via `wgpu::Instance::new()`.
    /// 2. Rebuild all pipelines and buffers.
    /// 3. Upload checkpoint fields to new GPU buffers.
    /// 4. Resume from `checkpoint.step + 1`.
    ReinitializeDevice { checkpoint: GpuCheckpoint },

    /// Timeout recovery: caller should retry the failed operation immediately
    /// (the backoff sleep has already been applied in `recover()`).
    Retry,
}
