//! GPU backend implementation.
//!
//! Production GPU acceleration through a provider-generic Hephaestus boundary.
//! The default provider is WGPU because the current shader implementations are
//! WGSL; CUDA satisfies the provider identity/acquisition seam independently,
//! and belongs behind [`GpuComputeProvider`] only once Kwavers owns CUDA
//! kernels for the operations exposed here.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │     GPUBackend (Public API)         │
//! ├─────────────────────────────────────┤
//! │  Initialization  │  Device Manager  │
//! ├──────────────────┼──────────────────┤
//! │ Provider Dispatch (WGPU/CUDA seam)  │
//! ├─────────────────────────────────────┤
//! │ Hephaestus typed buffers and kernels│
//! │ Compute Shaders (WGSL today)        │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Provider-generic:** WGPU is the default provider; CUDA uses the same trait seam
//! - **Explicit availability:** device acquisition failures are surfaced to callers
//! - **Memory management:** Hephaestus typed transfer and pooled buffers
//! - **Pipeline caching:** Hephaestus caches monomorphized kernel pipelines
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers_gpu::backend::GPUBackend;
//!
//! // Create default WGPU-backed GPU backend
//! let mut backend = GPUBackend::new()?;
//!
//! // Execute provider-native operations
//! backend.dispatch_element_wise_multiply(&a, &b, &mut out)?;
//!
//! // Synchronize and retrieve results
//! backend.synchronize()?;
//! ```
//!
//! ## Performance Metadata
//!
//! Provider performance reporting is evidence-bound. WGPU reports unknown peak
//! throughput as `0.0` because portable WGPU adapter metadata does not expose a
//! calibrated FLOP/s model; providers with benchmark-backed models can override
//! the generic estimate method.
//!
//! ## References
//!
//! - Hephaestus: local Atlas GPU provider crates
//! - WGPU: <https://wgpu.rs/>
//! - WGSL Spec: <https://www.w3.org/TR/WGSL/>
//! - k-Wave GPU: CUDA implementation patterns
//! - WebGPU API: <https://www.w3.org/TR/webgpu/>

pub mod init;
pub mod performance_monitor;
pub mod physics_kernels;
pub mod provider;
pub mod realtime_loop;

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::backend::traits::{
    BackendCapabilities, BackendType, ComputeBackend, ComputeDevice,
};
use leto::Array3 as LetoArray3;
use physics_kernels::PhysicsKernelRegistry;
use realtime_loop::{RealtimeConfig, RealtimeSimulationOrchestrator};
use std::collections::HashMap;

pub use performance_monitor::{BudgetAnalysis, GpuStepMetrics};
pub use physics_kernels::{PhysicsKernel, WorkgroupConfig};
#[cfg(feature = "cuda-provider")]
pub use provider::CudaElementWiseProvider;
pub use provider::{
    ElementWiseMultiplyProvider, GpuComputeProvider, GpuKernelProvider, GpuProviderBackend,
    SpatialDerivativeProvider, WgpuComputeProvider,
};
pub use realtime_loop::{GpuRealtimeSimulationStatistics, StepResult};

/// GPU backend generic over a concrete Hephaestus provider.
///
/// Provides GPU-accelerated operations for ultrasound simulation through `P`.
/// Automatically handles device selection, memory management, and pipeline compilation.
#[derive(Debug)]
pub struct GPUBackend<P = WgpuComputeProvider>
where
    P: GpuComputeProvider,
{
    /// Concrete GPU provider implementation.
    provider: P,
}

impl GPUBackend<WgpuComputeProvider> {
    /// Create a new GPU backend
    ///
    /// Attempts to initialize WGPU with the following priority:
    /// 1. High-performance discrete GPU
    /// 2. Integrated GPU
    /// 3. Software renderer (fallback)
    ///
    /// Returns error if no suitable backend is available.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new() -> KwaversResult<Self> {
        WgpuComputeProvider::new().map(Self::from_provider)
    }
}

impl<P> GPUBackend<P>
where
    P: GpuComputeProvider,
{
    /// Create a backend from a concrete GPU provider.
    #[must_use]
    pub const fn from_provider(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the concrete GPU provider.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Get device name for logging
    pub fn device_name(&self) -> &str {
        self.provider.device_name()
    }

    /// Get available GPU memory (estimated)
    pub fn available_memory(&self) -> usize {
        self.provider.available_memory()
    }

    /// Execute provider-native element-wise multiplication.
    ///
    /// # Errors
    ///
    /// Propagates provider transfer, dispatch, or readback failures.
    pub fn dispatch_element_wise_multiply(
        &self,
        a: &LetoArray3<P::Scalar>,
        b: &LetoArray3<P::Scalar>,
        out: &mut LetoArray3<P::Scalar>,
    ) -> KwaversResult<()> {
        self.provider.element_wise_multiply(a, b, out)
    }

    /// Execute provider-native spatial derivative.
    ///
    /// # Errors
    ///
    /// Propagates provider dispatch failures or invalid derivative directions.
    pub fn dispatch_spatial_derivative(
        &self,
        field: &LetoArray3<P::Scalar>,
        direction: usize,
        out: &mut LetoArray3<P::Scalar>,
    ) -> KwaversResult<()> {
        self.provider
            .apply_spatial_derivative(field, direction, out)
    }
}

impl<P> ComputeBackend for GPUBackend<P>
where
    P: GpuComputeProvider,
{
    type Scalar = P::Scalar;

    fn backend_type(&self) -> BackendType {
        BackendType::GPU(self.provider.provider_kind())
    }

    fn capabilities(&self) -> BackendCapabilities {
        self.provider.capabilities()
    }

    fn is_available(&self) -> bool {
        self.provider.is_available()
    }

    fn synchronize(&self) -> KwaversResult<()> {
        self.provider.synchronize()
    }

    fn devices(&self) -> Vec<ComputeDevice> {
        self.provider.devices()
    }

    fn select_device(&mut self, device_id: usize) -> KwaversResult<()> {
        if device_id == 0 {
            Ok(())
        } else {
            Err(KwaversError::GpuError(format!(
                "device_id: GPU backend only has device 0, got {}",
                device_id
            )))
        }
    }

    fn element_wise_multiply(
        &self,
        a: &LetoArray3<Self::Scalar>,
        b: &LetoArray3<Self::Scalar>,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        self.provider.element_wise_multiply(a, b, out)
    }

    fn apply_spatial_derivative(
        &self,
        field: &LetoArray3<Self::Scalar>,
        direction: usize,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        self.provider
            .apply_spatial_derivative(field, direction, out)
    }

    fn estimate_performance(&self, problem_size: (usize, usize, usize)) -> f64 {
        self.provider.estimate_performance(problem_size)
    }
}

impl<P> GPUBackend<P>
where
    P: GpuComputeProvider,
{
    /// Create a real-time multiphysics simulation orchestrator
    ///
    /// Initializes a GPU-accelerated real-time loop with performance monitoring,
    /// physics kernel management, and budget enforcement for <10ms per-step execution.
    ///
    /// # Arguments
    ///
    /// * `config` - Real-time configuration (budget_ms, adaptive timestepping, CFL safety)
    ///
    /// # Returns
    ///
    /// A new RealtimeSimulationOrchestrator ready for GPU timesteps
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut backend = GPUBackend::new()?;
    /// let config = RealtimeConfig {
    ///     budget_ms: 10.0,
    ///     adaptive_timestepping: true,
    ///     cfl_safety_factor: 0.9,
    ///     ..Default::default()
    /// };
    /// let mut orchestrator = backend.create_realtime_orchestrator(config)?;
    /// ```
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn create_realtime_orchestrator(
        &self,
        config: RealtimeConfig,
    ) -> KwaversResult<RealtimeSimulationOrchestrator> {
        let kernel_registry = PhysicsKernelRegistry::new();
        RealtimeSimulationOrchestrator::new(config, kernel_registry)
    }

    /// Execute a single GPU multiphysics scheduling timestep.
    ///
    /// Validates the `leto` field state against the registered kernel schedule,
    /// records per-kernel execution estimates, and advances realtime budget
    /// accounting. Concrete WGPU or CUDA command dispatch stays behind the
    /// provider-specific kernel implementations.
    ///
    /// # Arguments
    ///
    /// * `fields` - Leto field arrays indexed by domain name
    /// * `dt` - Timestep size (seconds)
    /// * `time` - Current simulation time (seconds)
    /// * `grid` - Computational grid with spacing and extents
    /// * `orchestrator` - Real-time orchestrator managing GPU execution
    ///
    /// # Returns
    ///
    /// StepResult containing execution time, budget status, and kernel count
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn multiphysics_step(
        &self,
        fields: &mut HashMap<String, LetoArray3<f64>>,
        dt: f64,
        time: f64,
        grid: &kwavers_grid::Grid,
        orchestrator: &mut RealtimeSimulationOrchestrator,
    ) -> KwaversResult<StepResult> {
        orchestrator.step(fields, dt, time, grid)
    }
}

#[cfg(test)]
mod tests;