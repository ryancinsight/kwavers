//! GPU context alias.
//!
//! Re-exports [`CoreGpuContext`] as the primary provider-generic GPU context
//! type. The default provider remains Hephaestus WGPU because the current
//! shader modules are WGSL, but callers can name another provider explicitly
//! once that provider satisfies [`crate::gpu::GpuDeviceProvider`].

use super::CoreGpuContext;
use hephaestus_wgpu::WgpuDevice;

/// Provider-generic GPU context for acoustic simulations.
///
/// `GpuBackend` preserves the existing WGPU default. `GpuBackend<P>` exposes
/// the provider parameter for WGPU, CUDA, and future Hephaestus-backed device
/// providers without adding backend-specific branches at call sites.
pub type GpuBackend<P = WgpuDevice> = CoreGpuContext<P>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuDeviceProvider;

    fn assert_alias_accepts_provider<P>()
    where
        P: GpuDeviceProvider,
    {
        let _ = core::mem::size_of::<GpuBackend<P>>();
    }

    #[test]
    fn gpu_backend_alias_exposes_provider_parameter() {
        assert_alias_accepts_provider::<WgpuDevice>();
    }

    #[cfg(feature = "cuda-provider")]
    #[test]
    fn gpu_backend_alias_accepts_cuda_provider() {
        assert_alias_accepts_provider::<hephaestus_cuda::CudaDevice>();
    }
}
