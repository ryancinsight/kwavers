//! Hephaestus provider initialization and device management.
//!
//! Handles backend-neutral device acquisition through Hephaestus. WGPU-specific
//! raw handles are exposed only on the WGPU specialization because the current
//! Kwavers shader kernels are WGSL.

use crate::gpu::{GpuDevice, GpuDeviceProvider};
use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::backend::traits::GpuProvider;

/// Provider context holding one acquired Hephaestus device.
#[derive(Debug)]
pub struct GpuProviderContext<P = WgpuDevice>
where
    P: GpuDeviceProvider,
{
    /// Hephaestus-owned provider device.
    device: GpuDevice<P>,
}

impl<P> GpuProviderContext<P>
where
    P: GpuDeviceProvider,
{
    /// Create a new provider context.
    ///
    /// Selection priority:
    /// 1. High-performance discrete GPU
    /// 2. Integrated GPU
    /// 3. Provider-specific software or compatibility device when available
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new() -> KwaversResult<Self> {
        Self::with_features_and_limits(
            P::acquisition_preference(),
            P::optional_features(),
            P::required_limits(),
        )
    }

    /// Create a provider context with operation-specific feature and limit
    /// requirements.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when the provider cannot acquire a matching device.
    pub fn with_features_and_limits(
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> KwaversResult<Self> {
        let device = P::try_acquire_device(
            P::acquisition_label(),
            device_preference,
            optional_features,
            required_limits,
        )
        .map_err(|e| KwaversError::GpuError(format!("GPU device: {e}")))?;

        Ok(Self {
            device: GpuDevice::from_provider(device),
        })
    }

    /// Get device name.
    pub fn device_name(&self) -> &str {
        &self.device.info().name
    }

    /// Get the provider identity.
    pub fn provider_kind(&self) -> GpuProvider {
        P::provider_kind()
    }

    /// Borrow the Hephaestus provider device.
    pub fn hephaestus_device(&self) -> &P {
        self.device.provider()
    }

    /// Synchronize queued provider work.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when device synchronization fails.
    pub fn synchronize(&self) -> KwaversResult<()> {
        self.device.synchronize()
    }
}

impl GpuProviderContext<WgpuDevice> {
    /// Get the raw WGPU device reference for WGSL shader modules.
    pub fn device(&self) -> &wgpu::Device {
        self.device.wgpu_device()
    }

    /// Get the raw WGPU queue reference for WGSL shader modules.
    pub fn queue(&self) -> &wgpu::Queue {
        self.device.wgpu_queue()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda-provider")]
    use hephaestus_cuda::CudaDevice;

    #[test]
    fn test_provider_context_creation() {
        // May fail if no GPU available
        match GpuProviderContext::<WgpuDevice>::new() {
            Ok(context) => {
                assert!(!context.device_name().is_empty());
                println!("GPU provider context created: {}", context.device_name());
            }
            Err(e) => {
                println!(
                    "GPU provider context creation failed (expected on some systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_wgpu_device_access() {
        if let Ok(context) = GpuProviderContext::<WgpuDevice>::new() {
            let device = context.device();
            let queue = context.queue();

            // Basic smoke tests
            assert!(device.limits().max_compute_invocations_per_workgroup > 0);
            queue.submit(std::iter::empty()); // Empty submission should work
        }
    }

    #[cfg(feature = "cuda-provider")]
    #[test]
    fn test_cuda_provider_context_is_type_valid() {
        assert!(std::mem::size_of::<GpuProviderContext<CudaDevice>>() > 0);
        assert_eq!(
            <CudaDevice as GpuDeviceProvider>::acquisition_label(),
            "kwavers-cuda-device"
        );
    }
}