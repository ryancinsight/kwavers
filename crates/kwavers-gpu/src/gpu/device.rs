//! Backend-neutral GPU device management.
//!
//! This module wraps the Hephaestus device-acquisition and capability traits so
//! kwavers code can bind to a generic GPU provider. WGPU-specific shader code
//! uses the `GpuDevice<WgpuDevice>` specialization for raw `wgpu` handles; a
//! CUDA provider can implement [`GpuDeviceProvider`] without changing generic
//! callers.

use hephaestus_core::{ComputeDevice, ComputeDeviceAcquisition};
pub use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};
#[cfg(feature = "cuda-provider")]
use hephaestus_cuda::CudaDevice;
use hephaestus_wgpu::WgpuDevice;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use kwavers_solver::backend::traits::GpuProvider;

/// Information about a GPU device.
///
/// Contains metadata useful for diagnostics and device selection. Backends
/// supply these values through [`GpuDeviceProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// PCI vendor ID, or `0` when the backend does not expose one.
    pub vendor: u32,
    /// Device type, such as `DiscreteGpu`, `IntegratedGpu`, or `Cuda`.
    pub device_type: String,
    /// Provider backend, such as `Vulkan`, `Dx12`, `Metal`, or `cuda`.
    pub backend: String,
}

/// Kwavers GPU provider contract.
///
/// This trait keeps high-level device management generic over Hephaestus
/// providers. WGPU, CUDA, and future providers supply metadata and inherit
/// acquisition, limits, feature checks, transfers, and synchronization from
/// Hephaestus' backend-neutral traits.
pub trait GpuDeviceProvider: ComputeDeviceAcquisition {
    /// Provider identity surfaced at the solver/backend boundary.
    fn provider_kind() -> GpuProvider;

    /// Label used when acquiring a provider device.
    fn acquisition_label() -> &'static str;

    /// Device preference used for the default Kwavers GPU context.
    fn acquisition_preference() -> DevicePreference {
        DevicePreference::HighPerformance
    }

    /// Optional provider features requested by the default Kwavers GPU context.
    fn optional_features() -> &'static [DeviceFeature];

    /// Minimum provider limits required by the default Kwavers GPU context.
    fn required_limits() -> DeviceLimits;

    /// Return provider metadata for diagnostics.
    fn device_info(&self) -> GpuDeviceInfo;

    /// Return whether this provider supports the atomic operations required by
    /// Kwavers core GPU kernels.
    fn supports_core_atomics() -> bool {
        false
    }
}

impl GpuDeviceProvider for WgpuDevice {
    fn provider_kind() -> GpuProvider {
        GpuProvider::Wgpu
    }

    fn acquisition_label() -> &'static str {
        "kwavers-wgpu-device"
    }

    fn optional_features() -> &'static [DeviceFeature] {
        &[DeviceFeature::ShaderF64]
    }

    fn required_limits() -> DeviceLimits {
        let base = minimal_compute_limits();
        DeviceLimits {
            max_buffer_size: base.max_buffer_size,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage,
            max_buffers_and_acceleration_structures_per_shader_stage: base
                .max_buffers_and_acceleration_structures_per_shader_stage,
            max_immediate_size: 0,
        }
    }

    fn device_info(&self) -> GpuDeviceInfo {
        match self.adapter_info() {
            Some(info) => GpuDeviceInfo {
                name: info.name.clone(),
                vendor: info.vendor,
                device_type: format!("{:?}", info.device_type),
                backend: format!("{:?}", info.backend),
            },
            None => GpuDeviceInfo {
                name: self.backend_name().to_string(),
                vendor: 0,
                device_type: "Unknown".to_string(),
                backend: self.backend_name().to_string(),
            },
        }
    }

    fn supports_core_atomics() -> bool {
        true
    }
}

#[cfg(feature = "cuda-provider")]
impl GpuDeviceProvider for CudaDevice {
    fn provider_kind() -> GpuProvider {
        GpuProvider::Cuda
    }

    fn acquisition_label() -> &'static str {
        "kwavers-cuda-device"
    }

    fn optional_features() -> &'static [DeviceFeature] {
        &[]
    }

    fn required_limits() -> DeviceLimits {
        let base = minimal_compute_limits();
        DeviceLimits {
            max_buffer_size: base.max_buffer_size,
            max_compute_workgroup_size_x: base.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: base.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: base.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: base.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: None,
            max_buffers_and_acceleration_structures_per_shader_stage: None,
            max_immediate_size: 0,
        }
    }

    fn device_info(&self) -> GpuDeviceInfo {
        GpuDeviceInfo {
            name: self.backend_name().to_string(),
            vendor: 0,
            device_type: "Cuda".to_string(),
            backend: self.backend_name().to_string(),
        }
    }
}

/// GPU device wrapper over a concrete Hephaestus provider.
///
/// The default provider is Hephaestus WGPU because the current kwavers shader
/// modules are WGSL. Generic code should use the backend-neutral methods on
/// this type; raw WGPU handles are intentionally available only on
/// `GpuDevice<WgpuDevice>`.
#[derive(Debug, Clone)]
pub struct GpuDevice<P = WgpuDevice>
where
    P: GpuDeviceProvider,
{
    provider: P,
    info: GpuDeviceInfo,
    limits: DeviceLimits,
}

impl<P> GpuDevice<P>
where
    P: GpuDeviceProvider,
{
    /// Acquire a device using a backend-neutral preference.
    ///
    /// # Errors
    ///
    /// Returns an error if the concrete provider cannot acquire a matching
    /// device or satisfy the requested baseline limits.
    pub async fn create(device_preference: DevicePreference) -> KwaversResult<Self> {
        Self::try_create(device_preference)
    }

    /// Acquire a device synchronously using a backend-neutral preference.
    ///
    /// # Errors
    ///
    /// Returns an error if the concrete provider cannot acquire a matching
    /// device or satisfy the requested baseline limits.
    pub fn try_create(device_preference: DevicePreference) -> KwaversResult<Self> {
        Self::try_create_with_features_and_limits(device_preference, &[], minimal_compute_limits())
    }

    /// Acquire a device with requested optional features and minimum limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the concrete provider cannot acquire a matching
    /// device, enable a requested optional feature, or satisfy `required_limits`.
    pub async fn create_with_features_and_limits(
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> KwaversResult<Self> {
        Self::try_create_with_features_and_limits(
            device_preference,
            optional_features,
            required_limits,
        )
    }

    /// Acquire a device synchronously with requested optional features and limits.
    ///
    /// # Errors
    ///
    /// Returns an error if the concrete provider cannot acquire a matching
    /// device, enable a requested optional feature, or satisfy `required_limits`.
    pub fn try_create_with_features_and_limits(
        device_preference: DevicePreference,
        optional_features: &[DeviceFeature],
        required_limits: DeviceLimits,
    ) -> KwaversResult<Self> {
        let provider = P::try_acquire_device(
            P::acquisition_label(),
            device_preference,
            optional_features,
            required_limits,
        )
        .map_err(|e| {
            KwaversError::Config(ConfigError::InvalidValue {
                parameter: "gpu_device".to_string(),
                value: format!("{e}"),
                constraint: "Failed to acquire Hephaestus GPU device".to_string(),
            })
        })?;

        Ok(Self::from_provider(provider))
    }

    /// Construct a device wrapper from an already-acquired provider.
    #[must_use]
    pub fn from_provider(provider: P) -> Self {
        let info = provider.device_info();
        let limits = provider.device_limits();
        Self {
            provider,
            info,
            limits,
        }
    }

    /// Borrow the concrete Hephaestus provider.
    #[must_use]
    pub const fn provider(&self) -> &P {
        &self.provider
    }

    /// Get device information.
    #[must_use]
    pub const fn info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Get backend-neutral device limits.
    #[must_use]
    pub const fn limits(&self) -> &DeviceLimits {
        &self.limits
    }

    /// Check whether the acquired provider enabled a backend-neutral feature.
    #[must_use]
    pub fn supports_feature(&self, feature: DeviceFeature) -> bool {
        self.provider.supports_device_feature(feature)
    }

    /// Synchronize queued provider work.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when provider synchronization fails.
    pub fn synchronize(&self) -> KwaversResult<()> {
        self.provider
            .synchronize()
            .map_err(|e| KwaversError::GpuError(format!("GPU synchronize: {e}")))
    }
}

impl GpuDevice<WgpuDevice> {
    /// Borrow the underlying WGPU device for WGSL shader modules.
    #[must_use]
    pub fn wgpu_device(&self) -> &wgpu::Device {
        self.provider.inner()
    }

    /// Borrow the underlying WGPU queue for WGSL shader modules.
    #[must_use]
    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        self.provider.queue()
    }
}

pub(crate) const fn minimal_compute_limits() -> DeviceLimits {
    DeviceLimits {
        max_buffer_size: 128 * 1024 * 1024,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_storage_size: 16 * 1024,
        max_storage_buffers_per_shader_stage: Some(8),
        max_buffers_and_acceleration_structures_per_shader_stage: Some(8),
        max_immediate_size: 0,
    }
}

#[cfg(test)]
mod provider_capability_tests {
    use super::*;

    #[test]
    fn wgpu_provider_declares_core_atomic_support() {
        assert!(
            <WgpuDevice as GpuDeviceProvider>::supports_core_atomics(),
            "WGPU provider preserves the core atomic capability previously reported by CoreGpuContext"
        );
    }

    #[test]
    fn baseline_limits_preserve_combined_binding_capacity() {
        let limits = minimal_compute_limits();
        assert_eq!(limits.max_storage_buffers_per_shader_stage, Some(8));
        assert_eq!(
            limits.max_buffers_and_acceleration_structures_per_shader_stage,
            Some(8)
        );
    }
}

#[cfg(all(test, feature = "cuda-provider"))]
mod cuda_tests {
    use super::*;

    #[test]
    fn cuda_device_satisfies_kwavers_provider_contract() {
        fn assert_provider<P: GpuDeviceProvider>() {}

        assert_provider::<CudaDevice>();
        assert_eq!(
            <CudaDevice as GpuDeviceProvider>::provider_kind(),
            GpuProvider::Cuda
        );
        assert_eq!(
            <CudaDevice as GpuDeviceProvider>::acquisition_label(),
            "kwavers-cuda-device"
        );
        assert_eq!(
            <CudaDevice as GpuDeviceProvider>::required_limits()
                .max_storage_buffers_per_shader_stage,
            None
        );
        assert!(
            <CudaDevice as GpuDeviceProvider>::optional_features().is_empty(),
            "CUDA provider acquisition must not inherit WGPU-only optional features"
        );
        assert!(
            !<CudaDevice as GpuDeviceProvider>::supports_core_atomics(),
            "CUDA core atomic support must not be claimed until Kwavers owns CUDA kernels for that contract"
        );
    }
}
