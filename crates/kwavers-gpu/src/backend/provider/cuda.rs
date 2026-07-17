//! Hephaestus CUDA implementation of the real elementwise operation family.

use super::shared::{
    dense_slice, dense_slice_mut, limit_bytes_to_usize, map_hephaestus_error,
    validate_elementwise_shapes,
};
use super::{ElementWiseMultiplyProvider, GpuKernelProvider, GpuProviderBackend};
use crate::backend::init::GpuProviderContext;
use hephaestus_core::{
    BlockWidth, ComputeDevice as HephaestusComputeDevice, ComputeDeviceCapabilities, DeviceFeature,
};
use hephaestus_cuda::CudaDevice;
use kwavers_core::error::KwaversResult;
use kwavers_solver::backend::traits::BackendCapabilities;
use leto::Array3 as LetoArray3;

/// Hephaestus-backed CUDA element-wise provider.
///
/// This is intentionally an operation-family provider, not a full
/// `GPUBackend` compute provider. CUDA implements this trait only for the real
/// Hephaestus CUDA element-wise kernel that exists today; spatial derivatives
/// and other Kwavers kernels must land as real CUDA implementations before CUDA
/// satisfies `GpuComputeProvider`.
#[derive(Debug)]
pub struct CudaElementWiseProvider {
    context: GpuProviderContext<CudaDevice>,
    capabilities: BackendCapabilities,
    initialized: bool,
}

impl CudaElementWiseProvider {
    /// Acquire a Hephaestus CUDA device for element-wise kernels.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when CUDA is unavailable or the selected device does
    /// not satisfy Kwavers' declared CUDA provider limits.
    pub fn new() -> KwaversResult<Self> {
        let context = GpuProviderContext::<CudaDevice>::new()?;
        let capabilities = Self::query_capabilities(&context);

        Ok(Self {
            context,
            capabilities,
            initialized: true,
        })
    }

    fn query_capabilities(context: &GpuProviderContext<CudaDevice>) -> BackendCapabilities {
        let device = context.hephaestus_device();
        let limits = device.device_limits();

        BackendCapabilities {
            supports_fft: false,
            supports_f64: device.supports_device_feature(DeviceFeature::ShaderF64),
            supports_f32: true,
            supports_async: true,
            max_parallelism: limits.max_compute_invocations_per_workgroup as usize,
            supports_unified_memory: false,
        }
    }

    fn provider_memory_bytes(&self) -> usize {
        limit_bytes_to_usize(
            self.context
                .hephaestus_device()
                .device_limits()
                .max_buffer_size,
        )
    }
}

impl GpuProviderBackend for CudaElementWiseProvider {
    type Device = CudaDevice;

    fn hephaestus_device(&self) -> &Self::Device {
        self.context.hephaestus_device()
    }

    fn device_name(&self) -> &str {
        self.context.device_name()
    }

    fn synchronize(&self) -> KwaversResult<()> {
        self.context.synchronize()
    }
}

impl GpuKernelProvider for CudaElementWiseProvider {
    type Scalar = f32;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        self.initialized
    }

    fn available_memory(&self) -> usize {
        self.provider_memory_bytes()
    }

    fn estimate_peak_performance(&self) -> f64 {
        0.0
    }
}

impl ElementWiseMultiplyProvider for CudaElementWiseProvider {
    fn element_wise_multiply(
        &self,
        a: &LetoArray3<Self::Scalar>,
        b: &LetoArray3<Self::Scalar>,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        validate_elementwise_shapes(a, b, out)?;

        let left_values = dense_slice("lhs", a)?;
        let right_values = dense_slice("rhs", b)?;
        let out_values = dense_slice_mut("out", out)?;
        let device = self.context.hephaestus_device();

        let left = device
            .upload(left_values)
            .map_err(|error| map_hephaestus_error("cuda elementwise lhs upload", error))?;
        let right = device
            .upload(right_values)
            .map_err(|error| map_hephaestus_error("cuda elementwise rhs upload", error))?;
        let output = device
            .alloc_zeroed::<f32>(out_values.len())
            .map_err(|error| map_hephaestus_error("cuda elementwise output allocation", error))?;

        hephaestus_cuda::binary_elementwise_into::<hephaestus_cuda::MulOp, f32>(
            device,
            &left,
            &right,
            &output,
            BlockWidth::DEFAULT,
        )
        .map_err(|error| map_hephaestus_error("cuda elementwise multiply dispatch", error))?;

        device
            .synchronize()
            .map_err(|error| map_hephaestus_error("cuda elementwise synchronize", error))?;
        device
            .download(&output, out_values)
            .map_err(|error| map_hephaestus_error("cuda elementwise readback", error))
    }
}
