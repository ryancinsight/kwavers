//! GPU provider traits and Hephaestus-backed implementations.

use super::buffers::WgpuBackendBufferManager;
use super::init::GpuProviderContext;
use super::pipeline::WgpuPipelineManager;
use crate::gpu::GpuDeviceProvider;
#[cfg(feature = "cuda-provider")]
use hephaestus_core::{
    BlockWidth, ComputeDevice as HephaestusComputeDevice, ComputeDeviceCapabilities, DeviceFeature,
};
#[cfg(feature = "cuda-provider")]
use hephaestus_cuda::CudaDevice;
use hephaestus_wgpu::WgpuDevice;
#[cfg(any(feature = "cuda-provider", test))]
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use kwavers_solver::backend::traits::{
    BackendCapabilities, BackendType, ComputeDevice, GpuProvider,
};
use leto::Array3 as LetoArray3;

/// Provider-level GPU identity and synchronization seam.
///
/// Implementations own concrete device acquisition through Hephaestus. WGPU,
/// CUDA, and future providers substitute by changing the associated device
/// type. Kernel dispatch is intentionally separated into
/// [`GpuComputeProvider`] so CUDA can satisfy the acquisition contract without
/// pretending WGSL kernels exist for it.
pub trait GpuProviderBackend: std::fmt::Debug {
    /// Concrete Hephaestus device type used by this provider.
    type Device: GpuDeviceProvider;

    /// Return the provider identity.
    fn provider_kind(&self) -> GpuProvider {
        <Self::Device as GpuDeviceProvider>::provider_kind()
    }

    /// Borrow the underlying Hephaestus device.
    fn hephaestus_device(&self) -> &Self::Device;

    /// Return the selected device name.
    fn device_name(&self) -> &str;

    /// Synchronize all queued provider work.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when the provider synchronization primitive fails.
    fn synchronize(&self) -> KwaversResult<()>;
}

impl<P> GpuProviderBackend for GpuProviderContext<P>
where
    P: GpuDeviceProvider + std::fmt::Debug,
{
    type Device = P;

    fn hephaestus_device(&self) -> &Self::Device {
        GpuProviderContext::hephaestus_device(self)
    }

    fn device_name(&self) -> &str {
        GpuProviderContext::device_name(self)
    }

    fn synchronize(&self) -> KwaversResult<()> {
        GpuProviderContext::synchronize(self)
    }
}

/// Provider-level scalar-kernel capability seam.
///
/// Implementations own real scalar transfer and capability reporting for a
/// concrete Hephaestus provider. Operation traits extend this surface so CUDA
/// can implement one real kernel family without pretending unrelated kernels
/// exist.
pub trait GpuKernelProvider: GpuProviderBackend {
    /// Scalar type accepted by this provider's real kernel implementations.
    type Scalar: Copy + Send + Sync + 'static;

    /// Return the provider capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    /// Return true when the provider is ready for dispatch.
    fn is_available(&self) -> bool;

    /// Return provider-reported device memory in bytes.
    ///
    /// Providers that do not expose live free-memory accounting return the
    /// strongest non-fabricated allocation bound available from their
    /// Hephaestus device contract.
    fn available_memory(&self) -> usize;

    /// Estimate peak provider throughput in FLOP/s.
    ///
    /// Providers return `0.0` when the underlying API does not report enough
    /// topology data to derive a hardware-specific value.
    fn estimate_peak_performance(&self) -> f64;
}

/// Provider execution seam for element-wise multiplication.
pub trait ElementWiseMultiplyProvider: GpuKernelProvider {
    /// Execute element-wise multiplication.
    ///
    /// # Errors
    ///
    /// Propagates provider transfer, dispatch, or readback failures.
    fn element_wise_multiply(
        &self,
        a: &LetoArray3<Self::Scalar>,
        b: &LetoArray3<Self::Scalar>,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()>;
}

/// Provider execution seam for spatial derivatives.
pub trait SpatialDerivativeProvider: GpuKernelProvider {
    /// Apply a spatial derivative.
    ///
    /// # Errors
    ///
    /// Propagates provider dispatch failures or invalid derivative directions.
    fn apply_spatial_derivative(
        &self,
        field: &LetoArray3<Self::Scalar>,
        direction: usize,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()>;
}

/// Provider-level GPU execution seam.
///
/// Implementations own real kernel dispatch for all operations exposed through
/// [`super::GPUBackend`]. CUDA should satisfy this composite trait only after
/// the concrete CUDA implementation covers each required operation.
pub trait GpuComputeProvider: ElementWiseMultiplyProvider + SpatialDerivativeProvider {
    /// Return provider-reported peak throughput for a problem size.
    ///
    /// The current default is problem-size independent because Hephaestus'
    /// shared device contract reports provider capability, not a calibrated
    /// kernel/runtime model. Providers with real benchmark-backed models can
    /// override this method.
    fn estimate_performance(&self, _problem_size: (usize, usize, usize)) -> f64 {
        self.estimate_peak_performance()
    }

    /// Return compute devices visible through this provider.
    fn devices(&self) -> Vec<ComputeDevice> {
        let capabilities = self.capabilities();
        vec![ComputeDevice {
            id: 0,
            name: self.device_name().to_string(),
            backend_type: BackendType::GPU(self.provider_kind()),
            total_memory: self.available_memory(),
            available_memory: self.available_memory(),
            compute_units: capabilities.max_parallelism,
            peak_performance: self.estimate_peak_performance(),
        }]
    }
}

impl<P> GpuComputeProvider for P where P: ElementWiseMultiplyProvider + SpatialDerivativeProvider {}

/// Hephaestus-backed WGPU provider.
#[derive(Debug)]
pub struct WgpuComputeProvider {
    context: GpuProviderContext<WgpuDevice>,
    buffer_manager: WgpuBackendBufferManager,
    pipeline_manager: WgpuPipelineManager,
    capabilities: BackendCapabilities,
    initialized: bool,
}

impl WgpuComputeProvider {
    /// Acquire a Hephaestus WGPU device and compile Kwavers kernels.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when device acquisition or pipeline compilation
    /// fails.
    pub fn new() -> KwaversResult<Self> {
        let context = GpuProviderContext::<WgpuDevice>::new()?;
        let capabilities = Self::query_capabilities(&context);
        let buffer_manager = WgpuBackendBufferManager::new(context.device());
        let pipeline_manager = WgpuPipelineManager::new(context.device())?;

        Ok(Self {
            context,
            buffer_manager,
            pipeline_manager,
            capabilities,
            initialized: true,
        })
    }

    fn query_capabilities(context: &GpuProviderContext<WgpuDevice>) -> BackendCapabilities {
        let limits = context.hephaestus_device().device_limits();

        BackendCapabilities {
            supports_fft: false,
            supports_f64: false,
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

    fn provider_peak_flops(&self) -> f64 {
        // WGPU exposes adapter class and limits, but not clock rate, memory
        // bandwidth, or compute-unit count in a portable form. Returning zero
        // preserves the "unknown, not fabricated" contract.
        0.0
    }
}

impl GpuKernelProvider for WgpuComputeProvider {
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
        self.provider_peak_flops()
    }
}

impl ElementWiseMultiplyProvider for WgpuComputeProvider {
    fn element_wise_multiply(
        &self,
        a: &LetoArray3<Self::Scalar>,
        b: &LetoArray3<Self::Scalar>,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        self.pipeline_manager.execute_element_wise_multiply(
            a,
            b,
            out,
            &self.context,
            &self.buffer_manager,
        )
    }
}

impl SpatialDerivativeProvider for WgpuComputeProvider {
    fn apply_spatial_derivative(
        &self,
        field: &LetoArray3<Self::Scalar>,
        direction: usize,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        self.pipeline_manager.execute_spatial_derivative(
            field,
            direction,
            out,
            &self.context,
            &self.buffer_manager,
        )
    }
}

impl GpuProviderBackend for WgpuComputeProvider {
    type Device = WgpuDevice;

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

/// Hephaestus-backed CUDA element-wise provider.
///
/// This is intentionally an operation-family provider, not a full
/// [`super::GPUBackend`] compute provider. CUDA implements this trait only for
/// the real Hephaestus CUDA element-wise kernel that exists today; spatial
/// derivatives and other Kwavers kernels must land as real CUDA implementations
/// before CUDA satisfies [`GpuComputeProvider`].
#[cfg(feature = "cuda-provider")]
#[derive(Debug)]
pub struct CudaElementWiseProvider {
    context: GpuProviderContext<CudaDevice>,
    capabilities: BackendCapabilities,
    initialized: bool,
}

#[cfg(feature = "cuda-provider")]
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

#[cfg(feature = "cuda-provider")]
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

#[cfg(feature = "cuda-provider")]
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

#[cfg(feature = "cuda-provider")]
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
            .map_err(|e| map_hephaestus_error("cuda elementwise lhs upload", e))?;
        let right = device
            .upload(right_values)
            .map_err(|e| map_hephaestus_error("cuda elementwise rhs upload", e))?;
        let output = device
            .alloc_zeroed::<f32>(out_values.len())
            .map_err(|e| map_hephaestus_error("cuda elementwise output allocation", e))?;

        hephaestus_cuda::binary_elementwise_into::<hephaestus_cuda::MulOp, f32>(
            device,
            &left,
            &right,
            &output,
            BlockWidth::DEFAULT,
        )
        .map_err(|e| map_hephaestus_error("cuda elementwise multiply dispatch", e))?;

        device
            .synchronize()
            .map_err(|e| map_hephaestus_error("cuda elementwise synchronize", e))?;
        device
            .download(&output, out_values)
            .map_err(|e| map_hephaestus_error("cuda elementwise readback", e))
    }
}

#[cfg(any(feature = "cuda-provider", test))]
fn validate_elementwise_shapes<T>(
    a: &LetoArray3<T>,
    b: &LetoArray3<T>,
    out: &LetoArray3<T>,
) -> KwaversResult<()> {
    if a.shape() != b.shape() {
        return Err(KwaversError::InvalidInput(format!(
            "rhs shape {:?} must match lhs shape {:?}",
            b.shape(),
            a.shape()
        )));
    }
    if out.shape() != a.shape() {
        return Err(KwaversError::InvalidInput(format!(
            "out shape {:?} must match lhs shape {:?}",
            out.shape(),
            a.shape()
        )));
    }

    Ok(())
}

#[cfg(any(feature = "cuda-provider", test))]
fn dense_slice<'a, T>(name: &str, array: &'a LetoArray3<T>) -> KwaversResult<&'a [T]> {
    array.as_slice().ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "{name} field must be a dense row-major Leto Array3"
        ))
    })
}

#[cfg(feature = "cuda-provider")]
fn dense_slice_mut<'a, T>(name: &str, array: &'a mut LetoArray3<T>) -> KwaversResult<&'a mut [T]> {
    array.as_slice_mut().ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "{name} field must be a dense row-major Leto Array3"
        ))
    })
}

#[cfg(feature = "cuda-provider")]
fn map_hephaestus_error(
    context: &'static str,
    error: hephaestus_core::HephaestusError,
) -> KwaversError {
    KwaversError::GpuError(format!("{context}: {error}"))
}

fn limit_bytes_to_usize(bytes: u64) -> usize {
    usize::try_from(bytes).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::{dense_slice, limit_bytes_to_usize, validate_elementwise_shapes};
    use leto::Array3 as LetoArray3;

    #[test]
    fn limit_bytes_to_usize_preserves_representable_values() {
        let value = 128 * 1024 * 1024;
        assert_eq!(limit_bytes_to_usize(value), value as usize);
    }

    #[test]
    fn limit_bytes_to_usize_saturates_unrepresentable_values() {
        if usize::BITS < u64::BITS {
            assert_eq!(limit_bytes_to_usize(u64::MAX), usize::MAX);
        } else {
            assert_eq!(limit_bytes_to_usize(u64::MAX), u64::MAX as usize);
        }
    }

    #[test]
    fn elementwise_shape_validation_rejects_mismatched_rhs() {
        let lhs = LetoArray3::<f32>::zeros([2, 2, 2]);
        let rhs = LetoArray3::<f32>::zeros([2, 2, 1]);
        let out = LetoArray3::<f32>::zeros([2, 2, 2]);

        let err = validate_elementwise_shapes(&lhs, &rhs, &out).unwrap_err();

        assert!(err.to_string().contains("rhs shape"));
        assert!(err.to_string().contains("[2, 2, 1]"));
        assert!(err.to_string().contains("[2, 2, 2]"));
    }

    #[test]
    fn elementwise_dense_slice_accepts_dense_leto_storage() {
        let values = LetoArray3::from_shape_vec([1, 2, 2], vec![1.0_f32, 2.0, 3.0, 4.0])
            .expect("invariant: shape matches value count");

        assert_eq!(
            dense_slice("values", &values).unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }
}
