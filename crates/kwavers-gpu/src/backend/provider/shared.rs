//! Provider-neutral operation contracts and host-array validation.

use crate::backend::init::GpuProviderContext;
use crate::gpu::GpuDeviceProvider;
use hephaestus_core::HephaestusError;
use kwavers_core::error::{KwaversError, KwaversResult};
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
/// `GPUBackend`. CUDA should satisfy this composite trait only after the
/// concrete CUDA implementation covers each required operation.
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

pub(super) fn validate_non_empty_shape(operation: &str, shape: [usize; 3]) -> KwaversResult<()> {
    if shape.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "{operation}: all dimensions must be non-zero; got {shape:?}"
        )));
    }

    Ok(())
}

pub(super) fn validate_elementwise_shapes<T>(
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

pub(super) fn dense_slice<'a, T>(name: &str, array: &'a LetoArray3<T>) -> KwaversResult<&'a [T]> {
    array.as_slice().ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "{name} field must be a dense row-major Leto Array3"
        ))
    })
}

pub(super) fn dense_slice_mut<'a, T>(
    name: &str,
    array: &'a mut LetoArray3<T>,
) -> KwaversResult<&'a mut [T]> {
    array.as_slice_mut().ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "{name} field must be a dense row-major Leto Array3"
        ))
    })
}

pub(super) fn map_hephaestus_error(context: &'static str, error: HephaestusError) -> KwaversError {
    KwaversError::GpuError(format!("{context}: {error}"))
}

pub(super) fn limit_bytes_to_usize(bytes: u64) -> usize {
    match usize::try_from(bytes) {
        Ok(value) => value,
        Err(_) => usize::MAX,
    }
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
