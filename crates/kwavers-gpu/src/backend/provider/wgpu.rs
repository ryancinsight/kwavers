//! Hephaestus WGPU implementation of the backend operation contracts.

use super::shared::{
    dense_slice, dense_slice_mut, limit_bytes_to_usize, map_hephaestus_error,
    validate_elementwise_shapes, validate_non_empty_shape,
};
use super::{
    ElementWiseMultiplyProvider, GpuKernelProvider, GpuProviderBackend, SpatialDerivativeProvider,
};
use crate::backend::init::GpuProviderContext;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DispatchGrid, MultiStorageDevice, MultiStorageKernel,
};
use hephaestus_wgpu::{
    binary_elementwise_into, MulOp, WgpuDevice, WgslMultiStorageKernel, WgslStorageBindingLayout,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_solver::backend::traits::BackendCapabilities;
use leto::Array3 as LetoArray3;

const DERIVATIVE_WORKGROUP_WIDTH: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DerivativeParams {
    nx: u32,
    ny: u32,
    nz: u32,
    direction: u32,
}

/// Hephaestus-backed WGPU provider.
#[derive(Debug)]
pub struct WgpuComputeProvider {
    context: GpuProviderContext<WgpuDevice>,
    derivative_kernel: WgslMultiStorageKernel,
    capabilities: BackendCapabilities,
    initialized: bool,
}

impl WgpuComputeProvider {
    /// Acquire a Hephaestus WGPU device and compile the spatial derivative.
    ///
    /// # Errors
    ///
    /// Returns a GPU error when device acquisition or provider-owned WGSL
    /// kernel construction fails.
    pub fn new() -> KwaversResult<Self> {
        let context = GpuProviderContext::<WgpuDevice>::new()?;
        let capabilities = Self::query_capabilities(&context);
        let derivative_kernel = WgslMultiStorageKernel::new(
            context.hephaestus_device(),
            "kwavers-spatial-derivative",
            include_str!("../shaders/operators.wgsl"),
            "spatial_derivative",
            &[
                WgslStorageBindingLayout::read_only(0),
                WgslStorageBindingLayout::read_write(1),
            ],
            2,
        )
        .map_err(|error| map_hephaestus_error("spatial derivative kernel construction", error))?;

        Ok(Self {
            context,
            derivative_kernel,
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
        validate_non_empty_shape("element_wise_multiply", a.shape())?;
        validate_elementwise_shapes(a, b, out)?;

        let device = self.context.hephaestus_device();
        let left = device
            .upload(dense_slice("lhs", a)?)
            .map_err(|error| map_hephaestus_error("elementwise lhs upload", error))?;
        let right = device
            .upload(dense_slice("rhs", b)?)
            .map_err(|error| map_hephaestus_error("elementwise rhs upload", error))?;
        let product = device
            .alloc_zeroed::<f32>(left.len())
            .map_err(|error| map_hephaestus_error("elementwise output allocation", error))?;

        binary_elementwise_into::<MulOp, f32>(device, &left, &right, &product, BlockWidth::DEFAULT)
            .map_err(|error| map_hephaestus_error("elementwise multiply dispatch", error))?;
        device
            .download(&product, dense_slice_mut("out", out)?)
            .map_err(|error| map_hephaestus_error("elementwise output readback", error))
    }
}

impl SpatialDerivativeProvider for WgpuComputeProvider {
    fn apply_spatial_derivative(
        &self,
        field: &LetoArray3<Self::Scalar>,
        direction: usize,
        out: &mut LetoArray3<Self::Scalar>,
    ) -> KwaversResult<()> {
        if direction > 2 {
            return Err(KwaversError::InvalidInput(format!(
                "spatial derivative direction must be 0, 1, or 2; got {direction}"
            )));
        }

        let shape = field.shape();
        validate_non_empty_shape("spatial_derivative", shape)?;
        if out.shape() != shape {
            return Err(KwaversError::InvalidInput(format!(
                "spatial_derivative: output shape {:?} must match input shape {shape:?}",
                out.shape()
            )));
        }

        let params = derivative_params(shape, direction)?;
        let grid = derivative_dispatch_grid(shape)?;
        let device = self.context.hephaestus_device();
        let input = device
            .upload(dense_slice("field", field)?)
            .map_err(|error| map_hephaestus_error("spatial derivative input upload", error))?;
        let output = device
            .alloc_zeroed::<f32>(input.len())
            .map_err(|error| map_hephaestus_error("spatial derivative output allocation", error))?;

        self.derivative_kernel
            .dispatch(
                device,
                [
                    WgpuDevice::storage_binding(0, &input),
                    WgpuDevice::storage_binding(1, &output),
                ],
                &params,
                grid,
            )
            .map_err(|error| map_hephaestus_error("spatial derivative dispatch", error))?;
        device
            .download(&output, dense_slice_mut("out", out)?)
            .map_err(|error| map_hephaestus_error("spatial derivative output readback", error))
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

fn derivative_params(shape: [usize; 3], direction: usize) -> KwaversResult<DerivativeParams> {
    let [nx, ny, nz] = shape;
    let nx = u32::try_from(nx).map_err(|_| derivative_dimension_error(shape))?;
    let ny = u32::try_from(ny).map_err(|_| derivative_dimension_error(shape))?;
    let nz = u32::try_from(nz).map_err(|_| derivative_dimension_error(shape))?;
    let direction = u32::try_from(direction).map_err(|_| {
        KwaversError::InvalidInput(format!(
            "spatial derivative direction {direction} does not fit the provider contract"
        ))
    })?;

    Ok(DerivativeParams {
        nx,
        ny,
        nz,
        direction,
    })
}

fn derivative_dispatch_grid(shape: [usize; 3]) -> KwaversResult<DispatchGrid> {
    let elements = shape.into_iter().try_fold(1_usize, |product, dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "spatial derivative element count overflows usize for shape {shape:?}"
            ))
        })
    })?;
    let elements = u32::try_from(elements).map_err(|_| {
        KwaversError::InvalidInput(format!(
            "spatial derivative element count exceeds the u32 WGSL index range for shape {shape:?}"
        ))
    })?;
    let groups = elements.div_ceil(DERIVATIVE_WORKGROUP_WIDTH as u32);

    Ok(DispatchGrid::new(groups, 1, 1))
}

fn derivative_dimension_error(shape: [usize; 3]) -> KwaversError {
    KwaversError::InvalidInput(format!(
        "spatial derivative shape {shape:?} exceeds the u32 WGSL parameter range"
    ))
}

#[cfg(test)]
mod tests {
    use super::{derivative_dispatch_grid, derivative_params};
    use hephaestus_core::DispatchGrid;

    #[test]
    fn derivative_dispatch_covers_a_partial_workgroup() {
        assert_eq!(
            derivative_dispatch_grid([257, 1, 1]).unwrap(),
            DispatchGrid::new(2, 1, 1)
        );
    }

    #[test]
    fn derivative_parameters_preserve_all_axes_and_direction() {
        let params = derivative_params([4, 3, 5], 2).unwrap();

        assert_eq!(params.nx, 4);
        assert_eq!(params.ny, 3);
        assert_eq!(params.nz, 5);
        assert_eq!(params.direction, 2);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn derivative_dispatch_rejects_a_wgsl_index_overflow() {
        let error = derivative_dispatch_grid([65_536, 65_536, 1]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Invalid input: spatial derivative element count exceeds the u32 WGSL index range for shape [65536, 65536, 1]"
        );
    }
}
