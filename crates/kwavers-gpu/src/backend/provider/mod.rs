//! GPU provider contracts and concrete Hephaestus implementations.

mod shared;
mod wgpu;

#[cfg(feature = "cuda-provider")]
mod cuda;

pub(crate) use shared::map_hephaestus_error;
pub use shared::{
    ElementWiseMultiplyProvider, GpuComputeProvider, GpuKernelProvider, GpuProviderBackend,
    SpatialDerivativeProvider,
};
pub use wgpu::WgpuComputeProvider;

#[cfg(feature = "cuda-provider")]
pub use cuda::CudaElementWiseProvider;
