//! GPU operation traits
//!
//! This module defines traits for GPU-accelerated operations,
//! following Interface Segregation Principle.

use crate::error::KwaversResult;
use ndarray::Array3;

/// Trait for GPU field operations following ISP
pub trait GpuFieldOps {
    /// Apply Laplacian operator on GPU
    fn gpu_laplacian(
        &self,
        field: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>>;

    /// Apply gradient operator on GPU
    fn gpu_gradient(
        &self,
        field: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)>;

    /// Apply divergence operator on GPU
    fn gpu_divergence(
        &self,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>>;

    /// Apply curl operator on GPU
    fn gpu_curl(
        &self,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)>;
}

/// Trait for GPU FFT operations
pub trait GpuFftOps {
    /// Forward 3D FFT on GPU
    fn gpu_fft3d_forward(
        &self,
        field: &Array3<f64>,
    ) -> KwaversResult<Array3<num_complex::Complex<f64>>>;

    /// Inverse 3D FFT on GPU
    fn gpu_fft3d_inverse(
        &self,
        field: &Array3<num_complex::Complex<f64>>,
    ) -> KwaversResult<Array3<f64>>;
}

/// Trait for GPU memory operations
pub trait GpuMemoryOps {
    /// Allocate GPU memory
    fn allocate(&mut self, size_bytes: usize) -> KwaversResult<GpuBuffer>;

    /// Deallocate GPU memory
    fn deallocate(&mut self, buffer: GpuBuffer) -> KwaversResult<()>;

    /// Copy data to GPU
    fn copy_to_device(&self, host_data: &[f64], device_buffer: &mut GpuBuffer)
        -> KwaversResult<()>;

    /// Copy data from GPU
    fn copy_from_device(
        &self,
        device_buffer: &GpuBuffer,
        host_data: &mut [f64],
    ) -> KwaversResult<()>;
}

/// GPU buffer abstraction
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer ID
    pub id: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Backend-specific handle
    pub handle: BufferHandle,
}

/// Backend-specific buffer handle
#[derive(Debug, Clone)]
pub enum BufferHandle {
    Cuda(usize),
    OpenCL(usize),
    WebGpu(usize),
}

/// Trait for GPU kernel compilation
pub trait GpuKernelOps {
    /// Compile a kernel from source
    fn compile_kernel(&self, source: &str, kernel_name: &str) -> KwaversResult<CompiledKernel>;

    /// Launch a compiled kernel
    fn launch_kernel(
        &self,
        kernel: &CompiledKernel,
        grid_size: (usize, usize, usize),
        block_size: (usize, usize, usize),
        args: &[KernelArg],
    ) -> KwaversResult<()>;
}

/// Compiled kernel representation
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
    pub handle: KernelHandle,
}

/// Backend-specific kernel handle
#[derive(Debug, Clone)]
pub enum KernelHandle {
    Cuda(usize),
    OpenCL(usize),
    WebGpu(usize),
}

/// Kernel argument type
#[derive(Debug, Clone)]
pub enum KernelArg<'a> {
    Buffer(&'a GpuBuffer),
    Scalar(f64),
    Int(i32),
    UInt(u32),
}
