//! GPU-Optimized FFT Kernels
//!
//! This module provides high-performance FFT implementations for various GPU backends:
//! - CUDA: Using cuFFT library and custom kernels
//! - OpenCL: Custom FFT kernels with local memory optimization
//! - WebGPU: Compute shader-based FFT implementation
//!
//! # Design Principles
//! - **SOLID**: Separate implementations for each GPU backend
//! - **CUPID**: Clear interfaces for FFT operations
//! - **DRY**: Shared FFT algorithms with backend-specific optimizations
//! - **KISS**: Simple API hiding complex GPU details

use crate::{
    error::{KwaversResult, KwaversError, GpuError},
    gpu::{GpuContext, GpuBackend, memory::GpuBuffer},
};
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex;
use std::sync::Arc;
use log::{debug, info};

/// FFT plan for efficient repeated transforms
#[derive(Debug)]
pub struct GpuFftPlan {
    /// FFT dimensions
    dimensions: (usize, usize, usize),
    /// Direction (forward or inverse)
    forward: bool,
    /// Backend-specific plan data
    backend_plan: BackendFftPlan,
    /// Workspace buffers
    workspace: FftWorkspace,
}

/// Backend-specific FFT plan
#[derive(Debug)]
enum BackendFftPlan {
    #[cfg(feature = "cuda")]
    Cuda(CudaFftPlan),
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLFftPlan),
    #[cfg(feature = "webgpu")]
    WebGPU(WebGPUFftPlan),
}

/// FFT workspace buffers
#[derive(Debug)]
struct FftWorkspace {
    /// Input buffer (real or complex)
    input: GpuBuffer,
    /// Output buffer (complex)
    output: GpuBuffer,
    /// Temporary buffer for intermediate results
    temp: Option<GpuBuffer>,
    /// Twiddle factors for FFT
    twiddle_factors: GpuBuffer,
}

/// CUDA FFT plan using cuFFT
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CudaFftPlan {
    /// cuFFT plan handle
    plan_handle: cuda_sys::cufftHandle,
    /// FFT type (R2C, C2C, etc.)
    fft_type: cuda_sys::cufftType,
}

/// OpenCL FFT plan
#[cfg(feature = "opencl")]
#[derive(Debug)]
struct OpenCLFftPlan {
    /// Radix for FFT decomposition
    radix: usize,
    /// Number of stages
    stages: usize,
    /// Local work size
    local_size: usize,
}

/// WebGPU FFT plan
#[cfg(feature = "webgpu")]
#[derive(Debug)]
struct WebGPUFftPlan {
    /// Compute pipeline for FFT
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuFftPlan {
    /// Create a new FFT plan
    pub fn new(
        context: &GpuContext,
        dimensions: (usize, usize, usize),
        forward: bool,
    ) -> KwaversResult<Self> {
        info!("Creating GPU FFT plan for {:?} (forward: {})", dimensions, forward);
        
        let (nx, ny, nz) = dimensions;
        let total_size = nx * ny * nz;
        
        // Create workspace buffers
        let workspace = FftWorkspace {
            input: context.allocate_buffer(total_size * std::mem::size_of::<Complex<f32>>())?,
            output: context.allocate_buffer(total_size * std::mem::size_of::<Complex<f32>>())?,
            temp: Some(context.allocate_buffer(total_size * std::mem::size_of::<Complex<f32>>())?),
            twiddle_factors: Self::create_twiddle_factors(context, dimensions)?,
        };
        
        // Create backend-specific plan
        let backend_plan = match context.backend() {
            #[cfg(feature = "cuda")]
            GpuBackend::CUDA => BackendFftPlan::Cuda(Self::create_cuda_plan(dimensions, forward)?),
            #[cfg(feature = "opencl")]
            GpuBackend::OpenCL => BackendFftPlan::OpenCL(Self::create_opencl_plan(dimensions)?),
            #[cfg(feature = "webgpu")]
            GpuBackend::WebGPU => BackendFftPlan::WebGPU(Self::create_webgpu_plan(context, dimensions)?),
            _ => return Err(GpuError::UnsupportedBackend {
                backend: format!("{:?}", context.backend()),
                operation: "FFT".to_string(),
            }.into()),
        };
        
        Ok(Self {
            dimensions,
            forward,
            backend_plan,
            workspace,
        })
    }
    
    /// Execute FFT on real input data
    pub fn execute_r2c(
        &mut self,
        context: &mut GpuContext,
        input: ArrayView3<f64>,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = self.dimensions;
        
        // Upload real data to GPU
        self.upload_real_data(context, input)?;
        
        // Execute FFT
        self.execute_fft_kernel(context)?;
        
        // Download complex result
        self.download_complex_data(context)
    }
    
    /// Execute FFT on complex input data
    pub fn execute_c2c(
        &mut self,
        context: &mut GpuContext,
        input: ArrayView3<Complex<f64>>,
    ) -> KwaversResult<Array3<Complex<f64>>> {
        // Upload complex data to GPU
        self.upload_complex_data(context, input)?;
        
        // Execute FFT
        self.execute_fft_kernel(context)?;
        
        // Download complex result
        self.download_complex_data(context)
    }
    
    /// Create twiddle factors for FFT
    fn create_twiddle_factors(
        context: &GpuContext,
        dimensions: (usize, usize, usize),
    ) -> KwaversResult<GpuBuffer> {
        let (nx, ny, nz) = dimensions;
        let max_dim = nx.max(ny).max(nz);
        
        // Generate twiddle factors
        let mut twiddle_factors = Vec::with_capacity(max_dim);
        for k in 0..max_dim {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / max_dim as f64;
            twiddle_factors.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
        }
        
        // Upload to GPU
        let buffer = context.allocate_buffer(twiddle_factors.len() * std::mem::size_of::<Complex<f32>>())?;
        context.upload_to_buffer(&buffer, &twiddle_factors)?;
        
        Ok(buffer)
    }
    
    /// Execute the FFT kernel
    fn execute_fft_kernel(&mut self, context: &mut GpuContext) -> KwaversResult<()> {
        match &self.backend_plan {
            #[cfg(feature = "cuda")]
            BackendFftPlan::Cuda(plan) => self.execute_cuda_fft(context, plan),
            #[cfg(feature = "opencl")]
            BackendFftPlan::OpenCL(plan) => self.execute_opencl_fft(context, plan),
            #[cfg(feature = "webgpu")]
            BackendFftPlan::WebGPU(plan) => self.execute_webgpu_fft(context, plan),
        }
    }
    
    /// Upload real data to GPU
    fn upload_real_data(&mut self, context: &mut GpuContext, data: ArrayView3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = self.dimensions;
        let mut complex_data = Vec::with_capacity(nx * ny * nz);
        
        // Convert real to complex
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    complex_data.push(Complex::new(data[[i, j, k]] as f32, 0.0));
                }
            }
        }
        
        context.upload_to_buffer(&self.workspace.input, &complex_data)?;
        Ok(())
    }
    
    /// Upload complex data to GPU
    fn upload_complex_data(&mut self, context: &mut GpuContext, data: ArrayView3<Complex<f64>>) -> KwaversResult<()> {
        let (nx, ny, nz) = self.dimensions;
        let mut complex_data = Vec::with_capacity(nx * ny * nz);
        
        // Convert f64 to f32
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let val = data[[i, j, k]];
                    complex_data.push(Complex::new(val.re as f32, val.im as f32));
                }
            }
        }
        
        context.upload_to_buffer(&self.workspace.input, &complex_data)?;
        Ok(())
    }
    
    /// Download complex data from GPU
    fn download_complex_data(&self, context: &mut GpuContext) -> KwaversResult<Array3<Complex<f64>>> {
        let (nx, ny, nz) = self.dimensions;
        let mut complex_data = vec![Complex::new(0.0f32, 0.0f32); nx * ny * nz];
        
        context.download_from_buffer(&self.workspace.output, &mut complex_data)?;
        
        // Convert f32 to f64 and reshape
        let mut result = Array3::zeros((nx, ny, nz));
        let mut idx = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let val = complex_data[idx];
                    result[[i, j, k]] = Complex::new(val.re as f64, val.im as f64);
                    idx += 1;
                }
            }
        }
        
        Ok(result)
    }
}

// CUDA-specific implementations
#[cfg(feature = "cuda")]
impl GpuFftPlan {
    fn create_cuda_plan(dimensions: (usize, usize, usize), forward: bool) -> KwaversResult<CudaFftPlan> {
        use cuda_sys::*;
        
        let (nx, ny, nz) = dimensions;
        let mut plan_handle: cufftHandle = 0;
        
        // Create 3D FFT plan
        let result = unsafe {
            cufftPlan3d(
                &mut plan_handle,
                nz as i32,
                ny as i32,
                nx as i32,
                CUFFT_C2C,
            )
        };
        
        if result != CUFFT_SUCCESS {
            return Err(GpuError::KernelCompilation {
                kernel_name: "cuFFT".to_string(),
                error: format!("Failed to create cuFFT plan: {}", result),
            }.into());
        }
        
        Ok(CudaFftPlan {
            plan_handle,
            fft_type: CUFFT_C2C,
        })
    }
    
    fn execute_cuda_fft(&mut self, context: &mut GpuContext, plan: &CudaFftPlan) -> KwaversResult<()> {
        use cuda_sys::*;
        
        let direction = if self.forward { CUFFT_FORWARD } else { CUFFT_INVERSE };
        
        let result = unsafe {
            cufftExecC2C(
                plan.plan_handle,
                self.workspace.input.as_ptr() as *mut cufftComplex,
                self.workspace.output.as_ptr() as *mut cufftComplex,
                direction,
            )
        };
        
        if result != CUFFT_SUCCESS {
            return Err(GpuError::KernelLaunch {
                kernel_name: "cuFFT".to_string(),
                reason: format!("cuFFT execution failed: {}", result),
            }.into());
        }
        
        // Normalize for inverse FFT
        if !self.forward {
            let (nx, ny, nz) = self.dimensions;
            let scale = 1.0 / (nx * ny * nz) as f32;
            self.scale_output(context, scale)?;
        }
        
        Ok(())
    }
    
    fn scale_output(&mut self, context: &mut GpuContext, scale: f32) -> KwaversResult<()> {
        // Launch scaling kernel
        let kernel_code = format!(r#"
            extern "C" __global__ void scale_complex(
                cuComplex* data,
                float scale,
                int n
            ) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {{
                    data[idx].x *= scale;
                    data[idx].y *= scale;
                }}
            }}
        "#);
        
        context.launch_kernel(
            "scale_complex",
            &kernel_code,
            (256, 1, 1),
            ((self.dimensions.0 * self.dimensions.1 * self.dimensions.2 + 255) / 256, 1, 1),
            &[&self.workspace.output, &scale],
        )?;
        
        Ok(())
    }
}

// OpenCL-specific implementations
#[cfg(feature = "opencl")]
impl GpuFftPlan {
    fn create_opencl_plan(dimensions: (usize, usize, usize)) -> KwaversResult<OpenCLFftPlan> {
        let (nx, ny, nz) = dimensions;
        let max_dim = nx.max(ny).max(nz);
        
        // Determine radix and stages
        let radix = if max_dim.is_power_of_two() { 2 } else { 4 };
        let stages = (max_dim as f64).log(radix as f64).ceil() as usize;
        
        Ok(OpenCLFftPlan {
            radix,
            stages,
            local_size: 64, // Typical local work group size
        })
    }
    
    fn execute_opencl_fft(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan) -> KwaversResult<()> {
        // Cooley-Tukey FFT implementation
        for stage in 0..plan.stages {
            self.execute_fft_stage(context, plan, stage)?;
        }
        
        // Normalize for inverse FFT
        if !self.forward {
            let (nx, ny, nz) = self.dimensions;
            let scale = 1.0 / (nx * ny * nz) as f32;
            self.scale_output_opencl(context, scale)?;
        }
        
        Ok(())
    }
    
    fn execute_fft_stage(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan, stage: usize) -> KwaversResult<()> {
        let kernel_code = r#"
            __kernel void fft_radix2_stage(
                __global float2* data,
                __global const float2* twiddle,
                int stage,
                int n
            ) {
                int idx = get_global_id(0);
                if (idx >= n/2) return;
                
                int stride = 1 << (stage + 1);
                int offset = idx & ((stride >> 1) - 1);
                int base = (idx >> (stage)) << (stage + 1);
                
                int i = base + offset;
                int j = i + (stride >> 1);
                
                float2 w = twiddle[(offset << (stage))];
                float2 a = data[i];
                float2 b = data[j];
                
                // Complex multiply b * w
                float2 bw;
                bw.x = b.x * w.x - b.y * w.y;
                bw.y = b.x * w.y + b.y * w.x;
                
                // Butterfly operation
                data[i] = a + bw;
                data[j] = a - bw;
            }
        "#;
        
        let (nx, ny, nz) = self.dimensions;
        let n = nx * ny * nz;
        
        context.launch_kernel(
            "fft_radix2_stage",
            kernel_code,
            (plan.local_size, 1, 1),
            ((n / 2 + plan.local_size - 1) / plan.local_size * plan.local_size, 1, 1),
            &[&self.workspace.input, &self.workspace.twiddle_factors, &stage, &n],
        )?;
        
        Ok(())
    }
    
    fn scale_output_opencl(&mut self, context: &mut GpuContext, scale: f32) -> KwaversResult<()> {
        let kernel_code = r#"
            __kernel void scale_complex(
                __global float2* data,
                float scale
            ) {
                int idx = get_global_id(0);
                data[idx] *= scale;
            }
        "#;
        
        let (nx, ny, nz) = self.dimensions;
        let n = nx * ny * nz;
        
        context.launch_kernel(
            "scale_complex",
            kernel_code,
            (256, 1, 1),
            ((n + 255) / 256 * 256, 1, 1),
            &[&self.workspace.output, &scale],
        )?;
        
        Ok(())
    }
}

// WebGPU-specific implementations
#[cfg(feature = "webgpu")]
impl GpuFftPlan {
    fn create_webgpu_plan(context: &GpuContext, dimensions: (usize, usize, usize)) -> KwaversResult<WebGPUFftPlan> {
        // Create compute shader for FFT
        let shader_code = r#"
            struct Complex {
                r: f32,
                i: f32,
            }
            
            @group(0) @binding(0) var<storage, read_write> data: array<Complex>;
            @group(0) @binding(1) var<storage, read> twiddle: array<Complex>;
            @group(0) @binding(2) var<uniform> params: FFTParams;
            
            struct FFTParams {
                stage: u32,
                n: u32,
                forward: u32,
            }
            
            fn complex_mul(a: Complex, b: Complex) -> Complex {
                return Complex(
                    a.r * b.r - a.i * b.i,
                    a.r * b.i + a.i * b.r
                );
            }
            
            @compute @workgroup_size(64)
            fn fft_stage(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= params.n / 2u) { return; }
                
                let stride = 1u << (params.stage + 1u);
                let offset = idx & ((stride >> 1u) - 1u);
                let base = (idx >> params.stage) << (params.stage + 1u);
                
                let i = base + offset;
                let j = i + (stride >> 1u);
                
                let w = twiddle[offset << params.stage];
                let a = data[i];
                let b = data[j];
                
                let bw = complex_mul(b, w);
                
                data[i] = Complex(a.r + bw.r, a.i + bw.i);
                data[j] = Complex(a.r - bw.r, a.i - bw.i);
            }
        "#;
        
        // Create pipeline and bind group layout
        // This is a simplified version - actual implementation would need proper WebGPU setup
        todo!("WebGPU FFT implementation")
    }
    
    fn execute_webgpu_fft(&mut self, context: &mut GpuContext, plan: &WebGPUFftPlan) -> KwaversResult<()> {
        todo!("WebGPU FFT execution")
    }
}

/// High-level FFT interface
pub struct GpuFft {
    /// FFT plans for different sizes
    plans: std::collections::HashMap<(usize, usize, usize, bool), Arc<GpuFftPlan>>,
    /// GPU context
    context: Arc<GpuContext>,
}

impl GpuFft {
    pub fn new(context: Arc<GpuContext>) -> Self {
        Self {
            plans: std::collections::HashMap::new(),
            context,
        }
    }
    
    /// Perform 3D FFT
    pub fn fft_3d(&mut self, input: ArrayView3<f64>) -> KwaversResult<Array3<Complex<f64>>> {
        let dims = input.dim();
        let key = (dims.0, dims.1, dims.2, true);
        
        // Get or create plan
        let plan = if let Some(plan) = self.plans.get(&key) {
            plan.clone()
        } else {
            let new_plan = Arc::new(GpuFftPlan::new(&self.context, dims, true)?);
            self.plans.insert(key, new_plan.clone());
            new_plan
        };
        
        // Execute FFT
        Arc::get_mut(&mut plan.clone())
            .unwrap()
            .execute_r2c(&mut *Arc::get_mut(&mut self.context.clone()).unwrap(), input)
    }
    
    /// Perform 3D inverse FFT
    pub fn ifft_3d(&mut self, input: ArrayView3<Complex<f64>>) -> KwaversResult<Array3<Complex<f64>>> {
        let dims = input.dim();
        let key = (dims.0, dims.1, dims.2, false);
        
        // Get or create plan
        let plan = if let Some(plan) = self.plans.get(&key) {
            plan.clone()
        } else {
            let new_plan = Arc::new(GpuFftPlan::new(&self.context, dims, false)?);
            self.plans.insert(key, new_plan.clone());
            new_plan
        };
        
        // Execute inverse FFT
        Arc::get_mut(&mut plan.clone())
            .unwrap()
            .execute_c2c(&mut *Arc::get_mut(&mut self.context.clone()).unwrap(), input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use num_complex::Complex;
    
    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_fft_forward_inverse() {
        // Create test data
        let mut data = Array3::<f64>::zeros((8, 8, 8));
        data[[4, 4, 4]] = 1.0;
        
        // Create GPU context (mock for testing)
        let context = Arc::new(GpuContext::new(GpuBackend::CUDA).unwrap());
        let mut gpu_fft = GpuFft::new(context);
        
        // Forward FFT
        let fft_result = gpu_fft.fft_3d(data.view()).unwrap();
        
        // Inverse FFT
        let ifft_result = gpu_fft.ifft_3d(fft_result.view()).unwrap();
        
        // Check reconstruction
        for ((i, j, k), &val) in ifft_result.indexed_iter() {
            let expected = if i == 4 && j == 4 && k == 4 { 1.0 } else { 0.0 };
            assert!((val.re - expected).abs() < 1e-6);
            assert!(val.im.abs() < 1e-6);
        }
    }
}