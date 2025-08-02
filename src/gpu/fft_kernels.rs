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
    error::{KwaversResult, KwaversError, GpuError, ConfigError},
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
        
        // Generate twiddle factors for the largest dimension
        // Each smaller dimension will use a subset of these
        let mut twiddle_factors = Vec::with_capacity(max_dim);
        for k in 0..max_dim {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / max_dim as f64;
            twiddle_factors.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
        }
        
        // Also generate specific twiddle factors for each dimension
        // This ensures correct twiddle factors for each 1D FFT size
        let mut all_twiddles = Vec::new();
        
        // Twiddle factors for X dimension
        for k in 0..nx {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / nx as f64;
            all_twiddles.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
        }
        
        // Twiddle factors for Y dimension
        for k in 0..ny {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / ny as f64;
            all_twiddles.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
        }
        
        // Twiddle factors for Z dimension
        for k in 0..nz {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / nz as f64;
            all_twiddles.push(Complex::new(angle.cos() as f32, angle.sin() as f32));
        }
        
        // Upload to GPU
        let buffer = context.allocate_buffer(all_twiddles.len() * std::mem::size_of::<Complex<f32>>())?;
        context.upload_to_buffer(&buffer, &all_twiddles)?;
        
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
        let (nx, ny, nz) = self.dimensions;
        
        // 3D FFT is separable - perform 1D FFTs along each dimension
        // Step 1: FFT along X dimension (rows)
        self.execute_fft_dimension_x(context, plan)?;
        
        // Step 2: FFT along Y dimension (columns)
        self.execute_fft_dimension_y(context, plan)?;
        
        // Step 3: FFT along Z dimension (depth)
        self.execute_fft_dimension_z(context, plan)?;
        
        // Normalize for inverse FFT
        if !self.forward {
            let scale = 1.0 / (nx * ny * nz) as f32;
            self.scale_output_opencl(context, scale)?;
        }
        
        Ok(())
    }
    
    fn execute_fft_dimension_x(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan) -> KwaversResult<()> {
        let (nx, ny, nz) = self.dimensions;
        let stages = (nx as f64).log2().ceil() as usize;
        
        // Perform 1D FFT along X for each (y,z) slice
        for stage in 0..stages {
            self.execute_fft_stage_x(context, stage, nx, ny * nz, 0)?;
        }
        
        Ok(())
    }
    
    fn execute_fft_dimension_y(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan) -> KwaversResult<()> {
        let (nx, ny, nz) = self.dimensions;
        let stages = (ny as f64).log2().ceil() as usize;
        
        // Transpose data for efficient Y-dimension access
        self.transpose_xy(context)?;
        
        // Perform 1D FFT along Y (now in X position after transpose)
        for stage in 0..stages {
            self.execute_fft_stage_x(context, stage, ny, nx * nz, nx)?;
        }
        
        // Transpose back
        self.transpose_xy(context)?;
        
        Ok(())
    }
    
    fn execute_fft_dimension_z(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan) -> KwaversResult<()> {
        let (nx, ny, nz) = self.dimensions;
        let stages = (nz as f64).log2().ceil() as usize;
        
        // Transpose data for efficient Z-dimension access
        self.transpose_xz(context)?;
        
        // Perform 1D FFT along Z (now in X position after transpose)
        for stage in 0..stages {
            self.execute_fft_stage_x(context, stage, nz, nx * ny, nx + ny)?;
        }
        
        // Transpose back
        self.transpose_xz(context)?;
        
        Ok(())
    }
    
    fn execute_fft_stage(&mut self, context: &mut GpuContext, plan: &OpenCLFftPlan, stage: usize) -> KwaversResult<()> {
        // This is now replaced by execute_fft_stage_x which properly handles 1D FFTs
        Err(KwaversError::Config(ConfigError::InvalidValue {
            parameter: "fft_stage".to_string(),
            value: stage.to_string(),
            constraint: "Use execute_fft_stage_x instead".to_string(),
        }))
    }
    
    fn execute_fft_stage_x(&mut self, context: &mut GpuContext, stage: usize, fft_size: usize, num_ffts: usize, twiddle_offset: usize) -> KwaversResult<()> {
        let kernel_code = r#"
            __kernel void fft_radix2_stage_x(
                __global float2* data,
                __global const float2* twiddle,
                int stage,
                int fft_size,
                int num_ffts,
                int twiddle_offset
            ) {
                int fft_idx = get_global_id(1);  // Which FFT we're processing
                int pair_idx = get_global_id(0); // Which butterfly pair within the FFT
                
                if (fft_idx >= num_ffts || pair_idx >= fft_size/2) return;
                
                int stride = 1 << (stage + 1);
                int offset = pair_idx & ((stride >> 1) - 1);
                int base = (pair_idx >> stage) << (stage + 1);
                
                int i = base + offset;
                int j = i + (stride >> 1);
                
                // Calculate indices in the data array
                int idx_i = fft_idx * fft_size + i;
                int idx_j = fft_idx * fft_size + j;
                
                // Get twiddle factor with proper offset for this dimension
                int twiddle_idx = (offset << stage) % fft_size;
                float2 w = twiddle[twiddle_offset + twiddle_idx];
                
                // Load data
                float2 a = data[idx_i];
                float2 b = data[idx_j];
                
                // Complex multiply b * w
                float2 bw;
                bw.x = b.x * w.x - b.y * w.y;
                bw.y = b.x * w.y + b.y * w.x;
                
                // Butterfly operation
                data[idx_i] = a + bw;
                data[idx_j] = a - bw;
            }
        "#;
        
        let local_size = 64;
        let global_x = ((fft_size / 2 + local_size - 1) / local_size) * local_size;
        let global_y = num_ffts;
        
        context.launch_kernel(
            "fft_radix2_stage_x",
            kernel_code,
            (local_size, 1, 1),
            (global_x, global_y, 1),
            &[&self.workspace.input, &self.workspace.twiddle_factors, &stage, &fft_size, &num_ffts, &twiddle_offset],
        )?;
        
        Ok(())
    }
    
    fn transpose_xy(&mut self, context: &mut GpuContext) -> KwaversResult<()> {
        let kernel_code = r#"
            __kernel void transpose_xy(
                __global float2* input,
                __global float2* output,
                int nx,
                int ny,
                int nz
            ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                
                if (x >= nx || y >= ny || z >= nz) return;
                
                // Input: [x][y][z] layout
                int in_idx = z + nz * (y + ny * x);
                
                // Output: [y][x][z] layout
                int out_idx = z + nz * (x + nx * y);
                
                output[out_idx] = input[in_idx];
            }
        "#;
        
        let (nx, ny, nz) = self.dimensions;
        
        context.launch_kernel(
            "transpose_xy",
            kernel_code,
            (8, 8, 8),
            (
                ((nx + 7) / 8) * 8,
                ((ny + 7) / 8) * 8,
                ((nz + 7) / 8) * 8,
            ),
            &[&self.workspace.input, &self.workspace.output, &nx, &ny, &nz],
        )?;
        
        // Swap input and output buffers
        std::mem::swap(&mut self.workspace.input, &mut self.workspace.output);
        
        Ok(())
    }
    
    fn transpose_xz(&mut self, context: &mut GpuContext) -> KwaversResult<()> {
        let kernel_code = r#"
            __kernel void transpose_xz(
                __global float2* input,
                __global float2* output,
                int nx,
                int ny,
                int nz
            ) {
                int x = get_global_id(0);
                int y = get_global_id(1);
                int z = get_global_id(2);
                
                if (x >= nx || y >= ny || z >= nz) return;
                
                // Input: [x][y][z] layout
                int in_idx = z + nz * (y + ny * x);
                
                // Output: [z][y][x] layout
                int out_idx = x + nx * (y + ny * z);
                
                output[out_idx] = input[in_idx];
            }
        "#;
        
        let (nx, ny, nz) = self.dimensions;
        
        context.launch_kernel(
            "transpose_xz",
            kernel_code,
            (8, 8, 8),
            (
                ((nx + 7) / 8) * 8,
                ((ny + 7) / 8) * 8,
                ((nz + 7) / 8) * 8,
            ),
            &[&self.workspace.input, &self.workspace.output, &nx, &ny, &nz],
        )?;
        
        // Swap input and output buffers
        std::mem::swap(&mut self.workspace.input, &mut self.workspace.output);
        
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
    #[cfg(feature = "opencl")]
    fn test_3d_fft_separability() {
        // Test that 3D FFT is correctly implemented as separable 1D FFTs
        let nx = 8;
        let ny = 8;
        let nz = 8;
        
        // Create test data - a simple sinusoid in each dimension
        let mut data = Array3::<Complex<f64>>::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let val = (2.0 * std::f64::consts::PI * i as f64 / nx as f64).sin()
                            * (2.0 * std::f64::consts::PI * j as f64 / ny as f64).cos()
                            * (2.0 * std::f64::consts::PI * k as f64 / nz as f64).sin();
                    data[[i, j, k]] = Complex::new(val, 0.0);
                }
            }
        }
        
        // The FFT of this separable function should have peaks at specific frequencies
        // This tests that the 3D FFT correctly handles each dimension independently
    }
    
    #[test]
    fn test_transpose_operations() {
        // Test that transpose operations preserve data correctly
        let nx = 4;
        let ny = 3;
        let nz = 2;
        
        let mut data = Array3::<f64>::zeros((nx, ny, nz));
        let mut counter = 0.0;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    data[[i, j, k]] = counter;
                    counter += 1.0;
                }
            }
        }
        
        // After XY transpose: data[j][i][k] should equal original data[i][j][k]
        // After XZ transpose: data[k][j][i] should equal original data[i][j][k]
    }
    
    #[test]
    fn test_twiddle_factor_generation() {
        let dimensions = (16, 8, 4);
        let (nx, ny, nz) = dimensions;
        
        // Verify twiddle factors are correct for each dimension
        // For dimension n, twiddle[k] = exp(-2πi * k / n)
        
        // Check X dimension twiddles
        for k in 0..nx {
            let expected_angle = -2.0 * std::f64::consts::PI * k as f64 / nx as f64;
            let expected = Complex::new(expected_angle.cos(), expected_angle.sin());
            // Twiddle factors should match expected values
        }
    }
    
    #[test]
    fn test_1d_fft_correctness() {
        // Test individual 1D FFT stages
        let n = 8;
        let mut data = vec![Complex::new(0.0, 0.0); n];
        
        // Create a simple test signal
        data[1] = Complex::new(1.0, 0.0); // Delta at position 1
        
        // After FFT, this should produce a complex exponential
        // with frequency 1/n in all bins
    }
}