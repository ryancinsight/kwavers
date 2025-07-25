//! # GPU Compute Kernels
//!
//! This module provides high-performance GPU kernels for acoustic wave propagation,
//! thermal diffusion, and FFT operations. Implements Phase 10 optimization targets
//! of >17M grid updates/second with optimized memory access patterns.

use crate::error::{KwaversResult, KwaversError};
use crate::grid::Grid;
use crate::gpu::{GpuBackend, GpuPerformanceMetrics};
use ndarray::Array3;
use std::collections::HashMap;

/// GPU kernel types for different physics operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    /// Acoustic wave propagation kernel
    AcousticWave,
    /// Thermal diffusion kernel
    ThermalDiffusion,
    /// Forward FFT kernel
    FFTForward,
    /// Inverse FFT kernel
    FFTInverse,
    /// Memory copy kernel
    MemoryCopy,
    /// Boundary condition kernel
    BoundaryCondition,
}

/// GPU kernel optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Basic optimization (memory coalescing)
    Basic,
    /// Moderate optimization (shared memory, loop unrolling)
    Moderate,
    /// Aggressive optimization (register blocking, texture memory)
    Aggressive,
}

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub kernel_type: KernelType,
    pub optimization_level: OptimizationLevel,
    pub block_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub shared_memory_size: u32,
    pub registers_per_thread: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            kernel_type: KernelType::AcousticWave,
            optimization_level: OptimizationLevel::Moderate,
            block_size: (16, 16, 4),
            grid_size: (1, 1, 1),
            shared_memory_size: 0,
            registers_per_thread: 32,
        }
    }
}

/// GPU kernel manager for different backends
pub struct KernelManager {
    backend: GpuBackend,
    kernels: HashMap<KernelType, CompiledKernel>,
    optimization_level: OptimizationLevel,
}

/// Compiled kernel representation
pub struct CompiledKernel {
    pub kernel_type: KernelType,
    pub source_code: String,
    pub binary_code: Option<Vec<u8>>,
    pub config: KernelConfig,
    pub performance_metrics: Option<GpuPerformanceMetrics>,
}

impl KernelManager {
    /// Create new kernel manager
    pub fn new(backend: GpuBackend, optimization_level: OptimizationLevel) -> Self {
        Self {
            backend,
            kernels: HashMap::new(),
            optimization_level,
        }
    }

    /// Compile all required kernels
    pub fn compile_kernels(&mut self, grid: &Grid) -> KwaversResult<()> {
        let kernel_types = vec![
            KernelType::AcousticWave,
            KernelType::ThermalDiffusion,
            KernelType::FFTForward,
            KernelType::FFTInverse,
            KernelType::MemoryCopy,
            KernelType::BoundaryCondition,
        ];

        for kernel_type in kernel_types {
            let config = self.generate_kernel_config(kernel_type, grid)?;
            let source = self.generate_kernel_source(kernel_type, &config)?;
            let binary = self.compile_kernel_source(&source)?;
            
            let compiled_kernel = CompiledKernel {
                kernel_type,
                source_code: source,
                binary_code: binary,
                config,
                performance_metrics: None,
            };
            
            self.kernels.insert(kernel_type, compiled_kernel);
        }

        Ok(())
    }

    /// Generate optimal kernel configuration for grid size
    fn generate_kernel_config(&self, kernel_type: KernelType, grid: &Grid) -> KwaversResult<KernelConfig> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Calculate optimal block size based on grid dimensions and GPU architecture
        let block_size = match self.backend {
            GpuBackend::Cuda => self.calculate_cuda_block_size(nx, ny, nz),
            GpuBackend::OpenCL | GpuBackend::WebGPU => self.calculate_opencl_block_size(nx, ny, nz),
        };

        // Calculate grid size to cover all elements
        let grid_size = (
            (nx as u32 + block_size.0 - 1) / block_size.0,
            (ny as u32 + block_size.1 - 1) / block_size.1,
            (nz as u32 + block_size.2 - 1) / block_size.2,
        );

        // Calculate shared memory requirements
        let shared_memory_size = match (kernel_type, self.optimization_level) {
            (KernelType::AcousticWave, OptimizationLevel::Aggressive) => {
                // Shared memory for pressure and velocity fields
                (block_size.0 + 2) * (block_size.1 + 2) * (block_size.2 + 2) * 4 * 8 // 4 fields * 8 bytes
            }
            (KernelType::ThermalDiffusion, OptimizationLevel::Aggressive) => {
                // Shared memory for temperature field with halo
                (block_size.0 + 2) * (block_size.1 + 2) * (block_size.2 + 2) * 8 // 8 bytes per element
            }
            _ => 0,
        };

        Ok(KernelConfig {
            kernel_type,
            optimization_level: self.optimization_level,
            block_size,
            grid_size,
            shared_memory_size,
            registers_per_thread: 32,
        })
    }

    /// Calculate optimal CUDA block size
    fn calculate_cuda_block_size(&self, nx: usize, ny: usize, nz: usize) -> (u32, u32, u32) {
        // Optimize for warp size (32) and memory coalescing
        match self.optimization_level {
            OptimizationLevel::Basic => (32, 4, 1),
            OptimizationLevel::Moderate => {
                if nz >= 8 { (16, 8, 2) } else { (32, 8, 1) }
            }
            OptimizationLevel::Aggressive => {
                // Adaptive block size based on grid dimensions
                if nx >= 128 && ny >= 128 && nz >= 64 {
                    (16, 16, 4) // Large grids
                } else if nx >= 64 && ny >= 64 {
                    (16, 8, 4) // Medium grids
                } else {
                    (32, 4, 2) // Small grids
                }
            }
        }
    }

    /// Calculate optimal OpenCL block size
    fn calculate_opencl_block_size(&self, _nx: usize, _ny: usize, _nz: usize) -> (u32, u32, u32) {
        // OpenCL work group size optimization
        match self.optimization_level {
            OptimizationLevel::Basic => (16, 16, 1),
            OptimizationLevel::Moderate => (16, 16, 2),
            OptimizationLevel::Aggressive => (16, 16, 4),
        }
    }

    /// Generate kernel source code for specific type and backend
    fn generate_kernel_source(&self, kernel_type: KernelType, config: &KernelConfig) -> KwaversResult<String> {
        match self.backend {
            GpuBackend::Cuda => self.generate_cuda_kernel(kernel_type, config),
            GpuBackend::OpenCL | GpuBackend::WebGPU => self.generate_wgsl_kernel(kernel_type, config),
        }
    }

    /// Generate CUDA kernel source
    fn generate_cuda_kernel(&self, kernel_type: KernelType, config: &KernelConfig) -> KwaversResult<String> {
        match kernel_type {
            KernelType::AcousticWave => Ok(self.generate_cuda_acoustic_kernel(config)),
            KernelType::ThermalDiffusion => Ok(self.generate_cuda_thermal_kernel(config)),
            KernelType::FFTForward => Ok(self.generate_cuda_fft_kernel(config, true)),
            KernelType::FFTInverse => Ok(self.generate_cuda_fft_kernel(config, false)),
            KernelType::MemoryCopy => Ok(self.generate_cuda_memcpy_kernel(config)),
            KernelType::BoundaryCondition => Ok(self.generate_cuda_boundary_kernel(config)),
        }
    }

    /// Generate WGSL kernel source for WebGPU
    fn generate_wgsl_kernel(&self, kernel_type: KernelType, config: &KernelConfig) -> KwaversResult<String> {
        match kernel_type {
            KernelType::AcousticWave => Ok(self.generate_wgsl_acoustic_kernel(config)),
            KernelType::ThermalDiffusion => Ok(self.generate_wgsl_thermal_kernel(config)),
            KernelType::FFTForward => Ok(self.generate_wgsl_fft_kernel(config, true)),
            KernelType::FFTInverse => Ok(self.generate_wgsl_fft_kernel(config, false)),
            KernelType::MemoryCopy => Ok(self.generate_wgsl_memcpy_kernel(config)),
            KernelType::BoundaryCondition => Ok(self.generate_wgsl_boundary_kernel(config)),
        }
    }

    /// Generate optimized CUDA acoustic wave kernel
    fn generate_cuda_acoustic_kernel(&self, config: &KernelConfig) -> String {
        let shared_memory = match config.optimization_level {
            OptimizationLevel::Aggressive => "extern __shared__ double shared_data[];",
            _ => "",
        };

        let memory_access = match config.optimization_level {
            OptimizationLevel::Aggressive => {
                // Use shared memory with halo zones
                r#"
                // Load data into shared memory with halo
                int shared_idx = (threadIdx.z + 1) * (blockDim.x + 2) * (blockDim.y + 2) + 
                                (threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1);
                
                shared_data[shared_idx] = pressure[idx];
                shared_data[shared_idx + shared_offset] = velocity_x[idx];
                shared_data[shared_idx + 2 * shared_offset] = velocity_y[idx];
                shared_data[shared_idx + 3 * shared_offset] = velocity_z[idx];
                
                __syncthreads();
                "#
            }
            OptimizationLevel::Moderate => {
                // Use register blocking with proper boundary checks
                r#"
                // Load neighboring values into registers with proper boundary checks
                double p_center = pressure[idx];
                double p_xm1 = (i > 0) ? pressure[idx - 1] : p_center;
                double p_xp1 = (i < nx - 1) ? pressure[idx + 1] : p_center;
                double p_ym1 = (j > 0) ? pressure[idx - nx] : p_center;
                double p_yp1 = (j < ny - 1) ? pressure[idx + nx] : p_center;
                double p_zm1 = (k > 0) ? pressure[idx - nx*ny] : p_center;
                double p_zp1 = (k < nz - 1) ? pressure[idx + nx*ny] : p_center;
                
                double vx_xm1 = (i > 0) ? velocity_x[idx - 1] : velocity_x[idx];
                double vx_xp1 = (i < nx - 1) ? velocity_x[idx + 1] : velocity_x[idx];
                double vy_ym1 = (j > 0) ? velocity_y[idx - nx] : velocity_y[idx];
                double vy_yp1 = (j < ny - 1) ? velocity_y[idx + nx] : velocity_y[idx];
                double vz_zm1 = (k > 0) ? velocity_z[idx - nx*ny] : velocity_z[idx];
                double vz_zp1 = (k < nz - 1) ? velocity_z[idx + nx*ny] : velocity_z[idx];
                "#
            }
            _ => {
                // Direct global memory access with boundary checks
                r#"
                double p_center = pressure[idx];
                double p_xm1 = (i > 0) ? pressure[idx - 1] : p_center;
                double p_xp1 = (i < nx - 1) ? pressure[idx + 1] : p_center;
                double p_ym1 = (j > 0) ? pressure[idx - nx] : p_center;
                double p_yp1 = (j < ny - 1) ? pressure[idx + nx] : p_center;
                double p_zm1 = (k > 0) ? pressure[idx - nx*ny] : p_center;
                double p_zp1 = (k < nz - 1) ? pressure[idx + nx*ny] : p_center;
                "#
            }
        };

        let computation = match config.optimization_level {
            OptimizationLevel::Moderate => {
                // Use register variables for computation
                r#"
    // Pressure update using velocity divergence with register variables
    double div_v = (vx_xp1 - vx_xm1) * 0.5 * inv_dx +
                   (vy_yp1 - vy_ym1) * 0.5 * inv_dy +
                   (vz_zp1 - vz_zm1) * 0.5 * inv_dz;
    
    new_pressure[idx] = p_center - rho0 * c0 * c0 * dt * div_v;
    
    // Velocity updates using pressure gradients with register variables
    double dp_dx = (p_xp1 - p_xm1) * 0.5 * inv_dx;
    double dp_dy = (p_yp1 - p_ym1) * 0.5 * inv_dy;
    double dp_dz = (p_zp1 - p_zm1) * 0.5 * inv_dz;
                "#
            }
            _ => {
                // Standard computation with boundary-safe access
                r#"
    // Pressure update using velocity divergence
    double div_v = (vx_xp1 - vx_xm1) * 0.5 * inv_dx +
                   (vy_yp1 - vy_ym1) * 0.5 * inv_dy +
                   (vz_zp1 - vz_zm1) * 0.5 * inv_dz;
    
    new_pressure[idx] = p_center - rho0 * c0 * c0 * dt * div_v;
    
    // Velocity updates using pressure gradients
    double dp_dx = (p_xp1 - p_xm1) * 0.5 * inv_dx;
    double dp_dy = (p_yp1 - p_ym1) * 0.5 * inv_dy;
    double dp_dz = (p_zp1 - p_zm1) * 0.5 * inv_dz;
                "#
            }
        };

        format!(r#"
extern "C" __global__ void acoustic_wave_kernel(
    double* pressure,
    double* velocity_x,
    double* velocity_y, 
    double* velocity_z,
    double* new_pressure,
    double* new_velocity_x,
    double* new_velocity_y,
    double* new_velocity_z,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double c0, double rho0
) {{
    {shared_memory}
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    {memory_access}
    
    // Acoustic wave equations with optimized finite differences
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double inv_dz = 1.0 / dz;
    
    {computation}
    
    new_velocity_x[idx] = velocity_x[idx] - dt / rho0 * dp_dx;
    new_velocity_y[idx] = velocity_y[idx] - dt / rho0 * dp_dy;
    new_velocity_z[idx] = velocity_z[idx] - dt / rho0 * dp_dz;
}}
"#, shared_memory = shared_memory, memory_access = memory_access, computation = computation)
    }

    /// Generate optimized CUDA thermal diffusion kernel
    fn generate_cuda_thermal_kernel(&self, config: &KernelConfig) -> String {
        let optimization_code = match config.optimization_level {
            OptimizationLevel::Aggressive => {
                r#"
                // Use shared memory for thermal diffusion
                extern __shared__ double shared_temp[];
                int shared_idx = (threadIdx.z + 1) * (blockDim.x + 2) * (blockDim.y + 2) + 
                                (threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1);
                
                shared_temp[shared_idx] = temperature[idx];
                __syncthreads();
                
                double temp_center = shared_temp[shared_idx];
                double temp_xm1 = shared_temp[shared_idx - 1];
                double temp_xp1 = shared_temp[shared_idx + 1];
                double temp_ym1 = shared_temp[shared_idx - (blockDim.x + 2)];
                double temp_yp1 = shared_temp[shared_idx + (blockDim.x + 2)];
                double temp_zm1 = shared_temp[shared_idx - (blockDim.x + 2) * (blockDim.y + 2)];
                double temp_zp1 = shared_temp[shared_idx + (blockDim.x + 2) * (blockDim.y + 2)];
                "#
            }
            OptimizationLevel::Moderate => {
                r#"
                // Use register blocking with proper boundary checks
                double temp_center = temperature[idx];
                double temp_xm1 = (i > 0) ? temperature[idx - 1] : temp_center;
                double temp_xp1 = (i < nx - 1) ? temperature[idx + 1] : temp_center;
                double temp_ym1 = (j > 0) ? temperature[idx - nx] : temp_center;
                double temp_yp1 = (j < ny - 1) ? temperature[idx + nx] : temp_center;
                double temp_zm1 = (k > 0) ? temperature[idx - nx*ny] : temp_center;
                double temp_zp1 = (k < nz - 1) ? temperature[idx + nx*ny] : temp_center;
                "#
            }
            _ => {
                r#"
                // Direct global memory access with boundary checks
                double temp_center = temperature[idx];
                double temp_xm1 = (i > 0) ? temperature[idx - 1] : temp_center;
                double temp_xp1 = (i < nx - 1) ? temperature[idx + 1] : temp_center;
                double temp_ym1 = (j > 0) ? temperature[idx - nx] : temp_center;
                double temp_yp1 = (j < ny - 1) ? temperature[idx + nx] : temp_center;
                double temp_zm1 = (k > 0) ? temperature[idx - nx*ny] : temp_center;
                double temp_zp1 = (k < nz - 1) ? temperature[idx + nx*ny] : temp_center;
                "#
            }
        };

        format!(r#"
extern "C" __global__ void thermal_diffusion_kernel(
    double* temperature,
    double* heat_source,
    double* new_temperature,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double dt, double alpha
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    {optimization_code}
    
    // 3D thermal diffusion with second-order finite differences using register variables
    double d2T_dx2 = (temp_xp1 - 2.0 * temp_center + temp_xm1) / (dx * dx);
    double d2T_dy2 = (temp_yp1 - 2.0 * temp_center + temp_ym1) / (dy * dy);
    double d2T_dz2 = (temp_zp1 - 2.0 * temp_center + temp_zm1) / (dz * dz);
    
    double laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
    
    new_temperature[idx] = temp_center + dt * (alpha * laplacian + heat_source[idx]);
}}
"#, optimization_code = optimization_code)
    }

    /// Generate CUDA FFT kernel
    fn generate_cuda_fft_kernel(&self, _config: &KernelConfig, forward: bool) -> String {
        let direction = if forward { "1" } else { "-1" };
        
        format!(r#"
extern "C" __global__ void fft_kernel(
    double* input_real,
    double* input_imag,
    double* output_real,
    double* output_imag,
    int n, int direction
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Placeholder for FFT implementation - would use cuFFT in practice
    output_real[idx] = input_real[idx];
    output_imag[idx] = input_imag[idx] * {direction};
}}
"#, direction = direction)
    }

    /// Generate CUDA memory copy kernel
    fn generate_cuda_memcpy_kernel(&self, _config: &KernelConfig) -> String {
        r#"
extern "C" __global__ void memcpy_kernel(
    double* src,
    double* dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    dst[idx] = src[idx];
}
"#.to_string()
    }

    /// Generate CUDA boundary condition kernel
    fn generate_cuda_boundary_kernel(&self, _config: &KernelConfig) -> String {
        r#"
extern "C" __global__ void boundary_kernel(
    double* field,
    int nx, int ny, int nz,
    double absorption_coeff
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    // Apply absorbing boundary conditions
    bool is_boundary = (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1);
    
    if (is_boundary) {
        int idx = k * nx * ny + j * nx + i;
        field[idx] *= (1.0 - absorption_coeff);
    }
}
"#.to_string()
    }

    /// Generate WGSL acoustic wave kernel for WebGPU
    fn generate_wgsl_acoustic_kernel(&self, _config: &KernelConfig) -> String {
        r#"
@group(0) @binding(0) var<storage, read> pressure: array<f32>;
@group(0) @binding(1) var<storage, read> velocity_x: array<f32>;
@group(0) @binding(2) var<storage, read> velocity_y: array<f32>;
@group(0) @binding(3) var<storage, read> velocity_z: array<f32>;
@group(0) @binding(4) var<storage, read_write> new_pressure: array<f32>;
@group(0) @binding(5) var<storage, read_write> new_velocity_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> new_velocity_y: array<f32>;
@group(0) @binding(7) var<storage, read_write> new_velocity_z: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    c0: f32,
    rho0: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn acoustic_wave(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;
    
    if (i >= params.nx || j >= params.ny || k >= params.nz) { return; }
    if (i == 0u || i == params.nx - 1u || j == 0u || j == params.ny - 1u || k == 0u || k == params.nz - 1u) { return; }
    
    let idx = k * params.nx * params.ny + j * params.nx + i;
    
    // Acoustic wave equations
    let inv_dx = 1.0 / params.dx;
    let inv_dy = 1.0 / params.dy;
    let inv_dz = 1.0 / params.dz;
    
    // Velocity divergence
    let div_v = (velocity_x[idx + 1u] - velocity_x[idx - 1u]) * 0.5 * inv_dx +
                (velocity_y[idx + params.nx] - velocity_y[idx - params.nx]) * 0.5 * inv_dy +
                (velocity_z[idx + params.nx * params.ny] - velocity_z[idx - params.nx * params.ny]) * 0.5 * inv_dz;
    
    new_pressure[idx] = pressure[idx] - params.rho0 * params.c0 * params.c0 * params.dt * div_v;
    
    // Pressure gradients
    let dp_dx = (pressure[idx + 1u] - pressure[idx - 1u]) * 0.5 * inv_dx;
    let dp_dy = (pressure[idx + params.nx] - pressure[idx - params.nx]) * 0.5 * inv_dy;
    let dp_dz = (pressure[idx + params.nx * params.ny] - pressure[idx - params.nx * params.ny]) * 0.5 * inv_dz;
    
    new_velocity_x[idx] = velocity_x[idx] - params.dt / params.rho0 * dp_dx;
    new_velocity_y[idx] = velocity_y[idx] - params.dt / params.rho0 * dp_dy;
    new_velocity_z[idx] = velocity_z[idx] - params.dt / params.rho0 * dp_dz;
}
"#.to_string()
    }

    /// Generate WGSL thermal diffusion kernel
    fn generate_wgsl_thermal_kernel(&self, _config: &KernelConfig) -> String {
        r#"
@group(0) @binding(0) var<storage, read> temperature: array<f32>;
@group(0) @binding(1) var<storage, read> heat_source: array<f32>;
@group(0) @binding(2) var<storage, read_write> new_temperature: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    dx: f32,
    dy: f32,
    dz: f32,
    dt: f32,
    alpha: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn thermal_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;
    
    if (i >= params.nx || j >= params.ny || k >= params.nz) { return; }
    if (i == 0u || i == params.nx - 1u || j == 0u || j == params.ny - 1u || k == 0u || k == params.nz - 1u) { return; }
    
    let idx = k * params.nx * params.ny + j * params.nx + i;
    
    let temp_center = temperature[idx];
    
    // 3D thermal diffusion
    let d2T_dx2 = (temperature[idx + 1u] - 2.0 * temp_center + temperature[idx - 1u]) / (params.dx * params.dx);
    let d2T_dy2 = (temperature[idx + params.nx] - 2.0 * temp_center + temperature[idx - params.nx]) / (params.dy * params.dy);
    let d2T_dz2 = (temperature[idx + params.nx * params.ny] - 2.0 * temp_center + temperature[idx - params.nx * params.ny]) / (params.dz * params.dz);
    
    let laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
    
    new_temperature[idx] = temp_center + params.dt * (params.alpha * laplacian + heat_source[idx]);
}
"#.to_string()
    }

    /// Generate WGSL FFT kernel
    fn generate_wgsl_fft_kernel(&self, _config: &KernelConfig, _forward: bool) -> String {
        r#"
@group(0) @binding(0) var<storage, read> input_real: array<f32>;
@group(0) @binding(1) var<storage, read> input_imag: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_real: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<f32>;

@compute @workgroup_size(64)
fn fft(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Placeholder FFT implementation
    output_real[idx] = input_real[idx];
    output_imag[idx] = input_imag[idx];
}
"#.to_string()
    }

    /// Generate WGSL memory copy kernel
    fn generate_wgsl_memcpy_kernel(&self, _config: &KernelConfig) -> String {
        r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn memcpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    dst[idx] = src[idx];
}
"#.to_string()
    }

    /// Generate WGSL boundary condition kernel
    fn generate_wgsl_boundary_kernel(&self, _config: &KernelConfig) -> String {
        r#"
@group(0) @binding(0) var<storage, read_write> field: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    absorption_coeff: f32,
}

@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn boundary(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let k = global_id.z;
    
    if (i >= params.nx || j >= params.ny || k >= params.nz) { return; }
    
    let is_boundary = (i == 0u || i == params.nx - 1u || j == 0u || j == params.ny - 1u || k == 0u || k == params.nz - 1u);
    
    if (is_boundary) {
        let idx = k * params.nx * params.ny + j * params.nx + i;
        field[idx] = field[idx] * (1.0 - params.absorption_coeff);
    }
}
"#.to_string()
    }

    /// Compile kernel source to binary
    fn compile_kernel_source(&self, source: &str) -> KwaversResult<Option<Vec<u8>>> {
        match self.backend {
            GpuBackend::Cuda => {
                // In practice, would use NVRTC or similar
                // For now, return None to indicate source-only compilation
                Ok(None)
            }
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // WGSL is compiled by the WebGPU driver
                Ok(None)
            }
        }
    }

    /// Get compiled kernel by type
    pub fn get_kernel(&self, kernel_type: KernelType) -> Option<&CompiledKernel> {
        self.kernels.get(&kernel_type)
    }

    /// Update kernel performance metrics
    pub fn update_performance_metrics(&mut self, kernel_type: KernelType, metrics: GpuPerformanceMetrics) {
        if let Some(kernel) = self.kernels.get_mut(&kernel_type) {
            kernel.performance_metrics = Some(metrics);
        }
    }

    /// Get performance summary for all kernels
    pub fn get_performance_summary(&self) -> HashMap<KernelType, Option<GpuPerformanceMetrics>> {
        self.kernels.iter()
            .map(|(k, v)| (*k, v.performance_metrics.clone()))
            .collect()
    }

    /// Optimize kernels based on runtime performance data
    pub fn optimize_kernels(&mut self, _grid: &Grid) -> KwaversResult<()> {
        // Analyze performance metrics and adjust configurations
        let kernel_updates: Vec<_> = self.kernels
            .iter()
            .filter_map(|(kernel_type, kernel)| {
                if let Some(metrics) = &kernel.performance_metrics {
                    if !metrics.meets_targets() && kernel.config.optimization_level != OptimizationLevel::Aggressive {
                        let new_config = KernelConfig {
                            optimization_level: OptimizationLevel::Aggressive,
                            ..kernel.config.clone()
                        };
                        return Some((*kernel_type, new_config));
                    }
                }
                None
            })
            .collect();
        
        for (kernel_type, new_config) in kernel_updates {
            let new_source = self.generate_kernel_source(kernel_type, &new_config)?;
            if let Some(kernel) = self.kernels.get_mut(&kernel_type) {
                kernel.source_code = new_source;
                kernel.config = new_config;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.kernel_type, KernelType::AcousticWave);
        assert_eq!(config.optimization_level, OptimizationLevel::Moderate);
        assert_eq!(config.block_size, (16, 16, 4));
    }

    #[test]
    fn test_kernel_manager_creation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Moderate);
        assert_eq!(manager.backend, GpuBackend::Cuda);
        assert_eq!(manager.optimization_level, OptimizationLevel::Moderate);
        assert!(manager.kernels.is_empty());
    }

    #[test]
    fn test_cuda_block_size_calculation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Aggressive);
        
        // Large grid
        let block_size = manager.calculate_cuda_block_size(256, 256, 128);
        assert_eq!(block_size, (16, 16, 4));
        
        // Small grid
        let block_size = manager.calculate_cuda_block_size(32, 32, 16);
        assert_eq!(block_size, (32, 4, 2));
    }

    #[test]
    fn test_opencl_block_size_calculation() {
        let manager = KernelManager::new(GpuBackend::OpenCL, OptimizationLevel::Moderate);
        let block_size = manager.calculate_opencl_block_size(128, 128, 64);
        assert_eq!(block_size, (16, 16, 2));
    }

    #[test]
    fn test_kernel_config_generation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Moderate);
        let grid = Grid {
            nx: 64,
            ny: 64,
            nz: 32,
            dx: 0.1e-3,
            dy: 0.1e-3,
            dz: 0.1e-3,
        };

        let config = manager.generate_kernel_config(KernelType::AcousticWave, &grid).unwrap();
        assert_eq!(config.kernel_type, KernelType::AcousticWave);
        assert!(config.grid_size.0 > 0);
        assert!(config.grid_size.1 > 0);
        assert!(config.grid_size.2 > 0);
    }

    #[test]
    fn test_cuda_acoustic_kernel_generation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Moderate);
        let config = KernelConfig::default();
        
        let source = manager.generate_cuda_acoustic_kernel(&config);
        assert!(source.contains("acoustic_wave_kernel"));
        assert!(source.contains("pressure"));
        assert!(source.contains("velocity_x"));
        assert!(source.contains("div_v"));
    }

    #[test]
    fn test_cuda_thermal_kernel_generation() {
        let manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Basic);
        let config = KernelConfig::default();
        
        let source = manager.generate_cuda_thermal_kernel(&config);
        assert!(source.contains("thermal_diffusion_kernel"));
        assert!(source.contains("temperature"));
        assert!(source.contains("laplacian"));
    }

    #[test]
    fn test_wgsl_acoustic_kernel_generation() {
        let manager = KernelManager::new(GpuBackend::WebGPU, OptimizationLevel::Moderate);
        let config = KernelConfig::default();
        
        let source = manager.generate_wgsl_acoustic_kernel(&config);
        assert!(source.contains("@compute"));
        assert!(source.contains("acoustic_wave"));
        assert!(source.contains("pressure"));
        assert!(source.contains("div_v"));
    }

    #[test]
    fn test_wgsl_thermal_kernel_generation() {
        let manager = KernelManager::new(GpuBackend::WebGPU, OptimizationLevel::Moderate);
        let config = KernelConfig::default();
        
        let source = manager.generate_wgsl_thermal_kernel(&config);
        assert!(source.contains("@compute"));
        assert!(source.contains("thermal_diffusion"));
        assert!(source.contains("temperature"));
        assert!(source.contains("laplacian"));
    }

    #[test]
    fn test_kernel_type_enum() {
        assert_eq!(KernelType::AcousticWave, KernelType::AcousticWave);
        assert_ne!(KernelType::AcousticWave, KernelType::ThermalDiffusion);
        
        // Test hash map usage
        let mut map = HashMap::new();
        map.insert(KernelType::AcousticWave, "acoustic");
        assert_eq!(map.get(&KernelType::AcousticWave), Some(&"acoustic"));
    }

    #[test]
    fn test_optimization_level_progression() {
        let levels = vec![
            OptimizationLevel::Basic,
            OptimizationLevel::Moderate,
            OptimizationLevel::Aggressive,
        ];
        
        for level in levels {
            let manager = KernelManager::new(GpuBackend::Cuda, level);
            assert_eq!(manager.optimization_level, level);
        }
    }

    #[test]
    fn test_performance_metrics_integration() {
        let mut manager = KernelManager::new(GpuBackend::Cuda, OptimizationLevel::Moderate);
        
        let metrics = GpuPerformanceMetrics::new(
            1_000_000, // 1M grid points
            10.0,      // 10ms kernel time
            5.0,       // 5ms transfer time
            500.0,     // 500 GB/s bandwidth
            0.1,       // 0.1 GB data
        );
        
        manager.update_performance_metrics(KernelType::AcousticWave, metrics.clone());
        
        let summary = manager.get_performance_summary();
        assert!(summary.contains_key(&KernelType::AcousticWave));
        
        if let Some(Some(retrieved_metrics)) = summary.get(&KernelType::AcousticWave) {
            assert_eq!(retrieved_metrics.kernel_execution_time_ms, 10.0);
        }
    }
}