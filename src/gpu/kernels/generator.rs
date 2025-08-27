//! Kernel code generators for different GPU backends

use super::config::KernelConfig;
use crate::grid::Grid;

/// CUDA kernel generator
pub struct CudaKernelGenerator;

impl CudaKernelGenerator {
    /// Generate kernel preamble with common definitions
    pub fn generate_preamble() -> String {
        r#"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

#define M_PI 3.14159265358979323846

// Helper functions
__device__ inline float safe_divide(float a, float b) {
    return (fabsf(b) > 1e-10f) ? a / b : 0.0f;
}

__device__ inline float clamp(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}
"#
        .to_string()
    }

    /// Generate kernel launch configuration
    pub fn generate_launch_config(config: &KernelConfig, grid: &Grid) -> String {
        format!(
            r#"
// Launch configuration
dim3 block_size({}, {}, {});
dim3 grid_size({}, {}, {});
size_t shared_mem_size = {};
"#,
            config.block_size.0,
            config.block_size.1,
            config.block_size.2,
            config.grid_size.0,
            config.grid_size.1,
            config.grid_size.2,
            config.shared_memory_size
        )
    }
}

/// OpenCL kernel generator
pub struct OpenCLKernelGenerator;

impl OpenCLKernelGenerator {
    /// Generate kernel preamble
    pub fn generate_preamble() -> String {
        r#"
// OpenCL kernel utilities
#define M_PI 3.14159265358979323846f

// Helper functions
float safe_divide(float a, float b) {
    return (fabs(b) > 1e-10f) ? a / b : 0.0f;
}

float clamp_value(float x, float min_val, float max_val) {
    return fmin(fmax(x, min_val), max_val);
}
"#
        .to_string()
    }

    /// Generate work group configuration
    pub fn generate_work_config(config: &KernelConfig) -> String {
        format!(
            r#"
// Work group configuration
__attribute__((reqd_work_group_size({}, {}, {})))
"#,
            config.block_size.0, config.block_size.1, config.block_size.2
        )
    }
}

/// WebGPU WGSL kernel generator
pub struct WebGPUKernelGenerator;

impl WebGPUKernelGenerator {
    /// Generate common WGSL utilities
    pub fn generate_utilities() -> String {
        r#"
// WGSL utility functions
fn safe_divide(a: f32, b: f32) -> f32 {
    if (abs(b) > 1e-10) {
        return a / b;
    } else {
        return 0.0;
    }
}

fn clamp_value(x: f32, min_val: f32, max_val: f32) -> f32 {
    return clamp(x, min_val, max_val);
}

fn compute_index_3d(x: u32, y: u32, z: u32, nx: u32, ny: u32) -> u32 {
    return x + y * nx + z * nx * ny;
}
"#
        .to_string()
    }

    /// Generate workgroup attributes
    pub fn generate_workgroup_attr(config: &KernelConfig) -> String {
        format!(
            "@compute @workgroup_size({}, {}, {})",
            config.block_size.0, config.block_size.1, config.block_size.2
        )
    }
}
