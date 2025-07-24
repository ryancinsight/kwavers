//! # GPU Compute Kernels
//!
//! This module provides optimized GPU compute kernels for various
//! physics operations in ultrasound simulation.

use crate::error::KwaversResult;

/// GPU kernel types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Acoustic wave propagation kernel
    AcousticWave,
    /// Thermal diffusion kernel
    ThermalDiffusion,
    /// Cavitation dynamics kernel
    Cavitation,
    /// FFT operations kernel
    FFT,
    /// Boundary condition kernel
    Boundary,
}

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub kernel_type: KernelType,
    pub grid_size: (usize, usize, usize),
    pub block_size: usize,
    pub shared_memory_bytes: usize,
    pub optimization_level: OptimizationLevel,
}

/// Kernel optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Basic optimization
    Basic,
    /// Moderate optimization with memory coalescing
    Moderate,
    /// Aggressive optimization with all techniques
    Aggressive,
}

impl KernelConfig {
    /// Create new kernel configuration
    pub fn new(kernel_type: KernelType, grid_size: (usize, usize, usize)) -> Self {
        let block_size = Self::optimal_block_size(kernel_type, grid_size);
        let shared_memory_bytes = Self::calculate_shared_memory(kernel_type, block_size);

        Self {
            kernel_type,
            grid_size,
            block_size,
            shared_memory_bytes,
            optimization_level: OptimizationLevel::Moderate,
        }
    }

    /// Calculate optimal block size for kernel type
    fn optimal_block_size(kernel_type: KernelType, grid_size: (usize, usize, usize)) -> usize {
        match kernel_type {
            KernelType::AcousticWave => {
                // Acoustic wave kernels benefit from larger blocks
                if grid_size.0 * grid_size.1 * grid_size.2 > 1_000_000 {
                    512
                } else {
                    256
                }
            }
            KernelType::ThermalDiffusion => {
                // Thermal diffusion uses moderate block sizes
                256
            }
            KernelType::Cavitation => {
                // Cavitation kernels need smaller blocks due to complexity
                128
            }
            KernelType::FFT => {
                // FFT kernels use power-of-2 sizes
                256
            }
            KernelType::Boundary => {
                // Boundary kernels are simple, use larger blocks
                512
            }
        }
    }

    /// Calculate shared memory requirements
    fn calculate_shared_memory(kernel_type: KernelType, block_size: usize) -> usize {
        match kernel_type {
            KernelType::AcousticWave => {
                // Need shared memory for finite difference stencils
                block_size * 4 * std::mem::size_of::<f64>() // 4 fields (p, vx, vy, vz)
            }
            KernelType::ThermalDiffusion => {
                // Shared memory for temperature values
                block_size * std::mem::size_of::<f64>()
            }
            KernelType::Cavitation => {
                // Complex cavitation calculations need more shared memory
                block_size * 8 * std::mem::size_of::<f64>() // Multiple bubble parameters
            }
            KernelType::FFT => {
                // FFT needs working space
                block_size * 2 * std::mem::size_of::<f64>() // Complex numbers
            }
            KernelType::Boundary => {
                // Minimal shared memory for boundary conditions
                block_size * std::mem::size_of::<f64>()
            }
        }
    }

    /// Get grid dimensions for kernel launch
    pub fn grid_dimensions(&self) -> (u32, u32, u32) {
        let total_threads = self.grid_size.0 * self.grid_size.1 * self.grid_size.2;
        let blocks_needed = (total_threads + self.block_size - 1) / self.block_size;
        
        // Optimize grid layout based on problem dimensions
        match self.kernel_type {
            KernelType::AcousticWave | KernelType::ThermalDiffusion => {
                // 3D grid layout for 3D problems
                let blocks_per_dim = ((blocks_needed as f64).powf(1.0/3.0).ceil() as u32).max(1);
                (blocks_per_dim, blocks_per_dim, blocks_per_dim)
            }
            _ => {
                // 1D grid layout for simpler kernels
                (blocks_needed as u32, 1, 1)
            }
        }
    }

    /// Get block dimensions for kernel launch
    pub fn block_dimensions(&self) -> (u32, u32, u32) {
        match self.kernel_type {
            KernelType::AcousticWave | KernelType::ThermalDiffusion => {
                // 3D block layout
                let threads_per_dim = ((self.block_size as f64).powf(1.0/3.0).ceil() as u32).max(1);
                (threads_per_dim, threads_per_dim, threads_per_dim)
            }
            _ => {
                // 1D block layout
                (self.block_size as u32, 1, 1)
            }
        }
    }

    /// Generate CUDA kernel source code
    pub fn generate_cuda_source(&self) -> KwaversResult<String> {
        match self.kernel_type {
            KernelType::AcousticWave => self.generate_acoustic_cuda_kernel(),
            KernelType::ThermalDiffusion => self.generate_thermal_cuda_kernel(),
            KernelType::Cavitation => self.generate_cavitation_cuda_kernel(),
            KernelType::FFT => self.generate_fft_cuda_kernel(),
            KernelType::Boundary => self.generate_boundary_cuda_kernel(),
        }
    }

    /// Generate acoustic wave CUDA kernel
    fn generate_acoustic_cuda_kernel(&self) -> KwaversResult<String> {
        let optimization_flags = match self.optimization_level {
            OptimizationLevel::Basic => "",
            OptimizationLevel::Moderate => "__launch_bounds__(256, 4)",
            OptimizationLevel::Aggressive => "__launch_bounds__(512, 2) __forceinline__",
        };

        Ok(format!(r#"
extern "C" {{
    {optimization_flags}
    __global__ void acoustic_update_kernel(
        float* pressure, float* velocity_x, float* velocity_y, float* velocity_z,
        unsigned int nx, unsigned int ny, unsigned int nz,
        float dx, float dy, float dz, float dt,
        float sound_speed, float density
    ) {{
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (idx >= nx || idy >= ny || idz >= nz) return;
        if (idx == 0 || idx == nx-1 || idy == 0 || idy == ny-1 || idz == 0 || idz == nz-1) return;
        
        unsigned int id = idx + idy * nx + idz * nx * ny;
        
        // Finite difference stencil for acoustic wave equation
        float dp_dx = (pressure[id + 1] - pressure[id - 1]) / (2.0f * dx);
        float dp_dy = (pressure[id + nx] - pressure[id - nx]) / (2.0f * dy);
        float dp_dz = (pressure[id + nx * ny] - pressure[id - nx * ny]) / (2.0f * dz);
        
        float dvx_dx = (velocity_x[id + 1] - velocity_x[id - 1]) / (2.0f * dx);
        float dvy_dy = (velocity_y[id + nx] - velocity_y[id - nx]) / (2.0f * dy);
        float dvz_dz = (velocity_z[id + nx * ny] - velocity_z[id - nx * ny]) / (2.0f * dz);
        
        // Update pressure (continuity equation)
        pressure[id] -= density * sound_speed * sound_speed * dt * (dvx_dx + dvy_dy + dvz_dz);
        
        // Update velocities (momentum equations)
        velocity_x[id] -= dt / density * dp_dx;
        velocity_y[id] -= dt / density * dp_dy;
        velocity_z[id] -= dt / density * dp_dz;
    }}
}}
"#, optimization_flags = optimization_flags))
    }

    /// Generate thermal diffusion CUDA kernel
    fn generate_thermal_cuda_kernel(&self) -> KwaversResult<String> {
        let optimization_flags = match self.optimization_level {
            OptimizationLevel::Basic => "",
            OptimizationLevel::Moderate => "__launch_bounds__(256, 4)",
            OptimizationLevel::Aggressive => "__launch_bounds__(512, 2) __forceinline__",
        };

        Ok(format!(r#"
extern "C" {{
    {optimization_flags}
    __global__ void thermal_update_kernel(
        float* temperature, const float* heat_source,
        unsigned int nx, unsigned int ny, unsigned int nz,
        float dx, float dy, float dz, float dt, float thermal_diffusivity
    ) {{
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (idx >= nx || idy >= ny || idz >= nz) return;
        if (idx == 0 || idx == nx-1 || idy == 0 || idy == ny-1 || idz == 0 || idz == nz-1) return;
        
        unsigned int id = idx + idy * nx + idz * nx * ny;
        
        // 3D Laplacian using finite differences
        float d2T_dx2 = (temperature[id + 1] - 2.0f * temperature[id] + temperature[id - 1]) / (dx * dx);
        float d2T_dy2 = (temperature[id + nx] - 2.0f * temperature[id] + temperature[id - nx]) / (dy * dy);
        float d2T_dz2 = (temperature[id + nx * ny] - 2.0f * temperature[id] + temperature[id - nx * ny]) / (dz * dz);
        
        // Heat equation: dT/dt = α∇²T + Q
        float laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2;
        temperature[id] += dt * (thermal_diffusivity * laplacian + heat_source[id]);
    }}
}}
"#, optimization_flags = optimization_flags))
    }

    /// Generate cavitation dynamics CUDA kernel
    fn generate_cavitation_cuda_kernel(&self) -> KwaversResult<String> {
        Ok(r#"
extern "C" {
    __global__ void cavitation_update_kernel(
        float* bubble_radius, float* bubble_velocity, const float* pressure,
        unsigned int nx, unsigned int ny, unsigned int nz,
        float dt, float surface_tension, float viscosity
    ) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nx * ny * nz) return;
        
        // Rayleigh-Plesset equation for bubble dynamics
        float R = bubble_radius[idx];
        float R_dot = bubble_velocity[idx];
        float P = pressure[idx];
        
        if (R > 0.0f) {
            float R_ddot = (P - 2.0f * surface_tension / R) / (R * 1000.0f) - 1.5f * R_dot * R_dot / R;
            
            bubble_velocity[idx] += dt * R_ddot;
            bubble_radius[idx] += dt * bubble_velocity[idx];
            
            // Prevent negative radius
            if (bubble_radius[idx] < 0.0f) {
                bubble_radius[idx] = 0.0f;
                bubble_velocity[idx] = 0.0f;
            }
        }
    }
}
"#.to_string())
    }

    /// Generate FFT CUDA kernel (placeholder)
    fn generate_fft_cuda_kernel(&self) -> KwaversResult<String> {
        Ok(r#"
extern "C" {
    __global__ void fft_kernel(
        float* real_part, float* imag_part,
        unsigned int n, int forward
    ) {
        // FFT implementation would go here
        // This is a placeholder for complex FFT operations
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        
        // Simple identity operation as placeholder
        if (forward) {
            // Forward FFT placeholder
        } else {
            // Inverse FFT placeholder
        }
    }
}
"#.to_string())
    }

    /// Generate boundary condition CUDA kernel
    fn generate_boundary_cuda_kernel(&self) -> KwaversResult<String> {
        Ok(r#"
extern "C" {
    __global__ void boundary_update_kernel(
        float* field, unsigned int nx, unsigned int ny, unsigned int nz,
        int boundary_type
    ) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (idx >= nx || idy >= ny || idz >= nz) return;
        
        unsigned int id = idx + idy * nx + idz * nx * ny;
        
        // Apply boundary conditions
        if (idx == 0 || idx == nx-1 || idy == 0 || idy == ny-1 || idz == 0 || idz == nz-1) {
            switch (boundary_type) {
                case 0: // Absorbing boundary
                    field[id] *= 0.9f;
                    break;
                case 1: // Rigid boundary
                    field[id] = 0.0f;
                    break;
                case 2: // Periodic boundary
                    // Periodic implementation would go here
                    break;
                default:
                    field[id] = 0.0f;
            }
        }
    }
}
"#.to_string())
    }

    /// Estimate kernel performance
    pub fn estimate_performance(&self) -> KernelPerformanceEstimate {
        let total_operations = self.grid_size.0 * self.grid_size.1 * self.grid_size.2;
        
        let ops_per_thread = match self.kernel_type {
            KernelType::AcousticWave => 20,      // Complex finite difference operations
            KernelType::ThermalDiffusion => 15,  // Laplacian calculation
            KernelType::Cavitation => 10,        // Bubble dynamics
            KernelType::FFT => 25,               // FFT operations
            KernelType::Boundary => 5,           // Simple boundary updates
        };

        let estimated_gflops = (total_operations * ops_per_thread) as f64 / 1e9;
        let memory_bandwidth_gb = (total_operations * std::mem::size_of::<f64>() * 2) as f64 / 1e9; // Read + Write

        KernelPerformanceEstimate {
            estimated_gflops,
            memory_bandwidth_gb,
            occupancy_estimate: self.estimate_occupancy(),
            shared_memory_utilization: self.shared_memory_bytes as f64 / (48.0 * 1024.0), // 48KB typical
        }
    }

    /// Estimate GPU occupancy
    fn estimate_occupancy(&self) -> f64 {
        let threads_per_block = self.block_size;
        let shared_mem_per_block = self.shared_memory_bytes;
        
        // Simplified occupancy model
        let max_threads_per_sm = 2048; // Typical for modern GPUs
        let max_shared_mem_per_sm = 48 * 1024; // 48KB typical
        
        let blocks_by_threads = max_threads_per_sm / threads_per_block;
        let blocks_by_shared_mem = if shared_mem_per_block > 0 {
            max_shared_mem_per_sm / shared_mem_per_block
        } else {
            blocks_by_threads
        };
        
        let max_blocks = blocks_by_threads.min(blocks_by_shared_mem);
        let occupancy = (max_blocks * threads_per_block) as f64 / max_threads_per_sm as f64;
        
        occupancy.min(1.0)
    }
}

/// Kernel performance estimate
#[derive(Debug, Clone)]
pub struct KernelPerformanceEstimate {
    pub estimated_gflops: f64,
    pub memory_bandwidth_gb: f64,
    pub occupancy_estimate: f64,
    pub shared_memory_utilization: f64,
}

impl KernelPerformanceEstimate {
    /// Check if performance meets Phase 9 targets
    pub fn meets_phase9_targets(&self) -> bool {
        self.estimated_gflops > 100.0 &&           // >100 GFLOPS
        self.occupancy_estimate > 0.75 &&          // >75% occupancy
        self.shared_memory_utilization < 0.8       // <80% shared memory usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_creation() {
        let config = KernelConfig::new(KernelType::AcousticWave, (128, 128, 128));
        
        assert_eq!(config.kernel_type, KernelType::AcousticWave);
        assert_eq!(config.grid_size, (128, 128, 128));
        assert!(config.block_size > 0);
        assert!(config.shared_memory_bytes > 0);
    }

    #[test]
    fn test_optimal_block_sizes() {
        let small_grid = (64, 64, 64);
        let large_grid = (256, 256, 256);
        
        let acoustic_small = KernelConfig::new(KernelType::AcousticWave, small_grid);
        let acoustic_large = KernelConfig::new(KernelType::AcousticWave, large_grid);
        
        assert_eq!(acoustic_small.block_size, 256);
        assert_eq!(acoustic_large.block_size, 512);
    }

    #[test]
    fn test_grid_dimensions() {
        let config = KernelConfig::new(KernelType::AcousticWave, (100, 100, 100));
        let (gx, gy, gz) = config.grid_dimensions();
        
        assert!(gx > 0 && gy > 0 && gz > 0);
        assert!(gx * gy * gz * config.block_size >= 1_000_000); // Should cover all grid points
    }

    #[test]
    fn test_cuda_kernel_generation() {
        let config = KernelConfig::new(KernelType::AcousticWave, (64, 64, 64));
        let source = config.generate_cuda_source().unwrap();
        
        assert!(source.contains("acoustic_update_kernel"));
        assert!(source.contains("extern \"C\""));
        assert!(source.contains("__global__"));
    }

    #[test]
    fn test_performance_estimation() {
        let config = KernelConfig::new(KernelType::AcousticWave, (128, 128, 128));
        let perf = config.estimate_performance();
        
        assert!(perf.estimated_gflops > 0.0);
        assert!(perf.memory_bandwidth_gb > 0.0);
        assert!(perf.occupancy_estimate >= 0.0 && perf.occupancy_estimate <= 1.0);
    }

    #[test]
    fn test_thermal_kernel_generation() {
        let config = KernelConfig::new(KernelType::ThermalDiffusion, (64, 64, 64));
        let source = config.generate_cuda_source().unwrap();
        
        assert!(source.contains("thermal_update_kernel"));
        assert!(source.contains("thermal_diffusivity"));
        assert!(source.contains("heat_source"));
    }

    #[test]
    fn test_optimization_levels() {
        let mut config = KernelConfig::new(KernelType::AcousticWave, (64, 64, 64));
        
        config.optimization_level = OptimizationLevel::Aggressive;
        let source = config.generate_cuda_source().unwrap();
        assert!(source.contains("__launch_bounds__"));
        assert!(source.contains("__forceinline__"));
    }
}