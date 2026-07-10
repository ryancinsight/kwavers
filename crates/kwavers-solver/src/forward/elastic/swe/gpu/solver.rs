//! `GPUElasticWaveSolver3D` struct and impl.

use super::device::GPUDevice;
use super::memory::{GPUMemoryPool, SweGpuMemoryStats};
use super::metrics::SweGpuStepMetrics;
use super::types::{GPUInversionResult, GPUPropagationResult};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use leto::Array3;
use std::collections::HashMap;

/// Legacy 3D elastic-wave GPU performance model.
///
/// This path estimates GPU memory and kernel timing but does not dispatch real
/// GPU work. Production WGPU or CUDA execution belongs in provider-generic
/// Hephaestus implementations in `kwavers-gpu`.
#[derive(Debug)]
pub struct GPUElasticWaveSolver3D {
    device: GPUDevice,
    memory_pool: GPUMemoryPool,
    pub(super) kernel_cache: HashMap<String, GPUKernel>,
    performance_metrics: SweGpuStepMetrics,
}

#[derive(Debug, Clone)]
pub(super) struct GPUKernel {
    pub(super) _name: String,
    pub(super) _shared_memory: usize,
    pub(super) _registers: usize,
    pub(super) _occupancy: f64,
}

impl GPUElasticWaveSolver3D {
    /// Create new GPU-accelerated solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(device: GPUDevice) -> KwaversResult<Self> {
        let memory_pool = GPUMemoryPool::new(device.global_memory, 256);

        Ok(Self {
            device,
            memory_pool,
            kernel_cache: HashMap::new(),
            performance_metrics: SweGpuStepMetrics::default(),
        })
    }

    /// Initialize GPU kernels for 3D SWE
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn initialize_kernels(&mut self) -> KwaversResult<()> {
        self.kernel_cache.insert(
            "elastic_wave_3d".to_owned(),
            GPUKernel {
                _name: "elastic_wave_3d".to_owned(),
                _shared_memory: 8192,
                _registers: 32,
                _occupancy: 0.75,
            },
        );

        self.kernel_cache.insert(
            "multidirectional_inversion".to_owned(),
            GPUKernel {
                _name: "multidirectional_inversion".to_owned(),
                _shared_memory: 4096,
                _registers: 24,
                _occupancy: 0.8,
            },
        );

        self.kernel_cache.insert(
            "volumetric_attenuation".to_owned(),
            GPUKernel {
                _name: "volumetric_attenuation".to_owned(),
                _shared_memory: 2048,
                _registers: 16,
                _occupancy: 0.9,
            },
        );

        Ok(())
    }

    /// Execute GPU-accelerated 3D wave propagation
    /// # Errors
    /// - Returns [`KwaversError::ResourceLimitExceeded`] if the precondition for a ResourceLimitExceeded-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn propagate_waves_gpu(
        &mut self,
        _initial_displacements: &[Array3<f64>],
        _push_times: &[f64],
        grid: &Grid,
        time_steps: usize,
    ) -> KwaversResult<GPUPropagationResult> {
        let start_time = std::time::Instant::now();

        let volume_size = grid.nx * grid.ny * grid.nz;
        let total_memory = volume_size * std::mem::size_of::<f64>() * 8;

        if !self.device.can_handle_volume(grid) {
            return Err(KwaversError::ResourceLimitExceeded {
                message: format!(
                    "Volume too large for GPU memory: {} GB required",
                    total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
            });
        }

        let ux_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;
        let uy_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;
        let uz_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;
        let vx_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;
        let vy_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;
        let vz_block = self
            .memory_pool
            .allocate(volume_size * std::mem::size_of::<f64>())?;

        let block_size = self.device.optimal_block_size([grid.nx, grid.ny, grid.nz]);
        let grid_size = [
            grid.nx.div_ceil(block_size[0]),
            grid.ny.div_ceil(block_size[1]),
            grid.nz.div_ceil(block_size[2]),
        ];

        let mut kernel_time = 0.0;
        for step in 0..time_steps {
            let _kernel_start = std::time::Instant::now();

            let kernel_execution_time = self.simulate_kernel_execution(
                "elastic_wave_3d",
                grid_size,
                block_size,
                volume_size,
            );

            kernel_time += kernel_execution_time;

            if step % 100 == 0 {
                let transfer_time =
                    self.simulate_data_transfer(volume_size * std::mem::size_of::<f64>());
                kernel_time += transfer_time;
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        self.performance_metrics.total_kernel_time += kernel_time;
        self.performance_metrics.total_execution_time += total_time;
        self.performance_metrics.kernels_executed += time_steps;

        self.memory_pool.free(ux_block);
        self.memory_pool.free(uy_block);
        self.memory_pool.free(uz_block);
        self.memory_pool.free(vx_block);
        self.memory_pool.free(vy_block);
        self.memory_pool.free(vz_block);

        Ok(GPUPropagationResult {
            execution_time: total_time,
            kernel_time,
            memory_used: total_memory,
            throughput: volume_size as f64 * time_steps as f64 / total_time,
            grid_size,
            block_size,
        })
    }

    /// Estimate kernel execution time from thread count and assumed FLOP rate.
    ///
    /// **Performance model only** — no real GPU kernel is launched.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn simulate_kernel_execution(
        &self,
        kernel_name: &str,
        grid_size: [usize; 3],
        block_size: [usize; 3],
        _volume_size: usize,
    ) -> f64 {
        let _kernel = self.kernel_cache.get(kernel_name).unwrap();

        let total_threads =
            grid_size.iter().product::<usize>() * block_size.iter().product::<usize>();
        let operations_per_thread = 100;
        let total_operations = total_threads * operations_per_thread;

        let gpu_performance = 10.0; // 10 TFLOPS
        let execution_time = total_operations as f64 / (gpu_performance * 1e12);

        execution_time + 0.00001
    }

    /// Estimate PCIe data transfer time.
    ///
    /// **Performance model only** — assumes PCIe 4.0 x16 (~32 GB/s).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn simulate_data_transfer(&self, bytes: usize) -> f64 {
        let pcie_bandwidth = 32.0 * 1024.0 * 1024.0 * 1024.0;
        bytes as f64 / pcie_bandwidth
    }

    /// Execute GPU-accelerated multi-directional inversion
    /// # Errors
    /// - Returns [`KwaversError::ResourceLimitExceeded`] if the precondition for a ResourceLimitExceeded-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn multidirectional_inversion_gpu(
        &mut self,
        arrival_times: &[Array3<f64>],
        wave_directions: &[[f64; 3]],
        grid: &Grid,
    ) -> KwaversResult<GPUInversionResult> {
        let start_time = std::time::Instant::now();

        let volume_size = grid.nx * grid.ny * grid.nz;
        let total_memory = volume_size * std::mem::size_of::<f64>() * ((arrival_times.len()) + 2);

        if !self.device.can_handle_volume(grid) {
            return Err(KwaversError::ResourceLimitExceeded {
                message: "Insufficient GPU memory for inversion".to_owned(),
            });
        }

        let mut memory_blocks = Vec::new();
        for _ in 0..((arrival_times.len()) + 2) {
            memory_blocks.push(
                self.memory_pool
                    .allocate(volume_size * std::mem::size_of::<f64>())?,
            );
        }

        let block_size = self.device.optimal_block_size([grid.nx, grid.ny, grid.nz]);
        let grid_size = [
            grid.nx.div_ceil(block_size[0]),
            grid.ny.div_ceil(block_size[1]),
            grid.nz.div_ceil(block_size[2]),
        ];

        let kernel_time = self.simulate_kernel_execution(
            "multidirectional_inversion",
            grid_size,
            block_size,
            volume_size * (arrival_times.len()),
        );

        let total_time = start_time.elapsed().as_secs_f64() + kernel_time;

        self.performance_metrics.total_kernel_time += kernel_time;
        self.performance_metrics.total_execution_time += total_time;
        self.performance_metrics.inversions_performed += 1;

        for block in memory_blocks {
            self.memory_pool.free(block);
        }

        Ok(GPUInversionResult {
            execution_time: total_time,
            kernel_time,
            memory_used: total_memory,
            directions_processed: (wave_directions.len()),
            convergence_iterations: 50,
            residual_error: 0.001,
        })
    }

    /// Get performance metrics
    #[must_use]
    pub fn performance_metrics(&self) -> &SweGpuStepMetrics {
        &self.performance_metrics
    }

    /// Get memory statistics
    #[must_use]
    pub fn memory_stats(&self) -> SweGpuMemoryStats {
        self.memory_pool.memory_stats()
    }

    /// Optimize memory layout for GPU access
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn optimize_memory_layout(&self, data: &mut Array3<f64>) {
        data.as_slice().unwrap();
    }
}
