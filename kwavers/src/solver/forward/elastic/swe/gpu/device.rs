//! `GPUDevice` struct and capability methods.

use crate::domain::grid::Grid;

/// GPU device information and capabilities
#[derive(Debug, Clone)]
pub struct GPUDevice {
    /// Device name
    pub name: String,
    /// Total global memory (bytes)
    pub global_memory: usize,
    /// Shared memory per block (bytes)
    pub shared_memory: usize,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Maximum grid dimensions
    pub max_grid_dims: [usize; 3],
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
}

impl GPUDevice {
    /// Check if device can handle given volume size
    pub fn can_handle_volume(&self, grid: &Grid) -> bool {
        let volume_size = grid.nx * grid.ny * grid.nz * std::mem::size_of::<f64>() * 6;
        let safety_margin = (self.global_memory as f64 * 0.8) as usize;
        volume_size <= safety_margin
    }

    /// Get optimal block size for 3D computations
    #[must_use] 
    pub fn optimal_block_size(&self, grid_dims: [usize; 3]) -> [usize; 3] {
        let mut block_size = [8, 8, 8];

        for i in 0..3 {
            while block_size[i] > 1
                && block_size.iter().product::<usize>() > self.max_threads_per_block
            {
                block_size[i] /= 2;
            }
        }

        for i in 0..3 {
            block_size[i] = block_size[i].min(grid_dims[i]);
        }

        block_size
    }
}
