//! # GPU Memory Management
//!
//! This module provides optimized GPU memory allocation and management
//! strategies for large-scale ultrasound simulations.

use crate::error::{KwaversResult, KwaversError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU memory pool for efficient allocation
pub struct GpuMemoryPool {
    allocated_buffers: HashMap<usize, GpuMemoryBlock>,
    free_blocks: Vec<GpuMemoryBlock>,
    total_allocated: usize,
    peak_usage: usize,
    allocation_count: usize,
}

/// GPU memory block information
#[derive(Debug, Clone)]
pub struct GpuMemoryBlock {
    pub id: usize,
    pub size: usize,
    pub device_ptr: usize,
    pub is_free: bool,
    pub allocation_time: std::time::Instant,
}

impl GpuMemoryPool {
    /// Create new memory pool
    pub fn new() -> Self {
        Self {
            allocated_buffers: HashMap::new(),
            free_blocks: Vec::new(),
            total_allocated: 0,
            peak_usage: 0,
            allocation_count: 0,
        }
    }

    /// Allocate memory block
    pub fn allocate(&mut self, size: usize) -> KwaversResult<usize> {
        // Try to find a suitable free block first
        if let Some(index) = self.find_free_block(size) {
            let mut block = self.free_blocks.remove(index);
            block.is_free = false;
            block.allocation_time = std::time::Instant::now();
            
            let id = block.id;
            self.allocated_buffers.insert(id, block);
            return Ok(id);
        }

        // Allocate new block
        let id = self.allocation_count;
        self.allocation_count += 1;

        let block = GpuMemoryBlock {
            id,
            size,
            device_ptr: 0, // Would be actual GPU pointer
            is_free: false,
            allocation_time: std::time::Instant::now(),
        };

        self.allocated_buffers.insert(id, block);
        self.total_allocated += size;
        
        if self.total_allocated > self.peak_usage {
            self.peak_usage = self.total_allocated;
        }

        Ok(id)
    }

    /// Free memory block
    pub fn free(&mut self, block_id: usize) -> KwaversResult<()> {
        if let Some(mut block) = self.allocated_buffers.remove(&block_id) {
            block.is_free = true;
            self.free_blocks.push(block);
            Ok(())
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: 0,
                available_bytes: 0,
                reason: format!("Invalid block ID: {}", block_id),
            }))
        }
    }

    /// Find suitable free block
    fn find_free_block(&self, required_size: usize) -> Option<usize> {
        self.free_blocks
            .iter()
            .enumerate()
            .find(|(_, block)| block.size >= required_size)
            .map(|(index, _)| index)
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> GpuMemoryStats {
        let active_blocks = self.allocated_buffers.len();
        let free_blocks = self.free_blocks.len();
        let current_usage = self.allocated_buffers.values()
            .map(|block| block.size)
            .sum::<usize>();

        GpuMemoryStats {
            total_allocated: self.total_allocated,
            current_usage,
            peak_usage: self.peak_usage,
            active_blocks,
            free_blocks,
            allocation_count: self.allocation_count,
        }
    }

    /// Compact memory by merging adjacent free blocks
    pub fn compact(&mut self) -> KwaversResult<()> {
        // Sort free blocks by device pointer
        self.free_blocks.sort_by_key(|block| block.device_ptr);

        // Merge adjacent blocks
        let mut merged_blocks = Vec::new();
        let mut current_block: Option<GpuMemoryBlock> = None;

        for block in self.free_blocks.drain(..) {
            match current_block {
                None => current_block = Some(block),
                Some(ref mut current) => {
                    if current.device_ptr + current.size == block.device_ptr {
                        // Merge blocks
                        current.size += block.size;
                    } else {
                        // Non-adjacent, push current and update
                        merged_blocks.push(current.clone());
                        *current = block;
                    }
                }
            }
        }

        if let Some(block) = current_block {
            merged_blocks.push(block);
        }

        self.free_blocks = merged_blocks;
        Ok(())
    }
}

/// GPU memory statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryStats {
    pub total_allocated: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub active_blocks: usize,
    pub free_blocks: usize,
    pub allocation_count: usize,
}

impl GpuMemoryStats {
    /// Get memory efficiency (0.0 to 1.0)
    pub fn efficiency(&self) -> f64 {
        if self.total_allocated == 0 {
            1.0
        } else {
            self.current_usage as f64 / self.total_allocated as f64
        }
    }

    /// Get fragmentation ratio (0.0 to 1.0, lower is better)
    pub fn fragmentation(&self) -> f64 {
        if self.free_blocks == 0 {
            0.0
        } else {
            self.free_blocks as f64 / (self.active_blocks + self.free_blocks) as f64
        }
    }
}

/// GPU memory transfer optimization
pub struct GpuMemoryTransfer {
    staging_buffers: HashMap<usize, usize>,
    transfer_stats: TransferStats,
}

/// Memory transfer statistics
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub host_to_device_time_ms: f64,
    pub device_to_host_time_ms: f64,
    pub transfer_count: usize,
}

impl GpuMemoryTransfer {
    /// Create new memory transfer manager
    pub fn new() -> Self {
        Self {
            staging_buffers: HashMap::new(),
            transfer_stats: TransferStats::default(),
        }
    }

    /// Optimized host to device transfer
    pub fn host_to_device_optimized(
        &mut self,
        host_data: &[f64],
        device_buffer: usize,
    ) -> KwaversResult<()> {
        let start_time = std::time::Instant::now();
        let data_size = host_data.len() * std::mem::size_of::<f64>();

        // Use staging buffer for large transfers
        if data_size > 1024 * 1024 { // 1MB threshold
            self.transfer_with_staging(host_data, device_buffer)?;
        } else {
            self.transfer_direct(host_data, device_buffer)?;
        }

        let transfer_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.transfer_stats.host_to_device_bytes += data_size;
        self.transfer_stats.host_to_device_time_ms += transfer_time;
        self.transfer_stats.transfer_count += 1;

        Ok(())
    }

    /// Direct memory transfer
    fn transfer_direct(&self, _host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
        // Implementation would depend on GPU backend
        Ok(())
    }

    /// Staged memory transfer for large data
    fn transfer_with_staging(&mut self, _host_data: &[f64], _device_buffer: usize) -> KwaversResult<()> {
        // Implementation would use staging buffers for optimal transfer
        Ok(())
    }

    /// Get transfer bandwidth in GB/s
    pub fn get_bandwidth_gbps(&self) -> f64 {
        let total_bytes = self.transfer_stats.host_to_device_bytes + self.transfer_stats.device_to_host_bytes;
        let total_time_s = (self.transfer_stats.host_to_device_time_ms + self.transfer_stats.device_to_host_time_ms) / 1000.0;
        
        if total_time_s > 0.0 {
            (total_bytes as f64) / (total_time_s * 1e9)
        } else {
            0.0
        }
    }

    /// Get transfer statistics
    pub fn get_stats(&self) -> &TransferStats {
        &self.transfer_stats
    }
}

/// GPU memory alignment utilities
pub struct GpuMemoryAlignment;

impl GpuMemoryAlignment {
    /// Align size to GPU memory requirements
    pub fn align_size(size: usize, alignment: usize) -> usize {
        (size + alignment - 1) / alignment * alignment
    }

    /// Calculate optimal buffer size for 3D arrays
    pub fn optimal_buffer_size(nx: usize, ny: usize, nz: usize, element_size: usize) -> usize {
        let total_elements = nx * ny * nz;
        let base_size = total_elements * element_size;
        
        // Align to 256-byte boundaries for optimal GPU access
        Self::align_size(base_size, 256)
    }

    /// Check if size is properly aligned
    pub fn is_aligned(size: usize, alignment: usize) -> bool {
        size % alignment == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_allocation() {
        let mut pool = GpuMemoryPool::new();
        
        // Test allocation
        let block1 = pool.allocate(1024).unwrap();
        let block2 = pool.allocate(2048).unwrap();
        
        assert_ne!(block1, block2);
        
        let stats = pool.get_stats();
        assert_eq!(stats.active_blocks, 2);
        assert_eq!(stats.total_allocated, 3072);
    }

    #[test]
    fn test_memory_pool_free() {
        let mut pool = GpuMemoryPool::new();
        
        let block = pool.allocate(1024).unwrap();
        pool.free(block).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.active_blocks, 0);
        assert_eq!(stats.free_blocks, 1);
    }

    #[test]
    fn test_memory_alignment() {
        assert_eq!(GpuMemoryAlignment::align_size(100, 64), 128);
        assert_eq!(GpuMemoryAlignment::align_size(128, 64), 128);
        assert_eq!(GpuMemoryAlignment::align_size(129, 64), 192);
        
        assert!(GpuMemoryAlignment::is_aligned(256, 64));
        assert!(!GpuMemoryAlignment::is_aligned(257, 64));
    }

    #[test]
    fn test_optimal_buffer_size() {
        let size = GpuMemoryAlignment::optimal_buffer_size(100, 100, 100, 8);
        assert!(size >= 100 * 100 * 100 * 8);
        assert!(GpuMemoryAlignment::is_aligned(size, 256));
    }

    #[test]
    fn test_memory_stats() {
        let mut pool = GpuMemoryPool::new();
        
        pool.allocate(1000).unwrap();
        pool.allocate(2000).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.efficiency(), 1.0); // All allocated memory is in use
        assert_eq!(stats.fragmentation(), 0.0); // No free blocks
    }

    #[test]
    fn test_transfer_stats() {
        let mut transfer = GpuMemoryTransfer::new();
        let data = vec![1.0; 1000];
        
        transfer.host_to_device_optimized(&data, 0).unwrap();
        
        let stats = transfer.get_stats();
        assert_eq!(stats.transfer_count, 1);
        assert_eq!(stats.host_to_device_bytes, 8000); // 1000 * 8 bytes
    }
}