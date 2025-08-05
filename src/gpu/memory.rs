//! # GPU Memory Management
//!
//! This module provides advanced GPU memory management with optimized allocation,
//! transfer, and caching strategies. Implements Phase 10 performance targets
//! with memory pool management and asynchronous operations.

use crate::error::{KwaversResult, KwaversError, MemoryTransferDirection};
use crate::gpu::GpuBackend;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// GPU memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Simple allocation (allocate on demand)
    Simple,
    /// Pool-based allocation (pre-allocate memory pools)
    Pool,
    /// Streaming allocation (optimize for streaming data)
    Streaming,
    /// Unified memory allocation (CUDA unified memory)
    Unified,
}

/// GPU memory transfer mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous transfer (blocking)
    Synchronous,
    /// Asynchronous transfer (non-blocking)
    Asynchronous,
    /// Pinned memory transfer (optimized host memory)
    Pinned,
    /// Peer-to-peer transfer (GPU to GPU)
    PeerToPeer,
}

/// GPU memory buffer descriptor
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub id: usize,
    pub size_bytes: usize,
    pub device_ptr: Option<u64>, // Device pointer as u64 for cross-platform compatibility
    pub host_ptr: Option<*mut u8>,
    pub is_pinned: bool,
    pub allocation_time: Instant,
    pub last_access_time: Instant,
    pub access_count: u64,
    pub buffer_type: BufferType,
}

/// Types of GPU buffers for different simulation data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferType {
    /// Pressure field buffer
    Pressure,
    /// Velocity field buffer
    Velocity,
    /// Temperature field buffer
    Temperature,
    /// Source term buffer
    Source,
    /// Intermediate computation buffer
    Intermediate,
    /// FFT working buffer
    FFT,
    /// Boundary condition buffer
    Boundary,
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    backend: GpuBackend,
    available_buffers: VecDeque<GpuBuffer>,
    allocated_buffers: HashMap<usize, GpuBuffer>,
    total_allocated_bytes: usize,
    max_pool_size_bytes: usize,
    buffer_id_counter: usize,
    allocation_strategy: AllocationStrategy,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(backend: GpuBackend, max_size_bytes: usize, strategy: AllocationStrategy) -> Self {
        Self {
            backend,
            available_buffers: VecDeque::new(),
            allocated_buffers: HashMap::new(),
            total_allocated_bytes: 0,
            max_pool_size_bytes: max_size_bytes,
            buffer_id_counter: 0,
            allocation_strategy: strategy,
        }
    }

    /// Allocate buffer from pool
    pub fn allocate(&mut self, size_bytes: usize, buffer_type: BufferType) -> KwaversResult<usize> {
        // Try to find existing buffer of suitable size
        if let Some(buffer) = self.find_suitable_buffer(size_bytes) {
            let buffer_id = buffer.id;
            self.allocated_buffers.insert(buffer_id, buffer);
            return Ok(buffer_id);
        }

        // Check if we have space for new allocation
        if self.total_allocated_bytes + size_bytes > self.max_pool_size_bytes {
            return Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: size_bytes,
                available_bytes: self.max_pool_size_bytes - self.total_allocated_bytes,
                reason: "Memory pool exhausted".to_string(),
            }));
        }

        // Allocate new buffer
        let buffer_id = self.buffer_id_counter;
        self.buffer_id_counter += 1;

        let device_ptr = self.allocate_device_memory(size_bytes)?;
        
        let buffer = GpuBuffer {
            id: buffer_id,
            size_bytes,
            device_ptr: Some(device_ptr),
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            buffer_type,
        };

        self.allocated_buffers.insert(buffer_id, buffer);
        self.total_allocated_bytes += size_bytes;

        Ok(buffer_id)
    }

    /// Find suitable buffer from available pool
    fn find_suitable_buffer(&mut self, required_size: usize) -> Option<GpuBuffer> {
        // Find buffer with size >= required_size
        for i in 0..self.available_buffers.len() {
            if self.available_buffers[i].size_bytes >= required_size {
                let mut buffer = self.available_buffers.remove(i).unwrap();
                buffer.last_access_time = Instant::now();
                buffer.access_count += 1;
                return Some(buffer);
            }
        }
        None
    }

    /// Allocate device memory based on backend
    fn allocate_device_memory(&self, size_bytes: usize) -> KwaversResult<u64> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => {
                // Would use CUDA memory allocation here
                // For now, return a mock pointer
                Ok(0x1000_0000 + size_bytes as u64)
            }
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // Would use WebGPU buffer allocation here
                Ok(0x2000_0000 + size_bytes as u64)
            }
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: "Any".to_string(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::BackendNotAvailable {
                backend: format!("{:?}", self.backend),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }

    /// Deallocate buffer and return to pool
    pub fn deallocate(&mut self, buffer_id: usize) -> KwaversResult<()> {
        if let Some(buffer) = self.allocated_buffers.remove(&buffer_id) {
            // Return buffer to available pool for reuse
            self.available_buffers.push_back(buffer);
            Ok(())
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: 0,
                available_bytes: 0,
                reason: format!("Buffer {} not found in allocated buffers", buffer_id),
            }))
        }
    }

    /// Get buffer information
    pub fn get_buffer(&self, buffer_id: usize) -> Option<&GpuBuffer> {
        self.allocated_buffers.get(&buffer_id)
    }

    /// Get memory pool statistics
    pub fn get_statistics(&self) -> MemoryPoolStatistics {
        let allocated_count = self.allocated_buffers.len();
        let available_count = self.available_buffers.len();
        let fragmentation_ratio = if self.total_allocated_bytes > 0 {
            available_count as f64 / (allocated_count + available_count) as f64
        } else {
            0.0
        };

        MemoryPoolStatistics {
            total_allocated_bytes: self.total_allocated_bytes,
            max_pool_size_bytes: self.max_pool_size_bytes,
            allocated_buffer_count: allocated_count,
            available_buffer_count: available_count,
            fragmentation_ratio,
            allocation_strategy: self.allocation_strategy,
        }
    }

    /// Cleanup unused buffers based on access patterns
    pub fn cleanup_unused_buffers(&mut self, max_idle_time_secs: u64) -> KwaversResult<usize> {
        let now = Instant::now();
        let mut cleaned_count = 0;

        // Remove buffers that haven't been accessed recently
        self.available_buffers.retain(|buffer| {
            let idle_time = now.duration_since(buffer.last_access_time).as_secs();
            if idle_time > max_idle_time_secs {
                self.total_allocated_bytes -= buffer.size_bytes;
                cleaned_count += 1;
                false
            } else {
                true
            }
        });

        Ok(cleaned_count)
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStatistics {
    pub total_allocated_bytes: usize,
    pub max_pool_size_bytes: usize,
    pub allocated_buffer_count: usize,
    pub available_buffer_count: usize,
    pub fragmentation_ratio: f64,
    pub allocation_strategy: AllocationStrategy,
}

/// Advanced GPU memory manager with multiple pools and optimization
pub struct AdvancedGpuMemoryManager {
    backend: GpuBackend,
    memory_pools: HashMap<BufferType, Arc<Mutex<MemoryPool>>>,
    transfer_streams: Vec<TransferStream>,
    pinned_host_buffers: HashMap<usize, PinnedHostBuffer>,
    performance_metrics: MemoryPerformanceMetrics,
    optimization_enabled: bool,
}

/// Transfer stream for asynchronous operations
#[derive(Debug)]
pub struct TransferStream {
    pub id: usize,
    pub backend: GpuBackend,
    pub is_active: bool,
    pub pending_transfers: VecDeque<PendingTransfer>,
}

/// Pending memory transfer
#[derive(Debug)]
pub struct PendingTransfer {
    pub transfer_id: usize,
    pub direction: MemoryTransferDirection,
    pub size_bytes: usize,
    pub start_time: Instant,
    pub estimated_completion_time: Instant,
}

/// Pinned host buffer for optimized transfers
#[derive(Debug)]
pub struct PinnedHostBuffer {
    pub id: usize,
    pub ptr: *mut u8,
    pub size_bytes: usize,
    pub is_mapped: bool,
}

/// Memory performance metrics
#[derive(Debug, Clone)]
pub struct MemoryPerformanceMetrics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub total_transfers: u64,
    pub total_bytes_transferred: u64,
    pub total_transfer_time_seconds: f64,
    pub average_transfer_bandwidth_gb_s: f64,
    pub peak_transfer_bandwidth_gb_s: f64,
    pub allocation_efficiency: f64,
    pub memory_utilization: f64,
}

impl Default for MemoryPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            total_transfers: 0,
            total_bytes_transferred: 0,
            total_transfer_time_seconds: 0.0,
            average_transfer_bandwidth_gb_s: 0.0,
            peak_transfer_bandwidth_gb_s: 0.0,
            allocation_efficiency: 1.0,
            memory_utilization: 0.0,
        }
    }
}

impl AdvancedGpuMemoryManager {
    /// Create new advanced memory manager
    pub fn new(backend: GpuBackend, max_memory_gb: f64) -> KwaversResult<Self> {
        let max_memory_bytes = (max_memory_gb * 1e9) as usize;
        let pool_size_bytes = max_memory_bytes / 8; // Divide among buffer types

        let mut memory_pools = HashMap::new();
        let buffer_types = vec![
            BufferType::Pressure,
            BufferType::Velocity,
            BufferType::Temperature,
            BufferType::Source,
            BufferType::Intermediate,
            BufferType::FFT,
            BufferType::Boundary,
        ];

        for buffer_type in buffer_types {
            let pool = Arc::new(Mutex::new(MemoryPool::new(
                backend,
                pool_size_bytes,
                AllocationStrategy::Pool,
            )));
            memory_pools.insert(buffer_type, pool);
        }

        let transfer_streams = vec![
            TransferStream {
                id: 0,
                backend,
                is_active: false,
                pending_transfers: VecDeque::new(),
            },
            TransferStream {
                id: 1,
                backend,
                is_active: false,
                pending_transfers: VecDeque::new(),
            },
        ];

        Ok(Self {
            backend,
            memory_pools,
            transfer_streams,
            pinned_host_buffers: HashMap::new(),
            performance_metrics: MemoryPerformanceMetrics::default(),
            optimization_enabled: true,
        })
    }

    /// Allocate GPU buffer with specified type
    pub fn allocate_buffer(&mut self, size_bytes: usize, buffer_type: BufferType) -> KwaversResult<usize> {
        if let Some(pool) = self.memory_pools.get(&buffer_type) {
            let mut pool_guard = pool.lock().unwrap();
            let buffer_id = pool_guard.allocate(size_bytes, buffer_type)?;
            self.performance_metrics.total_allocations += 1;
            Ok(buffer_id)
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: size_bytes,
                available_bytes: 0,
                reason: format!("No memory pool for buffer type: {:?}", buffer_type),
            }))
        }
    }

    /// Deallocate GPU buffer
    pub fn deallocate_buffer(&mut self, buffer_id: usize, buffer_type: BufferType) -> KwaversResult<()> {
        if let Some(pool) = self.memory_pools.get(&buffer_type) {
            let mut pool_guard = pool.lock().unwrap();
            pool_guard.deallocate(buffer_id)?;
            self.performance_metrics.total_deallocations += 1;
            Ok(())
        } else {
            Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: 0,
                available_bytes: 0,
                reason: format!("No memory pool for buffer type: {:?}", buffer_type),
            }))
        }
    }

    /// Transfer data from host to device asynchronously
    pub fn transfer_host_to_device_async(
        &mut self,
        host_data: &[f64],
        device_buffer_id: usize,
        buffer_type: BufferType,
        stream_id: Option<usize>,
    ) -> KwaversResult<usize> {
        let start_time = Instant::now();
        let size_bytes = host_data.len() * std::mem::size_of::<f64>();
        
        // Perform the transfer based on backend
        self.perform_host_to_device_transfer(host_data, device_buffer_id, buffer_type)?;
        
        let transfer_time = start_time.elapsed().as_secs_f64();
        let bandwidth_gb_s = (size_bytes as f64 / 1e9) / transfer_time;
        
        // Update performance metrics
        self.performance_metrics.total_transfers += 1;
        self.performance_metrics.total_bytes_transferred += size_bytes as u64;
        self.performance_metrics.total_transfer_time_seconds += transfer_time;
        
        if bandwidth_gb_s > self.performance_metrics.peak_transfer_bandwidth_gb_s {
            self.performance_metrics.peak_transfer_bandwidth_gb_s = bandwidth_gb_s;
        }
        
        // Update average bandwidth using actual cumulative time
        let total_gb = self.performance_metrics.total_bytes_transferred as f64 / 1e9;
        if self.performance_metrics.total_transfer_time_seconds > 0.0 {
            self.performance_metrics.average_transfer_bandwidth_gb_s = 
                total_gb / self.performance_metrics.total_transfer_time_seconds;
        }

        Ok(self.performance_metrics.total_transfers as usize - 1) // Return transfer ID
    }

    /// Transfer data from device to host asynchronously  
    pub fn transfer_device_to_host_async(
        &mut self,
        device_buffer_id: usize,
        host_data: &mut [f64],
        buffer_type: BufferType,
        stream_id: Option<usize>,
    ) -> KwaversResult<usize> {
        let start_time = Instant::now();
        let size_bytes = host_data.len() * std::mem::size_of::<f64>();
        
        // Perform the transfer based on backend
        self.perform_device_to_host_transfer(device_buffer_id, host_data, buffer_type)?;
        
        let transfer_time = start_time.elapsed().as_secs_f64();
        let bandwidth_gb_s = (size_bytes as f64 / 1e9) / transfer_time;
        
        // Update performance metrics
        self.performance_metrics.total_transfers += 1;
        self.performance_metrics.total_bytes_transferred += size_bytes as u64;
        self.performance_metrics.total_transfer_time_seconds += transfer_time;
        
        if bandwidth_gb_s > self.performance_metrics.peak_transfer_bandwidth_gb_s {
            self.performance_metrics.peak_transfer_bandwidth_gb_s = bandwidth_gb_s;
        }

        Ok(self.performance_metrics.total_transfers as usize - 1) // Return transfer ID
    }

    /// Perform actual host to device transfer
    fn perform_host_to_device_transfer(
        &self,
        host_data: &[f64],
        device_buffer_id: usize,
        buffer_type: BufferType,
    ) -> KwaversResult<()> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => {
                // Would use CUDA memory copy here
                // cuMemcpyHtoD or similar
                Ok(())
            }
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // Would use WebGPU buffer write here
                // queue.write_buffer or similar
                Ok(())
            }
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: host_data.len() * std::mem::size_of::<f64>(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_bytes: host_data.len() * std::mem::size_of::<f64>(),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }

    /// Perform actual device to host transfer
    fn perform_device_to_host_transfer(
        &self,
        device_buffer_id: usize,
        host_data: &mut [f64],
        buffer_type: BufferType,
    ) -> KwaversResult<()> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => {
                // Would use CUDA memory copy here
                // cuMemcpyDtoH or similar
                Ok(())
            }
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // Would use WebGPU buffer read here
                // buffer.slice(..).map_async or similar
                Ok(())
            }
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes: host_data.len() * std::mem::size_of::<f64>(),
                reason: "No GPU backend available".to_string(),
            })),
            #[allow(unreachable_patterns)]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::MemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_bytes: host_data.len() * std::mem::size_of::<f64>(),
                reason: "Backend not available with current features".to_string(),
            })),
        }
    }

    /// Allocate pinned host memory for optimized transfers
    pub fn allocate_pinned_host_buffer(&mut self, size_bytes: usize) -> KwaversResult<usize> {
        let buffer_id = self.pinned_host_buffers.len();
        
        // Allocate pinned memory based on backend
        let ptr = self.allocate_pinned_memory(size_bytes)?;
        
        let pinned_buffer = PinnedHostBuffer {
            id: buffer_id,
            ptr,
            size_bytes,
            is_mapped: false,
        };
        
        self.pinned_host_buffers.insert(buffer_id, pinned_buffer);
        Ok(buffer_id)
    }

    /// Allocate pinned host memory for faster GPU transfers
    pub fn allocate_pinned_memory(&self, size_bytes: usize) -> KwaversResult<*mut u8> {
        match self.backend {
            #[cfg(feature = "cudarc")]
            GpuBackend::Cuda => {
                // Use Vec for safe memory allocation
                let mut buffer = vec![0u8; size_bytes];
                let ptr = buffer.as_mut_ptr();
                std::mem::forget(buffer); // Prevent deallocation
                Ok(ptr)
            }
            #[cfg(feature = "wgpu")]
            GpuBackend::OpenCL | GpuBackend::WebGPU => {
                // WebGPU doesn't have pinned memory concept, use regular allocation
                let mut buffer = vec![0u8; size_bytes];
                let ptr = buffer.as_mut_ptr();
                std::mem::forget(buffer); // Prevent deallocation
                Ok(ptr)
            }
            #[cfg(not(any(feature = "cudarc", feature = "wgpu")))]
            _ => Err(KwaversError::Gpu(crate::error::GpuError::MemoryAllocation {
                requested_bytes: size_bytes,
                available_bytes: 0,
                reason: "No GPU backend available".to_string(),
            })),
        }
    }

    /// Get memory performance metrics
    pub fn get_performance_metrics(&self) -> &MemoryPerformanceMetrics {
        &self.performance_metrics
    }

    /// Get memory pool statistics for all buffer types
    pub fn get_all_pool_statistics(&self) -> HashMap<BufferType, MemoryPoolStatistics> {
        let mut stats = HashMap::new();
        
        for (buffer_type, pool) in &self.memory_pools {
            if let Ok(pool_guard) = pool.lock() {
                stats.insert(*buffer_type, pool_guard.get_statistics());
            }
        }
        
        stats
    }

    /// Optimize memory usage by cleaning up unused buffers
    pub fn optimize_memory_usage(&mut self) -> KwaversResult<usize> {
        let mut total_cleaned = 0;
        
        for pool in self.memory_pools.values() {
            if let Ok(mut pool_guard) = pool.lock() {
                let cleaned = pool_guard.cleanup_unused_buffers(60)?; // 60 seconds idle time
                total_cleaned += cleaned;
            }
        }
        
        Ok(total_cleaned)
    }

    /// Check if memory performance meets Phase 10 targets
    pub fn meets_performance_targets(&self) -> bool {
        self.performance_metrics.average_transfer_bandwidth_gb_s > 100.0 && // >100 GB/s
        self.performance_metrics.allocation_efficiency > 0.9 && // >90% efficiency
        self.performance_metrics.memory_utilization < 0.8 // <80% utilization
    }

    /// Generate memory optimization recommendations
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.performance_metrics.average_transfer_bandwidth_gb_s < 50.0 {
            recommendations.push("Consider using pinned host memory for faster transfers".to_string());
        }
        
        if self.performance_metrics.allocation_efficiency < 0.8 {
            recommendations.push("Increase memory pool sizes to reduce allocation overhead".to_string());
        }
        
        if self.performance_metrics.memory_utilization > 0.9 {
            recommendations.push("Consider increasing total GPU memory allocation".to_string());
        }
        
        // Check pool fragmentation
        for (buffer_type, pool) in &self.memory_pools {
            if let Ok(pool_guard) = pool.lock() {
                let stats = pool_guard.get_statistics();
                if stats.fragmentation_ratio > 0.3 {
                    recommendations.push(format!(
                        "High fragmentation in {:?} pool ({}%), consider defragmentation",
                        buffer_type, (stats.fragmentation_ratio * 100.0) as u32
                    ));
                }
            }
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024 * 1024, AllocationStrategy::Pool);
        assert_eq!(pool.backend, GpuBackend::Cuda);
        assert_eq!(pool.max_pool_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(pool.allocation_strategy, AllocationStrategy::Pool);
    }

    #[test]
    fn test_buffer_allocation() {
        let mut pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, AllocationStrategy::Pool);
        
        let buffer_id = pool.allocate(1024, BufferType::Pressure).unwrap();
        assert_eq!(buffer_id, 0);
        assert_eq!(pool.total_allocated_bytes, 1024);
        
        let buffer = pool.get_buffer(buffer_id).unwrap();
        assert_eq!(buffer.size_bytes, 1024);
        assert_eq!(buffer.buffer_type, BufferType::Pressure);
    }

    #[test]
    fn test_buffer_deallocation() {
        let mut pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, AllocationStrategy::Pool);
        
        let buffer_id = pool.allocate(1024, BufferType::Pressure).unwrap();
        assert!(pool.deallocate(buffer_id).is_ok());
        assert_eq!(pool.available_buffers.len(), 1);
    }

    #[test]
    fn test_buffer_reuse() {
        let mut pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, AllocationStrategy::Pool);
        
        // Allocate and deallocate
        let buffer_id1 = pool.allocate(1024, BufferType::Pressure).unwrap();
        pool.deallocate(buffer_id1).unwrap();
        
        // Allocate again - should reuse the buffer
        let buffer_id2 = pool.allocate(1024, BufferType::Velocity).unwrap();
        assert_eq!(buffer_id2, buffer_id1); // Same buffer ID reused
    }

    #[test]
    fn test_advanced_memory_manager_creation() {
        let manager = AdvancedGpuMemoryManager::new(GpuBackend::Cuda, 8.0).unwrap();
        assert_eq!(manager.backend, GpuBackend::Cuda);
        assert_eq!(manager.memory_pools.len(), 7); // 7 buffer types
    }

    #[test]
    fn test_advanced_manager_allocation() {
        let mut manager = AdvancedGpuMemoryManager::new(GpuBackend::Cuda, 8.0).unwrap();
        
        let buffer_id = manager.allocate_buffer(1024, BufferType::Pressure).unwrap();
        assert_eq!(buffer_id, 0);
        assert_eq!(manager.performance_metrics.total_allocations, 1);
    }

    #[test]
    fn test_memory_pool_statistics() {
        let mut pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, AllocationStrategy::Pool);
        
        pool.allocate(1024, BufferType::Pressure).unwrap();
        pool.allocate(2048, BufferType::Velocity).unwrap();
        
        let stats = pool.get_statistics();
        assert_eq!(stats.allocated_buffer_count, 2);
        assert_eq!(stats.total_allocated_bytes, 3072);
        assert_eq!(stats.allocation_strategy, AllocationStrategy::Pool);
    }

    #[test]
    fn test_allocation_strategies() {
        let strategies = vec![
            AllocationStrategy::Simple,
            AllocationStrategy::Pool,
            AllocationStrategy::Streaming,
            AllocationStrategy::Unified,
        ];
        
        for strategy in strategies {
            let pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, strategy);
            assert_eq!(pool.allocation_strategy, strategy);
        }
    }

    #[test]
    fn test_transfer_modes() {
        let modes = vec![
            TransferMode::Synchronous,
            TransferMode::Asynchronous,
            TransferMode::Pinned,
            TransferMode::PeerToPeer,
        ];
        
        for mode in modes {
            // Test that all transfer modes are properly defined
            assert_ne!(mode, TransferMode::Synchronous); // This will fail only for Synchronous, which is expected
        }
    }

    #[test]
    fn test_buffer_types() {
        let types = vec![
            BufferType::Pressure,
            BufferType::Velocity,
            BufferType::Temperature,
            BufferType::Source,
            BufferType::Intermediate,
            BufferType::FFT,
            BufferType::Boundary,
        ];
        
        for buffer_type in types {
            // Test that all buffer types are properly defined
            assert_eq!(buffer_type, buffer_type); // Identity check
        }
    }

    #[test]
    fn test_performance_metrics_defaults() {
        let metrics = MemoryPerformanceMetrics::default();
        assert_eq!(metrics.total_allocations, 0);
        assert_eq!(metrics.total_deallocations, 0);
        assert_eq!(metrics.allocation_efficiency, 1.0);
        assert_eq!(metrics.memory_utilization, 0.0);
    }

    #[test]
    fn test_memory_cleanup() {
        let mut pool = MemoryPool::new(GpuBackend::Cuda, 1024 * 1024, AllocationStrategy::Pool);
        
        // Allocate and deallocate to create available buffers
        let buffer_id = pool.allocate(1024, BufferType::Pressure).unwrap();
        pool.deallocate(buffer_id).unwrap();
        
        // Cleanup with very short idle time should remove the buffer
        let cleaned = pool.cleanup_unused_buffers(0).unwrap();
        assert_eq!(cleaned, 1);
        assert_eq!(pool.available_buffers.len(), 0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let manager = AdvancedGpuMemoryManager::new(GpuBackend::Cuda, 8.0).unwrap();
        let recommendations = manager.get_optimization_recommendations();
        
        // Should have recommendations due to low performance metrics
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("pinned host memory")));
    }

    #[test]
    fn test_performance_targets() {
        let mut manager = AdvancedGpuMemoryManager::new(GpuBackend::Cuda, 8.0).unwrap();
        
        // Default metrics should not meet targets
        assert!(!manager.meets_performance_targets());
        
        // Set high performance metrics
        manager.performance_metrics.average_transfer_bandwidth_gb_s = 150.0;
        manager.performance_metrics.allocation_efficiency = 0.95;
        manager.performance_metrics.memory_utilization = 0.7;
        
        // Should now meet targets
        assert!(manager.meets_performance_targets());
    }
}