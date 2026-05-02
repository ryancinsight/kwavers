use super::kernel::CudaKernelManager;
use super::memory::{GpuMemoryManager, MemoryPool, MemoryPoolType, MemoryStats};

#[test]
fn test_gpu_memory_manager_creation() {
    let manager = GpuMemoryManager::new();
    assert!(manager.is_ok());
}

#[test]
fn test_memory_pool_allocation() {
    let mut pool = MemoryPool::new(MemoryPoolType::Temporary, 1024 * 1024, 256);

    let block = pool.allocate(1024);
    assert!(block.is_ok());

    let block = block.unwrap();
    assert_eq!(block.size, 1024);

    assert!(pool.deallocate(block).is_ok());
}

#[test]
fn test_cuda_kernel_manager() {
    let manager = CudaKernelManager::new();
    assert!(manager.is_ok());

    let mut manager = manager.unwrap();

    assert!(manager.compile_ptx("test", "dummy_ptx").is_ok());

    assert!(manager.modules.contains_key("test"));
}

#[test]
fn test_memory_stats() {
    let stats = MemoryStats::default();

    assert_eq!(stats.peak_gpu_memory, 0);
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.deallocation_count, 0);
}
