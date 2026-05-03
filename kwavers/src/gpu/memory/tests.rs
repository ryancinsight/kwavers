use super::*;

#[test]
fn test_memory_pool_allocation() {
    let mut pool = MemoryPool::new(MemoryPoolType::Temporary);
    let handle = pool.allocate(0, 1024).unwrap();

    assert_eq!(handle.block.size, 1024);
    assert_eq!(pool.total_allocated, 1024);

    pool.deallocate(handle).unwrap();
    assert_eq!(pool.total_allocated, 0);
}

#[test]
fn test_unified_memory_manager() {
    let mut manager = UnifiedMemoryManager::new();

    let handle1 = manager
        .allocate(0, MemoryPoolType::Temporary, 2048)
        .unwrap();
    let handle2 = manager
        .allocate(0, MemoryPoolType::Collocation, 4096)
        .unwrap();

    let stats = manager.statistics();
    assert_eq!(stats.allocated_bytes, 6144);

    manager.deallocate(handle1).unwrap();
    manager.deallocate(handle2).unwrap();

    let stats_after = manager.statistics();
    assert_eq!(stats_after.allocated_bytes, 0);
}

#[test]
fn test_memory_compression() {
    let mut manager = UnifiedMemoryManager::new();
    let handle = manager
        .allocate(0, MemoryPoolType::Persistent, 8192)
        .unwrap();

    let ratio = manager.compress(&handle).unwrap();
    assert!(ratio < 1.0); // Should compress

    manager.decompress(&handle).unwrap();
    manager.deallocate(handle).unwrap();
}
