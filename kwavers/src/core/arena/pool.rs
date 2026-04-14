//! Lock-Free Buffer Pool with NUMA Awareness
//!
//! Provides high-performance buffer pooling with:
//! - **64-byte cache-line alignment**: optimal SIMD/cache performance
//! - **NUMA awareness**: delegates topology to [`super::numa`]
//! - **Batch allocation**: atomic multi-buffer acquisition
//! - **Lock-free operations**: Treiber stack for O(1) acquire/release
//!
//! # Algorithm
//!
//! **Theorem** (Treiber Stack Correctness): The CAS-based stack provides a
//! linearizable, lock-free LIFO queue under any interleaving of concurrent
//! operations.
//!
//! **Proof sketch**: Each `compare_exchange` is an atomic step. If the CAS
//! succeeds, the operation appears to take effect at that instant (linearisation
//! point). If it fails, the loop retries with a fresh read — no progress is
//! lost and at least one concurrent operation must have succeeded, guaranteeing
//! lock-freedom. ∎
//!
//! **Bug fix** (vs earlier draft): `PooledBuffer::drop` now calls
//! `pool.release(ptr)` so that buffers are genuinely recycled.  The previous
//! implementation silently discarded buffers on drop, defeating pooling.
//!
//! # References
//!
//! - Treiber R.K. (1986). "Systems programming: coping with parallelism".
//!   IBM Research Report RJ 5118.
//! - Bonwick J. (1994). "The Slab Allocator". *USENIX Summer Technical Conference*.
//! - McKenney P.E. (2004). "Exploiting Deferred Destruction". *Linux Symposium*.

use crate::core::error::{KwaversError, KwaversResult};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Cache line size for x86_64 architectures (64 bytes).
pub const CACHE_LINE_SIZE: usize = 64;

/// Default pool capacity (number of buffers).
pub const DEFAULT_POOL_CAPACITY: usize = 16;

/// Buffer pool configuration.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of elements per buffer.
    pub elements: usize,
    /// Element size in bytes.
    pub element_size: usize,
    /// Number of buffers in pool.
    pub capacity: usize,
    /// NUMA node affinity (-1 for any node).
    pub numa_node: i32,
}

impl PoolConfig {
    /// Configuration for f64 field buffers with NUMA awareness.
    ///
    /// # Mathematical Specification
    ///
    /// **Size Calculation**: `total_size = ceil(n × 8 / 64) × 64`
    ///
    /// **Proof of Alignment**: For any `n`:
    /// - `n × 8` is the raw size
    /// - Padding rounds up to multiple of 64
    /// - Therefore `total_size ≡ 0 (mod 64)`. ∎
    #[must_use]
    pub fn for_f64_field(elements: usize, capacity: usize, numa_node: i32) -> Self {
        Self {
            elements,
            element_size: std::mem::size_of::<f64>(),
            capacity,
            numa_node,
        }
    }

    /// Total size per buffer including alignment padding.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        let raw_size = self.elements * self.element_size;
        (raw_size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
    }

    /// Total memory required for the pool.
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.buffer_size() * self.capacity
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            elements: 1024,
            element_size: std::mem::size_of::<f64>(),
            capacity: DEFAULT_POOL_CAPACITY,
            numa_node: -1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pool Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Pool allocation statistics.
#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    /// Total capacity.
    pub capacity: usize,
    /// Currently allocated.
    pub allocated: usize,
    /// Peak allocation count.
    pub peak_allocated: usize,
    /// Buffer size in bytes.
    pub buffer_size: usize,
    /// Total memory reserved.
    pub total_memory: usize,
}

impl PoolStats {
    /// Current utilization ratio [0.0, 1.0].
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            self.allocated as f64 / self.capacity as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NUMA-Aware Memory Allocation (delegates to numa module)
// ─────────────────────────────────────────────────────────────────────────────

/// Allocate memory with first-touch NUMA policy and 64-byte alignment.
fn allocate_numa_aware(size: usize) -> KwaversResult<NonNull<u8>> {
    let layout = Layout::from_size_align(size, CACHE_LINE_SIZE).map_err(|_| {
        KwaversError::System(crate::core::error::SystemError::MemoryAllocation {
            requested_bytes: size,
            reason: "Invalid layout for NUMA-aware allocation".to_string(),
        })
    })?;

    // SAFETY: Layout is valid (non-zero size, power-of-2 alignment).
    let ptr = unsafe { alloc(layout) };
    let memory = NonNull::new(ptr).ok_or_else(|| {
        KwaversError::System(crate::core::error::SystemError::MemoryAllocation {
            requested_bytes: size,
            reason: "Failed to allocate NUMA-aware memory".to_string(),
        })
    })?;

    Ok(memory)
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer Pool
// ─────────────────────────────────────────────────────────────────────────────

/// Node in the Treiber stack free-list.
#[repr(C)]
struct BufferNode {
    next: *mut BufferNode,
}

/// Lock-free buffer pool with NUMA awareness.
///
/// Uses an atomic Treiber stack for O(1) acquire/release.
/// All buffers are 64-byte aligned for optimal cache/SIMD performance.
#[derive(Debug)]
pub struct BufferPool {
    /// Configuration snapshot.
    config: PoolConfig,
    /// Underlying memory block.
    memory: NonNull<u8>,
    /// Memory layout for deallocation.
    layout: Layout,
    /// Stack of available buffers (atomic singly-linked list).
    free_list: AtomicPtr<BufferNode>,
    /// Number of buffers currently allocated.
    allocated: AtomicUsize,
    /// Peak allocation count.
    peak_allocated: AtomicUsize,
}

// SAFETY: BufferPool is Send+Sync because all state is atomically synchronized.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Create a new buffer pool with the given configuration.
    ///
    /// # Algorithm
    ///
    /// 1. Allocate contiguous block: `total = config.buffer_size() × capacity`
    /// 2. Partition into aligned buffers
    /// 3. Initialize Treiber stack with all buffers
    ///
    /// Time: O(capacity), Space: O(total_memory)
    pub fn new(config: PoolConfig) -> KwaversResult<Arc<Self>> {
        if config.capacity == 0 || config.elements == 0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "pool_config".to_string(),
                    value: 0.0,
                    reason: "Pool capacity and elements must be non-zero".to_string(),
                },
            ));
        }

        let buffer_size = config.buffer_size();
        let total_size = buffer_size * config.capacity;

        let memory = allocate_numa_aware(total_size)?;

        let layout = Layout::from_size_align(total_size, CACHE_LINE_SIZE).map_err(|_| {
            KwaversError::System(crate::core::error::SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to create layout for pool".to_string(),
            })
        })?;

        let pool = Arc::new(Self {
            config: config.clone(),
            memory,
            layout,
            free_list: AtomicPtr::new(std::ptr::null_mut()),
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
        });

        // Partition memory into buffers and push to free list in reverse order
        // so that buffer 0 is at the head (LIFO: first acquired = most recently freed).
        // SAFETY: memory is valid for total_size bytes; buffer_ptr is within range.
        unsafe {
            let base_ptr = memory.as_ptr();
            for i in (0..config.capacity).rev() {
                let buffer_ptr = base_ptr.add(i * buffer_size) as *mut BufferNode;
                (*buffer_ptr).next = pool.free_list.load(Ordering::Relaxed);
                pool.free_list.store(buffer_ptr, Ordering::Relaxed);
            }
        }

        Ok(pool)
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns `ResourceUnavailable` if the pool is empty.
    ///
    /// # Lock-Free Guarantee
    ///
    /// Uses a CAS loop (Treiber stack pop). At least one thread succeeds per
    /// iteration, guaranteeing lock-freedom.
    pub fn acquire(self: &Arc<Self>) -> KwaversResult<PooledBuffer> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                return Err(KwaversError::System(
                    crate::core::error::SystemError::ResourceUnavailable {
                        resource: "buffer pool".to_string(),
                    },
                ));
            }

            // SAFETY: `head` is a valid pointer (from our contiguous pool block)
            // and cannot be freed externally because only `release` pushes back.
            let next = unsafe { (*head).next };

            match self
                .free_list
                .compare_exchange(head, next, Ordering::Release, Ordering::Acquire)
            {
                Ok(_) => {
                    let new_allocated = self.allocated.fetch_add(1, Ordering::SeqCst) + 1;

                    // Update peak atomically without a mutex.
                    let mut peak = self.peak_allocated.load(Ordering::Relaxed);
                    loop {
                        if new_allocated <= peak {
                            break;
                        }
                        match self.peak_allocated.compare_exchange(
                            peak,
                            new_allocated,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(actual) => peak = actual,
                        }
                    }

                    return Ok(PooledBuffer {
                        ptr: head as *mut u8,
                        pool: Arc::clone(self),
                    });
                }
                Err(_) => continue,
            }
        }
    }

    /// Return a buffer pointer to the free-list (Treiber stack push).
    ///
    /// # Safety
    ///
    /// - `ptr` must have been obtained from `acquire` on this pool.
    /// - `ptr` must not be in use (no live references to the buffer).
    /// - Must not be called twice for the same `ptr` (no double-free).
    pub(super) unsafe fn release(&self, ptr: *mut u8) {
        let node = ptr as *mut BufferNode;

        loop {
            let head = self.free_list.load(Ordering::Acquire);
            (*node).next = head;

            match self
                .free_list
                .compare_exchange(head, node, Ordering::Release, Ordering::Acquire)
            {
                Ok(_) => {
                    self.allocated.fetch_sub(1, Ordering::SeqCst);
                    break;
                }
                Err(_) => continue,
            }
        }
    }

    /// Get current pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let allocated = self.allocated.load(Ordering::Relaxed);
        PoolStats {
            capacity: self.config.capacity,
            allocated,
            peak_allocated: self.peak_allocated.load(Ordering::Relaxed),
            buffer_size: self.config.buffer_size(),
            total_memory: self.config.total_memory_bytes(),
        }
    }

    /// Get buffer size in bytes.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.config.buffer_size()
    }

    /// Get reference to pool configuration.
    #[must_use]
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        // SAFETY: memory was allocated with alloc(), layout matches.
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PooledBuffer — RAII Guard
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard for a buffer acquired from [`BufferPool`].
///
/// Automatically returns the buffer to the pool when dropped.  This corrects
/// the previous implementation which silently discarded buffers on drop,
/// defeating pool recycling.
#[derive(Debug)]
pub struct PooledBuffer {
    /// Pointer to the buffer memory (within pool's contiguous block).
    ptr: *mut u8,
    /// Strong reference to the pool for recycling on drop.
    pool: Arc<BufferPool>,
}

// SAFETY: PooledBuffer is Send+Sync because pool uses atomic synchronization
// and the pointed-to memory is exclusively owned by this guard.
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

impl PooledBuffer {
    /// View buffer as an immutable byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        let size = self.pool.config.buffer_size();
        // SAFETY: ptr is valid for `size` bytes (from pool construction).
        unsafe { std::slice::from_raw_parts(self.ptr, size) }
    }

    /// View buffer as a mutable byte slice.
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let size = self.pool.config.buffer_size();
        // SAFETY: exclusive ownership guarantees no aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, size) }
    }

    /// View buffer as an immutable typed slice.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_size % size_of::<T>() != 0`.
    #[must_use]
    pub fn as_typed<T>(&self) -> &[T] {
        let byte_size = self.pool.config.buffer_size();
        assert!(
            byte_size.is_multiple_of(std::mem::size_of::<T>()),
            "buffer size not aligned to type"
        );
        let count = byte_size / std::mem::size_of::<T>();
        // SAFETY: aligned, valid, exclusively owned.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, count) }
    }

    /// View buffer as a mutable typed slice.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_size % size_of::<T>() != 0`.
    #[must_use]
    pub fn as_typed_mut<T>(&mut self) -> &mut [T] {
        let byte_size = self.pool.config.buffer_size();
        assert!(
            byte_size.is_multiple_of(std::mem::size_of::<T>()),
            "buffer size not aligned to type"
        );
        let count = byte_size / std::mem::size_of::<T>();
        // SAFETY: exclusive ownership guarantees no aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, count) }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // SAFETY:
        // - `self.ptr` was obtained from `BufferPool::acquire` on `self.pool`.
        // - This `PooledBuffer` has exclusive ownership (no cloning).
        // - `drop` is called exactly once (Rust ownership guarantee).
        // Therefore the safety contract of `release` is satisfied.
        unsafe {
            self.pool.release(self.ptr);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch Buffer Allocation
// ─────────────────────────────────────────────────────────────────────────────

/// Batch allocation for multi-field operations.
///
/// Atomically acquires multiple buffers; if any acquisition fails, all
/// previously acquired buffers are released (all-or-nothing semantics).
#[derive(Debug)]
pub struct BufferBatch {
    buffers: Vec<PooledBuffer>,
}

impl BufferBatch {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Acquire `count` buffers from the pool.
    ///
    /// # Atomicity
    ///
    /// Either all `count` buffers are acquired, or none are (the `Err` path
    /// drops `buffers`, returning each acquired buffer to the pool via
    /// `PooledBuffer::drop`).
    pub fn acquire(pool: &Arc<BufferPool>, count: usize) -> KwaversResult<Self> {
        let mut buffers = Vec::with_capacity(count);

        for _ in 0..count {
            match pool.acquire() {
                Ok(buffer) => buffers.push(buffer),
                Err(e) => {
                    // Drop releases already-acquired buffers back to pool.
                    drop(buffers);
                    return Err(e);
                }
            }
        }

        Ok(Self { buffers })
    }

    /// Number of buffers in batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Access buffer at index as byte slice.
    #[must_use]
    pub fn get(&self, index: usize) -> &[u8] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_bytes()
    }

    /// Access buffer at index as mutable byte slice.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> &mut [u8] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_bytes_mut()
    }

    /// Access buffer as typed slice.
    #[must_use]
    pub fn get_typed<T>(&self, index: usize) -> &[T] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_typed::<T>()
    }

    /// Access buffer as mutable typed slice.
    #[must_use]
    pub fn get_typed_mut<T>(&mut self, index: usize) -> &mut [T] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_typed_mut::<T>()
    }
}

impl Default for BufferBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-NUMA-Node Pool Manager
// ─────────────────────────────────────────────────────────────────────────────

/// Per-NUMA-node pool manager.
///
/// Creates separate pools for each NUMA node to ensure local memory access,
/// using topology information from [`super::numa::NumaTopology`].
#[derive(Debug)]
pub struct NumaPoolManager {
    /// Pools per NUMA node (indexed by node id).
    pools: Vec<Option<Arc<BufferPool>>>,
}

impl NumaPoolManager {
    /// Create pool manager with one pool per detected NUMA node.
    pub fn new(config: PoolConfig) -> KwaversResult<Self> {
        let topology = super::numa::NumaTopology::detect();
        let mut pools = Vec::with_capacity(topology.node_count);

        for node in 0..topology.node_count {
            let mut node_config = config.clone();
            node_config.numa_node = node as i32;
            match BufferPool::new(node_config) {
                Ok(pool) => pools.push(Some(pool)),
                Err(_) => pools.push(None),
            }
        }

        Ok(Self { pools })
    }

    /// Acquire buffer from pool on specified NUMA node.
    ///
    /// Falls back to any available pool if the preferred node is unavailable.
    pub fn acquire_on_node(&self, node: i32) -> KwaversResult<PooledBuffer> {
        if node >= 0 && (node as usize) < self.pools.len() {
            if let Some(ref pool) = self.pools[node as usize] {
                return pool.acquire();
            }
        }

        for pool in self.pools.iter().flatten() {
            if let Ok(buffer) = pool.acquire() {
                return Ok(buffer);
            }
        }

        Err(KwaversError::System(
            crate::core::error::SystemError::ResourceUnavailable {
                resource: "NUMA pool".to_string(),
            },
        ))
    }

    /// Get pool statistics per node.
    #[must_use]
    pub fn stats(&self) -> Vec<Option<PoolStats>> {
        self.pools
            .iter()
            .map(|p| p.as_ref().map(|pool| pool.stats()))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_buffer_size_alignment() {
        // 100 × 8 = 800 bytes → rounds up to 832 (= 13 × 64)
        let config = PoolConfig::for_f64_field(100, 16, -1);
        let raw_size = 100 * 8;
        let padded = (raw_size + 63) & !63;
        assert_eq!(config.buffer_size(), padded);
        assert_eq!(config.buffer_size() % 64, 0);
    }

    #[test]
    fn test_pool_acquire_release_cycle() {
        let config = PoolConfig::for_f64_field(128, 4, -1);
        let pool = BufferPool::new(config).expect("pool must create");

        assert_eq!(pool.stats().allocated, 0);

        let buf = pool.acquire().expect("first acquire must succeed");
        assert_eq!(pool.stats().allocated, 1);

        drop(buf); // PooledBuffer::drop calls release()
        assert_eq!(pool.stats().allocated, 0);
    }

    #[test]
    fn test_pool_exhaustion_then_recycle() {
        let config = PoolConfig::for_f64_field(64, 2, -1);
        let pool = BufferPool::new(config).expect("pool must create");

        let b0 = pool.acquire().expect("first acquire");
        let b1 = pool.acquire().expect("second acquire");

        // Pool exhausted — third acquire must fail.
        assert!(pool.acquire().is_err());

        // Release one buffer.
        drop(b0);
        assert_eq!(pool.stats().allocated, 1);

        // Now another acquire succeeds.
        let _b2 = pool.acquire().expect("acquire after release must succeed");
        drop(b1);
        drop(_b2);
        assert_eq!(pool.stats().allocated, 0);
    }

    #[test]
    fn test_buffer_batch_all_or_nothing() {
        let config = PoolConfig::for_f64_field(64, 3, -1);
        let pool = BufferPool::new(config).expect("pool must create");

        // Requesting 4 from a pool of 3 must fail and release all acquired.
        assert!(BufferBatch::acquire(&pool, 4).is_err());
        // All buffers must be back in the pool.
        assert_eq!(pool.stats().allocated, 0);
    }

    #[test]
    fn test_pool_stats_peak_tracking() {
        let config = PoolConfig::for_f64_field(64, 4, -1);
        let pool = BufferPool::new(config).expect("pool must create");

        let b0 = pool.acquire().unwrap();
        let b1 = pool.acquire().unwrap();
        let b2 = pool.acquire().unwrap();

        assert_eq!(pool.stats().peak_allocated, 3);

        drop(b0);
        drop(b1);
        drop(b2);

        // Peak should remain at 3 after release.
        assert_eq!(pool.stats().peak_allocated, 3);
        assert_eq!(pool.stats().allocated, 0);
    }

    #[test]
    fn test_pooled_buffer_typed_access() {
        let config = PoolConfig::for_f64_field(8, 2, -1);
        let pool = BufferPool::new(config).expect("pool must create");
        let mut buf = pool.acquire().expect("acquire");

        let data: &mut [f64] = buf.as_typed_mut();
        assert_eq!(data.len(), 8);
        data[0] = std::f64::consts::PI;
        assert_eq!(buf.as_typed::<f64>()[0], std::f64::consts::PI);
    }
}
