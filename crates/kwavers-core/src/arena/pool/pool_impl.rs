use super::{allocate_numa_aware, PoolConfig, PoolStats, CACHE_LINE_SIZE};
use crate::error::{KwaversError, KwaversResult};
use std::alloc::{dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;

/// Node in the Treiber stack free-list.
#[repr(C)]
struct BufferNode {
    next: *mut Self,
}

/// Lock-free buffer pool with NUMA awareness.
///
/// Uses an atomic Treiber stack for O(1) acquire/release.
/// All buffers are 64-byte aligned for optimal cache/SIMD performance.
#[derive(Debug)]
pub struct BufferPool {
    config: PoolConfig,
    memory: NonNull<u8>,
    layout: Layout,
    free_list: AtomicPtr<BufferNode>,
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

// SAFETY: BufferPool is Send+Sync because all state is atomically synchronized.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Create a new buffer pool.
    ///
    /// 1. Allocate contiguous block: `total = config.buffer_size() × capacity`
    /// 2. Partition into aligned buffers
    /// 3. Initialize Treiber stack with all buffers
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: PoolConfig) -> KwaversResult<Arc<Self>> {
        if config.capacity == 0 || config.elements == 0 {
            return Err(KwaversError::Validation(
                crate::error::ValidationError::InvalidValue {
                    parameter: "pool_config".to_owned(),
                    value: 0.0,
                    reason: "Pool capacity and elements must be non-zero".to_owned(),
                },
            ));
        }

        let buffer_size = config.buffer_size();
        let total_size = buffer_size * config.capacity;

        let memory = allocate_numa_aware(total_size)?;

        let layout = Layout::from_size_align(total_size, CACHE_LINE_SIZE).map_err(|_| {
            KwaversError::System(crate::error::SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to create layout for pool".to_owned(),
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

        // Partition memory into buffers and push to free list in reverse order.
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

    /// Acquire a buffer from the pool (Treiber stack pop).
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
    pub fn acquire(self: &Arc<Self>) -> KwaversResult<PooledBuffer> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);

            if head.is_null() {
                return Err(KwaversError::System(
                    crate::error::SystemError::ResourceUnavailable {
                        resource: "buffer pool".to_owned(),
                    },
                ));
            }

            // SAFETY: `head` is a valid pointer from our contiguous pool block.
            let next = unsafe { (*head).next };

            match self
                .free_list
                .compare_exchange(head, next, Ordering::Release, Ordering::Acquire)
            {
                Ok(_) => {
                    let new_allocated = self.allocated.fetch_add(1, Ordering::SeqCst) + 1;

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
    /// - `ptr` must not be in use (no live references).
    /// - Must not be called twice for the same `ptr`.
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

/// RAII guard for a buffer acquired from [`BufferPool`].
///
/// Automatically returns the buffer to the pool when dropped.
#[derive(Debug)]
pub struct PooledBuffer {
    ptr: *mut u8,
    pool: Arc<BufferPool>,
}

// SAFETY: PooledBuffer is Send+Sync; pool uses atomic sync, memory exclusively owned.
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
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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
        // SAFETY: ptr from BufferPool::acquire; exclusive ownership; called once.
        unsafe {
            self.pool.release(self.ptr);
        }
    }
}
