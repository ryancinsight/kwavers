use super::{
    pool_impl::{BufferPool, PooledBuffer},
    PoolConfig, PoolStats,
};
use crate::core::error::{KwaversError, KwaversResult};
use std::sync::Arc;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    /// Acquire `count` buffers from the pool.
    ///
    /// Either all `count` buffers are acquired, or none are.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn acquire(pool: &Arc<BufferPool>, count: usize) -> KwaversResult<Self> {
        let mut buffers = Vec::with_capacity(count);

        for _ in 0..count {
            match pool.acquire() {
                Ok(buffer) => buffers.push(buffer),
                Err(e) => {
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
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn get(&self, index: usize) -> &[u8] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_bytes()
    }

    /// Access buffer at index as mutable byte slice.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> &mut [u8] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_bytes_mut()
    }

    /// Access buffer as typed slice.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn get_typed<T>(&self, index: usize) -> &[T] {
        assert!(index < self.buffers.len(), "buffer index out of bounds");
        self.buffers[index].as_typed::<T>()
    }

    /// Access buffer as mutable typed slice.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
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

/// Per-NUMA-node pool manager.
///
/// Creates separate pools for each NUMA node to ensure local memory access.
#[derive(Debug)]
pub struct NumaPoolManager {
    pools: Vec<Option<Arc<BufferPool>>>,
}

impl NumaPoolManager {
    /// Create pool manager with one pool per detected NUMA node.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(config: PoolConfig) -> KwaversResult<Self> {
        let topology = super::super::numa::NumaTopology::detect();
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

    /// Acquire buffer from pool on specified NUMA node, with fallback.
    /// # Errors
    /// - Returns [`KwaversError::System`] if the precondition for a System-class constraint is violated.
    ///
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
                resource: "NUMA pool".to_owned(),
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
