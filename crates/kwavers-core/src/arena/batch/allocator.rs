use super::{BatchFieldConfig, SoAFieldBuffer, CACHE_LINE_SIZE};
use crate::error::KwaversResult;

/// Batch field allocator with pooled reuse of SoA buffers.
#[derive(Debug)]
pub struct BatchFieldAllocator {
    pools: std::collections::HashMap<(usize, usize), Vec<SoAFieldBuffer<f64>>>,
    preferred_numa: Option<u32>,
}

impl BatchFieldAllocator {
    /// Create new batch field allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
            preferred_numa: None,
        }
    }

    /// Set NUMA node preference.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.preferred_numa = Some(node);
        self
    }

    /// Allocate or retrieve from pool.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn allocate(&mut self, config: BatchFieldConfig) -> KwaversResult<SoAFieldBuffer<f64>> {
        let key = (config.field_elements, config.num_fields);
        if let Some(pool) = self.pools.get_mut(&key) {
            if let Some(buffer) = pool.pop() {
                return Ok(buffer);
            }
        }
        let mut cfg = config;
        cfg.numa_node = self.preferred_numa;
        SoAFieldBuffer::new(cfg)
    }

    /// Return buffer to pool for reuse.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn release(&mut self, buffer: SoAFieldBuffer<f64>) {
        let key = (buffer.field_elements, buffer.num_fields);
        self.pools.entry(key).or_default().push(buffer);
    }

    /// Pre-allocate buffers for common configurations.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn preallocate(
        &mut self,
        configs: &[(usize, usize)],
        count_per_config: usize,
    ) -> KwaversResult<()> {
        for &(field_elems, num_fields) in configs {
            let pool = self.pools.entry((field_elems, num_fields)).or_default();
            for _ in 0..count_per_config {
                let config = BatchFieldConfig {
                    field_elements: field_elems,
                    num_fields,
                    alignment: CACHE_LINE_SIZE,
                    numa_node: self.preferred_numa,
                };
                pool.push(SoAFieldBuffer::new(config)?);
            }
        }
        Ok(())
    }

    /// Clear all pools.
    pub fn clear_pools(&mut self) {
        self.pools.clear();
    }

    /// Total pooled buffers.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }
}

impl Default for BatchFieldAllocator {
    fn default() -> Self {
        Self::new()
    }
}
