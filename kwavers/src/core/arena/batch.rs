//! Batch Field Allocation for Multi-Field Operations
//!
//! Provides pre-allocation strategies for simulation fields to eliminate
//! per-step allocations in hot loops and improve cache locality through
//! Structure of Arrays (SoA) layouts.
//!
//! # Mathematical Specification
//!
//! **Batch Allocation**: Given `n` fields of size `m` elements each,
//! allocate a single contiguous block `B` of size `n × m × sizeof(T)`.
//!
//! **SoA Layout**: For fields F₁, F₂, …, Fₙ:
//! ```text
//! Memory: [F₁[0..m] | F₂[0..m] | … | Fₙ[0..m]]
//! ```
//!
//! **Cache Efficiency**: Sequential access to Fᵢ achieves stride-1 pattern
//! with prefetch distance `d = cache_line_size / sizeof(T)` elements.
//!
//! # Design Principles
//!
//! - **SRP**: Single responsibility per allocation strategy.
//! - **DIP**: Abstract allocation policy, concrete implementations.
//! - **SSOT**: One canonical location for batch allocation logic.
//!
//! # References
//!
//! - Drepper U. (2007). "What Every Programmer Should Know About Memory".
//! - Calder B. et al. (1998). "Cache-Conscious Data Placement", ASPLOS.

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache line size in bytes (x86_64).
pub const CACHE_LINE_SIZE: usize = 64;

// ─── SoAFieldBuffer ───────────────────────────────────────────────────────────

/// Structure of Arrays field layout for cache-efficient access.
///
/// Stores multiple field components contiguously in memory:
/// `[field0[0..n] | field1[0..n] | field2[0..n] | …]`
///
/// ## Cache Locality
///
/// - Stride-1 access when iterating over individual fields
/// - Prefetch-friendly sequential memory layout
/// - Reduced TLB pressure from a single large allocation vs multiple small ones
#[derive(Debug)]
pub struct SoAFieldBuffer<T> {
    memory: NonNull<u8>,
    layout: Layout,
    field_elements: usize,
    num_fields: usize,
    _phantom: PhantomData<T>,
}

/// Field layout configuration for batch allocation.
#[derive(Debug, Clone, Copy)]
pub struct BatchFieldConfig {
    /// Number of elements per field.
    pub field_elements: usize,
    /// Number of fields to allocate.
    pub num_fields: usize,
    /// Alignment in bytes (default: `CACHE_LINE_SIZE`).
    pub alignment: usize,
    /// NUMA node for first-touch policy (`None` = OS default).
    pub numa_node: Option<u32>,
}

impl Default for BatchFieldConfig {
    fn default() -> Self {
        Self {
            field_elements: 0,
            num_fields: 0,
            alignment: CACHE_LINE_SIZE,
            numa_node: None,
        }
    }
}

impl BatchFieldConfig {
    /// Create configuration for a 3-D field batch.
    pub fn for_3d_fields(nx: usize, ny: usize, nz: usize, num_fields: usize) -> Self {
        Self {
            field_elements: nx * ny * nz,
            num_fields,
            alignment: CACHE_LINE_SIZE,
            numa_node: None,
        }
    }

    /// Create configuration for a 2-D field batch.
    pub fn for_2d_fields(nx: usize, ny: usize, num_fields: usize) -> Self {
        Self {
            field_elements: nx * ny,
            num_fields,
            alignment: CACHE_LINE_SIZE,
            numa_node: None,
        }
    }

    /// Set NUMA node preference.
    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }

    /// Total memory required in bytes.
    pub fn total_bytes(&self) -> usize {
        self.field_elements
            .checked_mul(self.num_fields)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .unwrap_or(usize::MAX)
    }

    /// Validate configuration.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.field_elements == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "field_elements".to_string(),
                value: 0.0,
                reason: "Field elements must be non-zero".to_string(),
            }));
        }
        if self.num_fields == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "num_fields".to_string(),
                value: 0.0,
                reason: "Number of fields must be non-zero".to_string(),
            }));
        }
        if !self.alignment.is_power_of_two() {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "alignment".to_string(),
                value: self.alignment as f64,
                reason: "Alignment must be a power of two".to_string(),
            }));
        }
        Ok(())
    }
}

impl SoAFieldBuffer<f64> {
    /// Create a new SoA field buffer.
    ///
    /// All fields are zero-initialized.  NUMA binding is applied on Linux via
    /// `mbind(2)` when `numa_node` is specified.
    pub fn new(config: BatchFieldConfig) -> KwaversResult<Self> {
        config.validate()?;

        let element_size = std::mem::size_of::<f64>();
        let total_size = config
            .field_elements
            .checked_mul(config.num_fields)
            .and_then(|n| n.checked_mul(element_size))
            .ok_or_else(|| {
                KwaversError::System(SystemError::MemoryAllocation {
                    requested_bytes: usize::MAX,
                    reason: "Size calculation overflow".to_string(),
                })
            })?;

        let layout = Layout::from_size_align(total_size, config.alignment).map_err(|_| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Invalid memory layout".to_string(),
            })
        })?;

        // SAFETY: Non-zero size and power-of-two alignment.
        let memory = unsafe { alloc(layout) };
        let memory = NonNull::new(memory).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to allocate SoA field buffer".to_string(),
            })
        })?;

        // SAFETY: Valid pointer to allocated memory.
        unsafe { std::ptr::write_bytes(memory.as_ptr(), 0, total_size) };

        // Apply NUMA binding via `super::numa::bind_memory_to_node`.
        if let Some(node) = config.numa_node {
            // SAFETY: memory is valid for total_size bytes.
            // Non-fatal: if binding fails, allocation still succeeds.
            let _ = unsafe {
                super::numa::bind_memory_to_node(memory.as_ptr(), total_size, node as usize)
            };
        }

        Ok(Self {
            memory,
            layout,
            field_elements: config.field_elements,
            num_fields: config.num_fields,
            _phantom: PhantomData,
        })
    }

    /// Get mutable slice for field `field_index`.
    ///
    /// # Panics
    ///
    /// Panics if `field_index >= num_fields`.
    #[inline]
    pub fn field_mut(&mut self, field_index: usize) -> &mut [f64] {
        assert!(
            field_index < self.num_fields,
            "field index {} out of bounds (num_fields = {})",
            field_index,
            self.num_fields
        );
        let start = field_index * self.field_elements;
        let ptr = self.memory.as_ptr() as *mut f64;
        // SAFETY: Bounds checked above; memory is valid and non-overlapping.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(start), self.field_elements) }
    }

    /// Get immutable slice for field `field_index`.
    ///
    /// # Panics
    ///
    /// Panics if `field_index >= num_fields`.
    #[inline]
    pub fn field(&self, field_index: usize) -> &[f64] {
        assert!(
            field_index < self.num_fields,
            "field index {} out of bounds",
            field_index
        );
        let start = field_index * self.field_elements;
        let ptr = self.memory.as_ptr() as *const f64;
        unsafe { std::slice::from_raw_parts(ptr.add(start), self.field_elements) }
    }

    /// Get all fields as mutable slices.
    pub fn all_fields_mut(&mut self) -> Vec<&mut [f64]> {
        let ptr = self.memory.as_ptr() as *mut f64;
        (0..self.num_fields)
            .map(|i| {
                let start = i * self.field_elements;
                unsafe { std::slice::from_raw_parts_mut(ptr.add(start), self.field_elements) }
            })
            .collect()
    }

    /// Fill all fields with `value`.
    pub fn fill(&mut self, value: f64) {
        let total = self.field_elements * self.num_fields;
        let ptr = self.memory.as_ptr() as *mut f64;
        for i in 0..total {
            unsafe { *ptr.add(i) = value };
        }
    }

    /// Number of elements per field.
    #[inline]
    pub fn field_size(&self) -> usize {
        self.field_elements
    }

    /// Number of fields.
    #[inline]
    pub fn num_fields(&self) -> usize {
        self.num_fields
    }

    /// Total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.layout.size()
    }

    /// Check whether each field start is cache-line aligned.
    pub fn is_cache_aligned(&self) -> bool {
        let field_stride = self.field_elements * std::mem::size_of::<f64>();
        field_stride.is_multiple_of(CACHE_LINE_SIZE) || self.field_elements == 1
    }
}

impl<T> Drop for SoAFieldBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: Matching pointer and layout from construction.
        unsafe { dealloc(self.memory.as_ptr(), self.layout) };
    }
}

// SoAFieldBuffer is NOT Send/Sync due to NonNull (thread-local by design).

// ─── TempBufferPool ───────────────────────────────────────────────────────────

/// Pool allocator for temporary computation buffers.
///
/// Pre-allocates buffers in three size classes to eliminate allocation in hot
/// loops.  Uses LIFO (stack) discipline for cache efficiency.
#[derive(Debug)]
pub struct TempBufferPool {
    small: Vec<Box<[f64]>>,  // ≤256 elements
    medium: Vec<Box<[f64]>>, // ≤4096 elements
    large: Vec<Box<[f64]>>,  // ≤65536 elements
    stat_allocated: AtomicUsize,
    stat_reused: AtomicUsize,
}

/// Buffer size classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSize {
    /// ~2 KB (256 × f64)
    Small,
    /// ~32 KB (4096 × f64)
    Medium,
    /// ~512 KB (65536 × f64)
    Large,
    /// Custom size (not pooled)
    Custom(usize),
}

impl BufferSize {
    /// Classify a required element count into a pool tier.
    pub fn classify(elements: usize) -> Self {
        if elements <= 256 {
            BufferSize::Small
        } else if elements <= 4096 {
            BufferSize::Medium
        } else if elements <= 65536 {
            BufferSize::Large
        } else {
            BufferSize::Custom(elements)
        }
    }

    /// Capacity in elements for this tier.
    pub fn capacity(&self) -> usize {
        match self {
            BufferSize::Small => 256,
            BufferSize::Medium => 4096,
            BufferSize::Large => 65536,
            BufferSize::Custom(n) => *n,
        }
    }
}

impl Default for TempBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl TempBufferPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            small: Vec::new(),
            medium: Vec::new(),
            large: Vec::new(),
            stat_allocated: AtomicUsize::new(0),
            stat_reused: AtomicUsize::new(0),
        }
    }

    /// Pre-allocate buffers in each size class.
    pub fn preallocate(&mut self, small_count: usize, medium_count: usize, large_count: usize) {
        for _ in 0..small_count {
            self.small
                .push(vec![0.0; BufferSize::Small.capacity()].into_boxed_slice());
        }
        for _ in 0..medium_count {
            self.medium
                .push(vec![0.0; BufferSize::Medium.capacity()].into_boxed_slice());
        }
        for _ in 0..large_count {
            self.large
                .push(vec![0.0; BufferSize::Large.capacity()].into_boxed_slice());
        }
    }

    /// Acquire a buffer with at least `min_elements` capacity.
    pub fn acquire(&mut self, min_elements: usize) -> Vec<f64> {
        let size_class = BufferSize::classify(min_elements);

        let buffer = match size_class {
            BufferSize::Small => self.small.pop(),
            BufferSize::Medium => self.medium.pop(),
            BufferSize::Large => self.large.pop(),
            BufferSize::Custom(_) => None,
        };

        match buffer {
            Some(b) => {
                self.stat_reused.fetch_add(1, Ordering::Relaxed);
                let mut vec = b.into_vec();
                vec.clear();
                vec.resize(min_elements, 0.0);
                vec
            }
            None => {
                self.stat_allocated.fetch_add(1, Ordering::Relaxed);
                vec![0.0; size_class.capacity().max(min_elements)]
            }
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, mut buffer: Vec<f64>) {
        let cap = buffer.capacity();
        buffer.clear();

        if cap == BufferSize::Large.capacity() {
            self.large.push(buffer.into_boxed_slice());
        } else if cap == BufferSize::Medium.capacity() {
            self.medium.push(buffer.into_boxed_slice());
        } else if cap == BufferSize::Small.capacity() {
            self.small.push(buffer.into_boxed_slice());
        }
        // Custom-sized buffers are dropped (not pooled).
    }

    /// Current pool sizes per tier (small, medium, large).
    pub fn pool_sizes(&self) -> (usize, usize, usize) {
        (self.small.len(), self.medium.len(), self.large.len())
    }

    /// Total new allocations (excluding pre-allocation).
    pub fn total_allocated(&self) -> usize {
        self.stat_allocated.load(Ordering::Relaxed)
    }

    /// Total reused buffers from pool.
    pub fn total_reused(&self) -> usize {
        self.stat_reused.load(Ordering::Relaxed)
    }

    /// Efficiency ratio: `reused / (reused + allocated)`.
    pub fn efficiency_ratio(&self) -> f64 {
        let reused = self.total_reused() as f64;
        let allocated = self.total_allocated() as f64;
        if reused + allocated > 0.0 {
            reused / (reused + allocated)
        } else {
            0.0
        }
    }

    /// Clear all pools.
    pub fn clear(&mut self) {
        self.small.clear();
        self.medium.clear();
        self.large.clear();
        self.stat_allocated.store(0, Ordering::Relaxed);
        self.stat_reused.store(0, Ordering::Relaxed);
    }
}

// ─── BatchFieldHandle ─────────────────────────────────────────────────────────

/// Pre-allocated field batch for wave simulation (SoA layout).
///
/// Pre-allocates:
/// - Primary fields: pressure (0), velocity_x (1), velocity_y (2), velocity_z (3)
/// - Temporary fields: 2 scratch buffers
#[derive(Debug)]
pub struct BatchFieldHandle {
    pub primary: SoAFieldBuffer<f64>,
    pub temp: SoAFieldBuffer<f64>,
    config: BatchFieldConfig,
}

impl BatchFieldHandle {
    /// Create batch allocation for wave simulation.
    pub fn for_wave_simulation(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        let primary_config = BatchFieldConfig::for_3d_fields(nx, ny, nz, 4);
        let temp_config = BatchFieldConfig::for_3d_fields(nx, ny, nz, 2);
        let primary = SoAFieldBuffer::new(primary_config)?;
        let temp = SoAFieldBuffer::new(temp_config)?;
        Ok(Self {
            primary,
            temp,
            config: primary_config,
        })
    }

    /// Get pressure field (index 0).
    #[inline]
    pub fn pressure(&self) -> &[f64] {
        self.primary.field(0)
    }

    /// Get mutable pressure field.
    #[inline]
    pub fn pressure_mut(&mut self) -> &mut [f64] {
        self.primary.field_mut(0)
    }

    /// Get velocity components (indices 1, 2, 3).
    pub fn velocity(&self) -> (&[f64], &[f64], &[f64]) {
        (
            self.primary.field(1),
            self.primary.field(2),
            self.primary.field(3),
        )
    }

    /// Get mutable velocity components.
    pub fn velocity_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [f64]) {
        // SAFETY: Non-overlapping field indices guarantee no aliasing.
        let config = self.config;
        let ptr = self.primary.memory.as_ptr() as *mut f64;
        unsafe {
            let vx = std::slice::from_raw_parts_mut(
                ptr.add(config.field_elements),
                config.field_elements,
            );
            let vy = std::slice::from_raw_parts_mut(
                ptr.add(2 * config.field_elements),
                config.field_elements,
            );
            let vz = std::slice::from_raw_parts_mut(
                ptr.add(3 * config.field_elements),
                config.field_elements,
            );
            (vx, vy, vz)
        }
    }

    /// Get mutable temporary buffer 0.
    #[inline]
    pub fn temp_0_mut(&mut self) -> &mut [f64] {
        self.temp.field_mut(0)
    }

    /// Get mutable temporary buffer 1.
    #[inline]
    pub fn temp_1_mut(&mut self) -> &mut [f64] {
        self.temp.field_mut(1)
    }

    /// Total memory usage in bytes.
    pub fn total_memory(&self) -> usize {
        self.primary.memory_usage() + self.temp.memory_usage()
    }

    /// Zero all fields.
    pub fn clear(&mut self) {
        self.primary.fill(0.0);
        self.temp.fill(0.0);
    }

    /// Whether fields are cache-line aligned.
    pub fn is_cache_efficient(&self) -> bool {
        self.primary.is_cache_aligned() && self.temp.is_cache_aligned()
    }
}

// ─── BatchFieldAllocator ──────────────────────────────────────────────────────

/// Batch field allocator with pooled reuse of SoA buffers.
#[derive(Debug)]
pub struct BatchFieldAllocator {
    pools: std::collections::HashMap<(usize, usize), Vec<SoAFieldBuffer<f64>>>,
    preferred_numa: Option<u32>,
}

impl BatchFieldAllocator {
    /// Create new batch field allocator.
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
            preferred_numa: None,
        }
    }

    /// Set NUMA node preference.
    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.preferred_numa = Some(node);
        self
    }

    /// Allocate or retrieve from pool.
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
    pub fn release(&mut self, buffer: SoAFieldBuffer<f64>) {
        let key = (buffer.field_elements, buffer.num_fields);
        self.pools.entry(key).or_default().push(buffer);
    }

    /// Pre-allocate buffers for common configurations.
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
    pub fn pool_size(&self) -> usize {
        self.pools.values().map(|v| v.len()).sum()
    }
}

impl Default for BatchFieldAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_validation() {
        let valid = BatchFieldConfig::for_3d_fields(10, 10, 10, 4);
        assert!(valid.validate().is_ok());

        let invalid_zero_elems = BatchFieldConfig {
            field_elements: 0,
            num_fields: 4,
            alignment: 64,
            numa_node: None,
        };
        assert!(invalid_zero_elems.validate().is_err());

        let invalid_alignment = BatchFieldConfig {
            field_elements: 100,
            num_fields: 4,
            alignment: 63,
            numa_node: None,
        };
        assert!(invalid_alignment.validate().is_err());
    }

    #[test]
    fn test_soa_buffer_allocation() {
        let config = BatchFieldConfig::for_3d_fields(10, 10, 10, 4);
        let mut buffer = SoAFieldBuffer::new(config).unwrap();

        assert_eq!(buffer.field_size(), 1000);
        assert_eq!(buffer.num_fields(), 4);
        assert!(buffer.is_cache_aligned());

        let field0 = buffer.field_mut(0);
        assert_eq!(field0.len(), 1000);
        field0[0] = 1.0;

        // Other fields must be independent.
        assert_eq!(buffer.field(1)[0], 0.0);
    }

    #[test]
    fn test_temp_buffer_pool() {
        let mut pool = TempBufferPool::new();
        pool.preallocate(2, 2, 2);

        let buffer = pool.acquire(100);
        assert_eq!(buffer.len(), 100);
        assert!(buffer.capacity() >= 256);

        pool.release(buffer);
        assert_eq!(pool.total_allocated(), 0);

        let _ = pool.acquire(65537); // Custom size → new allocation
        assert_eq!(pool.total_allocated(), 1);
    }

    #[test]
    fn test_batch_field_handle_wave_sim() {
        let mut handle = BatchFieldHandle::for_wave_simulation(8, 8, 8).unwrap();

        assert_eq!(handle.pressure().len(), 512);
        let (vx, vy, vz) = handle.velocity();
        assert_eq!(vx.len(), 512);
        assert_eq!(vy.len(), 512);
        assert_eq!(vz.len(), 512);

        handle.clear();
        assert_eq!(handle.pressure()[0], 0.0);
        assert!(handle.is_cache_efficient());
    }

    #[test]
    fn test_buffer_size_classification() {
        assert_eq!(BufferSize::classify(100), BufferSize::Small);
        assert_eq!(BufferSize::classify(1000), BufferSize::Medium);
        assert_eq!(BufferSize::classify(10000), BufferSize::Large);
        assert_eq!(BufferSize::classify(100000), BufferSize::Custom(100000));
    }
}
