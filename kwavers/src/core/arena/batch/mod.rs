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
//! # References
//!
//! - Drepper U. (2007). "What Every Programmer Should Know About Memory".
//! - Calder B. et al. (1998). "Cache-Conscious Data Placement", ASPLOS.

mod allocator;
mod pool;
#[cfg(test)]
mod tests;

pub use allocator::BatchFieldAllocator;
pub use pool::{BufferSize, TempBufferPool};

use crate::core::error::{KwaversError, KwaversResult, SystemError, ValidationError};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

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
    pub(super) memory: NonNull<u8>,
    pub(super) layout: Layout,
    pub(super) field_elements: usize,
    pub(super) num_fields: usize,
    pub(super) _phantom: PhantomData<T>,
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
