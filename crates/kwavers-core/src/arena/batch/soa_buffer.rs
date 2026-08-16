use super::config::BatchFieldConfig;
use crate::error::{KwaversError, KwaversResult, SystemError};
use mnemosyne_arena::{allocate_large_or_huge, deallocate_large_or_huge};
use mnemosyne_backend::MemoryBackendWrapper;
use mnemosyne_core::constants::SEGMENT_ALIGN;
use mnemosyne_core::types::Segment;
use std::marker::PhantomData;
use std::ptr::NonNull;
use themis::NumaNodeId;

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
    pub(super) field_elements: usize,
    pub(super) num_fields: usize,
    pub(super) allocated_size: usize,
    pub(super) segment_ptr: *mut Segment,
    pub(super) _phantom: PhantomData<T>,
}

impl SoAFieldBuffer<f64> {
    /// Create a new SoA field buffer.
    ///
    /// All fields are zero-initialized. NUMA binding is applied on Linux via
    /// `mbind(2)` when `numa_node` is specified.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
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
                    reason: "Size calculation overflow".to_owned(),
                })
            })?;

        if total_size == 0 || !config.alignment.is_power_of_two() {
            return Err(KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Invalid memory layout".to_owned(),
            }));
        }

        // SAFETY: non-zero size, power-of-two alignment (validated above).
        // mnemosyne allocates at least `SEGMENT_ALIGN` alignment.
        let user_ptr = unsafe {
            allocate_large_or_huge::<MemoryBackendWrapper>(total_size, SEGMENT_ALIGN, false)
        };
        let memory = NonNull::new(user_ptr).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to allocate SoA field buffer".to_owned(),
            })
        })?;

        // SAFETY: Valid pointer to allocated memory.
        // Segment pointer is stored in metadata slot by allocate_large_or_huge.
        let segment_ptr = unsafe { *((memory.as_ptr() as *mut *mut Segment).sub(1)) };
        unsafe { std::ptr::write_bytes(memory.as_ptr(), 0, total_size) };

        // Apply NUMA binding via `mnemosyne_heap::numa::bind_to_node`.
        if let Some(node) = config.numa_node {
            // SAFETY: memory is valid for total_size bytes.
            // Non-fatal: if binding fails, allocation still succeeds.
            let _ = unsafe {
                mnemosyne_heap::numa::bind_to_node(
                    memory.as_ptr(),
                    total_size,
                    NumaNodeId::new(node),
                )
            };
        }

        // Round up to alignment boundary for actual allocation size.
        let allocated_size = (total_size + SEGMENT_ALIGN - 1) & !(SEGMENT_ALIGN - 1);

        Ok(Self {
            memory,
            field_elements: config.field_elements,
            num_fields: config.num_fields,
            allocated_size,
            segment_ptr,
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
    #[must_use]
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
    #[must_use]
    pub fn field_size(&self) -> usize {
        self.field_elements
    }

    /// Number of fields.
    #[inline]
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.num_fields
    }

    /// Total memory usage in bytes.
    /// Reports the actual allocated size (rounded up to alignment).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.allocated_size
    }

    /// Check whether each field start is cache-line aligned.
    #[must_use]
    pub fn is_cache_aligned(&self) -> bool {
        let field_stride = self.field_elements * std::mem::size_of::<f64>();
        field_stride.is_multiple_of(super::CACHE_LINE_SIZE) || self.field_elements == 1
    }
}

impl<T> Drop for SoAFieldBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: segment_ptr was written by allocate_large_or_huge and is valid.
        // Recovered via the standard metadata-slot pattern.
        let _released = unsafe {
            deallocate_large_or_huge::<MemoryBackendWrapper>(self.memory.as_ptr(), self.segment_ptr)
        };
    }
}

// SoAFieldBuffer is NOT Send/Sync due to NonNull (thread-local by design).
