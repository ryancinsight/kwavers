use leto::{ArrayView3, ArrayViewMut3, Layout as LetoLayout};
use moirai_parallel::{for_each_chunk_mut_with, Adaptive};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

use super::{cache_aligned_size, CACHE_LINE_SIZE};
use crate::error::{KwaversError, KwaversResult, SystemError, ValidationError};

// OPTIMIZED FIELD STORAGE - SoA IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Thread-safe, cache-line-aligned field storage using Structure-of-Arrays
///
/// Stores multiple scalar fields in contiguous, cache-aligned memory blocks.
/// Each field is stored as a separate contiguous array (SoA pattern) for
/// optimal vectorization and cache utilization.
///
/// # Memory Layout
///
/// ```text
/// Field 0: [f64, f64, f64, ...] (64-byte aligned)
/// Padding: cache line alignment
/// Field 1: [f64, f64, f64, ...] (64-byte aligned)
/// Padding: cache line alignment
/// Field 2: [f64, f64, f64, ...] (64-byte aligned)
/// ...
/// ```
///
/// # Cache Efficiency Proof
///
/// **Theorem**: SoA layout achieves optimal cache utilization for single-field operations.
///
/// **Proof**:
/// - Cache line contains $N_L = 64 / 8 = 8$ f64 elements
/// - Sequential access to field $i$: elements $[0, N_L)$ in first cache miss
/// - Subsequent elements $[N_L, 2N_L)$ already prefetch-ready
/// - Cache miss rate: $1 / N_L = 0.125$ (12.5%)
/// - Compared to AoS: accessing only field $i$ needs $1/N_{fields}$ utilization per miss
///
/// **QED**.
#[derive(Debug)]
pub struct SoAFieldStorage {
    /// Contiguous memory buffer for all fields
    memory: NonNull<u8>,

    /// Allocated layout (needed for dealloc)
    layout: Layout,

    /// Number of fields stored
    num_fields: usize,

    /// Total elements per field
    num_elements: usize,

    /// Stride between field starts (cache-aligned)
    field_stride: usize,
}

impl SoAFieldStorage {
    /// Create new SoA storage for `num_fields` each with `num_elements` f64 values
    ///
    /// # Arguments
    /// * `num_fields` - Number of scalar fields (e.g., pressure, velocity_x, velocity_y, ...)
    /// * `num_elements` - Total elements per field (nx * ny * nz)
    ///
    /// # Mathematical Specification
    ///
    /// **Preconditions**:
    /// - $N_{fields} > 0$
    /// - $N_{elements} > 0$
    /// - $N_{fields} \times \text{aligned}(N_{elements} \times S, 64) \leq \text{usize::MAX}$
    ///
    /// **Postconditions**:
    /// - $\forall i \in \[0, N_{\text{fields}}): \text{field}_i \text{ is 64-byte aligned}$
    /// - element `field_i[j]` accessible for all `j` in `0..N_elements`
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(num_fields: usize, num_elements: usize) -> KwaversResult<Self> {
        if num_fields == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "num_fields".to_owned(),
                value: 0.0,
                reason: "SoA storage requires at least one field".to_owned(),
            }));
        }

        if num_elements == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "num_elements".to_owned(),
                value: 0.0,
                reason: "SoA storage requires at least one element".to_owned(),
            }));
        }

        // Each field is cache-line aligned
        let field_size_bytes = cache_aligned_size(num_elements, std::mem::size_of::<f64>());
        let total_size = num_fields
            .checked_mul(field_size_bytes)
            .and_then(|s| s.checked_add(CACHE_LINE_SIZE)) // Extra padding
            .ok_or_else(|| {
                KwaversError::System(SystemError::MemoryAllocation {
                    requested_bytes: num_fields * field_size_bytes,
                    reason: "SoA storage size overflow".to_owned(),
                })
            })?;

        let layout = Layout::from_size_align(total_size, CACHE_LINE_SIZE).map_err(|_| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to create layout for SoA storage".to_owned(),
            })
        })?;

        // SAFETY: Layout is valid with non-zero size and power-of-2 alignment
        let memory = unsafe { alloc(layout) };
        let memory = NonNull::new(memory).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_size,
                reason: "Failed to allocate SoA storage".to_owned(),
            })
        })?;

        // Zero-initialize for cache warming (NUMA first-touch)
        // SAFETY: Memory is valid for total_size bytes
        unsafe {
            std::ptr::write_bytes(memory.as_ptr(), 0, total_size);
        }

        Ok(Self {
            memory,
            layout,
            num_fields,
            num_elements,
            field_stride: field_size_bytes,
        })
    }

    /// Get a mutable slice for field `index`
    ///
    /// # Safety
    /// Returns `&mut [f64]` with lifetime tied to `&mut self`.
    /// Caller must ensure field data is not accessed mutably and immutably at the same time.
    #[inline]
    #[must_use]
    pub fn field_mut(&mut self, index: usize) -> Option<&mut [f64]> {
        if index >= self.num_fields {
            return None;
        }

        // SAFETY:
        // 1. offset is within allocated bounds (checked at construction)
        // 2. num_elements is within field capacity
        // 3. Each field is properly aligned (enforced by construction)
        let offset = index * self.field_stride;
        let ptr = unsafe { self.memory.as_ptr().add(offset) as *mut f64 };

        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.num_elements) })
    }

    /// Get an immutable slice for field `index`
    #[inline]
    #[must_use]
    pub fn field(&self, index: usize) -> Option<&[f64]> {
        if index >= self.num_fields {
            return None;
        }

        let offset = index * self.field_stride;
        let ptr = unsafe { self.memory.as_ptr().add(offset) as *const f64 };

        Some(unsafe { std::slice::from_raw_parts(ptr, self.num_elements) })
    }

    /// Create a 3D view of field `index` with given dimensions
    #[inline]
    #[must_use]
    pub fn field_view3(
        &self,
        index: usize,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Option<ArrayView3<'_, f64>> {
        let slice = self.field(index)?;
        LetoLayout::<3>::c_contiguous([nx, ny, nz])
            .ok()
            .filter(|l| l.size() == slice.len())
            .map(|layout| ArrayView3::new(layout, slice))
    }

    /// Create a mutable 3D view of field `index` with given dimensions
    #[inline]
    pub fn field_view3_mut(
        &mut self,
        index: usize,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Option<ArrayViewMut3<'_, f64>> {
        let slice = self.field_mut(index)?;
        LetoLayout::<3>::c_contiguous([nx, ny, nz])
            .ok()
            .filter(|l| l.size() == slice.len())
            .map(|layout| ArrayViewMut3::new(layout, slice))
    }

    /// Perform sequential first-touch initialization
    ///
    /// Writes to each field to establish NUMA affinity. Sequential version
    /// for single-threaded context.
    ///
    /// For parallel first-touch, use `first_touch_field_parallel` on individual fields.
    pub fn first_touch_sequential(&mut self) {
        for field_idx in 0..self.num_fields {
            if let Some(field) = self.field_mut(field_idx) {
                field.fill(0.0);
            }
        }
    }

    /// Parallel first-touch for a specific field index
    ///
    /// This method allows parallel initialization without borrow checker issues
    /// by operating on individual fields.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn first_touch_field_parallel(&mut self, field_idx: usize) {
        if let Some(field) = self.field_mut(field_idx) {
            const CHUNK_SIZE: usize = 512;
            for_each_chunk_mut_with::<Adaptive, _, _>(field, CHUNK_SIZE, |chunk| {
                chunk.fill(0.0);
            });
        }
    }

    /// Copy data from source slices into this SoA storage
    ///
    /// # Preconditions
    /// - `sources.len() == self.num_fields`
    /// - Each `sources`i`.len() >= self.num_elements`
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn copy_from_slices(&mut self, sources: &[&[f64]]) -> KwaversResult<()> {
        if sources.len() != self.num_fields {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "sources.len()".to_owned(),
                value: sources.len() as f64,
                reason: format!(
                    "Expected {} source slices, got {}",
                    self.num_fields,
                    sources.len()
                ),
            }));
        }

        let num_elements = self.num_elements;

        for (idx, src) in sources.iter().enumerate() {
            if src.len() < num_elements {
                return Err(KwaversError::Validation(ValidationError::InvalidValue {
                    parameter: format!("sources[{}].len()", idx),
                    value: src.len() as f64,
                    reason: format!("Source slice too small, need {} elements", num_elements),
                }));
            }

            if let Some(dst) = self.field_mut(idx) {
                // SAFETY: Both slices have length >= num_elements
                dst.copy_from_slice(&src[..num_elements]);
            }
        }

        Ok(())
    }

    /// Transfer data from SoA to AoS format
    ///
    /// Outputs to `dest` where `dest`i`` contains all fields at element i
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn to_aos(&self, dest: &mut [f64]) -> KwaversResult<()> {
        let elements_aos = self.num_fields * self.num_elements;
        if dest.len() < elements_aos {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "dest.len()".to_owned(),
                value: dest.len() as f64,
                reason: format!("AoS destination too small, need {} elements", elements_aos),
            }));
        }

        // AoS layout: [f0[0], f1[0], ..., fn[0], f0[1], f1[1], ...]
        for elem_idx in 0..self.num_elements {
            for field_idx in 0..self.num_fields {
                if let Some(field) = self.field(field_idx) {
                    let aos_idx = elem_idx * self.num_fields + field_idx;
                    dest[aos_idx] = field[elem_idx];
                }
            }
        }

        Ok(())
    }

    /// Get number of stored fields
    #[inline]
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.num_fields
    }

    /// Get elements per field
    #[inline]
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Get total allocated size in bytes
    #[inline]
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for SoAFieldStorage {
    fn drop(&mut self) {
        // SAFETY: Memory was allocated with this layout, properly aligned
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}

// SAFETY: SoA storage is Send if the pointer is Send
unsafe impl Send for SoAFieldStorage {}

// SAFETY: SoA storage is Sync if accessed properly (no concurrent mutable access)
unsafe impl Sync for SoAFieldStorage {}

#[cfg(test)]
mod tests;