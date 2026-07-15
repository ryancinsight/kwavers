use leto::{ArrayView3, ArrayViewMut3, Layout as LetoLayout};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{cache_aligned_size, CACHE_LINE_SIZE};
use crate::error::{KwaversError, KwaversResult, SystemError, ValidationError};

// BATCH FIELD ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════

/// Pool allocator for temporary computation fields
///
/// Pre-allocates multiple fields and reuses them, eliminating
/// per-step allocations in hot loops.
#[derive(Debug)]
pub struct FieldPool {
    /// Contiguous memory for all pooled fields
    memory: NonNull<u8>,

    /// Allocation layout
    layout: Layout,

    /// Field capacity (max concurrent allocations)
    capacity: usize,

    /// Currently available field count
    available: AtomicUsize,

    /// Elements per field
    elements_per_field: usize,

    /// Stride between fields (cache-aligned)
    field_stride: usize,

    /// Bitmap of available slots (true = available)
    slot_bitmap: Vec<std::sync::atomic::AtomicBool>,
}

impl FieldPool {
    /// Create a new field pool with given capacity
    ///
    /// # Arguments
    /// * `capacity` - Number of fields to pre-allocate
    /// * `elements_per_field` - Size of each field in elements
    ///
    /// # Mathematical Specification
    ///
    /// **Precondition**: $\text{capacity} > 0 \land \text{elements} > 0$
    /// **Postcondition**: $\forall i \in [0, \text{capacity}): \text{slot}_i \text{ is available}$
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(capacity: usize, elements_per_field: usize) -> KwaversResult<Self> {
        if capacity == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "capacity".to_owned(),
                value: 0.0,
                reason: "Field pool requires positive capacity".to_owned(),
            }));
        }

        if elements_per_field == 0 {
            return Err(KwaversError::Validation(ValidationError::InvalidValue {
                parameter: "elements_per_field".to_owned(),
                value: 0.0,
                reason: "Field pool requires positive field size".to_owned(),
            }));
        }

        let field_stride = cache_aligned_size(elements_per_field, std::mem::size_of::<f64>());
        let total_bytes = capacity.checked_mul(field_stride).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: capacity * field_stride,
                reason: "Field pool size overflow".to_owned(),
            })
        })?;

        let layout = Layout::from_size_align(total_bytes, CACHE_LINE_SIZE).map_err(|_| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_bytes,
                reason: "Invalid layout for field pool".to_owned(),
            })
        })?;

        let memory = unsafe { alloc(layout) };
        let memory = NonNull::new(memory).ok_or_else(|| {
            KwaversError::System(SystemError::MemoryAllocation {
                requested_bytes: total_bytes,
                reason: "Failed to allocate field pool".to_owned(),
            })
        })?;

        // Zero-initialize and first-touch
        unsafe {
            std::ptr::write_bytes(memory.as_ptr(), 0, total_bytes);
        }

        // Initialize all slots as available
        let mut slot_bitmap = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slot_bitmap.push(std::sync::atomic::AtomicBool::new(true));
        }

        Ok(Self {
            memory,
            layout,
            capacity,
            available: AtomicUsize::new(capacity),
            elements_per_field,
            field_stride,
            slot_bitmap,
        })
    }

    /// Acquire a field buffer from the pool
    ///
    /// Returns `None` if pool is exhausted.
    /// On success, returns mutable slice to the field buffer.
    ///
    /// # Performance
    /// - Time: $O(\text{capacity})$ worst case (scan for free slot)
    /// - Typical: $O(1)$ when slots available near current allocation point
    pub fn acquire(&self) -> Option<FieldBufferGuard<'_>> {
        let current = self.available.load(Ordering::Relaxed);
        if current == 0 {
            return None;
        }

        // Linear scan for available slot (cache-friendly, typically finds quickly)
        for slot_idx in 0..self.capacity {
            let slot_state = &self.slot_bitmap[slot_idx];

            // Try to acquire this slot (CAS: true -> false)
            if slot_state
                .compare_exchange(true, false, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // Acquired slot
                self.available.fetch_sub(1, Ordering::Release);

                // SAFETY: Slot offset is within bounds, properly aligned
                let offset = slot_idx * self.field_stride;
                let ptr = unsafe { self.memory.as_ptr().add(offset) as *mut f64 };
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.elements_per_field) };

                return Some(FieldBufferGuard {
                    data: slice,
                    pool: self,
                    slot_idx,
                });
            }
        }

        // No slots available (race condition, decrement already happened)
        None
    }

    /// Return a buffer to the pool.
    /// Called automatically by [`FieldBufferGuard`]`::drop`.
    fn release(&self, slot_idx: usize) {
        // Mark slot as available
        self.slot_bitmap[slot_idx].store(true, Ordering::Release);
        self.available.fetch_add(1, Ordering::Release);
    }

    /// Get current available slot count
    #[inline]
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.available.load(Ordering::Relaxed)
    }

    /// Get total capacity
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all buffers (reset to zeros)
    ///
    /// # Performance
    /// - Time: $O(\text{capacity} \times \text{elements})$
    pub fn clear_all(&mut self) {
        // Sequential clear (parallel would require careful synchronization)
        for slot_idx in 0..self.capacity {
            let offset = slot_idx * self.field_stride;
            let ptr = unsafe { self.memory.as_ptr().add(offset) as *mut f64 };
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.elements_per_field) };
            slice.fill(0.0);
        }
    }
}

impl Drop for FieldPool {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.memory.as_ptr(), self.layout);
        }
    }
}

// SAFETY: FieldPool can be Send as long as memory isn't shared across threads unsafely
unsafe impl Send for FieldPool {}

/// RAII guard for pooled field buffer
#[derive(Debug)]
pub struct FieldBufferGuard<'a> {
    /// Data slice (valid until guard dropped)
    data: &'a mut [f64],
    /// Back-reference to pool for release
    pool: &'a FieldPool,
    /// Index of this slot
    slot_idx: usize,
}

impl<'a> FieldBufferGuard<'a> {
    /// Get mutable slice to field data
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }

    /// Get immutable slice to field data
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    /// Convert to 3D view
    #[inline]
    #[must_use]
    pub fn as_view3(&self, nx: usize, ny: usize, nz: usize) -> Option<ArrayView3<'_, f64>> {
        LetoLayout::<3>::c_contiguous([nx, ny, nz])
            .ok()
            .filter(|l| l.size() == self.data.len())
            .map(|layout| ArrayView3::new(layout, self.data))
    }

    /// Convert to mutable 3D view
    #[inline]
    pub fn as_view3_mut(
        &mut self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Option<ArrayViewMut3<'_, f64>> {
        LetoLayout::<3>::c_contiguous([nx, ny, nz])
            .ok()
            .filter(|l| l.size() == self.data.len())
            .map(|layout| ArrayViewMut3::new(layout, self.data))
    }
}

impl<'a> Drop for FieldBufferGuard<'a> {
    fn drop(&mut self) {
        self.pool.release(self.slot_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_pool() {
        let pool = FieldPool::new(4, 100).expect("pool must create");

        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available_count(), 4);

        // Acquire buffers
        let buf1 = pool.acquire().expect("acquire 1");
        let buf2 = pool.acquire().expect("acquire 2");
        assert_eq!(pool.available_count(), 2);

        // Use buffers
        let mut guard1 = buf1;
        guard1.as_mut_slice()[0] = 42.0;

        let mut guard2 = buf2;
        guard2.as_mut_slice()[99] = 99.0;

        // Release by dropping
        drop(guard1);
        assert_eq!(pool.available_count(), 3);

        // Re-acquire should reuse
        let buf3 = pool.acquire().expect("re-acquire");
        assert_eq!(pool.available_count(), 2);

        let _ = buf3;

        // Drop remaining
        drop(guard2);
        drop(buf3);
        assert_eq!(pool.available_count(), 4);
    }
}
