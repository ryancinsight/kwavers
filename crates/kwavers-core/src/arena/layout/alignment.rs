use super::CACHE_LINE_SIZE;

// MEMORY ALIGNMENT UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Align size up to the nearest multiple of alignment (power of 2)
///
/// # Mathematical Specification
///
/// For alignment $A = 2^k$:
/// $$ \text{aligned}(n, A) = \lceil n / A \rceil \times A = (n + A - 1) \land \lnot(A - 1) $$
///
/// # Safety
/// - Requires $A$ to be a power of 2
/// - Requires $n + A - 1$ to not overflow
#[inline]
#[must_use]
pub const fn align_up(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "alignment must be power of 2");
    (size + align - 1) & !(align - 1)
}

/// Align pointer to cache line boundary
///
/// # Safety
/// Caller must ensure pointer is valid and within bounds for realignment
#[inline]
#[must_use]
pub fn align_to_cache_line<T>(ptr: *mut T) -> *mut T {
    let align = CACHE_LINE_SIZE;
    let addr = ptr as usize;
    let aligned_addr = align_up(addr, align);
    aligned_addr as *mut T
}

/// Size in bytes aligned to cache line
#[inline]
#[must_use]
pub const fn cache_aligned_size(n_elements: usize, element_size: usize) -> usize {
    let raw_size = n_elements * element_size;
    align_up(raw_size, CACHE_LINE_SIZE)
}

/// Compute padding bytes needed for alignment
#[inline]
#[must_use]
pub const fn padding_needed(size: usize, align: usize) -> usize {
    align_up(size, align) - size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);

        assert_eq!(align_up(3, 4), 4);
        assert_eq!(align_up(4, 4), 4);
        assert_eq!(align_up(8, 8), 8);
    }

    #[test]
    fn test_cache_aligned_size() {
        // 1 element = 8 bytes -> align to 64 = 64
        assert_eq!(cache_aligned_size(1, 8), 64);

        // 8 elements = 64 bytes -> already aligned
        assert_eq!(cache_aligned_size(8, 8), 64);

        // 9 elements = 72 bytes -> align to 128
        assert_eq!(cache_aligned_size(9, 8), 128);

        // 1000 elements = 8000 bytes -> already 64-aligned
        let size_1000 = cache_aligned_size(1000, 8);
        assert_eq!(size_1000, 8000);
    }
}
