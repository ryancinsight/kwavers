use super::align_up;

// LAYOUT OPTIMIZATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Reorder struct fields to minimize padding (struct packing)
///
/// Sorts fields by descending alignment requirement to eliminate
/// internal padding in structures.
///
/// # Mathematical Proof
///
/// **Theorem**: Sorting fields by descending alignment minimizes struct size.
///
/// **Proof**:
/// - Field with alignment $A_i$ requires $\text{offset} \equiv 0 \pmod{A_i}$
/// - Padding = aligned_offset - current_offset
/// - To minimize padding, place high-alignment fields first where
///   they align naturally without padding
/// - Inductive: assume optimal for $n-1$ fields, $n$th field adds minimal
///   padding when placed at start (largest alignment constraint first)
///
/// **QED**.
pub fn optimal_field_order<T: FieldLayoutInfo>(fields: &[T]) -> Vec<usize> {
    let mut indices: Vec<(usize, usize)> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| (i, f.alignment()))
        .collect();

    // Sort by alignment descending
    indices.sort_by_key(|b| std::cmp::Reverse(b.1));

    indices.into_iter().map(|(i, _)| i).collect()
}

/// Trait for types that provide layout information
pub trait FieldLayoutInfo {
    /// Alignment requirement in bytes
    fn alignment(&self) -> usize;

    /// Size in bytes
    fn size(&self) -> usize;
}

/// Calculate struct size with optimal field ordering
///
/// Returns minimum possible size by reordering fields.
/// # Panics
/// - Panics if an internal precondition is violated.
///
#[must_use] 
pub fn packed_struct_size(field_sizes: &[usize], field_alignments: &[usize]) -> usize {
    assert_eq!(field_sizes.len(), field_alignments.len());

    let mut field_info: Vec<(usize, usize)> = field_sizes
        .iter()
        .zip(field_alignments.iter())
        .map(|(&s, &a)| (s, a))
        .collect();

    // Sort by alignment descending
    field_info.sort_by_key(|b| std::cmp::Reverse(b.1));

    let mut offset = 0;
    let mut max_align = 1;

    for (size, align) in field_info {
        // Align to field's requirement
        offset = align_up(offset, align);
        offset += size;
        max_align = max_align.max(align);
    }

    // Final struct alignment
    align_up(offset, max_align)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_struct_size() {
        // Example: struct { u8, u64, u32 }
        // Naive: 1 + 7pad + 8 + 4 + 4pad = 24
        // Optimal (descending align): u64(8,8), u32(4,4), u8(1,1)
        // = 0:u64(8), 8:u32(4), 12:u8(1), 13->16: pad = 16 bytes
        let sizes = vec![1, 8, 4];
        let aligns = vec![1, 8, 4];
        let packed = packed_struct_size(&sizes, &aligns);

        assert_eq!(packed, 16);
    }
}
