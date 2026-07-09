use leto::{Array3, ArrayView3};

/// Borrow one field block from a stacked monolithic state without allocation.
///
/// Contract: for `0 <= i < block_rows`,
/// `view[[i, j, k]] == stacked[[block * block_rows + i, j, k]]`.
/// The returned `ArrayView3` shares storage with `stacked`; callers that only
/// need read access use this instead of materializing owned field blocks.
pub(in crate::multiphysics::monolithic) fn field_block_view(
    stacked: &Array3<f64>,
    block_rows: usize,
    block: usize,
) -> ArrayView3<'_, f64> {
    stacked.slice(s![block * block_rows..(block + 1) * block_rows, .., ..])
}
