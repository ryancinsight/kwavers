//! Utility functions shared by [`super::VolumeOperator`] submodules.

/// Number of rows per Rayon chunk for a row-parallel operation.
///
/// Targets `4 × num_threads` chunks so each thread handles four chunks,
/// balancing load imbalance against task-spawn overhead.
pub(super) fn row_chunk_len(row_count: usize) -> usize {
    let target_chunks = rayon::current_num_threads().max(1) * 4;
    row_count.div_ceil(target_chunks).max(1)
}

/// Euclidean distance between two 3-D points [m].
pub(super) fn distance(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}
