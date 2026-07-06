//! Matrix-free finite-frequency Born sensitivity operator (anatomy-neutral).
//!
//! ## Distance-table precomputation
//!
//! The Born sensitivity kernel for row `r` at column `col` is:
//!
//! ```text
//! A[r, col] = V · atten(col, path) · harmonic(path) · cos(k · path) / sqrt(ds · dr)
//! ```
//!
//! where `ds = dist(source_elem, voxel[col])` and `dr = dist(receiver_elem, voxel[col])`.
//! Naïvely each evaluation requires three `sqrt` calls.  `VolumeOperator::new` pre-fills
//! `elem_dist` and `elem_sqrt_dist` tables so the hot inner loop uses only two memory
//! reads and one product (zero sqrt per kernel evaluation after construction).
//!
//! ## Parallel reduction pattern
//!
//! All four parallel operators (`diagonal`, `migration`, `normal_residual`,
//! `apply_normal`) use Moirai's `fold_reduce_with` pattern:
//!
//! - `fold`: each Moirai worker shard processes a chunk of row indices,
//!   accumulating into a task-local partial result.  For `normal_residual` and
//!   `apply_normal` the fold state is a `(partial, row_values)` tuple so the
//!   `ncols`-sized row buffer is allocated once per shard rather than once per
//!   row.
//! - `reduce`: Moirai combines task partials, avoiding an intermediate
//!   `Vec<Vec<f64>>` collection barrier.
//!
//! `collect() + add_partials()` (the prior pattern) forced a serial O(n_tasks × ncols)
//! accumulation after a `Vec<Vec<f64>>` collection barrier.  `fold + reduce` pipelines
//! that step, lowering both peak memory (no intermediate collection Vec) and
//! the serial critical-path length.

mod construction;
mod helpers;
mod kernel;
mod operators;

pub(crate) use super::VolumeVoxel;

#[derive(Debug)]
pub struct VolumeOperator<'a> {
    active: &'a [VolumeVoxel],
    voxel_volume_m3: f64,
    row_contexts: Vec<RowContext>,
    /// Flat element-to-voxel Euclidean distances.
    ///
    /// `elem_dist[elem_idx * n_active + col]` = distance from element `elem_idx` to
    /// active voxel `col` [m]. Populated once in `new()` via Moirai chunk
    /// dispatch.
    elem_dist: Vec<f64>,
    /// Element-wise square roots of `elem_dist`.
    ///
    /// `elem_sqrt_dist[i] = sqrt(elem_dist[i])`.  Stored so the spreading denominator
    /// `sqrt(ds·dr) = sqrt(ds) × sqrt(dr)` requires no sqrt in the hot path.
    elem_sqrt_dist: Vec<f64>,
    n_active: usize,
}

/// Per-row parameters for the Born sensitivity kernel.
///
/// Stores element *indices* rather than full `ElementPosition` copies so the
/// distance-table lookup `elem_dist[source_idx * n_active + col]` uses the
/// cached value directly.
#[derive(Clone, Copy, Debug)]
struct RowContext {
    source_idx: usize,
    receiver_idx: usize,
    frequency_mhz: f64,
    harmonic_path_scale: f64,
    attenuation_model: bool,
    k: f64,
}
