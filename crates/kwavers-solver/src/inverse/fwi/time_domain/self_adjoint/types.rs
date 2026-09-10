//! The engine's inputs: time stepping, acquisition geometry, and the
//! per-axis spacings its stencils are built from.

use kwavers_grid::Grid;
use leto::ArrayView2;

/// Time-stepping parameters for the self-adjoint engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelfAdjointConfig {
    /// Number of time samples (`N`).
    pub nt: usize,
    /// Time step \[s\].
    pub dt: f64,
}

/// Acquisition geometry: source voxels + per-source signal, receiver voxels.
///
/// `source_signal` is `(n_rows, nt)` with `n_rows == (source_voxels.len())` for a
/// per-voxel signal, or `n_rows == 1` to broadcast one signal to every source
/// voxel. Receiver traces are returned/consumed in `receiver_voxels` order.
#[derive(Clone, Copy)]
pub(crate) struct Acquisition<'a> {
    pub source_voxels: &'a [(usize, usize, usize)],
    pub source_signal: ArrayView2<'a, f64>,
    pub receiver_voxels: &'a [(usize, usize, usize)],
}

/// Per-axis inverse-square spacings; a degenerate axis (`n == 1`) contributes 0.
pub(super) struct Spacing {
    pub(super) inv_dx2: f64,
    pub(super) inv_dy2: f64,
    pub(super) inv_dz2: f64,
}

impl Spacing {
    pub(super) fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = grid.dimensions();
        Self {
            inv_dx2: if nx > 1 {
                1.0 / (grid.dx * grid.dx)
            } else {
                0.0
            },
            inv_dy2: if ny > 1 {
                1.0 / (grid.dy * grid.dy)
            } else {
                0.0
            },
            inv_dz2: if nz > 1 {
                1.0 / (grid.dz * grid.dz)
            } else {
                0.0
            },
        }
    }
}
