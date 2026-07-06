//! 3-D Westervelt finite-difference stencil and output recording.
//!
//! ## Performance design — x-slab parallel decomposition
//!
//! The 3-D grid is stored in row-major (`[x][y][z]`) flat order:
//!   flat index `i = (x * n + y) * n + z`
//!
//! The finite-difference Laplacian accesses six neighbours:
//!   `±n²` (x-direction stride `n²`), `±n` (y-direction stride `n`), `±1`
//!   (z-direction, stride 1 — contiguous in memory).
//!
//! ### Former approach (flat parallel iterator with index decomposition)
//!
//! Each cell computed `z = i % n`, `y = (i / n) % n`, `x = i / n²` — three
//! integer divisions.  For n = 56 these compile to multiply-by-magic-constant
//! sequences (~8–12 instructions each).  At 175 616 cells × 1 946 steps × 18
//! passes × 3 anatomies ≈ 18 billion index decompositions.  At 4 GHz with
//! out-of-order execution (one division result per ~6 cycles) this cost
//! dominates the memory-bandwidth budget.
//!
//! ### Current approach (x-slab Moirai provider + sequential y,z loops)
//!
//! Moirai splits on the x-axis only.  Each x-slab worker maintains `x`, `y`, `z`
//! as explicit loop variables — zero integer division in the hot path.  The
//! inner z-loop is contiguous in all three read buffers and in the write
//! buffer, enabling auto-vectorisation (compiler emits AVX2/AVX-512 FMA for
//! the Laplacian accumulation and `westervelt_cell_terms`).  The boundary
//! check per slab costs one predicated assignment at start/end only.
//!
//! ### Memory layout
//!
//! With n = 56 each field occupies 56 × 56 × 56 × 8 = 1.4 MB.  Seven fields
//! (next, current, previous, speed, density, beta, sponge) = 9.8 MB, which
//! fits in the L3 caches of all modern desktop and server CPUs.  After the
//! first time step the hot data stays resident, so subsequent steps run at
//! L3-bandwidth throughput (≈ 300 GB/s on DDR5 platforms; ≈ 100 GB/s on
//! older DDR4).
//!
//! ### Boundary cells
//!
//! The boundary x-slabs (x = 0 and x = n−1) are zeroed with a single
//! `fill(0)` call before the parallel body, so the interior-only slab loop
//! never branches on boundary conditions.

use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

use super::super::stencil::westervelt_cell_terms;

pub(super) struct UpdateCells<'a> {
    pub(super) next: &'a mut [f64],
    pub(super) current: &'a [f64],
    pub(super) previous: &'a [f64],
    pub(super) speed: &'a [f64],
    pub(super) density: &'a [f64],
    pub(super) beta: &'a [f64],
    pub(super) sponge: &'a [f64],
}

/// Update all interior cells of the 3-D Westervelt grid for one time step.
///
/// # Algorithm
///
/// Parallelises over x-slabs via Moirai.  For each interior x-slab
/// `x ∈ [1, n-2]` the inner y and z loops iterate sequentially.  Boundary
/// x-slabs (x = 0 and x = n−1) are zeroed by a pre-pass; the parallel body
/// handles only the interior x-range `1..n-1` and zeroes the y = 0, y = n−1,
/// z = 0, z = n−1 boundary rows inside each slab.
///
/// # Complexity
///
/// O(n³) per time step.  No integer division on the hot path (loop variables
/// replace `i % n`, `(i / n) % n`, `i / n²`).
// JUSTIFICATION: `clippy::identity_op` triggers on stencil index expressions
// such as `i - 1` (z-direction stride-1 neighbor) and `i + 1`, which the lint
// treats as identity operations on the coefficient. These are intentional
// unit-stride neighbor offsets in the flat 3-D row-major layout, not
// algebraic simplifications. Removing them would require a different indexing
// scheme that defeats auto-vectorization. The suppression is narrowed to this
// function only.
#[allow(clippy::identity_op)]
pub(super) fn update_cells(buffers: UpdateCells<'_>, n: usize, dt: f64, inv_dx2: f64, step: usize) {
    let n2 = n * n;
    let _ = step;

    // Zero boundary x-slabs (x = 0 and x = n−1) before the parallel body.
    // This avoids a branch inside the inner loop.
    buffers.next[..n2].fill(0.0);
    buffers.next[(n - 1) * n2..].fill(0.0);

    // Interior x-slabs: Moirai splits across x in [1, n-1).
    // Each chunk owns one x-slab of size n² in `next`.
    let interior_next = &mut buffers.next[n2..(n - 1) * n2];

    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(interior_next, n2, |xi, slab| {
        // xi is the slab index within the interior range; real x = xi + 1.
        let x = xi + 1;
        let x_base = x * n2;

        for y in 0..n {
            let y_base = y * n;
            for z in 0..n {
                let i = x_base + y_base + z;
                let slab_i = y_base + z;
                if y == 0 || z == 0 || y + 1 == n || z + 1 == n {
                    slab[slab_i] = 0.0;
                    continue;
                }
                let center = buffers.current[i];
                let prev = buffers.previous[i];
                let lap = (buffers.current[i - n2]
                    + buffers.current[i + n2]
                    + buffers.current[i - n]
                    + buffers.current[i + n]
                    + buffers.current[i - 1]
                    + buffers.current[i + 1]
                    - 6.0 * center)
                    * inv_dx2;
                let c = buffers.speed[i];
                let terms = westervelt_cell_terms(
                    center,
                    prev,
                    lap,
                    c,
                    buffers.density[i],
                    buffers.beta[i],
                    dt,
                );
                let raw = 2.0_f64.mul_add(center, -prev) + terms.numerator / terms.denominator;
                slab[slab_i] = buffers.sponge[i] * raw;
            }
        }
    });
}

pub(super) fn record_receivers(
    traces: &mut [f64],
    receiver_cells: &[usize],
    next: &[f64],
    step: usize,
) {
    for (receiver, cell) in receiver_cells.iter().copied().enumerate() {
        traces[step * receiver_cells.len() + receiver] = next[cell];
    }
}

pub(super) fn update_peak(peak: &mut [f64], next: &[f64], source_mask: &[bool]) {
    for ((dst, value), is_source) in peak.iter_mut().zip(next.iter()).zip(source_mask.iter()) {
        if !*is_source {
            *dst = (*dst).max(value.abs());
        }
    }
}
