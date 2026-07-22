//! Tiled k-space processing using `LendingIterator` and `Tiles<T, N>` from leto.
//!
//! In PSTD solvers the pressure field spans hundreds of megabytes for
//! production grids (e.g. 256³). Processing it tile-by-tile keeps each working
//! set inside L2 cache and enables compiler auto-vectorization per tile.
//! This example demonstrates the leto `Tiles<f64, 3>` GAT iterator for
//! zero-copy, cache-blocked field traversal.
//!
//! ## Physics context
//!
//! A PSTD pressure-field update step computes
//!
//! ```text
//! p_new`i` = p`i` - rho*c² * dt * div(u)
//! ```
//!
//! where `div(u)` is evaluated spectrally. When the field must also be
//! inspected for diagnostics (energy, peak pressure, receiver sampling) the
//! naïve approach walks the entire array once per diagnostic. Tiling fuses
//! those passes: a single tile-loop simultaneously accumulates energy and
//! peak while keeping the active working set in L2.
//!
//! ## Theorem — energy conservation across tiles
//!
//! For a partition of an N-element field into K non-overlapping tiles:
//!
//! ```text
//! Σ_{k=1..K}  Σ_{j ∈ tile_k}  p`J`²  =  Σ_{j=0..N-1} p`J`²
//! ```
//!
//! i.e., summing squared pressures tile-by-tile is identical to a single flat
//! pass.  Tiling is a *pure* cache optimization with zero numerical effect.
//!
//! ## GAT advantage
//!
//! Standard `Iterator` cannot yield views *borrowed from the iterator itself*
//! because `Item` must outlive `&mut self`.  `LendingIterator` (GAT-based)
//! ties the item lifetime to `&'this self`, enabling the streaming loop:
//!
//! ```text
//! while let Some(tile) = tiles.next() {
//!     // tile: ArrayView<'_, f64, 3>  — zero-copy borrow from backing array
//!     process_tile(&tile);
//!     // tile dropped here; cache lines from this tile may be evicted
//! }
//! ```
//!
//! # Chapter reference
//!
//! Part VI — Atlas Stack Integration, §SIMD and Tiling.

use leto::{Array3, LendingIterator, Tiles};

// ── Grid and tile dimensions ───────────────────────────────────────────────
const NX: usize = 32;
const NY: usize = 32;
const NZ: usize = 32;
const TILE_X: usize = 8;
const TILE_Y: usize = 8;
const TILE_Z: usize = 8;

// ── Physics constants ─────────────────────────────────────────────────────
const DX: f64 = 0.5e-3; // 0.5 mm grid spacing
const SIGMA_CELLS: f64 = 4.0; // Gaussian half-width in grid cells
const AMPLITUDE: f64 = 1.0; // Peak pressure [Pa]

fn main() {
    println!("Tiled k-space processing demo (LendingIterator + Tiles from leto)");
    println!("grid: {NX}³, tile: {TILE_X}³, dx: {:.3} mm", DX * 1e3);
    println!();

    // ── Initialize Gaussian pressure pulse ────────────────────────────────
    let sigma = SIGMA_CELLS * DX;
    let cx = 0.5 * (NX as f64 - 1.0) * DX;
    let cy = 0.5 * (NY as f64 - 1.0) * DX;
    let cz = 0.5 * (NZ as f64 - 1.0) * DX;

    let mut pressure = Array3::<f64>::zeros([NX, NY, NZ]);
    for i in 0..NX {
        for j in 0..NY {
            for k in 0..NZ {
                let x = i as f64 * DX;
                let y = j as f64 * DX;
                let z = k as f64 * DX;
                let r2 = (x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2);
                pressure[[i, j, k]] = AMPLITUDE * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    // ── Reference — single flat pass ──────────────────────────────────────
    let energy_ref: f64 = pressure.iter().map(|&p| p * p).sum();
    let peak_ref: f64 = pressure.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // ── Tiled pass via LendingIterator ────────────────────────────────────
    // `view.data()` and `view.layout()` give the slice and layout without
    // copying; ownership stays with `pressure`.
    let view = pressure.view();
    let mut tiles = Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z])
        .expect("tile shape is non-zero");

    let expected_tiles = (NX / TILE_X) * (NY / TILE_Y) * (NZ / TILE_Z);
    assert_eq!(tiles.total_tiles(), expected_tiles, "tile-grid count");

    let mut energy_tiled = 0.0_f64;
    let mut peak_tiled = f64::NEG_INFINITY;
    let mut tile_count = 0usize;

    // GAT streaming loop — `tile` borrows from `tiles` for one iteration
    while let Some(tile) = tiles.next() {
        // Each tile is a zero-copy ArrayView<'_, f64, 3>
        energy_tiled += tile.iter().map(|&p| p * p).sum::<f64>();
        peak_tiled = peak_tiled.max(tile.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        tile_count += 1;
    }

    // ── Results ───────────────────────────────────────────────────────────
    println!("tile coverage");
    println!("  tiles processed: {tile_count}  (expected {expected_tiles})");
    println!();
    println!("acoustic energy [Pa²·m³/dx³]");
    println!("  reference (flat):    {energy_ref:.12e}");
    println!("  tiled accumulation:  {energy_tiled:.12e}");
    let energy_error = (energy_tiled - energy_ref).abs() / energy_ref;
    println!("  relative error:      {energy_error:.2e}");
    println!();
    println!("peak pressure [Pa]");
    println!("  reference: {peak_ref:.12e}");
    println!("  tiled:     {peak_tiled:.12e}");

    // ── Theorem verification ──────────────────────────────────────────────
    assert_eq!(tile_count, expected_tiles, "all tiles must be visited");
    assert!(
        energy_error < 1e-13,
        "tiled energy {energy_tiled:.12e} ≠ reference {energy_ref:.12e} \
         (relative error {energy_error:.2e})"
    );
    assert!(
        (peak_tiled - peak_ref).abs() < 1e-12 * AMPLITUDE,
        "tiled peak {peak_tiled:.12e} ≠ reference {peak_ref:.12e}"
    );

    println!();
    println!("Energy conservation: PASS  (error = {energy_error:.2e})");
    println!("Peak pressure:       PASS");
    println!();
    println!("LendingIterator/Tiles: zero-copy tiling, no element copied.");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiles cover every element of a small 2×2×2 array exactly once.
    #[test]
    fn tiles_cover_every_element() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let arr = Array3::from_shape_vec([2, 2, 2], data).expect("shape matches data");
        let view = arr.view();
        let mut tiles = Tiles::new(view.data(), view.layout(), [1, 1, 1]).expect("non-zero tile");

        let mut collected: Vec<f64> = Vec::new();
        while let Some(tile) = tiles.next() {
            for &v in tile.iter() {
                collected.push(v);
            }
        }

        let mut sorted = collected.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f64> = (0..8).map(|i| i as f64).collect();
        assert_eq!(sorted, expected);
    }

    /// Tiled energy equals flat energy to machine precision.
    #[test]
    fn tiled_energy_equals_flat_energy() {
        let pressure = Array3::<f64>::zeros([NX, NY, NZ]);
        // Fill with deterministic non-trivial values
        let mut p = pressure;
        for i in 0..NX {
            for j in 0..NY {
                for k in 0..NZ {
                    p[[i, j, k]] = ((i + j * NY + k) as f64).sin();
                }
            }
        }
        let flat: f64 = p.iter().map(|&v| v * v).sum();
        let view = p.view();
        let mut tiles =
            Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z]).expect("non-zero");
        let tiled: f64 = {
            let mut acc = 0.0;
            while let Some(tile) = tiles.next() {
                acc += tile.iter().map(|&v| v * v).sum::<f64>();
            }
            acc
        };
        assert!(
            (tiled - flat).abs() < 1e-12 * flat.max(1.0),
            "tiled {tiled:.12e} ≠ flat {flat:.12e}"
        );
    }

    /// `count_remaining` reports the correct number before exhaustion.
    #[test]
    fn count_remaining_is_correct() {
        let arr = Array3::<f64>::zeros([NX, NY, NZ]);
        let view = arr.view();
        let mut tiles =
            Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z]).expect("non-zero");
        let expected = (NX / TILE_X) * (NY / TILE_Y) * (NZ / TILE_Z);
        assert_eq!(tiles.count_remaining(), expected);
        assert!(tiles.next().is_none());
    }
}
