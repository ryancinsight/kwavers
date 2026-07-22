# Example: Tiled K-Space Processing

**Source**: `crates/kwavers/examples/tiled_kspace_processing.rs`

## Overview

Demonstrates `leto::Tiles<f64, 3>` (a GAT-based streaming iterator) for
zero-copy, cache-blocked processing of a 3-D PSTD pressure field.

## Physics motivation

PSTD pressure fields for production grids (e.g. 256³) do not fit in L2 cache.
Processing the field tile-by-tile keeps each working set inside cache and
enables compiler auto-vectorization per tile.

The example initializes a Gaussian pressure pulse on a 32³ grid, then walks
it in 8³ tiles using `LendingIterator`, computing per-tile acoustic energy.

## Theorem — energy conservation across tiles

For a partition of an N-element field into K non-overlapping tiles:

```text
Σ_{k=1..K}  Σ_{j ∈ tile_k}  p[j]²  =  Σ_{j=0..N-1} p[j]²
```

Tiling is a *pure* cache optimization — no element is copied, no energy is
lost or gained.  The example asserts this identity to machine precision.

## Running

```bash
cargo run --example tiled_kspace_processing
```

## Expected output

```
Tiled k-space processing demo (LendingIterator + Tiles from leto)
grid: 32³, tile: 8³, dx: 0.500 mm

tile coverage
  tiles processed: 64  (expected 64)

acoustic energy [Pa²·m³/dx³]
  reference (flat):    1.178097...e+00
  tiled accumulation:  1.178097...e+00
  relative error:      0.00e+00

peak pressure [Pa]
  reference: 1.000000...e+00
  tiled:     1.000000...e+00

Energy conservation: PASS  (error = 0.00e+00)
Peak pressure:       PASS

LendingIterator/Tiles: zero-copy tiling, no element copied.
```

## Key API

```rust
use leto::{Array3, LendingIterator, Tiles};

let arr = Array3::<f64>::zeros([NX, NY, NZ]);
// ... fill with physics data ...

let view = arr.view();
let mut tiles = Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z])
    .expect("non-zero tile shape");

println!("total tiles: {}", tiles.total_tiles());

while let Some(tile) = tiles.next() {
    // tile: ArrayView<'_, f64, 3>  — zero-copy borrow from `arr`
    let energy: f64 = tile.iter().map(|&p| p * p).sum();
}
```

## GAT advantage

Standard `Iterator` cannot yield views *borrowed from the iterator itself*:
`Item` must outlive `&mut self`.  `LendingIterator<Item<'this>>` ties the
yielded lifetime to `&'this self`, enabling the streaming loop above
without any data copies.

Once `LendingIterator` stabilizes in std (RFC 3301) this trait becomes
redundant — until then it is the correct GAT-native pattern for zero-copy
window/tile iterators.

## See also

- [`migration_gat_tiles.md`](../migration_gat_tiles.md) — full GAT tiling chapter
- [`pstd_fdtd_comparison.rs`](pstd_fdtd_comparison.md) — PSTD solver comparison using the same grid setup
