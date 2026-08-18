# Example: Tiled K-Space Processing

**Source**: `crates/kwavers/examples/tiled_kspace_processing.rs`

## Overview

Demonstrates `leto::Tiles<f64, 3>` for zero-copy, cache-blocked processing of
a 3-D PSTD pressure field.

## Physics motivation

PSTD pressure fields for production grids (e.g. 256³) do not fit in L2 cache.
Processing the field tile-by-tile keeps each working set inside cache and
enables compiler auto-vectorization per tile.

The example initializes a Gaussian pressure pulse on a 32³ grid, then walks
it in 8³ tiles, computing per-tile acoustic energy.

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
Tiled k-space processing demo (Tiles from leto)
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

Tiles: zero-copy tiling, no element copied.
```

## Key API

```rust
use leto::{Array3, Tiles};

let arr = Array3::<f64>::zeros([NX, NY, NZ]);
// ... fill with physics data ...

let view = arr.view();
let mut tiles = Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z])
    .expect("non-zero tile shape");

println!("total tiles: {}", tiles.total_tiles());

for tile in &mut tiles {
    // tile: ArrayView<'_, f64, 3>  — zero-copy borrow from `arr`
    let energy: f64 = tile.iter().map(|&p| p * p).sum();
}
```

## Why a plain `Iterator`

A tile borrows the *parent slice* for `'a`, not the `Tiles` value itself, so
its item lifetime is independent of `&mut self` and no GAT is required.
Declaring it as a plain [`Iterator`] is what earns `for` loops,
`zip`/`enumerate`/`rev`, `ExactSizeIterator::len` and the parallel bridges; a
streaming-iterator signature would forfeit all of them for no capability.
`Tiles` additionally implements `DoubleEndedIterator` and
`ExactSizeIterator` — `len()` is exact because `Tiles::new` rejects a layout
addressing outside its data, so iteration can never terminate early.

## See also

- [`pstd_fdtd_comparison.rs`](pstd_fdtd_comparison.md) — PSTD solver comparison using the same grid setup
