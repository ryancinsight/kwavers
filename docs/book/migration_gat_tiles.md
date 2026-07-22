# Chapter 43 — GAT Tiling: LendingIterator and Tiles

## Motivation

The standard `Iterator` trait requires `Item` to have a lifetime independent of
`&self`:

```rust
trait Iterator {
    type Item;         // lifetime unrelated to Self
    fn next(&mut self) -> Option<Self::Item>;
}
```

This prevents returning *views borrowed from the iterator itself* — the
streaming iterator problem.  For a tile iterator over a 3-D pressure field,
we want each yielded `ArrayView<'_, T, 3>` to borrow the backing slice for
exactly one iteration, then be discarded.  Standard `Iterator` cannot express
this.

## LendingIterator (GAT-based)

`leto::LendingIterator` uses a Generic Associated Type (GAT) to tie the item
lifetime to `&mut self`:

```rust
pub trait LendingIterator {
    type Item<'this> where Self: 'this;
    fn next(&mut self) -> Option<Self::Item<'_>>;
}
```

The driver loop is a `while let` (standard `for` is not supported because `for`
requires `IntoIterator` / std `Iterator`):

```rust
while let Some(item) = iter.next() {
    // item lifetime: tied to this iteration
}
```

## Tiles — non-overlapping rectangular tiles

`leto::Tiles<'a, T, const N: usize>` implements `LendingIterator` and partitions
an N-dimensional array into non-overlapping tiles of a fixed shape:

| Array shape | Tile shape | Tile grid | Notes |
|---|---|---|---|
| `[32, 32, 32]` | `[8, 8, 8]` | `[4, 4, 4]` = 64 tiles | exact partition |
| `[10, 10]` | `[3, 3]` | `[4, 4]` = 16 tiles | boundary tiles clipped |

The last tile along each axis is automatically clipped when the array size is
not divisible by the tile size.

### Construction

```rust
use leto::{Array3, LendingIterator, Tiles};

let arr = Array3::<f64>::zeros([NX, NY, NZ]);
let view = arr.view();

let mut tiles = Tiles::new(view.data(), view.layout(), [TILE_X, TILE_Y, TILE_Z])
    .expect("tile shape must be non-zero");

println!("total tiles: {}", tiles.total_tiles());
```

`Tiles::new` returns `None` when any `tile_shape[i] == 0` — a defensive
constructor consistent with the zero-cost abstraction principle.

### Zero-copy usage

```rust
let mut total_energy = 0.0_f64;
while let Some(tile) = tiles.next() {
    // tile: ArrayView<'_, f64, 3>
    // — a zero-copy view into `arr`; no elements are copied
    total_energy += tile.iter().map(|&p| p * p).sum::<f64>();
}
```

## Cache-blocking pattern

For a large production grid (256³ at f64 = 128 MiB), a tile of shape
`[16, 16, 16]` uses 32 kB — fitting comfortably in L2 cache.  Iterating
tile-by-tile minimizes cache evictions during field traversal.

```text
PSTD inner loop (old — L3 bound for large grids)
  for i in 0..NX { for j in 0..NY { for k in 0..NZ { process(p[i,j,k]); } } }

PSTD inner loop (new — L2 resident via Tiles)
  while let Some(tile) = tiles.next() {
      for &p in tile.iter() { process(p); }
  }
```

## What changed from ndarray

| Before (`ndarray::windows`) | After (`leto::Tiles`) |
|---|---|
| `Array3::windows([w,w,w])` | `Tiles::new(…, [t,t,t])` |
| `Item = ArrayView3<'_, f64>` but overlapping | non-overlapping, zero remainder |
| standard `Iterator` (item lifetime `'a`) | GAT `LendingIterator` (item lifetime `'this`) |
| requires heap clone for borrowed view | zero-copy, `ArrayView<'_, T, N>` |

## Stability note

Once `LendingIterator` stabilizes in the standard library (RFC 3301), this
trait will become an alias for `std::iter::LendingIterator`.  Until then, the
`leto` crate re-exports its own definition so downstream crates compile on
stable Rust.

## Example

See [`tiled_kspace_processing.rs`](examples/tiled_kspace_processing.md) for a
complete working example including a theorem-level correctness proof.
