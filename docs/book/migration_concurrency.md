# Concurrency: Moirai for Parallel Execution

`moirai` replaces `rayon` and `tokio` as the Atlas concurrency runtime.

## What Was Replaced

| Before | After |
|---|---|
| `rayon::par_iter()` | `moirai::map_collect_index_with::<Adaptive, _, _>` |
| `tokio::spawn(async { … })` | `moirai` async executor |
| `(0..n).into_par_iter().map(f)` | `moirai_parallel::for_each_chunk_mut_enumerated_with` |

## Example: parallel field fill

`kwavers-medium` uses `moirai_parallel` to fill voxel arrays in parallel:

```rust
use moirai_parallel::{for_each_chunk_mut_enumerated_with, Adaptive};

pub const FIELD_CHUNK_SIZE: usize = 4096;

pub fn fill_from_function<F>(array: &mut Array3<f64>, grid: &Grid, f: F)
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    let slab = ny * nz;
    if let Some(values) = array.as_slice_mut() {
        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(
            values,
            FIELD_CHUNK_SIZE,
            |chunk_index, chunk| {
                let base = chunk_index * FIELD_CHUNK_SIZE;
                for (lane, value) in chunk.iter_mut().enumerate() {
                    let flat = base + lane;
                    let i = flat / slab;
                    // …compute x, y, z from i, j, k…
                    *value = f(x, y, z);
                }
            },
        );
    }
}
```

## Example: parallel DICOM load

`kwavers-io` uses `moirai::map_collect_index_with` for parallel slice loading:

```rust
let slice_pixels = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
    slices.len(),
    |i| decode_slice(&slices[i]),
).into_iter().collect::<Result<Vec<_>>>()?;
```

## Adaptive Scheduling

The `Adaptive` strategy chooses thread count and chunk size at runtime based
on workload size and core count — no fixed `num_threads` in call sites.
