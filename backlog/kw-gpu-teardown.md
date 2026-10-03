<a id="kw-gpu-teardown"></a>

## KW-GPU-TEARDOWN — Dissolve the remaining internal `kwavers/src` into the layered crates [arch] — todo

priority: architecture; needs: none; scope: `crates/kwavers/src`, `crates/kwavers-gpu`, `crates/kwavers-solver` GPU surfaces

- **Outcome:** `kwavers` is a thin facade. The bulk is GPU: a `kwavers-gpu` leaf owns the `ComputeBackend`/`FdtdGpuAccelerator` surfaces that stay in solver and consolidates all three scattered GPU paths, with wgpu-v26 bit-rot repaired as part of the move (user decision 2026-06-03).
- **Delivered so far:** the `kwavers-gpu` scaffold.
- **Acceptance:** no implementation left in `kwavers/src` beyond re-exports, and one owner for the GPU paths.
- **Next step:** enumerate what `crates/kwavers/src` still implements and move the GPU paths first.
