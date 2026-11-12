<a id="kw-gpu-plan-capacity"></a>

## KW-GPU-PLAN-CAPACITY — A GPU PSTD request beyond device capacity has no tested outcome [patch] — todo

priority: correctness; needs: none; basis: b3b2c3a372c8ceaf6f330637383e4d2cd11c43c4; scope: `crates/kwavers-gpu/src/pstd_gpu/pipeline/`, `crates/kwavers-simulation/src/dispatch/pstd/gpu.rs`, `crates/kwavers-simulation/src/solver_adapters/gpu_pstd.rs`

- **Origin.** Split from `KW-GPU-048`, which #1053 closes on its three acceptance clauses. Its driver also asked that a consumer "receive an explicit unsupported error for the CT-scale GPU constraint"; no acceptance clause held that, and ADR 037 says "CT-scale planning still needs a per-plan allocation-capacity check".
- **Not known:** what a request returns when its field buffers exceed what the device can allocate: a typed error, a panic or a lost device. No run was made.
- **Read at the basis.** `validate_pstd_grid_shape` (`crates/kwavers-gpu/src/pstd_gpu/mod.rs`) refuses a grid whose point count does not fit `u32` shader addressing: a bound on the count, not on the bytes a device can allocate. `git grep -n -i` under `crates/kwavers-gpu/src/pstd_gpu` for `capacity`, `max_buffer_size`, `max_storage_buffer`, `too large for` and `exceeds the` finds the storage-buffer count limits (`pipeline/auto_device.rs` lines 64, 84 and 88; `tests/fixtures.rs` line 78) and host `Vec` capacities, and no comparison of field bytes with a device limit. The search is of those five patterns only.
- **Acceptance:** a request whose field size exceeds the device limit returns a typed error that names the limit and the requested size, before the run allocates field buffers; a test asserts the variant and the message; the Leto solver is not run in its place.
- **Next step (spike):** on a host with an adapter, read the acquired device's `max_buffer_size`, request a grid whose pressure field exceeds it, and record how `GpuPstdSolver::with_auto_device` and the first run end.
