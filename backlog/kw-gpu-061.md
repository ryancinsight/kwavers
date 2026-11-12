<a id="kw-gpu-061"></a>

## KW-GPU-061 — Extend GPU PSTD FFT lattice [minor] — todo

priority: feature; needs: none; scope: `crates/kwavers-gpu/src/pstd_gpu/`, GPU PSTD consumer validation in `kwavers-simulation`

- **Acceptance:** the Hephaestus-acquired WGPU PSTD provider accepts every power-of-two axis through 1,024, rejects 2,048 before allocation, and keeps the final-field readback contract intact. The shader declares at most 12 KiB of workgroup storage and the acquisition contract requires that amount explicitly.
- **Evidence target:** value-semantic dimension contracts, a shader/host ABI regression, GPU-feature Nextest, and a consumer gate. Construction refuses a plan whose packed PML or shift buffer exceeds a buffer limit of the device (`crates/kwavers-gpu/src/pstd_gpu/pipeline/capacity.rs`).
- **Next step:** re-validate the axis limit and the 12 KiB workgroup declaration at the default branch. `accepts_axis_exceeding_retired_local_fft_limit` (`crates/kwavers-simulation/src/solver_adapters/gpu_pstd/tests.rs`) constructs the adapter for a 2,048-point axis and runs no batch; the acceptance's "rejects 2,048 before allocation" is to be reconciled with it.
