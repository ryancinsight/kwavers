<a id="kw-gpu-061"></a>

## KW-GPU-061 — Extend GPU PSTD FFT lattice [minor] — todo

priority: feature; needs: none; scope: `crates/kwavers-gpu/src/pstd_gpu/`, GPU PSTD consumer validation in `kwavers-simulation`

- **Acceptance:** the Hephaestus-acquired WGPU PSTD provider accepts every power-of-two axis through 1,024, rejects 2,048 before allocation, and keeps the final-field readback contract intact. The shader declares at most 12 KiB of workgroup storage and the acquisition contract requires that amount explicitly.
- **Evidence target:** value-semantic dimension contracts, a shader/host ABI regression, GPU-feature Nextest, and a consumer gate. The adapter tests accept FFT axes of 1,024 and 2,048 points (`crates/kwavers-simulation/src/solver_adapters/gpu_pstd/tests.rs`); whole-grid provider capacity is `KW-GPU-PLAN-CAPACITY`.
- **Next step:** re-validate the axis limit and the 12 KiB workgroup declaration at the default branch.
