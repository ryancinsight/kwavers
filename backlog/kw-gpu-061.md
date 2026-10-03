<a id="kw-gpu-061"></a>

## KW-GPU-061 — Extend GPU PSTD FFT lattice [minor] — in-progress

priority: feature; needs: none; scope: `crates/kwavers-gpu/src/pstd_gpu/`, GPU PSTD consumer validation in `kwavers-simulation`

- **Acceptance:** the Hephaestus-acquired WGPU PSTD provider accepts every power-of-two axis through 1,024, rejects 2,048 before allocation, and keeps the final-field readback contract intact. The shader declares at most 12 KiB of workgroup storage and the acquisition contract requires that amount explicitly.
- **Evidence target:** value-semantic dimension contracts, a shader/host ABI regression, GPU-feature Nextest, and a consumer gate. KW-GPU-048 records per-axis FFT support reaching 1,024; whole-grid provider capacity remains a per-plan constraint.
- **Next step:** re-validate the axis limit and the 12 KiB workgroup declaration at the default branch.
