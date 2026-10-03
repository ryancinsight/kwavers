<a id="atlas-kwavers-hephaestus-fdtd-107"></a>

## ATLAS-KWAVERS-HEPHAESTUS-FDTD-107 — Route collocated FDTD through Hephaestus [minor] [arch] — blocked 2026-08-18

priority: architecture; needs: none; scope: `crates/kwavers-gpu/src/{gpu,validation/gpu_cpu_equivalence}/`

- **Outcome:** delete the consumer-owned collocated raw-WGPU FDTD path and validate the real Hephaestus `Fdtd3dOps` provider against an independent f32 CPU stencil.
- **Acceptance:** provider-owned typed buffers and kernels execute velocity then pressure updates; the CPU oracle uses the same contract in native f32; provider acquisition and dispatch failures stay explicit; no CPU fallback or consumer-owned collocated FDTD shader remains; focused Nextest, feature-enabled check/Clippy, doctests and the Hephaestus provider contract pass.
- **State:** implementation complete on the consumer side (ryancinsight/hephaestus#213 added the typed contract and sequential-step differential coverage; the consumer exact-head benchmark regression passes). The final matrix stopped at Cargo resolution: Kwavers `main` required Apollo `0.27.0` while Apollo's default was `0.26.0`.
- **Re-open trigger:** Apollo `0.27.0` on Apollo's default branch; do not add a fallback to Apollo `0.26.0`. The trigger may already have fired: re-validate first.
- **Residuals:** the separate pressure-only `gpu::compute::fdtd_gpu` dispatcher and the disconnected f64 solver accelerator seam stay tracked boundaries; neither is evidence for this provider integration.
