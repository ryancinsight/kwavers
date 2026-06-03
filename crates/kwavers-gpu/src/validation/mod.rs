//! GPU validation: differential CPU-vs-GPU equivalence checks.
//!
//! These live with the GPU backend (not in the algorithm crates) because they
//! exercise the concrete wgpu `GPUBackend` against a CPU reference taken from
//! `kwavers_solver`. Gated on `gpu` — the comparison needs the device.

#[cfg(feature = "gpu")]
pub mod gpu_cpu_equivalence;
