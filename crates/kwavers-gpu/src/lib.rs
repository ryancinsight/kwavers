#![doc = include_str!("../README.md")]
#![allow(clippy::module_inception)]

// GPU allocation profiling/tracking. Pure bookkeeping over the ungated
// kwavers_core GpuError; no wgpu dependency, so it is available unconditionally.
pub mod profiling;

// The consolidated GPU implementation: kernels, buffers, and devices.
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::*;

// Concrete ComputeBackend implementation. Solver owns only the trait; this leaf
// crate owns provider-specific implementations.
#[cfg(feature = "gpu")]
pub mod backend;

// Provider implementations for beamforming algorithms whose operation traits
// remain in kwavers-analysis.
#[cfg(feature = "gpu")]
pub mod beamforming;

// Provider-owned visualization transfer: concrete WGPU device/buffer/queue
// ownership implementing kwavers-analysis' provider-neutral seam.
#[cfg(feature = "visualization")]
pub mod visualization;

// CPU-vs-GPU differential equivalence validation, moved out of solver with the
// backend it exercises.
pub mod validation;

// GPU-resident PSTD solver, k-space corrected pseudospectral. Solver keeps only
// the CPU PSTD; GPU concretions live here.
#[cfg(feature = "gpu")]
pub mod pstd_gpu;
