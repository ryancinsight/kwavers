//! `kwavers-gpu` - provider-generic GPU compute backend for Kwavers.
//!
//! This crate is the single home for GPU concretions: Hephaestus-backed device
//! acquisition, provider-specific buffers and kernels, and the concrete
//! implementations of the [`kwavers_solver::backend::ComputeBackend`] and
//! `kwavers_solver::forward::fdtd::FdtdGpuAccelerator` trait surfaces.
//!
//! ## Why a separate leaf crate
//!
//! The compute-backend and FDTD-accelerator traits live in `kwavers-solver`.
//! Per the dependency-inversion rule, the algorithm crates depend only on those
//! abstractions; concrete WGPU/WGSL code lives here, downstream of solver, and
//! is injected at the application boundary. Adding CUDA means adding another
//! Hephaestus provider implementation in this crate, not changing algorithm
//! layers.
//!
//! Migration status: this crate is consolidating the three previously scattered
//! GPU code paths (the old `kwavers::gpu` facade monolith,
//! `kwavers-solver::backend::gpu`, and the `forward::{fdtd,pstd}` GPU kernels)
//! into one Hephaestus-backed provider boundary. The currently implemented
//! provider is WGPU because the production kernels are WGSL.

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

// CPU-vs-GPU differential equivalence validation, moved out of solver with the
// backend it exercises.
pub mod validation;

// GPU-resident PSTD solver, k-space corrected pseudospectral. Solver keeps only
// the CPU PSTD; GPU concretions live here.
#[cfg(feature = "gpu")]
pub mod pstd_gpu;
