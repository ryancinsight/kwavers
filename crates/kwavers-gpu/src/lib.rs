//! `kwavers-gpu` — wgpu-backed compute backend for kwavers.
//!
//! This crate is the single home for all GPU concretions: the wgpu device and
//! buffer management, the WGSL compute kernels, and the concrete implementations
//! of the [`kwavers_solver::backend::ComputeBackend`] and
//! `kwavers_solver::forward::fdtd::FdtdGpuAccelerator` trait surfaces.
//!
//! ## Why a separate leaf crate
//!
//! The compute-backend and FDTD-accelerator *traits* live in `kwavers-solver`.
//! Per the dependency-inversion rule, the algorithm crates depend only on those
//! abstractions; the concrete `wgpu`/WGSL code lives here, downstream of solver,
//! and is injected at the application boundary. Adding a new device target means
//! a new `impl` in this crate — never a change in the algorithm layers.
//!
//! Migration status: this crate is being populated by consolidating the three
//! previously-scattered GPU code paths (the old `kwavers::gpu` facade monolith,
//! `kwavers-solver::backend::gpu`, and the `forward::{fdtd,pstd}` GPU kernels)
//! into one repaired, wgpu-v26-correct backend.

#![allow(clippy::module_inception)]

// GPU allocation profiling/tracking. Pure bookkeeping over the (ungated)
// kwavers_core GpuError; no wgpu dependency, so it is available unconditionally.
pub mod profiling;

// The consolidated GPU implementation (kernels, buffers, devices). Gated on `gpu`.
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::*;

// Concrete `ComputeBackend` implementation (the wgpu `GPUBackend`), consolidated
// out of `kwavers_solver::backend::gpu` so the algorithm crate holds only the
// trait. Implements `kwavers_solver::backend::ComputeBackend`.
#[cfg(feature = "gpu")]
pub mod backend;

// CPU-vs-GPU differential equivalence validation (moved out of solver with the
// backend it exercises).
pub mod validation;
