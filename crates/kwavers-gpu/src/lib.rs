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

// The consolidated GPU implementation. Gated behind `gpu` while the wgpu-v26
// repair is in progress; the default build compiles this crate to (effectively)
// empty so the workspace stays green.
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::*;
